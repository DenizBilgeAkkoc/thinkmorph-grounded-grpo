"""Paired native text sampling with independent per-rollout RNG streams.

Only prefix caches built by the singleton native path are accepted. Images,
VAE/SDE work, production traces and the live A100 run are outside this module.
CPU tests alone do not certify CUDA likelihood parity or backward gradients.
"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import math

import torch
import torch.nn.functional as F

from .paired_packing import pack_contexts, insertion_indexes, compact_cache, detached_cache


LAYOUT_REVISION = 'packed_active_text_v1'


class LaneRandomStream:
    """One rollout's complete RNG state, used by one orchestrator thread only.

    Singleton VAE/image operations run inside scope(). Text sampling receives a
    separate Torch generator; adopt_generator() writes its progressed state back
    into this lane without changing any other lane or the process RNG state.
    """
    def __init__(self, state):
        self.state = deepcopy(state)
        self.active = False

    @classmethod
    def from_seed(cls, seed):
        from .randomness import capture_rng_state, restore_rng_state, seed_everything
        if type(seed) is not int:
            raise ValueError('A lane seed must be an integer')
        caller = capture_rng_state()
        try:
            seed_everything(seed)
            return cls(capture_rng_state())
        finally:
            restore_rng_state(caller)

    @contextmanager
    def scope(self):
        from .randomness import capture_rng_state, restore_rng_state
        if self.active:
            raise RuntimeError('The same lane RNG scope cannot be nested')
        caller = capture_rng_state()
        entered = False
        try:
            restore_rng_state(self.state)
            self.active = entered = True
            yield
        finally:
            try:
                if entered:
                    self.state = capture_rng_state()
            finally:
                self.active = False
                restore_rng_state(caller)

    def generator(self, device):
        if self.active:
            raise RuntimeError('Create the text generator outside the lane RNG scope')
        device = torch.device(device)
        generator = torch.Generator(device=device)
        if device.type == 'cpu':
            state = self.state['torch']
        elif device.type == 'cuda':
            index = device.index if device.index is not None else torch.cuda.current_device()
            if index >= len(self.state['cuda']):
                raise ValueError('Lane CUDA RNG topology differs from the sampling device')
            state = self.state['cuda'][index]
        else:
            raise ValueError('Only native CPU/CUDA RNG streams are supported')
        generator.set_state(state)
        return generator

    def adopt_generator(self, generator):
        if self.active:
            raise RuntimeError('Adopt the text generator outside the lane RNG scope')
        device = generator.device
        if device.type == 'cpu':
            self.state['torch'] = generator.get_state().clone()
        elif device.type == 'cuda':
            index = device.index if device.index is not None else torch.cuda.current_device()
            if index >= len(self.state['cuda']):
                raise ValueError('Lane CUDA RNG topology differs from the sampling device')
            self.state['cuda'][index] = generator.get_state().clone()
        else:
            raise ValueError('Only native CPU/CUDA RNG streams are supported')

    def snapshot(self):
        return deepcopy(self.state)

    def restore_process(self):
        """Explicitly leave the final logical lane's state after a completed pair."""
        from .randomness import restore_rng_state
        if self.active:
            raise RuntimeError('Cannot publish the process RNG inside a lane scope')
        restore_rng_state(self.state)


def _lane_result(index, draws, logps, *, bos_id, eos_id, max_length, temperature):
    # This deliberately matches the existing recorder's EOS-at-cap rule.
    ended = draws[-1] == eos_id and len(draws) < max_length
    targets = list(draws if ended else draws[:-1])
    retained_logps = list(logps if ended else logps[:-1])
    return dict(lane_index=index, draw_ids=list(draws), draw_logps=list(logps),
                input_ids=([bos_id] + targets[:-1]) if targets else [],
                target_ids=targets, temperature=temperature,
                sampling_token_logps=retained_logps,
                sampling_logp=sum(retained_logps), sampled_tokens=len(draws),
                stop='eos' if ended else 'token_limit',
                decoded_token_ids=[bos_id] + list(draws[:-1]))


@torch.no_grad()
def sample_paired_text(model, priors, *, bos_id, eos_id, max_length, temperature,
                       generators, progress=None):
    """Sample 1–2 text lanes with independent draws and actual behavior logps.

    All forwards, including final cap-only unscored draws, enter the returned
    active_lane_schedule. The caller owns generator states and text decoding.
    Cached prefixes are detached and never mutated. This no-grad sampler makes
    no claim that existing scalar replay can validate its packed likelihoods.
    """
    if len(priors) not in (1, 2) or len(generators) != len(priors):
        raise ValueError('Expected one or two singleton priors and one generator per lane')
    if type(max_length) is not int or max_length < 1:
        raise ValueError('max_length must be a positive integer')
    if any(type(value) is not int or value < 0 for value in (bos_id, eos_id)):
        raise ValueError('BOS/EOS IDs must be nonnegative integers')
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Expected one positive finite native temperature')
    if model.training:
        raise ValueError('Paired sampling requires the evaluation-mode native policy')
    if (any(not isinstance(g, torch.Generator) for g in generators)
            or len({id(g) for g in generators}) != len(generators)):
        raise ValueError('Each lane needs its own independent Torch generator')
    device = next(model.parameters()).device
    if any(g.device != device for g in generators):
        raise ValueError('Generator device must match the native policy sampling device')
    context = pack_contexts(priors)
    packed = model.prepare_start_tokens(context['kv_lens'], context['ropes'], {'bos_token_id': bos_id})
    tokens, positions, lengths = (packed['packed_start_tokens'],
        packed['packed_query_position_ids'], packed['key_values_lens'])
    cache = detached_cache(context['past_key_values'])
    active = list(range(len(priors)))
    draws, logps = [[] for _ in priors], [[] for _ in priors]
    schedule = []
    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with autocast:
        for step in range(max_length):
            schedule.append(list(active))
            queries, keys = insertion_indexes(lengths)
            embeddings = model.language_model.model.embed_tokens(tokens)
            output = model.language_model.forward_inference(
                packed_query_sequence=embeddings, query_lens=torch.ones_like(tokens),
                packed_query_position_ids=positions, packed_query_indexes=queries,
                past_key_values=cache, key_values_lens=lengths,
                packed_key_value_indexes=keys, update_past_key_values=True,
                is_causal=True, **({'mode': 'und'} if model.use_moe else {}))
            logits = model.language_model.lm_head(output.packed_query_sequence)
            raw = F.softmax(logits / temperature, dim=-1)
            if raw.ndim != 2 or raw.shape[0] != len(active):
                raise ValueError('Native text logits do not match active lanes')
            sampled, selected_logps = [], []
            for row, lane in enumerate(active):
                # Keep the native singleton multinomial shape and random stream.
                probabilities = raw[row:row + 1]
                token = torch.multinomial(probabilities, num_samples=1, generator=generators[lane])
                distribution = probabilities.detach().double()
                distribution = distribution / distribution.sum(dim=-1, keepdim=True)
                selected_logps.append(distribution.gather(-1, token).log().sum())
                sampled.append(token.reshape(()))
            token_values = torch.stack(sampled).cpu().tolist()
            score_values = torch.stack(selected_logps).cpu().tolist()
            for row, lane in enumerate(active):
                if not math.isfinite(score_values[row]):
                    raise FloatingPointError('Nonfinite sampled-token likelihood')
                draws[lane].append(token_values[row])
                logps[lane].append(score_values[row])
            cache = detached_cache(output.past_key_values)
            lengths, positions = lengths + 1, positions + 1
            keep = [row for row, lane in enumerate(active)
                    if token_values[row] != eos_id and step + 1 < max_length]
            if progress is not None:
                progress(dict(step=step, active_lanes=list(active), sampled_tokens=list(token_values)))
            if not keep:
                break
            if len(keep) != len(active):
                cache = compact_cache(cache, lengths, keep)
                lengths, positions = lengths[keep], positions[keep]
            tokens = torch.stack([sampled[row] for row in keep])
            active = [active[row] for row in keep]
    return dict(layout_revision=LAYOUT_REVISION, active_lane_schedule=schedule,
                lanes=[_lane_result(index, values, logps[index], bos_id=bos_id, eos_id=eos_id,
                                   max_length=max_length, temperature=temperature)
                       for index, values in enumerate(draws)],
                model_forward_count=len(schedule), total_sampled_tokens=sum(map(len, draws)))
