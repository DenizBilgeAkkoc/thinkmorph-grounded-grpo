"""Experimental full-causal text scoring with consistent old/new arithmetic.

This scorer is not numerically identical to native incremental sampling. Both
old-policy snapshots and gradient-enabled scores MUST use this implementation,
the same detached prefixes, exact tokens, lane order and checkpoint_chunk_size.
The caller retains the strict archived-policy identity guard and takes the old
snapshot before any optimizer update. This module never updates model weights.

Normal attach_lora initialization supplies native differentiable attention and
the existing replay-only decoder-layer checkpoint wrapper. Imports here do not
load native model classes before runtime.load_native selects their checkout.
"""
import torch
import torch.nn.functional as F

from .paired_replay import LAYOUT_REVISION, _autocast, _pack_prefixes, _validate


PARALLEL_REPLAY_REVISION = "parallel_fixed_prefix_text_v1"
DEFAULT_CHECKPOINT_CHUNK_SIZE = 128


def parallel_token_logprobs(model, contexts, events, *, active_lane_schedule,
                           bos_id=None, layout_revision=LAYOUT_REVISION,
                           checkpoint_chunk_size=DEFAULT_CHECKPOINT_CHUNK_SIZE):
    """Return one float64 token-score tensor per original paired lane.

    Natural EOS is scored. For a cap, only the retained input/target pairs are
    evaluated; the final discarded draw contributes no query or likelihood.
    Validation still requires the complete recorded draw schedule. A cap of
    one yields an empty score tensor and no transformer call for that lane.

    Positive checkpoint_chunk_size controls LM-head chunk shape and checkpoint
    recomputation. Zero uses one whole-lane head call without head checkpointing.
    The same value is required for old and new scores, including no-grad calls.
    Existing decoder-layer checkpointing is left unchanged.
    """
    from torch.utils.checkpoint import checkpoint

    if type(checkpoint_chunk_size) is not int or checkpoint_chunk_size < 0:
        raise ValueError("checkpoint_chunk_size must be a nonnegative integer")
    _, temperature = _validate(model, contexts, events, active_lane_schedule, bos_id, layout_revision)
    device = next(model.parameters()).device
    empty = lambda: torch.empty(0, device=device, dtype=torch.float64,
                                requires_grad=torch.is_grad_enabled())
    scores = [empty() for _ in events]
    active = [lane for lane, event in enumerate(events) if event["target_ids"]]
    if not active:
        return scores

    # Packing only nonempty scored lanes is deterministic from the saved action,
    # and is therefore identical for no-grad old and gradient-enabled new calls.
    prefix = _pack_prefixes([contexts[lane] for lane in active])
    inputs, positions, queries, keys, qlens, klens = [], [], [], [], [], []
    offset = 0
    for lane in active:
        event, prior = events[lane], contexts[lane]
        n, length, position = len(event["input_ids"]), int(prior["kv_lens"][0]), int(prior["ropes"][0])
        if length < 0 or position < 0:
            raise ValueError("Negative prefix length or RoPE position")
        inputs.extend(event["input_ids"])
        positions.extend(range(position, position + n))
        keys.extend(range(offset, offset + length))
        queries.extend(range(offset + length, offset + length + n))
        qlens.append(n); klens.append(length); offset += length + n

    def tensor(values, dtype=torch.long):
        return torch.tensor(values, device=device, dtype=dtype)

    with _autocast(model):
        output = model.language_model.forward_inference(
            packed_query_sequence=model.language_model.model.embed_tokens(tensor(inputs)),
            query_lens=tensor(qlens, torch.int), packed_query_position_ids=tensor(positions),
            packed_query_indexes=tensor(queries), past_key_values=prefix,
            key_values_lens=tensor(klens, torch.int), packed_key_value_indexes=tensor(keys),
            update_past_key_values=False, is_causal=True,
            **({"mode": "und"} if model.use_moe else {}),
        )
        # Binding temperature here also makes checkpoint recomputation immune
        # to any later loop-variable rebinding.
        def chunk_score(hidden, targets, temperature=temperature):
            logits = model.language_model.lm_head(hidden).float() / temperature
            return F.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).flatten()

        offset = 0
        for lane, n in zip(active, qlens):
            hidden = output.packed_query_sequence[offset:offset + n]
            targets = tensor(events[lane]["target_ids"])
            width = checkpoint_chunk_size or n
            pieces = []
            for start in range(0, n, width):
                args = (hidden[start:start + width], targets[start:start + width])
                part = (checkpoint(chunk_score, *args, use_reentrant=False)
                        if torch.is_grad_enabled() and checkpoint_chunk_size else chunk_score(*args))
                pieces.append(part.double())
            scores[lane] = torch.cat(pieces)
            offset += n
    return scores


@torch.no_grad()
def snapshot_parallel_token_logprobs(model, contexts, events, *, active_lane_schedule,
                                    bos_id=None, layout_revision=LAYOUT_REVISION,
                                    checkpoint_chunk_size=DEFAULT_CHECKPOINT_CHUNK_SIZE):
    """Take detached old-policy values using the identical full-causal scorer.

    The returned clones cannot retain a computation graph or share score
    storage with a later forward. They are NOT recorded native sampler scores;
    the trainer must label their source as parallel_old_policy_snapshot.
    """
    return [score.detach().clone() for score in parallel_token_logprobs(
        model, contexts, events, active_lane_schedule=active_lane_schedule,
        bos_id=bos_id, layout_revision=layout_revision,
        checkpoint_chunk_size=checkpoint_chunk_size,
    )]
