"""Matched-packing native text replay with action-local KV gradients.

Matches packed_active_text_v1. Prefix KV is detached once; action KV remains
differentiable through every incremental step and lane compaction. The final
cap-only draw stays in the execution schedule but has no likelihood/loss term.
Nested token checkpoints bound the growing KV intermediates retained during
one layer's backward recomputation. Chunk boundaries carry differentiable KV
as explicit inputs and outputs; they never truncate the action's gradient.
"""
from contextlib import nullcontext
from copy import copy
import math

import torch
import torch.nn.functional as F


LAYOUT_REVISION = "packed_active_text_v1"
DEFAULT_CHECKPOINT_CHUNK_SIZE = 128


def _autocast(model):
    device = next(model.parameters()).device
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _validate(model, priors, events, schedule, bos_id, layout_revision):
    if layout_revision != LAYOUT_REVISION:
        raise ValueError("Unsupported paired replay layout revision")
    if not 1 <= len(events) <= 8 or len(priors) != len(events):
        raise ValueError("Paired replay requires one to eight matched priors/events")
    if model.training:
        raise ValueError("Paired replay requires eval mode and zero dropout")
    decoder = model.language_model.model
    if getattr(decoder, "enable_taylorseer", False) or any(
            getattr(layer, "enable_taylorseer", False) for layer in decoder.layers):
        raise ValueError("Paired replay forbids TaylorSeer")
    temperatures = [event["temperature"] for event in events]
    if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 for t in temperatures):
        raise ValueError("Sampling temperature must be finite and positive")
    if len(set(temperatures)) != 1:
        raise ValueError("Native paired replay requires a shared scalar temperature")
    candidate_bos = {event["input_ids"][0] for event in events if event["input_ids"]}
    if bos_id is None:
        if len(candidate_bos) != 1:
            raise ValueError("Explicit bos_id required for an empty capped action")
        bos_id = next(iter(candidate_bos))
    if type(bos_id) is not int or any(value != bos_id for value in candidate_bos):
        raise ValueError("Paired BOS tokens differ")
    counts = []
    for event in events:
        inputs, targets = event["input_ids"], event["target_ids"]
        if not isinstance(inputs, list) or not isinstance(targets, list):
            raise ValueError("Recorded exact-token actions must be lists")
        if any(type(token) is not int or token < 0 for token in inputs + targets):
            raise ValueError("Invalid recorded token ID")
        expected_inputs = [bos_id] + targets[:-1] if targets else []
        if inputs != expected_inputs:
            raise ValueError("Invalid exact-token input/target shift")
        count, stop = event.get("sampled_tokens"), event.get("stop")
        if type(count) is not int or count < 1 or stop not in ("eos", "token_limit"):
            raise ValueError("Missing draw count or stopping convention")
        if len(targets) != count - int(stop == "token_limit"):
            raise ValueError("Target count disagrees with EOS/cap draw count")
        draws = event.get("draw_ids")
        if draws is not None and (len(draws) != count or draws[:len(targets)] != targets):
            raise ValueError("Recorded draws disagree with retained targets")
        counts.append(count)
    expected_schedule = [[lane for lane, count in enumerate(counts) if step < count]
                         for step in range(max(counts))]
    if schedule != expected_schedule:
        raise ValueError("Active-lane schedule must include every draw, including the final unscored cap draw")
    for prior in priors:
        if len(prior["kv_lens"]) != 1 or len(prior["ropes"]) != 1:
            raise ValueError("Replay prefixes must be singleton contexts")
    return bos_id, temperatures[0]


def _pack_prefixes(priors):
    cache = copy(priors[0]["past_key_values"])
    for field in ("key_cache", "value_cache"):
        source = getattr(priors[0]["past_key_values"], field)
        keys = set(source)
        if any(set(getattr(prior["past_key_values"], field)) != keys for prior in priors):
            raise ValueError("Prefix cache layer sets differ")
        packed = {}
        for layer in source:
            values = [getattr(prior["past_key_values"], field)[layer] for prior in priors]
            if all(value is None for value in values):
                if any(prior["kv_lens"][0] != 0 for prior in priors):
                    raise ValueError("Nonempty prefix has no KV tensor")
                packed[layer] = None
                continue
            exemplar = next(value for value in values if value is not None)
            tensors = []
            for prior, value in zip(priors, values):
                if value is None:
                    if prior["kv_lens"][0] != 0:
                        raise ValueError("Nonempty prefix has no KV tensor")
                    value = exemplar[:0]
                if value.shape[0] != prior["kv_lens"][0] or value.shape[1:] != exemplar.shape[1:]:
                    raise ValueError("Prefix length/shape does not match cache storage")
                tensors.append(value.detach())
            packed[layer] = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        setattr(cache, field, packed)
    return cache


def _copy_cache(cache):
    result = copy(cache)
    result.key_cache = dict(cache.key_cache)
    result.value_cache = dict(cache.value_cache)
    return result


def _compact(cache, lengths, keep, *, layer_index=None):
    """Keep graph edges to prior action tokens; only initial prefixes detach."""
    result = _copy_cache(cache)
    layers = list(cache.key_cache) if layer_index is None else [layer_index]
    for field in ("key_cache", "value_cache"):
        target = getattr(result, field)
        for layer in layers:
            value = target[layer]
            if value is None:
                raise ValueError("Cannot compact an uninitialized action cache")
            pieces = value.split(lengths)
            target[layer] = torch.cat([pieces[index] for index in keep], dim=0)
    return result


def _plan(model, priors, events, schedule, bos_id):
    """Keep native CPU index dtypes and GPU target-token inputs after step zero."""
    device = next(model.parameters()).device
    packed = model.prepare_start_tokens([p["kv_lens"][0] for p in priors],
        [p["ropes"][0] for p in priors], {"bos_token_id": bos_id})
    lengths, positions = packed["key_values_lens"], packed["packed_query_position_ids"]
    targets = [torch.tensor(event["target_ids"], device=device, dtype=torch.long) for event in events]
    plan, offset = [], 0
    for step, lanes in enumerate(schedule):
        tokens = packed["packed_start_tokens"] if step == 0 else torch.stack([targets[lane][step - 1] for lane in lanes])
        old_indexes = torch.arange(int(lengths.sum()), device=lengths.device, dtype=torch.long)
        segments = old_indexes.split(lengths.tolist())
        keys = torch.cat([part + lane for lane, part in enumerate(segments)])
        queries = torch.cumsum(lengths, dim=0) + torch.arange(len(lanes), device=lengths.device, dtype=lengths.dtype)
        after = lengths + 1
        next_lanes = schedule[step + 1] if step + 1 < len(schedule) else lanes
        keep = [lanes.index(lane) for lane in next_lanes]
        plan.append(dict(step=step, lanes=list(lanes), tokens=tokens, positions=positions,
            lengths=lengths, queries=queries, keys=keys, offset=offset,
            lengths_after=after.tolist(), keep=keep, compact=next_lanes != lanes))
        offset += len(lanes)
        lengths, positions = after[keep], (positions + 1)[keep]
    return plan, targets


def _finish_scores(rows, events, model):
    device = next(model.parameters()).device
    return [torch.stack(row) if row else torch.empty(0, dtype=torch.float64, device=device,
            requires_grad=torch.is_grad_enabled()) for row in rows]


def _collect_scores(rows, distribution, settings, targets, events):
    for index, lane in enumerate(settings["lanes"]):
        step = settings["step"]
        if step < len(events[lane]["target_ids"]):
            rows[lane].append(distribution[index, targets[lane][step]].log())


def paired_token_logprobs_reference(model, priors, events, *, active_lane_schedule,
                                    bos_id=None, layout_revision=LAYOUT_REVISION):
    """Token-major native reference retaining the full graph; use short probes."""
    bos_id, temperature = _validate(model, priors, events, active_lane_schedule, bos_id, layout_revision)
    prefix = _pack_prefixes(priors)
    cache = _copy_cache(prefix)
    plan, targets = _plan(model, priors, events, active_lane_schedule, bos_id)
    rows = [[] for _ in events]
    with _autocast(model):
        for settings in plan:
            hidden = model.language_model.model.embed_tokens(settings["tokens"])
            output = model.language_model.forward_inference(packed_query_sequence=hidden,
                query_lens=torch.ones_like(settings["tokens"]), packed_query_position_ids=settings["positions"],
                packed_query_indexes=settings["queries"], past_key_values=cache,
                key_values_lens=settings["lengths"], packed_key_value_indexes=settings["keys"],
                update_past_key_values=True, is_causal=True, **({"mode": "und"} if model.use_moe else {}))
            cache = output.past_key_values
            logits = model.language_model.lm_head(output.packed_query_sequence)
            distribution = F.softmax(logits / temperature, dim=-1).double()
            distribution = distribution / distribution.sum(dim=-1, keepdim=True)
            _collect_scores(rows, distribution, settings, targets, events)
            if settings["compact"]:
                cache = _compact(cache, settings["lengths_after"], settings["keep"])
    return _finish_scores(rows, events, model)


def paired_token_logprobs(model, priors, events, *, active_lane_schedule,
                          bos_id=None, checkpoint_layers=True,
                          checkpoint_chunk_size=DEFAULT_CHECKPOINT_CHUNK_SIZE,
                          layout_revision=LAYOUT_REVISION):
    """Layerwise causal replay with the sampler's packed shape at EVERY step."""
    from torch.utils.checkpoint import checkpoint
    if type(checkpoint_chunk_size) is not int or checkpoint_chunk_size < 0:
        raise ValueError("checkpoint_chunk_size must be a nonnegative integer")
    bos_id, temperature = _validate(model, priors, events, active_lane_schedule, bos_id, layout_revision)
    prefix = _pack_prefixes(priors)
    plan, targets = _plan(model, priors, events, active_lane_schedule, bos_id)
    decoder = model.language_model.model
    with _autocast(model):
        embeddings = [decoder.embed_tokens(settings["tokens"]) for settings in plan]
        metadata = []
        for hidden, settings in zip(embeddings, plan):
            cos, sin = decoder.rotary_emb(hidden, settings["positions"].to(hidden.device).unsqueeze(0))
            metadata.append(dict(query_lens=torch.ones_like(settings["tokens"]).to(hidden.device),
                packed_query_position_embeddings=(cos.squeeze(0), sin.squeeze(0)),
                packed_query_indexes=settings["queries"].to(hidden.device),
                key_values_lens=settings["lengths"].to(hidden.device),
                packed_key_value_indexes=settings["keys"].to(hidden.device)))
        sequence = torch.cat(embeddings, dim=0)

        def layer_function(layer, layer_index):
            def chunk_function(first, last):
                # Bind the exact step range in a separate function invocation:
                # backward may recompute this after the outer loop has advanced.
                chunk_plan = plan[first:last]
                chunk_metadata = metadata[first:last]
                offset = chunk_plan[0]["offset"]

                def run_chunk(hidden, initial_key, initial_value):
                    cache = _copy_cache(prefix)
                    cache.key_cache[layer_index] = initial_key
                    cache.value_cache[layer_index] = initial_value
                    rows = []
                    for settings, kwargs in zip(chunk_plan, chunk_metadata):
                        begin = settings["offset"] - offset
                        width = len(settings["lanes"])
                        row, cache = layer(packed_query_sequence=hidden[begin:begin + width],
                            past_key_values=cache, update_past_key_values=True, is_causal=True,
                            **kwargs, **({"mode": "und"} if model.use_moe else {}))
                        rows.append(row)
                        if settings["compact"]:
                            cache = _compact(cache, settings["lengths_after"], settings["keep"],
                                layer_index=layer_index)
                    return (torch.cat(rows, dim=0), cache.key_cache[layer_index],
                            cache.value_cache[layer_index])
                return run_chunk

            def run(hidden):
                use_chunks = (torch.is_grad_enabled() and checkpoint_layers and
                              checkpoint_chunk_size > 0 and len(plan) > checkpoint_chunk_size)
                chunk_size = checkpoint_chunk_size if use_chunks else len(plan)
                key, value = prefix.key_cache[layer_index], prefix.value_cache[layer_index]
                rows = []
                for first in range(0, len(plan), chunk_size):
                    last = min(first + chunk_size, len(plan))
                    begin = plan[first]["offset"]
                    end = plan[last - 1]["offset"] + len(plan[last - 1]["lanes"])
                    function = chunk_function(first, last)
                    arguments = (hidden[begin:end], key, value)
                    if use_chunks:
                        row, key, value = checkpoint(function, *arguments, use_reentrant=False)
                    else:
                        row, key, value = function(*arguments)
                    # These KV outputs feed the next chunk without detach. The
                    # checkpoint saves boundary tensors, not every merged KV
                    # intermediate from all preceding action steps.
                    rows.append(row)
                return torch.cat(rows, dim=0)
            return run

        for index, layer in enumerate(decoder.layers):
            actual_index = getattr(getattr(layer, "self_attn", None), "layer_idx", index)
            if actual_index != index or index not in prefix.key_cache:
                raise ValueError("Native layer/cache index mismatch")
            function = layer_function(layer, index)
            sequence = checkpoint(function, sequence, use_reentrant=False) if (
                torch.is_grad_enabled() and checkpoint_layers) else function(sequence)
        scores = [[] for _ in events]
        for settings in plan:
            begin, width = settings["offset"], len(settings["lanes"])
            # Do not turn the LM head back into singleton row calls: its packed
            # GEMM/softmax shapes are part of the behavior-policy arithmetic.
            hidden = decoder.norm(sequence[begin:begin + width])
            logits = model.language_model.lm_head(hidden)
            distribution = F.softmax(logits / temperature, dim=-1).double()
            distribution = distribution / distribution.sum(dim=-1, keepdim=True)
            _collect_scores(scores, distribution, settings, targets, events)
    return _finish_scores(scores, events, model)


def paired_logprobs(*args, **kwargs):
    return [scores.sum() for scores in paired_token_logprobs(*args, **kwargs)]


def verify_behavior_logprobs(scores, events, *, tolerance=1e-5):
    """Unchanged absolute summed-action gate; also report per-token differences."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Behavior likelihood tolerance must be finite and positive")
    if len(scores) != len(events):
        raise ValueError("Behavior-score lane count mismatch")
    result = []
    for lane, (actual, event) in enumerate(zip(scores, events)):
        expected = torch.tensor(event["sampling_token_logps"], dtype=torch.float64, device=actual.device)
        if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise ValueError("Invalid per-lane behavior likelihood evidence")
        recorded = float(event["sampling_logp"])
        if not math.isfinite(recorded) or abs(float(expected.sum()) - recorded) > 1e-8:
            raise ValueError("Recorded summed/per-token behavior likelihoods disagree")
        delta = abs(float(actual.detach().sum()) - recorded)
        if delta > tolerance:
            raise ValueError(f"Paired lane {lane} sampling/replay likelihood mismatch: {delta}")
        result.append(dict(lane=lane, summed_logp_delta=delta,
            max_token_logp_delta=float((actual.detach() - expected).abs().max()) if actual.numel() else 0.,
            tokens=actual.numel(), tolerance=tolerance))
    return result
