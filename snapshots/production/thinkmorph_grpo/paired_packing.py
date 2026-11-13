"""Native packed-cache helpers for independent rollout text lanes.

These helpers detach sampling/prefix caches. Gradient replay uses its separate
graph-preserving compaction implementation.
"""
from copy import copy
import torch


def detached_cache(cache):
    result = copy(cache)
    for field in ('key_cache', 'value_cache'):
        setattr(result, field, {k: v.detach() if v is not None else None
                               for k, v in getattr(cache, field).items()})
    return result


def pack_contexts(contexts):
    if any(len(c['kv_lens']) != 1 or len(c['ropes']) != 1 for c in contexts):
        raise ValueError('Only singleton source contexts may be packed')
    cache = copy(contexts[0]['past_key_values'])
    for field in ('key_cache', 'value_cache'):
        layers = {}
        for layer in getattr(cache, field):
            values = [getattr(c['past_key_values'], field)[layer] for c in contexts]
            if any(v is None for v in values):
                raise ValueError('Probe requires nonempty prefixes in every layer')
            if any(v.shape[0] != c['kv_lens'][0] for v, c in zip(values, contexts)):
                raise ValueError('Prefix length does not match cache storage')
            layers[layer] = (values[0].detach() if len(values) == 1 else
                             torch.cat([v.detach() for v in values], dim=0))
        setattr(cache, field, layers)
    return dict(past_key_values=cache,
                kv_lens=[c['kv_lens'][0] for c in contexts],
                ropes=[c['ropes'][0] for c in contexts])


def insertion_indexes(lengths):
    """Native per-segment shift before inserting one query into each sequence."""
    indexes = torch.arange(int(lengths.sum()), dtype=torch.long,
                           device=lengths.device)
    segments = indexes.split(lengths.tolist())
    keys = torch.cat([segment + lane for lane, segment in enumerate(segments)])
    queries = torch.cumsum(lengths, dim=0) + torch.arange(
        len(lengths), dtype=lengths.dtype, device=lengths.device)
    return queries, keys


def compact_cache(cache, lengths, keep):
    result = copy(cache)
    for field in ('key_cache', 'value_cache'):
        layers = {}
        for layer, values in getattr(cache, field).items():
            parts = values.split(lengths.tolist())
            layers[layer] = torch.cat([parts[n].detach() for n in keep], dim=0)
        setattr(result, field, layers)
    return result
