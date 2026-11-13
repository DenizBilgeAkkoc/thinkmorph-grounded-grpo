"""Detached native prefix reconstruction, exact-token text replay and Flow-GRPO replay."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from types import MethodType
import inspect
import json
import hashlib
from pathlib import Path
import time
from unittest.mock import patch
import torch
import torch.nn.functional as F
from PIL import Image
from .sde import sde_step
from .guidance import require_no_cfg

TEXT_LIKELIHOOD_BACKEND = "native_incremental"
TEXT_REPLAY_MEMORY_STRATEGY = "layerwise_singleton_checkpoint"

_ACTIVE_CONTEXT_CACHE = ContextVar("thinkmorph_replay_context_cache", default=None)
_REFERENCE_MODEL = ContextVar("thinkmorph_reference_model", default=None)


def _copy_detached_context(ctx):
    """Copy mutable containers; native cache extension replaces tensor entries."""
    result = dict(ctx)
    result["kv_lens"], result["ropes"] = list(ctx["kv_lens"]), list(ctx["ropes"])
    cache = copy(ctx["past_key_values"])
    cache.key_cache = {k: v.detach() if v is not None else None
                       for k, v in cache.key_cache.items()}
    cache.value_cache = {k: v.detach() if v is not None else None
                         for k, v in cache.value_cache.items()}
    result["past_key_values"] = cache
    return result


class ReplayContextCache:
    """Update-local, bounded reuse of exact recorded prefixes within one rollout.

    Keep only the newest policy prefix and newest frozen-reference prefix. Never
    share across rollout folders: equal images may have different VAE samples.
    Caller-owned KV dictionaries are copied, so text replay and native prefill
    cannot extend a retained cache by mutation. Tensors are detached and shared;
    native attention builds new tensors when extending its KV cache.
    """
    def __init__(self, infer):
        self.infer = infer
        self.folder = None
        self.entries = {}
        self.counters = dict(requests=0, full_hits=0, prefix_hits=0, misses=0,
                             reused_events=0, rebuilt_events=0, peak_entries=0,
                             rebuild_seconds=0.)

    @staticmethod
    def event_keys(events, folder):
        keys = []
        for event in events:
            # Include exact text/flags and the recorded VAE latent filename.
            # Content digests prevent reuse even if an asset is replaced within
            # the filesystem's timestamp resolution during an update.
            assets = []
            for name in ("file", "vae_latent_file"):
                if name in event:
                    path = (folder / event[name]).resolve()
                    assets.append((str(path), hashlib.sha256(path.read_bytes()).hexdigest()))
            keys.append((json.dumps(event, sort_keys=True, separators=(",", ":")), tuple(assets)))
        return tuple(keys)

    def get(self, events, folder):
        folder = Path(folder).resolve()
        if folder != self.folder:
            self.entries.clear()
            self.folder = folder
        keys = self.event_keys(events, folder)
        role = "reference" if _REFERENCE_MODEL.get() is self.infer.model else "policy"
        self.counters["requests"] += 1
        saved = self.entries.get(role)
        if saved is not None and keys[:len(saved[0])] == saved[0]:
            count = len(saved[0])
            self.counters["full_hits" if count == len(keys) else "prefix_hits"] += 1
            self.counters["reused_events"] += count
            ctx = _copy_detached_context(saved[1])
        else:
            count = 0
            self.counters["misses"] += 1
            ctx = self.infer.init_gen_context()
        started = time.monotonic()
        ctx = _extend_context(self.infer, events[count:], folder, ctx)
        self.counters["rebuild_seconds"] += time.monotonic() - started
        self.counters["rebuilt_events"] += len(events) - count
        self.entries[role] = (keys, _copy_detached_context(ctx))
        self.counters["peak_entries"] = max(self.counters["peak_entries"], len(self.entries))
        return ctx

    def stats(self):
        return dict(self.counters)

    def clear(self):
        self.entries.clear()
        self.folder = None


@contextmanager
def replay_context_cache(infer):
    """Scope cache lifetime to one update; release it even on validation failure."""
    cache = ReplayContextCache(infer)
    token = _ACTIVE_CONTEXT_CACHE.set(cache)
    try:
        yield cache
    finally:
        cache.clear()
        _ACTIVE_CONTEXT_CACHE.reset(token)


@torch.no_grad()
def rebuild_context(infer, events, folder):
    cache = _ACTIVE_CONTEXT_CACHE.get()
    if cache is not None and cache.infer is infer:
        return cache.get(events, folder)
    return _extend_context(infer, events, folder, infer.init_gen_context())


@torch.no_grad()
def _extend_context(infer, events, folder, ctx):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for event in events:
            if event["kind"] == "text":
                ctx = infer.update_context_text(event["text"], ctx)
            else:
                image = Image.open(Path(folder) / event["file"]).convert("RGB")
                if event["vae"]:
                    latent = torch.load(Path(folder) / event["vae_latent_file"], map_location="cpu", weights_only=True)
                    def encode(input_image):
                        return latent.to(input_image.device)
                    with patch.object(infer.vae_model, "encode", encode):
                        ctx = infer.update_context_image(image, ctx, vae=True, vit=event["vit"])
                else:
                    ctx = infer.update_context_image(image, ctx, vae=False, vit=event["vit"])
    return ctx


def text_token_logprobs(model, prior, event):
    """Uncheckpointed native singleton diagnostic with action-local KV gradients.

    Full-sequence teacher forcing differs numerically from the released native
    incremental sampler. Preserve its token shapes, index dtypes, autocast and
    normalized sampling distribution. Only the pre-action prefix is detached;
    every new key/value remains connected to the preceding action tokens.
    """
    device = next(model.parameters()).device
    inputs, targets = event["input_ids"], event["target_ids"]
    if len(inputs) != len(targets) or (inputs and inputs[1:] != targets[:-1]):
        raise ValueError("Invalid exact-token replay")
    if len(prior["kv_lens"]) != 1 or len(prior["ropes"]) != 1:
        raise ValueError("Native text replay requires a single sequence")
    if not targets:
        return torch.empty(0, device=device, dtype=torch.float64,
                           requires_grad=torch.is_grad_enabled())
    if event["temperature"] <= 0:
        raise ValueError("Text sampling temperature must be positive")
    cache = copy(prior["past_key_values"])
    # Native attention replaces dictionary entries when extending its cache.
    # Independent dictionaries protect the caller; detached prefix tensors may
    # share storage because native attention does not modify them in place.
    cache.key_cache = {k: v.detach() if v is not None else None
                       for k, v in cache.key_cache.items()}
    cache.value_cache = {k: v.detach() if v is not None else None
                         for k, v in cache.value_cache.items()}
    packed = model.prepare_start_tokens(prior["kv_lens"], prior["ropes"],
                                       {"bos_token_id": inputs[0]})
    tokens = packed["packed_start_tokens"]
    positions = packed["packed_query_position_ids"]
    lengths = packed["key_values_lens"]
    indexes = packed["packed_key_value_indexes"]
    target_ids = torch.tensor(targets, device=device, dtype=torch.long)
    scores = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(targets)):
            embeddings = model.language_model.model.embed_tokens(tokens)
            query_indexes = torch.cumsum(lengths, dim=0) + torch.arange(
                len(lengths), device=lengths.device, dtype=lengths.dtype)
            output = model.language_model.forward_inference(
                packed_query_sequence=embeddings,
                query_lens=torch.ones_like(tokens),
                packed_query_position_ids=positions,
                packed_query_indexes=query_indexes,
                past_key_values=cache,
                key_values_lens=lengths,
                packed_key_value_indexes=indexes,
                update_past_key_values=True, is_causal=True,
                **({"mode": "und"} if model.use_moe else {}))
            cache = output.past_key_values
            logits = model.language_model.lm_head(output.packed_query_sequence)
            probabilities = F.softmax(logits / event["temperature"], dim=-1)
            # Match the actual torch.multinomial distribution and bridge recorder,
            # including normalization of rounded probability entries in float64.
            probabilities = probabilities.double()
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            scores.append(probabilities[0, target_ids[i]].log())
            tokens = target_ids[i:i+1]
            indexes = torch.cat([indexes, torch.tensor([indexes[-1] + 1],
                                                       device=indexes.device)])
            lengths = lengths + 1
            positions = positions + 1
    return torch.stack(scores)


def text_logprob(model, prior, event):
    """Sum native incremental token likelihoods, retaining action-local gradients."""
    return text_token_logprobs_layerwise(model, prior, event).sum()


def text_token_logprobs_layerwise(model, prior, event, *, checkpoint_layers=True):
    """Checkpoint each decoder layer's causal singleton loop.

    Reordering the independent layer/token loops preserves native singleton
    arithmetic. Each pure checkpoint reconstructs its own cache from the shared
    detached prefix and keeps all action-local KV dependencies differentiable.
    This avoids retaining a copy of the full image prefix for every token in
    every layer; backward materializes one layer's graph at a time.
    """
    from torch.utils.checkpoint import checkpoint
    inputs, targets = event["input_ids"], event["target_ids"]
    if len(inputs) != len(targets) or (inputs and inputs[1:] != targets[:-1]):
        raise ValueError("Invalid exact-token replay")
    if len(prior["kv_lens"]) != 1 or len(prior["ropes"]) != 1:
        raise ValueError("Native text replay requires a single sequence")
    device = next(model.parameters()).device
    if not targets:
        return torch.empty(0, device=device, dtype=torch.float64,
                           requires_grad=torch.is_grad_enabled())
    if event["temperature"] <= 0:
        raise ValueError("Text sampling temperature must be positive")
    decoder = model.language_model.model
    if getattr(decoder, "enable_taylorseer", False) or any(
            getattr(layer, "enable_taylorseer", False) for layer in decoder.layers):
        raise ValueError("Text replay requires native full inference without TaylorSeer")
    prefix = copy(prior["past_key_values"])
    prefix.key_cache = {k: v.detach() if v is not None else None
                        for k, v in prefix.key_cache.items()}
    prefix.value_cache = {k: v.detach() if v is not None else None
                          for k, v in prefix.value_cache.items()}
    packed = model.prepare_start_tokens(prior["kv_lens"], prior["ropes"],
                                       {"bos_token_id": inputs[0]})
    target_ids = torch.tensor(targets, device=device, dtype=torch.long)
    tokens = [packed["packed_start_tokens"]] + [target_ids[i:i+1] for i in range(len(targets)-1)]
    metadata = []
    lengths, indexes = packed["key_values_lens"], packed["packed_key_value_indexes"]
    positions = packed["packed_query_position_ids"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embedded = [decoder.embed_tokens(token) for token in tokens]
        for row, token in zip(embedded, tokens):
            # The normal decoder __call__ Accelerate hook moves these native
            # CPU index tensors to the hidden-state device before rotary/layers.
            cos, sin = decoder.rotary_emb(row, positions.to(row.device).unsqueeze(0))
            metadata.append(dict(query_lens=torch.ones_like(token).to(row.device),
                packed_query_position_embeddings=(cos.squeeze(0), sin.squeeze(0)),
                packed_query_indexes=(torch.cumsum(lengths, dim=0) + torch.arange(
                    len(lengths), device=lengths.device, dtype=lengths.dtype)).to(row.device),
                key_values_lens=lengths.to(row.device), packed_key_value_indexes=indexes.to(row.device)))
            indexes = torch.cat([indexes, torch.tensor([indexes[-1] + 1], device=indexes.device)])
            lengths, positions = lengths + 1, positions + 1
        sequence = torch.cat(embedded)
        def layer_function(layer):
            def run_layer(hidden):
                cache = copy(prefix)
                cache.key_cache, cache.value_cache = dict(prefix.key_cache), dict(prefix.value_cache)
                rows = []
                for i, settings in enumerate(metadata):
                    row, cache = layer(packed_query_sequence=hidden[i:i+1],
                        past_key_values=cache, update_past_key_values=True, is_causal=True,
                        **settings, **({"mode": "und"} if model.use_moe else {}))
                    rows.append(row)
                return torch.cat(rows)
            return run_layer
        for layer in decoder.layers:
            run_layer = layer_function(layer)
            sequence = (checkpoint(run_layer, sequence, use_reentrant=False)
                        if torch.is_grad_enabled() and checkpoint_layers else run_layer(sequence))
        scores = []
        for i in range(len(targets)):
            hidden = decoder.norm(sequence[i:i+1])
            logits = model.language_model.lm_head(hidden)
            probabilities = F.softmax(logits / event["temperature"], dim=-1).double()
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            scores.append(probabilities[0, target_ids[i]].log())
    return torch.stack(scores)


def text_logprob_full_sequence(model, prior, event):
    """Legacy full-sequence diagnostic; not numerically equal to native sampling."""
    from torch.utils.checkpoint import checkpoint
    device = next(model.parameters()).device
    inputs = torch.tensor(event["input_ids"], device=device, dtype=torch.long)
    targets = torch.tensor(event["target_ids"], device=device, dtype=torch.long)
    n, L, R = len(inputs), prior["kv_lens"][0], prior["ropes"][0]
    if n != len(targets):
        raise ValueError("Invalid exact-token replay")
    if not n:
        # A native cap of one discards its only sampled token. The probability
        # of the resulting empty action is one and has no policy gradient.
        return torch.zeros((), device=device, requires_grad=torch.is_grad_enabled())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = model.language_model.model.embed_tokens(inputs)
        output = model.language_model.forward_inference(
            packed_query_sequence=embeddings,
            query_lens=torch.tensor([n], device=device, dtype=torch.int),
            packed_query_position_ids=torch.arange(R, R+n, device=device),
            packed_query_indexes=torch.arange(L, L+n, device=device),
            past_key_values=prior["past_key_values"],
            key_values_lens=torch.tensor([L], device=device, dtype=torch.int),
            packed_key_value_indexes=torch.arange(L, device=device),
            update_past_key_values=False, is_causal=True,
            **({"mode": "und"} if model.use_moe else {}))
        def chunk_score(hidden, target):
            logits = model.language_model.lm_head(hidden).float() / event["temperature"]
            lp = F.log_softmax(logits, dim=-1)
            return lp.gather(-1, target[:, None]).sum()
        scores = []
        for start in range(0, n, 128):
            hidden = output.packed_query_sequence[start:start+128]
            target = targets[start:start+128]
            scores.append(checkpoint(chunk_score, hidden, target, use_reentrant=False)
                          if torch.is_grad_enabled() else chunk_score(hidden, target))
    return torch.stack(scores).sum()


@torch.no_grad()
def flow_inputs(infer, event, folder):
    require_no_cfg(event["settings"], "Saved rollout")
    conditional = rebuild_context(infer, event["contexts"]["conditional"], folder)
    model = infer.model
    packed = model.prepare_vae_latent(curr_kvlens=conditional["kv_lens"],
        curr_rope=conditional["ropes"], image_sizes=[event["image_shape"]], new_token_ids=infer.new_token_ids)
    packed.pop("packed_init_noises")
    packed["past_key_values"] = conditional["past_key_values"]
    settings = event["settings"]
    for key in ("cfg_text_scale", "cfg_img_scale", "cfg_renorm_min", "cfg_renorm_type"):
        packed[key] = settings[key]
    return packed


@torch.no_grad()
def reference_flow_inputs(infer, event, folder):
    """Build frozen-SFT prefix values, not only a detached policy cache."""
    with reference_adapters(infer.model):
        return flow_inputs(infer, event, folder)


def flow_logprob(model, inputs, event, state, transition):
    require_no_cfg(event["settings"], "Saved rollout")
    require_no_cfg(state.get("guidance", {}), "Saved transition")
    require_no_cfg(inputs, "GRPO replay")
    ts, i = state["schedule"], transition["index"]
    x = transition["x"]
    flow = dict(inputs)
    lo, hi = event["settings"]["cfg_interval"]
    if not lo < ts[i] <= hi:
        flow["cfg_text_scale"] = flow["cfg_img_scale"] = 1.0
    model.language_model.model.enable_taylorseer = False
    # Unwrap only the native method's @no_grad decorator. Its prediction/CFG
    # computation remains unchanged with CFG off; prefix KV tensors stay detached.
    native_flow = inspect.unwrap(type(model)._forward_flow)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        velocity = native_flow(model, x_t=x, timestep=ts[i].expand(x.shape[0]), **flow).to(x.device)
    _, logp, _, _ = sde_step(velocity, ts[i], ts[i+1], x, sigma_max=ts[1],
        noise_level=state["noise_level"], previous_sample=transition["next_x"])
    return logp, velocity


def enable_replay_checkpointing(model):
    """Same replay-only checkpoint policy as the custom Qwen2Model implementation."""
    from torch.utils.checkpoint import checkpoint
    for layer in model.language_model.model.layers:
        if getattr(layer, "_thinkmorph_replay_checkpointed", False):
            continue
        original = layer.forward_inference
        def make_wrapper(original):
            def wrapped(layer, *args, **kwargs):
                sequence = kwargs.get("packed_query_sequence")
                use_checkpoint = (torch.is_grad_enabled() and sequence is not None
                    and not kwargs.get("update_past_key_values", True)
                    and not getattr(layer, "enable_taylorseer", False)
                    and (kwargs.get("mode") == "gen" or sequence.shape[0] > 256))
                if not use_checkpoint:
                    return original(*args, **kwargs)
                rest = {k:v for k,v in kwargs.items() if k != "packed_query_sequence"}
                def forward_hidden(hidden):
                    return original(*args, packed_query_sequence=hidden, **rest)[0]
                hidden = checkpoint(forward_hidden, sequence, use_reentrant=False)
                return hidden, rest.get("past_key_values")
            return wrapped
        layer.forward_inference = MethodType(make_wrapper(original), layer)
        layer._thinkmorph_replay_checkpointed = True


def attach_lora(model, rank=64, alpha=128):
    from peft import LoraConfig, inject_adapter_in_model
    from .native_autograd import enable_native_autograd
    enable_native_autograd(model)
    enable_replay_checkpointing(model)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=["self_attn.q_proj_moe_gen", "self_attn.k_proj_moe_gen",
            "self_attn.v_proj_moe_gen", "self_attn.o_proj_moe_gen",
            "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj",
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])
    model.requires_grad_(False)
    # In-place injection keeps the native model's embed_tokens attribute path.
    inject_adapter_in_model(cfg, model.language_model)
    groups = {"image": [], "text": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise RuntimeError("Unexpected non-adapter trainable parameter: " + name)
            parameter.data = parameter.data.float()
            groups["image" if "_moe_gen" in name else "text"].append(parameter)
    if not all(groups.values()):
        raise RuntimeError("Both native experts must receive adapters")
    model.eval()  # Dropout is zero; native inference methods are differentiable.
    return groups


@contextmanager
def reference_adapters(model):
    from peft.tuners.tuners_utils import BaseTunerLayer
    layers = [m for m in model.modules() if isinstance(m, BaseTunerLayer)]
    prior = [m.disable_adapters for m in layers]
    token = _REFERENCE_MODEL.set(model)
    try:
        for m in layers:
            m.enable_adapters(False)
        yield
    finally:
        try:
            for m, was_disabled in zip(layers, prior):
                m.enable_adapters(not was_disabled)
        finally:
            _REFERENCE_MODEL.reset(token)
