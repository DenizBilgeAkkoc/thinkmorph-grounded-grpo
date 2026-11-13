# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
# Derived from ThinkMorph c1a48ad, PackedAttentionMoT.forward_inference.
# Adaptations: out-of-place output projection for autograd and deterministic
# FlashAttention backward. The latter preserves native forward arithmetic and
# removes measured run-to-run gradient variation; original source hash is pinned.
from typing import Optional
from types import MethodType
import hashlib
import inspect
import textwrap
import torch
from modeling.bagel.qwen2_navit import NaiveCache, PackedAttentionMoT
from modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from flash_attn import flash_attn_varlen_func

ORIGINAL_METHOD_SHA256 = '9168fd293a6291b8b31f6b1ff6433f49a46db9ddfb5cb10bdae74e305f18ba8b'

def forward_inference(
    self,
    packed_query_sequence: torch.Tensor,
    query_lens: torch.Tensor,
    packed_query_position_embeddings: torch.Tensor,
    packed_query_indexes: torch.Tensor,
    past_key_values: Optional[NaiveCache] = None,
    key_values_lens: Optional[torch.Tensor] = None,
    packed_key_value_indexes: Optional[torch.Tensor] = None,
    update_past_key_values=True,
    is_causal=True,
    mode="und",
    packed_vae_token_indexes=None,
    packed_text_indexes=None,
):
    if mode == 'und':
        packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)
    elif mode == 'gen':
        packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
        packed_query_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_heads * self.head_dim))
        packed_key_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))
        packed_value_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))

        packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
        packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

        packed_query_states[packed_text_indexes] = self.q_proj(packed_text_query_sequence)
        packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(packed_vae_query_sequence)

        packed_key_states[packed_text_indexes] = self.k_proj(packed_text_query_sequence)
        packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(packed_vae_query_sequence)

        packed_value_states[packed_text_indexes] = self.v_proj(packed_text_query_sequence)
        packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(packed_vae_query_sequence)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = packed_query_states.to(torch.float32)
        packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
        packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_vae_token_indexes])

        packed_key_states = packed_key_states.to(torch.float32)
        packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
        packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_vae_token_indexes])

    packed_cos, packed_sin = packed_query_position_embeddings
    packed_query_states, packed_key_states = apply_rotary_pos_emb(
        packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
    )

    packed_query_states = packed_query_states.to(torch.bfloat16)
    packed_key_states = packed_key_states.to(torch.bfloat16)
    packed_value_states = packed_value_states.to(torch.bfloat16)

    if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
        past_key_states = past_key_values.key_cache[self.layer_idx]
        past_value_states = past_key_values.value_cache[self.layer_idx]

        seqlens = sum(query_lens) + sum(key_values_lens)
        merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
        merged_value_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
        merged_key_states[packed_query_indexes] = packed_key_states
        merged_key_states[packed_key_value_indexes] = past_key_states
        merged_value_states[packed_query_indexes] = packed_value_states
        merged_value_states[packed_key_value_indexes] = past_value_states
        key_values_lens = key_values_lens + query_lens
    else:
        merged_key_states = packed_key_states
        merged_value_states = packed_value_states
        key_values_lens = query_lens

    cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
    cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

    packed_attn_output = flash_attn_varlen_func(
        q=packed_query_states,
        k=merged_key_states,
        v=merged_value_states,
        cu_seqlens_q=cu_seqlens_q.to(torch.int32),
        cu_seqlens_k=cu_seqlens_k.to(torch.int32),
        max_seqlen_q=max(query_lens).item(),
        max_seqlen_k=max(key_values_lens).item(),
        causal=is_causal,
        deterministic=True,
    )
    packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
    if mode == 'und':
        packed_attn_output = self.o_proj(packed_attn_output)
    elif mode == 'gen':
        projected_output = torch.zeros_like(packed_attn_output)
        projected_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
        projected_output[packed_vae_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes])
        packed_attn_output = projected_output

    if update_past_key_values:
        past_key_values.key_cache[self.layer_idx] = merged_key_states
        past_key_values.value_cache[self.layer_idx] = merged_value_states

    return packed_attn_output, past_key_values


def enable_native_autograd(model):
    source = textwrap.dedent(inspect.getsource(PackedAttentionMoT.forward_inference))
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_METHOD_SHA256:
        raise RuntimeError("Native attention changed; review the minimal autograd adaptation")
    count = 0
    for module in model.modules():
        if isinstance(module, PackedAttentionMoT):
            module.forward_inference = MethodType(forward_inference, module)
            count += 1
    if count == 0:
        raise RuntimeError("No native MoT attention modules found")
    return count
