# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton stand-ins for FlashMLA's sparse kernels on GPUs below sm_90.

FlashMLA's sparse prefill and FP8 decode kernels are built for Hopper and
Blackwell only. These keep the call signatures the DeepSeek-V4/V4.1 FlashMLA
layers use: decode dequantizes the selected FP8 cache slots with the model's
own Triton gather, then both run the Triton BF16 sparse MLA kernel and fold in
the attention sink from its log-sum-exp.
"""

from collections.abc import Callable

import torch

from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface

_BLOCK_N = 16


def _attend(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None,
    out: torch.Tensor,
) -> None:
    o, _, lse = triton_bf16_mla_sparse_interface(
        q=q,
        kv=kv,
        indices=indices.unsqueeze(1),
        sm_scale=sm_scale,
        d_v=q.shape[-1],
        block_dpe=0,
    )
    if attn_sink is not None:
        o = o * torch.sigmoid(lse - attn_sink[: q.shape[1]]).unsqueeze(-1)
    out.copy_(o.to(out.dtype))


def _pad_indices(indices: torch.Tensor, lens: torch.Tensor | None) -> torch.Tensor:
    k = indices.shape[-1]
    if lens is not None:
        pos = torch.arange(k, device=indices.device)
        indices = torch.where(pos < lens.reshape(-1, 1), indices, -1)
    pad = -k % _BLOCK_N
    if pad:
        indices = torch.nn.functional.pad(indices, (0, pad), value=-1)
    return indices.to(torch.int32).contiguous()


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
):
    assert out is not None
    if q.shape[0] > 0:
        _attend(
            q,
            kv,
            _pad_indices(indices.squeeze(1), topk_length),
            sm_scale,
            attn_sink,
            out,
        )
    return out, None, None


def flash_mla_with_kvcache(
    gather: Callable[..., None],
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    softmax_scale: float,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    **_unused,
):
    """FP8 sparse decode; ``gather`` is the model's dequantize_and_gather_k_cache.

    ``q``/``out`` are ``[tokens, 1, heads, 512]``, caches
    ``[blocks, block_size, 1, bytes]`` and indices global cache slots.
    """
    assert out is not None
    num_tokens = q.shape[0]
    if num_tokens == 0:
        return out, None
    head_dim = q.shape[-1]

    def dequant(cache, idx, lens):
        cache = cache.squeeze(-2)
        idx = idx.reshape(num_tokens, -1)
        k = idx.shape[1]
        if lens is not None:
            pos = torch.arange(k, device=idx.device)
            idx = torch.where(pos < lens.reshape(-1, 1), idx, -1)
        flat = idx.reshape(-1).to(torch.int64)
        valid = flat >= 0
        safe = torch.where(valid, flat, 0)
        block_size = cache.shape[1]
        rows = torch.empty(
            flat.shape[0], 1, head_dim, dtype=torch.bfloat16, device=q.device
        )
        # One single-slot request per index: its sequence position is the
        # slot's offset in its block and its block table row that block.
        gather(
            rows,
            cache,
            seq_lens=(safe % block_size + 1).to(torch.int32),
            gather_lens=torch.ones_like(flat, dtype=torch.int32),
            block_table=(safe // block_size).to(torch.int32).unsqueeze(1),
            block_size=block_size,
            offset=0,
        )
        return rows.view(num_tokens, k, head_dim), valid.view(num_tokens, k)

    parts = [dequant(k_cache, indices, topk_length)]
    if extra_k_cache is not None and extra_indices_in_kvcache is not None:
        parts.append(
            dequant(extra_k_cache, extra_indices_in_kvcache, extra_topk_length)
        )
    kv = torch.cat([p[0] for p in parts], dim=1)
    valid = torch.cat([p[1] for p in parts], dim=1)
    k_total = kv.shape[1]
    pos = torch.arange(k_total, device=q.device, dtype=torch.int32)
    tok = torch.arange(num_tokens, device=q.device, dtype=torch.int32).unsqueeze(1)
    sel = torch.where(valid, tok * k_total + pos, -1)
    _attend(
        q.squeeze(1),
        kv.view(-1, 1, head_dim),
        _pad_indices(sel, None),
        softmax_scale,
        attn_sink,
        out.squeeze(1),
    )
    return out, None
