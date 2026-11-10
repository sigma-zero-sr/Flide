"""The one kernel the Gemma 4 vision tower needs beside flide's: its per-head norm with the axial rotary embedding.

flide.kernels.pointwise.gemma_norm_rope rotates over the whole head (pairs d and d + hd/2). The vision tower rotates
each half of the head by itself, one half per image axis (transformers' apply_multidimensional_rope): pairs d and
d + part/2 inside each `part` channels.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_parts_kernel(X, COS, SIN, W, OUT, T, HD, sx_t, sx_h, sc_t, so_t, so_h, eps, PART: tl.constexpr,
                            BT: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    mt = t < T
    md = d < HD
    m = mt[:, None] & md[None, :]
    base = X + t[:, None] * sx_t + h * sx_h
    x = tl.load(base + d[None, :], mask=m, other=0.0).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(x * x, 1) / HD + eps)
    y = x * inv[:, None] * tl.load(W + d, mask=md, other=0.0).to(tl.float32)[None, :]
    half = PART // 2
    first = (d % PART) < half
    partner = tl.where(first, d + half, d - half)
    yp = tl.load(base + partner[None, :], mask=m, other=0.0).to(tl.float32) * inv[:, None]
    yp = yp * tl.load(W + partner, mask=md, other=0.0).to(tl.float32)[None, :]
    c = tl.load(COS + t[:, None] * sc_t + d[None, :], mask=m, other=1.0).to(tl.float32)
    s = tl.load(SIN + t[:, None] * sc_t + d[None, :], mask=m, other=0.0).to(tl.float32)
    y = y * c + tl.where(first, -1.0, 1.0)[None, :] * yp * s
    tl.store(OUT + t[:, None] * so_t + h * so_h + d[None, :], y.to(OUT.dtype.element_ty), mask=m)


def norm_rope_parts(x, cos, sin, w, eps: float, parts: int):
    """Per-head RMSNorm of x [t, heads, hd] with weight w, then the rotary embedding over each of `parts` equal
    runs of the head's channels by itself (cos and sin [t, hd]). Any strides with a contiguous last dimension;
    returns a contiguous [t, heads, hd]."""
    T, N, HD = x.shape
    if HD % (2 * parts):
        raise ValueError(f"a head of {HD} channels does not split into {parts} rotated parts")
    out = torch.empty(T, N, HD, device=x.device, dtype=x.dtype)
    if T:
        block = triton.next_power_of_2(HD)
        BT = max(1, 2048 // block)
        _norm_rope_parts_kernel[(triton.cdiv(T, BT), N)](x, cos, sin, w, out, T, HD, x.stride(0), x.stride(1),
                                                         cos.stride(0), out.stride(0), out.stride(1), eps,
                                                         PART=HD // parts, BT=BT, BLOCK=block, num_warps=4)
    return out
