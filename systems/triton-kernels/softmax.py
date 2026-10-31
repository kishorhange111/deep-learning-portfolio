"""
Row-wise softmax in Triton, two ways.

1. `softmax` (fused): one program per row loads the whole row into registers, then computes the
   max, the exponentials, the sum and the division without touching global memory in between.
   PyTorch eager runs max, subtract, exp, sum and divide as five kernels, each a full read and write
   of the tensor. The fused kernel reads the input once and writes the output once, which is what
   matters for a memory-bound operation.
2. `softmax_online`: rows longer than fit in one block are processed in tiles. The max and the
   normaliser are updated online (Milakov & Gimelshein, 2018): when a new tile raises the running max
   from m to m', the running sum is rescaled by exp(m - m'). This is the same trick that lets
   FlashAttention compute softmax(QK^T)V without materialising the full attention matrix.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(in_ptr + row * in_stride + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)  # subtract the max for numerical stability
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_stride + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _softmax_online_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols, TILE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, TILE)
    # Running max (per lane) starts at a large *finite* negative value, so m - m_new is never
    # -inf - (-inf) = NaN when a lane has only seen masked (-inf) elements so far.
    m = tl.full([TILE], -1e30, tl.float32)
    s = tl.zeros([TILE], tl.float32)  # running sum of exp(x - m), per lane
    for start in range(0, n_cols, TILE):  # pass 1: max and normaliser in one sweep
        cols = start + offs
        x = tl.load(in_ptr + row * in_stride + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        s = s * tl.exp(m - m_new) + tl.exp(x - m_new)  # rescale the old sum to the new max
        m = m_new
    row_max = tl.max(m, axis=0)
    total = tl.sum(s * tl.exp(m - row_max), axis=0)  # combine the per-lane partial results
    for start in range(0, n_cols, TILE):  # pass 2: write normalised values
        cols = start + offs
        x = tl.load(in_ptr + row * in_stride + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
        out = tl.exp(x - row_max) / total
        tl.store(out_ptr + row * out_stride + cols, out.to(out_ptr.dtype.element_ty), mask=cols < n_cols)


def _num_warps(block):
    return 4 if block < 2048 else (8 if block < 8192 else 16)


def softmax(x):
    """Softmax over the last dimension with the fused single-pass kernel (rows up to 65,536 wide)."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    n_rows, n_cols = x2.shape
    block = triton.next_power_of_2(n_cols)
    if block > 65536:
        raise ValueError("row too long for the fused kernel; use softmax_online")
    out = torch.empty_like(x2)
    _softmax_kernel[(n_rows,)](out, x2, x2.stride(0), out.stride(0), n_cols, BLOCK=block, num_warps=_num_warps(block))
    return out.reshape(shape)


def softmax_online(x, tile=1024):
    """Softmax over the last dimension for rows of any length (two passes over tiles, online normaliser)."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    out = torch.empty_like(x2)
    tile = min(tile, triton.next_power_of_2(x2.shape[1]))
    _softmax_online_kernel[(x2.shape[0],)](out, x2, x2.stride(0), out.stride(0), x2.shape[1], TILE=tile)
    return out.reshape(shape)


def softmax_naive(x):
    """The eager PyTorch composition the fused kernel replaces (5 kernels, 5 memory round trips)."""
    x = x - x.max(dim=-1, keepdim=True).values
    num = torch.exp(x)
    return num / num.sum(dim=-1, keepdim=True)
