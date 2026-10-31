"""
FlashAttention-2 in Triton: forward and backward, usable as a drop-in differentiable op.

Standard attention materialises S = QK^T, a (T x T) matrix per head, in GPU memory. That costs
O(T^2) memory and, worse, O(T^2) reads and writes of slow HBM. FlashAttention (Dao et al., 2022;
FlashAttention-2, Dao 2023) tiles Q, K and V into blocks that fit in on-chip SRAM and streams over
the key blocks with an online softmax:

    for each key block j:   S_ij = Q_i K_j^T * scale
                            m_new = max(m, rowmax(S_ij))
                            P_ij  = exp(S_ij - m_new)
                            l     = l * exp(m - m_new) + rowsum(P_ij)
                            acc   = acc * exp(m - m_new) + P_ij V_j
    O_i = acc / l,  and the log-sum-exp L_i = m + log(l) is kept for the backward pass.

Memory is O(T), and the T x T matrix never exists. The backward pass recomputes P_ij from Q, K and
L instead of storing it (recomputation is cheaper than HBM traffic):

    dV_j = sum_i P_ij^T dO_i            dP_ij = dO_i V_j^T
    dS_ij = P_ij * (dP_ij - D_i),  D_i = rowsum(dO_i * O_i)
    dQ_i = scale * sum_j dS_ij K_j      dK_j = scale * sum_i dS_ij^T Q_i

Exponentials are computed base 2 (exp2), with log2(e) folded into the scale, because exp2 maps to
a single hardware instruction.
"""

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Out,
    L,
    qk_scale,
    T,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m, pid_bh = tl.program_id(0), tl.program_id(1)
    base = pid_bh.to(tl.int64) * T * HEAD_DIM  # tensors are contiguous (B*H, T, D)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=offs_m[:, None] < T, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    hi = tl.minimum((pid_m + 1) * BLOCK_M, T) if CAUSAL else T  # causal: skip key blocks entirely in the future
    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        k = tl.load(K + base + cols[:, None] * HEAD_DIM + offs_d[None, :], mask=cols[:, None] < T, other=0.0)
        v = tl.load(V + base + cols[:, None] * HEAD_DIM + offs_d[None, :], mask=cols[:, None] < T, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        valid = cols[None, :] < T
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= cols[None, :])
        s = tl.where(valid, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))  # finite: key 0 is always visible in the first block
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(L + pid_bh.to(tl.int64) * T + offs_m, m_i + tl.log2(l_i), mask=offs_m < T)  # base-2 log-sum-exp
    out_ptrs = Out + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < T)


@triton.jit
def _attn_bwd_dkdv(
    Q,
    K,
    V,
    DO,
    L,
    Delta,
    DK,
    DV,
    qk_scale,
    sm_scale,
    T,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """One program per key block: loop over the query blocks that can see it, accumulate dK and dV."""
    pid_n, pid_bh = tl.program_id(0), tl.program_id(1)
    base = pid_bh.to(tl.int64) * T * HEAD_DIM
    row_base = pid_bh.to(tl.int64) * T
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    k = tl.load(K + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=offs_n[:, None] < T, other=0.0)
    v = tl.load(V + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=offs_n[:, None] < T, other=0.0)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    lo = pid_n * BLOCK_N if CAUSAL else 0  # causal: earlier queries never attend to this block
    for start_m in range(lo, T, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        row_ok = offs_m < T
        q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=row_ok[:, None], other=0.0)
        do = tl.load(DO + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=row_ok[:, None], other=0.0)
        lse = tl.load(L + row_base + offs_m, mask=row_ok, other=0.0)
        delta = tl.load(Delta + row_base + offs_m, mask=row_ok, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        valid = row_ok[:, None] & (offs_n[None, :] < T)
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)  # recomputed attention probabilities
        dv += tl.dot(tl.trans(p).to(do.dtype), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds).to(q.dtype), q)
    dk = dk * sm_scale
    tl.store(
        DK + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < T
    )
    tl.store(
        DV + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < T
    )


@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    DO,
    L,
    Delta,
    DQ,
    qk_scale,
    sm_scale,
    T,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """One program per query block: loop over the visible key blocks, accumulate dQ."""
    pid_m, pid_bh = tl.program_id(0), tl.program_id(1)
    base = pid_bh.to(tl.int64) * T * HEAD_DIM
    row_base = pid_bh.to(tl.int64) * T
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    row_ok = offs_m < T
    q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    do = tl.load(DO + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    lse = tl.load(L + row_base + offs_m, mask=row_ok, other=0.0)
    delta = tl.load(Delta + row_base + offs_m, mask=row_ok, other=0.0)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    hi = tl.minimum((pid_m + 1) * BLOCK_M, T) if CAUSAL else T
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=offs_n[:, None] < T, other=0.0)
        v = tl.load(V + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=offs_n[:, None] < T, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        valid = row_ok[:, None] & (offs_n[None, :] < T)
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    dq = dq * sm_scale
    tl.store(DQ + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], dq.to(DQ.dtype.element_ty), mask=row_ok[:, None])


def _default_blocks(head_dim, device):
    if device.type != "cuda":  # Triton's CPU interpreter (tests): small tiles keep it fast
        return 16, 16, 1, 1
    return (128 if head_dim <= 64 else 64), 64, (4 if head_dim <= 64 else 8), 3


class FlashAttentionFn(torch.autograd.Function):
    """autograd wrapper: forward saves (q, k, v, o, lse); backward runs the two recomputation kernels."""

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, block_m, block_n):
        B, H, T, D = q.shape
        assert D in (16, 32, 64, 128), "head_dim must be 16, 32, 64 or 128"
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        bm, bn, warps, stages = _default_blocks(D, q.device)
        bm, bn = block_m or bm, block_n or bn
        o = torch.empty_like(q)
        lse = torch.empty((B * H, T), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(T, bm), B * H)
        _attn_fwd[grid](
            q,
            k,
            v,
            o,
            lse,
            sm_scale * LOG2E,
            T,
            BLOCK_M=bm,
            BLOCK_N=bn,
            HEAD_DIM=D,
            CAUSAL=causal,
            num_warps=warps,
            num_stages=stages,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal, ctx.sm_scale, ctx.blocks = causal, sm_scale, (block_m, block_n)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, H, T, D = q.shape
        do = do.contiguous()
        bm = ctx.blocks[0] or (16 if q.device.type != "cuda" else 64)
        bn = ctx.blocks[1] or (16 if q.device.type != "cuda" else 64)
        warps = 1 if q.device.type != "cuda" else 4
        delta = (do.float() * o.float()).sum(-1).reshape(B * H, T)  # D_i = rowsum(dO_i * O_i): one cheap fused op
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        qk_scale = ctx.sm_scale * LOG2E
        _attn_bwd_dkdv[(triton.cdiv(T, bn), B * H)](
            q,
            k,
            v,
            do,
            lse,
            delta,
            dk,
            dv,
            qk_scale,
            ctx.sm_scale,
            T,
            BLOCK_M=bm,
            BLOCK_N=bn,
            HEAD_DIM=D,
            CAUSAL=ctx.causal,
            num_warps=warps,
        )
        _attn_bwd_dq[(triton.cdiv(T, bm), B * H)](
            q,
            k,
            v,
            do,
            lse,
            delta,
            dq,
            qk_scale,
            ctx.sm_scale,
            T,
            BLOCK_M=bm,
            BLOCK_N=bn,
            HEAD_DIM=D,
            CAUSAL=ctx.causal,
            num_warps=warps,
        )
        return dq, dk, dv, None, None, None, None


def flash_attention(q, k, v, causal=True, sm_scale=None, block_m=None, block_n=None):
    """softmax(q k^T * scale [+ causal mask]) v for q, k, v of shape (batch, heads, seq, head_dim)."""
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    return FlashAttentionFn.apply(q, k, v, causal, sm_scale, block_m, block_n)


def attention_reference(q, k, v, causal=True, sm_scale=None):
    """Naive attention that materialises the full T x T score matrix (the baseline)."""
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = (q.float() @ k.float().transpose(-1, -2)) * sm_scale
    if causal:
        T = q.shape[-2]
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)
