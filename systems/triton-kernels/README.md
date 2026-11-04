# GPU kernels in Triton: fused softmax and FlashAttention-2 (forward + backward)

**Status: kernels written and verified against PyTorch on CPU (Triton interpreter). GPU benchmarks have not been run yet.** Run `bench.py` on a CUDA GPU to fill in `results/`.

| Kernel | File | Idea |
|---|---|---|
| Fused softmax | [`softmax.py`](softmax.py) | One program per row keeps the row in registers. Max, exp, sum and divide take one read and one write of HBM instead of five round trips. |
| Online softmax | [`softmax.py`](softmax.py) | Rows of any length, processed in tiles. The running max and normaliser are rescaled by exp(m − m′) when the max grows (Milakov & Gimelshein, 2018). |
| FlashAttention-2 forward | [`flash_attention.py`](flash_attention.py) | Tiles Q, K and V through SRAM with an online softmax. The T×T attention matrix is never written to memory, so memory is O(T) instead of O(T²). Causal blocks above the diagonal are skipped. |
| FlashAttention-2 backward | [`flash_attention.py`](flash_attention.py) | Recomputes P from Q, K and the saved log-sum-exp instead of storing it. One kernel accumulates dK/dV per key block, another dQ per query block. |

`flash_attention(q, k, v, causal=True)` is a `torch.autograd.Function`, so it can replace `F.scaled_dot_product_attention` in a model that trains. It supports head dims 16/32/64/128, any sequence length, and a custom softmax scale.

## Why these two
Softmax is the simplest kernel where **memory traffic, not arithmetic, is the bottleneck**. Fusing it is the canonical example of why kernel fusion matters. FlashAttention applies the same insight to attention, which is why long-context training is affordable at all. Writing the backward pass means deriving the gradient of a tiled online softmax:

```
dV_j = Σ_i P_ijᵀ dO_i      dP_ij = dO_i V_jᵀ      dS_ij = P_ij ∘ (dP_ij − D_i),  D_i = rowsum(dO_i ∘ O_i)
dQ_i = scale · Σ_j dS_ij K_j                     dK_j = scale · Σ_i dS_ijᵀ Q_i
```

## Correctness (`pytest tests/test_triton_kernels.py`, 20 tests, ~5 s on CPU)
The tests run the real kernel code on CPU through Triton's interpreter (`TRITON_INTERPRET=1`), so they run in CI without a GPU.
- **Softmax:** fused and online softmax match `torch.softmax` across odd shapes, strided inputs, logits of ±1e4, and rows whose leading tiles are entirely −∞.
- **Attention forward:** matches materialised attention, causal and non-causal, with sequence lengths that don't divide the block size.
- **Attention backward:** gradients for dQ, dK and dV match autograd through the naive implementation, and `torch.autograd.gradcheck` (finite differences, float64) passes.

## Benchmarks (run on a GPU)
```bash
pip install torch triton matplotlib
python bench.py
```
`bench.py` runs:
- a bf16 correctness check at T = 1000;
- softmax throughput (GB/s) vs. `torch.softmax` and the 5-kernel eager version, for row lengths 256–32k;
- attention forward and forward+backward TFLOP/s and peak memory vs. PyTorch SDPA (Flash backend) and naive attention, for T = 512–8192.

Outputs go to `results/benchmark.md`, `results/softmax.png` and `results/attention.png`.

**What to expect:**
- Fused softmax should be several times faster than the eager composition and close to `torch.softmax`.
- Naive attention runs out of memory at long sequences while both Flash implementations stay O(T).
- This kernel will very likely be slower than PyTorch's FlashAttention, which is hand-tuned CUDA. The gap should be measured and explained: no autotuning, no warp specialisation, no TMA/async copies on Hopper, and a simple two-kernel backward.

## References
Dao et al., *FlashAttention* (2022) · Dao, *FlashAttention-2* (2023) · Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018) · Tillet et al., *Triton* (2019) and the Triton tutorials (fused softmax, fused attention).
