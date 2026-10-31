"""
GPU benchmarks: the Triton kernels against PyTorch.

  softmax    (4096 rows x N): fused Triton vs. torch.softmax vs. the 5-kernel eager composition.
             Reported in GB/s of useful traffic (read input + write output once) against the
             GPU's memory bandwidth, since softmax is memory-bound.
  attention  (B=4, H=16, D=64, causal): our FlashAttention-2 vs. PyTorch SDPA with the Flash backend
             vs. naive attention that materialises T x T. Reported in TFLOP/s (forward and
             forward+backward) and in peak memory.

    python bench.py              # needs a CUDA GPU; writes results/*.md and results/*.png
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from flash_attention import attention_reference, flash_attention  # noqa: E402
from softmax import softmax, softmax_naive  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def bench_ms(fn):
    """Median runtime in ms (triton.testing.do_bench handles warm-up, L2 flushing and repeats)."""
    import triton.testing

    return triton.testing.do_bench(fn)


def softmax_bench(rows=4096, sizes=(256, 512, 1024, 2048, 4096, 8192, 16384, 32768), dtype=torch.float16):
    """GB/s per provider and row length."""
    out = []
    for n in sizes:
        x = torch.randn(rows, n, device="cuda", dtype=dtype)
        gbytes = 2 * x.numel() * x.element_size() / 1e9
        row = {"N": n}
        for name, fn in [
            ("triton fused", softmax),
            ("torch.softmax", lambda t: torch.softmax(t, -1)),
            ("eager (5 kernels)", softmax_naive),
        ]:
            row[name] = gbytes / (bench_ms(lambda fn=fn, x=x: fn(x)) / 1e3)
        out.append(row)
        print(row)
    return out


def attention_flops(B, H, T, D, causal, mode):
    """4 B H T^2 D for the two matmuls (halved if causal); backward = 2.5x forward (FlashAttention-2 paper)."""
    f = 4 * B * H * T * T * D * (0.5 if causal else 1.0)
    return f if mode == "fwd" else 3.5 * f


def attention_bench(B=4, H=16, D=64, seqs=(512, 1024, 2048, 4096, 8192), causal=True, dtype=torch.bfloat16):
    """TFLOP/s (fwd, fwd+bwd) and peak memory per provider and sequence length."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def sdpa(q, k, v):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

    providers = {
        "ours (Triton FA2)": lambda q, k, v: flash_attention(q, k, v, causal=causal),
        "PyTorch SDPA (Flash)": sdpa,
        "naive (T x T)": lambda q, k, v: attention_reference(q, k, v, causal=causal),
    }
    out = []
    for T in seqs:
        q, k, v = (torch.randn(B, H, T, D, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3))
        do = torch.randn_like(q)
        row = {"T": T}
        for name, fn in providers.items():
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                ms_f = bench_ms(lambda fn=fn, q=q, k=k, v=v: fn(q, k, v))

                def fwd_bwd(fn=fn, q=q, k=k, v=v, do=do):
                    o = fn(q, k, v)
                    torch.autograd.grad(o, (q, k, v), do)

                ms_fb = bench_ms(fwd_bwd)
                row[name] = {
                    "fwd_tflops": attention_flops(B, H, T, D, causal, "fwd") / ms_f / 1e9,
                    "fwd_bwd_tflops": attention_flops(B, H, T, D, causal, "fwd_bwd") / ms_fb / 1e9,
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
            except torch.OutOfMemoryError:
                row[name] = None  # the naive version runs out of memory at long sequences
        out.append(row)
        print(row)
    return out


def correctness_on_gpu(dtype=torch.bfloat16):
    """Max abs error vs. the fp32 reference at realistic sizes (the CPU tests use tiny ones)."""
    q, k, v = (torch.randn(2, 8, 1000, 64, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3))
    o = flash_attention(q, k, v)
    ref = attention_reference(q.float(), k.float(), v.float())
    do = torch.randn_like(o)
    grads = torch.autograd.grad(o, (q, k, v), do)
    ref_grads = torch.autograd.grad(ref, (q, k, v), do.float())
    return {
        "fwd_max_abs_err": (o.float() - ref).abs().max().item(),
        "bwd_max_abs_err": max((a.float() - b).abs().max().item() for a, b in zip(grads, ref_grads)),
    }


def plot(soft, attn, gpu):
    """Line charts: softmax GB/s vs N, attention fwd+bwd TFLOP/s vs T."""
    from common.plotting import ColorMap, apply_style

    plt = apply_style()
    colors = ColorMap(["triton fused", "torch.softmax", "eager (5 kernels)"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in ("triton fused", "torch.softmax", "eager (5 kernels)"):
        ax.plot([r["N"] for r in soft], [r[name] for r in soft], marker="o", color=colors[name], label=name)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("row length N (4096 rows, fp16)")
    ax.set_ylabel("GB/s")
    ax.set_title(f"Softmax throughput on {gpu}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "softmax.png"), dpi=150)
    plt.close(fig)

    names = ["ours (Triton FA2)", "PyTorch SDPA (Flash)", "naive (T x T)"]
    colors = ColorMap(names)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in names:
        pts = [(r["T"], r[name]["fwd_bwd_tflops"]) for r in attn if r.get(name)]
        if pts:
            ax.plot(*zip(*pts), marker="o", color=colors[name], label=name)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length T (B=4, H=16, D=64, causal, bf16)")
    ax.set_ylabel("TFLOP/s (forward + backward)")
    ax.set_title(f"Attention throughput on {gpu}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "attention.png"), dpi=150)
    plt.close(fig)


def write_markdown(soft, attn, check, gpu):
    """results/benchmark.md with both tables and the correctness check."""
    lines = [
        f"GPU: **{gpu}** · correctness vs fp32 reference (bf16, T=1000): forward max error "
        f"{check['fwd_max_abs_err']:.2e}, backward {check['bwd_max_abs_err']:.2e}",
        "",
        "| Softmax N | Triton fused (GB/s) | torch.softmax | eager 5-kernel |",
        "|---|---|---|---|",
    ]
    for r in soft:
        lines.append(
            f"| {r['N']} | {r['triton fused']:.0f} | {r['torch.softmax']:.0f} | {r['eager (5 kernels)']:.0f} |"
        )
    lines += [
        "",
        "| Attention T | ours fwd / fwd+bwd (TFLOP/s) | SDPA Flash fwd / fwd+bwd | naive fwd / fwd+bwd | "
        "peak memory ours / SDPA / naive (GB) |",
        "|---|---|---|---|---|",
    ]

    def cell(r, n):
        return f"{r[n]['fwd_tflops']:.0f} / {r[n]['fwd_bwd_tflops']:.0f}" if r.get(n) else "OOM"

    def mem(r, n):
        return f"{r[n]['peak_mem_gb']:.1f}" if r.get(n) else "OOM"

    names = ["ours (Triton FA2)", "PyTorch SDPA (Flash)", "naive (T x T)"]
    for r in attn:
        lines.append(
            f"| {r['T']} | {cell(r, names[0])} | {cell(r, names[1])} | {cell(r, names[2])} | "
            f"{' / '.join(mem(r, n) for n in names)} |"
        )
    with open(os.path.join(RESULTS, "benchmark.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    """Run everything and write results/."""
    if not torch.cuda.is_available():
        sys.exit("bench.py needs a CUDA GPU (the CPU tests in tests/test_triton_kernels.py check correctness)")
    os.makedirs(RESULTS, exist_ok=True)
    gpu = torch.cuda.get_device_name()
    check = correctness_on_gpu()
    soft, attn = softmax_bench(), attention_bench()
    with open(os.path.join(RESULTS, "benchmark.json"), "w") as f:
        json.dump({"gpu": gpu, "correctness": check, "softmax": soft, "attention": attn}, f, indent=1)
    write_markdown(soft, attn, check, gpu)
    plot(soft, attn, gpu)


if __name__ == "__main__":
    main()
