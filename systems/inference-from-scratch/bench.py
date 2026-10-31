"""
Inference benchmarks.

  kv         ms per generated token vs. sequence position, naive vs. KV cache (random weights are
             fine: speed does not depend on what the weights are)
  batch      decode throughput vs. batch size with the KV cache: decoding is memory-bound, so one
             read of the weights serves the whole batch and throughput grows almost linearly at first
  spec       speculative decoding with a trained draft and target from the pretraining project:
             acceptance rate, tokens per target call vs. theory, and wall-clock speed-up vs. plain
             KV decoding, for k in {1, 2, 4, 6, 8} at temperature 0 and 1

    python bench.py kv batch                    # any GPU (or CPU, slowly)
    python bench.py spec --target_ckpt target.pt --draft_ckpt draft.pt   # data from the pretraining project
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from decoding import expected_tokens_per_round, generate_kv, generate_speculative  # noqa: E402
from kv_model import GPT, GPTConfig  # noqa: E402

RESULTS = os.path.join(HERE, "results")
SIZES = {"85M": (12, 768, 12), "302M": (24, 1024, 16)}


def device_and_dtype():
    """bf16 on GPU, fp32 on CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def random_model(size, device, dtype):
    """A model of the given size with random weights (for speed-only benchmarks)."""
    n_layer, d_model, n_head = SIZES[size]
    return GPT(GPTConfig(n_layer=n_layer, d_model=d_model, n_head=n_head)).to(device, dtype).eval()


def sync(device):
    """Wait for queued GPU work so timings are real."""
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def per_token_latency(model, positions, device, cached):
    """Time to produce one token when `pos` tokens already exist, for each pos."""
    out = []
    for pos in positions:
        ctx = torch.randint(0, 50000, (1, pos), device=device)
        if cached:
            cache = model.new_cache(1, pos + 1)
            model(ctx, cache)

            def step(cache=cache, pos=pos, ctx=ctx):
                cache.truncate(pos)  # drop the previous rep's token, then decode one position
                model(ctx[:, -1:], cache)
        else:

            def step(ctx=ctx):
                model(torch.cat([ctx, ctx[:, -1:]], 1))  # recompute all pos + 1 positions

        for _ in range(3):
            step()
        sync(device)
        t0 = time.perf_counter()
        reps = 10
        for _ in range(reps):
            step()
        sync(device)
        out.append(1000 * (time.perf_counter() - t0) / reps)
    return out


def bench_kv(args):
    """Naive vs. cached per-token latency as the sequence grows."""
    device, dtype = device_and_dtype()
    model = random_model(args.size, device, dtype)
    positions = [64, 128, 256, 512, 1000]
    naive, cached = (
        per_token_latency(model, positions, device, False),
        per_token_latency(model, positions, device, True),
    )
    rows = [{"position": p, "naive_ms": a, "kv_ms": b, "speedup": a / b} for p, a, b in zip(positions, naive, cached)]
    lines = ["| Tokens already generated | Naive (ms/token) | KV cache (ms/token) | Speed-up |", "|---|---|---|---|"]
    lines += [f"| {r['position']} | {r['naive_ms']:.2f} | {r['kv_ms']:.2f} | {r['speedup']:.1f}× |" for r in rows]
    return rows, lines


@torch.no_grad()
def bench_batch(args):
    """Decode tokens/s vs. batch size (KV cache, 128-token prompt, 128 new tokens)."""
    device, dtype = device_and_dtype()
    model = random_model(args.size, device, dtype)
    rows = []
    for b in [1, 2, 4, 8, 16, 32, 64]:
        prompt = torch.randint(0, 50000, (b, 128), device=device)
        generate_kv(model, prompt, 8)  # warm-up
        sync(device)
        t0 = time.perf_counter()
        generate_kv(model, prompt, 128)
        sync(device)
        rows.append({"batch": b, "tok_per_s": b * 128 / (time.perf_counter() - t0)})
    base = rows[0]["tok_per_s"]
    lines = ["| Batch | Tokens/s | vs. batch 1 |", "|---|---|---|"]
    lines += [f"| {r['batch']} | {r['tok_per_s']:,.0f} | {r['tok_per_s'] / base:.1f}× |" for r in rows]
    return rows, lines


def load_prompts(data_dir, n_prompts, length, seed=0):
    """Token windows from the validation shard (same distribution the models were trained on)."""
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    shard = np.memmap(os.path.join(data_dir, meta["val"][0]["path"]), dtype=np.uint16, mode="r")
    starts = np.random.default_rng(seed).integers(0, len(shard) - length, n_prompts)
    return [torch.tensor(np.asarray(shard[s : s + length], dtype=np.int64))[None] for s in starts]


def bench_spec(args):
    """Speculative decoding vs. plain KV decoding with trained models."""
    device, dtype = device_and_dtype()
    target = GPT.from_checkpoint(args.target_ckpt, device).to(dtype)
    draft = GPT.from_checkpoint(args.draft_ckpt, device).to(dtype)
    prompts = [p.to(device) for p in load_prompts(args.data_dir, args.n_prompts, args.prompt_len)]
    rows = []
    for temperature in (0.0, 1.0):
        gen = torch.Generator(device=device).manual_seed(0)
        base_s = sum(generate_kv(target, p, args.new_tokens, temperature, generator=gen)[1].seconds for p in prompts)
        for k in (1, 2, 4, 6, 8):
            stats = [
                generate_speculative(target, draft, p, args.new_tokens, k, temperature, generator=gen)[1]
                for p in prompts
            ]
            alpha = sum(s.accepted for s in stats) / sum(s.proposed for s in stats)
            per_call = sum(s.new_tokens for s in stats) / sum(s.target_forwards for s in stats)
            rows.append(
                {
                    "temperature": temperature,
                    "k": k,
                    "acceptance": alpha,
                    "tokens_per_target_call": per_call,
                    "theory": expected_tokens_per_round(alpha, k),
                    "speedup": base_s / sum(s.seconds for s in stats),
                }
            )
            print(rows[-1])
    lines = [
        "| Temperature | k | Acceptance α | Tokens per target call | Theory (1−α^(k+1))/(1−α) | Wall-clock speed-up |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['temperature']:.0f} | {r['k']} | {r['acceptance']:.2f} | {r['tokens_per_target_call']:.2f} | "
        f"{r['theory']:.2f} | {r['speedup']:.2f}× |"
        for r in rows
    ]
    return rows, lines


def plot(name, rows):
    """One figure per benchmark."""
    from common.plotting import ColorMap, apply_style

    plt = apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    if name == "kv":
        c = ColorMap(["naive", "KV cache"])
        ax.plot(
            [r["position"] for r in rows], [r["naive_ms"] for r in rows], marker="o", color=c["naive"], label="naive"
        )
        ax.plot(
            [r["position"] for r in rows], [r["kv_ms"] for r in rows], marker="o", color=c["KV cache"], label="KV cache"
        )
        ax.set_xlabel("tokens already in the sequence")
        ax.set_ylabel("ms per new token")
        ax.set_title("Per-token latency: recompute vs. cache")
        ax.legend()
    elif name == "batch":
        ax.plot([r["batch"] for r in rows], [r["tok_per_s"] for r in rows], marker="o", color="#2a78d6")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch size")
        ax.set_ylabel("generated tokens / s")
        ax.set_title("Decode throughput vs. batch size")
    else:
        c = ColorMap(["temperature 0", "temperature 1"])
        for t in (0.0, 1.0):
            pts = [(r["k"], r["speedup"]) for r in rows if r["temperature"] == t]
            ax.plot(*zip(*pts), marker="o", color=c[f"temperature {t:.0f}"], label=f"temperature {t:.0f}")
        ax.axhline(1.0, color="#898781", linewidth=1)
        ax.set_xlabel("draft tokens per round k")
        ax.set_ylabel("speed-up vs. plain KV decoding")
        ax.set_title("Speculative decoding")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"{name}.png"), dpi=150)
    plt.close(fig)


def main(argv=None):
    """Run the requested benchmarks and write results/<name>.md, .json, .png."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("which", nargs="+", choices=["kv", "batch", "spec"])
    p.add_argument("--size", default="85M", choices=list(SIZES))
    p.add_argument("--target_ckpt")
    p.add_argument("--draft_ckpt")
    p.add_argument(
        "--data_dir", default=os.path.join(HERE, "..", "gpt-pretraining-scaling-laws", "data", "fineweb_edu")
    )
    p.add_argument("--n_prompts", type=int, default=20)
    p.add_argument("--prompt_len", type=int, default=128)
    p.add_argument("--new_tokens", type=int, default=128)
    args = p.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    for name in args.which:
        rows, lines = {"kv": bench_kv, "batch": bench_batch, "spec": bench_spec}[name](args)
        text = f"Device: {gpu}\n\n" + "\n".join(lines) + "\n"
        print(text)
        with open(os.path.join(RESULTS, f"{name}.md"), "w") as f:
            f.write(text)
        with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
            json.dump({"device": gpu, "rows": rows}, f, indent=1)
        plot(name, rows)


if __name__ == "__main__":
    main()
