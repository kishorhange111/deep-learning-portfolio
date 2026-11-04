"""
Extending a RoPE model's context: plain extrapolation vs position interpolation, NTK-aware scaling
and YaRN, without and with a short long-context fine-tune.

LLMs are pretrained at one context length and then stretched to longer ones (Code Llama, Qwen,
Llama 3 and others use these methods). This study does the same at CPU scale:
  1. Train a 4-layer byte-level GPT with rotary position embeddings on 128-byte windows of Python
     source (the standard library), ~8M bytes.
  2. Without any training, evaluate it on windows of 128, 256, 512 and 1,024 bytes with each method
     (none = plain extrapolation, pi, ntk, yarn; scale factor = length / 128).
  3. Fine-tune for 200 steps on 512-byte windows with each method active (the same bytes per step as
     pretraining), plus two controls: plain RoPE at 512 bytes, and 200 more steps at 128 bytes (to
     separate "long-context training" from "more training"). Evaluate again at 512 and 1,024 bytes,
     with the scale factor growing with the length (dynamic) and kept at the fine-tuning factor (fixed).

Statistics: every setting is scored on the same 64 held-out windows; differences are paired
bootstrap 95 % intervals over windows. Per-position loss curves show where each method breaks.

    python train.py      # ~25 min on one CPU core -> results/study.json
    python analyze.py    # figures and results/summary.md
    QUICK=1 python train.py
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # <repo>/src
from dl_common.byte_corpus import fixed_windows, load_corpus, random_windows, synthetic_corpus  # noqa: E402
from dl_common.bytelm import ROPE_METHODS, ByteGPT, ByteGPTConfig, lr_schedule  # noqa: E402


def parse_args(argv=None):
    """Options; QUICK=1 makes everything tiny."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_len", type=int, default=128)
    p.add_argument("--eval_lens", nargs="*", type=int, default=[128, 256, 512, 1024])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--ft_len", type=int, default=512)
    p.add_argument("--ft_steps", type=int, default=200)
    p.add_argument("--ft_lr", type=float, default=1e-3)
    p.add_argument("--eval_windows", type=int, default=64)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--checkpoint", default=os.path.join(HERE, "checkpoints", "rope_base.pt"))
    p.add_argument("--out", default=os.path.join(HERE, "results", "study.json"))
    args = p.parse_args(argv)
    if os.environ.get("QUICK"):
        args.train_len, args.eval_lens, args.steps, args.batch = 16, [16, 32, 64], 20, 8
        args.n_layer, args.d_model, args.ft_len, args.ft_steps, args.eval_windows = 1, 32, 32, 5, 8
        args.synthetic = True
    return args


def train(model, data, steps, batch, seq_len, lr, seed, warmup):
    """AdamW with warmup and cosine decay on random windows. Returns seconds taken."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95)
    )
    rng, t0 = np.random.default_rng(seed), time.time()
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * lr_schedule(step, steps, warmup)
        x, y = random_windows(data, batch, seq_len, rng)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return time.time() - t0


@torch.no_grad()
def position_losses(model, x, y, batch=8):
    """(windows, positions) cross-entropy in bits for every byte of every window."""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        logits, _ = model(torch.from_numpy(x[i : i + batch]))
        ce = F.cross_entropy(logits.transpose(1, 2), torch.from_numpy(y[i : i + batch]), reduction="none")
        out.append(ce.numpy() / math.log(2))
    return np.concatenate(out)


def evaluate(model, windows, train_len, methods, lengths, fixed_factor=None):
    """{method: {length: {mean bits/byte, per-window means, per-position curve}}} on the first `length`
    bytes of each held-out window. The scale factor is length / train_len ("dynamic"), or
    `fixed_factor` at every length (the factor a model was fine-tuned with)."""
    x_all, y_all = windows
    res = {}
    for method in methods:
        res[method] = {}
        for length in lengths:
            model.set_rope_scaling(method, fixed_factor or max(1.0, length / train_len))
            ce = position_losses(model, x_all[:, :length], y_all[:, :length])
            res[method][str(length)] = {
                "bpb": float(ce.mean()),
                "per_window": ce.mean(1).tolist(),
                "per_position": ce.mean(0).tolist(),
            }
        model.set_rope_scaling("none", 1.0)
    return res


def main(argv=None):
    """Pretrain (or load), evaluate every method zero-shot, fine-tune with each, evaluate again."""
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    train_data, val_data, info = synthetic_corpus() if args.synthetic else load_corpus()
    cfg = ByteGPTConfig(
        seq_len=args.train_len, n_layer=args.n_layer, d_model=args.d_model, n_head=4, d_ff=4 * args.d_model, pos="rope"
    )
    torch.manual_seed(0)
    base = ByteGPT(cfg)
    out = {"settings": {k: v for k, v in vars(args).items() if k not in ("out", "checkpoint")}, "corpus": info}
    if os.path.exists(args.checkpoint) and not os.environ.get("QUICK"):
        ckpt = torch.load(args.checkpoint)
        base.load_state_dict(ckpt["model"])
        out["pretrain_seconds"] = ckpt["seconds"]
    else:
        out["pretrain_seconds"] = train(
            base, train_data, args.steps, args.batch, args.train_len, args.lr, seed=0, warmup=max(1, args.steps // 20)
        )
        if not os.environ.get("QUICK"):
            os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
            torch.save({"model": base.state_dict(), "seconds": out["pretrain_seconds"]}, args.checkpoint)
    print(f"pretrained ({out['pretrain_seconds']:.0f}s)", flush=True)

    windows = fixed_windows(val_data, args.eval_windows, max(args.eval_lens), seed=1)
    out["zero_shot"] = evaluate(base, windows, args.train_len, ROPE_METHODS, args.eval_lens)
    print(
        "zero-shot: "
        + ", ".join(f"{m} {out['zero_shot'][m][str(max(args.eval_lens))]['bpb']:.3f}" for m in ROPE_METHODS),
        flush=True,
    )

    # fine-tune at ft_len with each method active (same bytes per step as pretraining)
    ft_batch = max(1, args.batch * args.train_len // args.ft_len)
    ft_lens = [n for n in args.eval_lens if n >= args.ft_len] or [args.ft_len]
    out["fine_tuned"] = {}
    variants = [(m, args.ft_len) for m in ROPE_METHODS] + [("none", args.train_len)]
    for method, length in variants:
        name = method if length == args.ft_len else "more_short_training"
        model = copy.deepcopy(base)
        model.set_rope_scaling(method, max(1.0, length / args.train_len))
        secs = train(
            model,
            train_data,
            args.ft_steps,
            ft_batch if length == args.ft_len else args.batch,
            length,
            args.ft_lr,
            seed=1,
            warmup=max(1, args.ft_steps // 10),
        )
        eval_methods = [method] if name != "more_short_training" else list(ROPE_METHODS)
        res = evaluate(model, windows, args.train_len, eval_methods, ft_lens)
        out["fine_tuned"][name] = {"seconds": secs, "results": res}
        if name != "more_short_training":  # also keep the fine-tuning scale factor at every length
            fixed = evaluate(model, windows, args.train_len, eval_methods, ft_lens, args.ft_len / args.train_len)
            out["fine_tuned"][name]["results_fixed"] = fixed
        print(
            f"fine-tuned {name}: " + ", ".join(f"{m} {r[str(max(ft_lens))]['bpb']:.3f}" for m, r in res.items()),
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
