"""
Post-training weight quantization from scratch: how far can a trained language model's weights be
squeezed, and how much does GPTQ's error compensation buy over plain rounding?

Steps:
  1. Train a 4-layer byte-level GPT on Python source (the standard library), ~7M tokens, or load
     it from checkpoints/ if it already exists.
  2. Quantize every linear layer inside the transformer blocks (embeddings and the output head stay
     in float, as is usual) with round-to-nearest (RTN) and with GPTQ, at 8, 4, 3 and 2 bits, per
     output channel or in groups of 64 or 32 input columns.
  3. Measure held-out bits per byte for each, plus the storage cost in effective bits per weight
     (scales and zero points included).
  4. Ablations: which layer types are most sensitive (quantize one type at a time), and how much
     calibration data GPTQ needs. GPTQ results use 3 calibration samples (different windows) and
     report mean +/- std.

    python train.py      # ~15 min on a laptop CPU -> results/quantization.json
    python analyze.py    # figures and results/summary.md
    QUICK=1 python train.py
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # <repo>/src
import weight_quant as wq  # noqa: E402

from dl_common.byte_corpus import fixed_windows, load_corpus, random_windows, synthetic_corpus  # noqa: E402
from dl_common.bytelm import ByteGPT, ByteGPTConfig, evaluate_bpb, lr_schedule  # noqa: E402


def parse_args(argv=None):
    """Options; QUICK=1 makes everything tiny."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--bits", nargs="*", type=int, default=[8, 4, 3, 2])
    p.add_argument("--groups", nargs="*", default=["channel", "64", "32"])
    p.add_argument("--calib_windows", type=int, default=128)
    p.add_argument("--calib_sizes", nargs="*", type=int, default=[4, 16, 64, 256])
    p.add_argument("--calib_seeds", type=int, default=3)
    p.add_argument("--eval_windows", type=int, default=1024)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--checkpoint", default=os.path.join(HERE, "checkpoints", "base_model.pt"))
    p.add_argument("--out", default=os.path.join(HERE, "results", "quantization.json"))
    args = p.parse_args(argv)
    if os.environ.get("QUICK"):
        args.steps, args.batch, args.seq_len, args.n_layer, args.d_model = 30, 8, 32, 2, 64
        args.bits, args.groups, args.calib_windows, args.calib_sizes = [8, 3], ["channel", "32"], 16, [4, 16]
        args.calib_seeds, args.eval_windows, args.synthetic = 2, 64, True
    return args


def train_base(args, data):
    """Train (or load) the float model. Returns (model, training info)."""
    train, val, info = data
    cfg = ByteGPTConfig(
        seq_len=args.seq_len, n_layer=args.n_layer, d_model=args.d_model, n_head=4, d_ff=4 * args.d_model
    )
    torch.manual_seed(0)
    model = ByteGPT(cfg)
    if os.path.exists(args.checkpoint) and not os.environ.get("QUICK"):
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt["model"])
        return model, ckpt["info"]
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    rng, t0 = np.random.default_rng(0), time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_schedule(step, args.steps, max(1, args.steps // 20))
        x, y = random_windows(train, args.batch, args.seq_len, rng)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(
                f"train step {step}: loss {loss.item() / np.log(2):.3f} bits/byte ({time.time() - t0:.0f}s)", flush=True
            )
    train_info = {
        "steps": args.steps,
        "tokens": args.steps * args.batch * args.seq_len,
        "seconds": time.time() - t0,
        "corpus": info,
    }
    if not os.environ.get("QUICK"):
        os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
        torch.save({"model": model.state_dict(), "info": train_info}, args.checkpoint)
    return model, train_info


def group_value(g):
    """'channel' -> None, '64' -> 64."""
    return None if g == "channel" else int(g)


def run_one(base, method, bits, group, calib, val_xy, only=None):
    """Quantize a copy of the model and evaluate it."""
    model = copy.deepcopy(base)
    t0 = time.time()
    errors = wq.quantize_model(model, method, bits, group, calib_x=calib, only=only)
    seconds = time.time() - t0
    return {"val_bpb": evaluate_bpb(model, *val_xy), "weight_rel_error": errors, "quantize_seconds": seconds}


def main(argv=None):
    """Train or load the model, then run every quantization setting and write results/quantization.json."""
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    data = synthetic_corpus() if args.synthetic else load_corpus()
    train, val, _ = data
    base, train_info = train_base(args, data)
    val_xy = fixed_windows(val, args.eval_windows, args.seq_len, seed=1)
    calibs = [
        fixed_windows(train, max(args.calib_windows, max(args.calib_sizes)), args.seq_len, seed=100 + s)[0]
        for s in range(args.calib_seeds)
    ]
    layers = wq.block_linears(base)
    out = {
        "settings": {k: v for k, v in vars(args).items() if k not in ("out", "checkpoint")},
        "train": train_info,
        "float_val_bpb": evaluate_bpb(base, *val_xy),
        "layer_shapes": {n: list(m.weight.shape) for n, m in layers.items()},
        "quantized_params": int(sum(m.weight.numel() for m in layers.values())),
        "float_params_kept": int(base.num_params() - sum(m.weight.numel() for m in layers.values())),
        "grid": [],
    }
    print(f"float model: {out['float_val_bpb']:.4f} bits/byte", flush=True)
    t0 = time.time()
    for bits in args.bits:
        for g in args.groups:
            group = group_value(g)
            eff = float(np.mean([wq.effective_bits(*m.weight.shape, bits, group) for m in layers.values()]))
            row = {"bits": bits, "group": g, "effective_bits": eff}
            row["rtn"] = run_one(base, "rtn", bits, group, None, val_xy)
            row["gptq"] = [run_one(base, "gptq", bits, group, c[: args.calib_windows], val_xy) for c in calibs]
            out["grid"].append(row)
            gq = np.mean([r["val_bpb"] for r in row["gptq"]])
            print(
                f"{bits} bits, group {g:>7}: RTN {row['rtn']['val_bpb']:.4f}  GPTQ {gq:.4f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )

    # Which layer types hurt most at low precision? Quantize one type at a time (RTN, 3 bits, per channel).
    sens_bits = 3 if 3 in args.bits else min(args.bits)
    out["sensitivity"] = {"bits": sens_bits, "layers": {}}
    for kind in ("attn.qkv", "attn.out", "mlp.fc", "mlp.proj"):
        out["sensitivity"]["layers"][kind] = run_one(base, "rtn", sens_bits, None, None, val_xy, only=kind)["val_bpb"]

    # How much calibration data does GPTQ need? (3 bits, group 64 if available)
    calib_group = "64" if "64" in args.groups else args.groups[-1]
    out["calibration"] = {"bits": sens_bits, "group": calib_group, "sizes": {}}
    for n in args.calib_sizes:
        out["calibration"]["sizes"][str(n)] = [
            run_one(base, "gptq", sens_bits, group_value(calib_group), c[:n], val_xy)["val_bpb"] for c in calibs
        ]
    out["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
