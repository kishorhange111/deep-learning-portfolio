"""
AdamW vs. Muon vs. Lion on a byte-level language model of Python source code.

Optimizer comparisons are easy to get wrong: a new optimizer at a tuned learning rate beats a
baseline at an untuned one, a single seed's noise passes for an effect, or a per-step win hides a
per-second loss. This study fixes the protocol before running:
  * the same model (a 4-layer byte-level GPT, 0.86M parameters), data order, schedule (50 warmup
    steps, cosine to 10 %) and step budget for every optimizer;
  * a learning-rate sweep for each optimizer over a 3x grid wide enough that the best value is
    inside it, never at an edge;
  * the best learning rate of each re-run with 3 seeds; the comparison is mean +/- std of the final
    validation bits per byte, with a Welch t-test;
  * efficiency reported per step and per second of wall-clock, because Muon's Newton-Schulz
    iterations cost extra matrix multiplies (timed in a separate, interleaved benchmark);
  * two fairness checks: AdamW with gradient clipping (standard for GPT training; Muon does not
    need it), and the best AdamW and Muon settings re-run for 3x the steps.

Data: the Python standard library's own .py files (no download), split by file into train and
validation (see dl_common/byte_corpus.py).

    python train.py                     # sweeps, 3 seeds at each best LR, horizon check (~1.5 hours on 2 cores)
    python train.py --only muon:0.03:0  # one run: optimizer:lr:seed
    python analyze.py                   # figures and results/summary.md
    QUICK=1 python train.py             # smoke test on a synthetic corpus
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
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # <repo>/src
import optimizers_from_scratch as ofs  # noqa: E402

from dl_common.byte_corpus import fixed_windows, load_corpus, random_windows, synthetic_corpus  # noqa: E402
from dl_common.bytelm import ByteGPT, ByteGPTConfig, evaluate_bpb, lr_schedule  # noqa: E402

RUNS_DIR = os.path.join(HERE, "results", "runs")
LR_GRID = {
    "adamw": [1e-3, 3e-3, 1e-2, 3e-2],
    "lion": [1e-4, 3e-4, 1e-3, 3e-3],
    "muon": [3e-3, 1e-2, 3e-2, 1e-1],
    "adamw+clip": [1e-3, 3e-3, 1e-2],  # fairness control: AdamW with gradient-norm clipping at 1.0
}
LONG_STEPS = 3000  # the horizon check: best LRs of AdamW and Muon with 3x the steps


def parse_args(argv=None):
    """Options; QUICK=1 makes everything tiny."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", default=None, help="runs as optimizer:lr:seed")
    p.add_argument("--optimizers", nargs="*", default=list(LR_GRID))
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_windows", type=int, default=128)
    p.add_argument("--final_eval_windows", type=int, default=1024)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--bench", action="store_true", help="only time a training step per optimizer (interleaved)")
    p.add_argument("--out_dir", default=RUNS_DIR)
    args = p.parse_args(argv)
    if os.environ.get("QUICK"):
        args.steps, args.warmup, args.batch, args.seq_len, args.n_layer, args.d_model = 12, 2, 4, 32, 1, 32
        args.eval_every, args.eval_windows, args.final_eval_windows, args.synthetic = 6, 8, 16, True
    return args


def run_name(opt, lr, seed, steps=1000):
    """File-name-safe run id ('+' becomes '-'; a non-default step budget is appended)."""
    return f"{opt.replace('+', '-')}_lr{lr:g}_s{seed}" + (f"_steps{steps}" if steps != 1000 else "")


def train_run(opt_name, lr, seed, data, args):
    """Train one model; return its validation curve, final validation bpb and timing."""
    train, val, info = data
    torch.manual_seed(seed)
    cfg = ByteGPTConfig(
        seq_len=args.seq_len, n_layer=args.n_layer, d_model=args.d_model, n_head=4, d_ff=4 * args.d_model
    )
    model = ByteGPT(cfg)
    base_name, clip = opt_name.removesuffix("+clip"), opt_name.endswith("+clip")
    opt = ofs.make_optimizer(base_name, model, lr)
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    rng = np.random.default_rng(seed)  # data order depends on the seed only, not on the optimizer
    vx, vy = fixed_windows(val, args.eval_windows, args.seq_len, seed=0)
    log, opt_seconds, t0 = [], 0.0, time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            log.append({"step": step, "val_bpb": evaluate_bpb(model, vx, vy), "seconds": time.time() - t0})
            if not np.isfinite(log[-1]["val_bpb"]):
                break
        if step == args.steps:
            break
        mult = lr_schedule(step, args.steps, args.warmup)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
        x, y = random_windows(train, args.batch, args.seq_len, rng)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        t_opt = time.perf_counter()
        opt.step()
        opt_seconds += time.perf_counter() - t_opt
        if not torch.isfinite(loss):
            log.append({"step": step, "val_bpb": float("nan"), "seconds": time.time() - t0})
            break
    fx, fy = fixed_windows(val, args.final_eval_windows, args.seq_len, seed=1)
    final = evaluate_bpb(model, fx, fy) if np.isfinite(log[-1]["val_bpb"]) else float("nan")
    return {
        "optimizer": opt_name,
        "lr": lr,
        "seed": seed,
        "log": log,
        "final_val_bpb": final,
        "diverged": not np.isfinite(final),
        "seconds": time.time() - t0,
        "optimizer_seconds": opt_seconds,
        "params": model.num_params(),
        "tokens": args.steps * args.batch * args.seq_len,
        "corpus": info,
        "settings": {k: v for k, v in vars(args).items() if k not in ("only", "out_dir")},
    }


def run_and_save(opt_name, lr, seed, data, args):
    """Run unless results exist; returns the record."""
    path = os.path.join(args.out_dir, run_name(opt_name, lr, seed, args.steps) + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    rec = train_run(opt_name, lr, seed, data, args)
    with open(path, "w") as f:
        json.dump(rec, f)
    print(
        f"{run_name(opt_name, lr, seed, args.steps):>22}: val {rec['final_val_bpb']:.4f} bpb  "
        f"{rec['seconds']:.0f}s (optimizer {rec['optimizer_seconds']:.1f}s)",
        flush=True,
    )
    return rec


def benchmark_step_time(args, data, rounds=5, steps_per_round=20):
    """Time a full training step (forward, backward, optimizer) for each optimizer, interleaving the
    optimizers round by round so they see the same machine load. Returns {optimizer: {...}}."""
    train = data[0]
    cfg = ByteGPTConfig(seq_len=args.seq_len, n_layer=args.n_layer, d_model=args.d_model, n_head=4, d_ff=4 * args.d_model)
    names = ["adamw", "lion", "muon"]
    setups = {}
    for name in names:
        torch.manual_seed(0)
        model = ByteGPT(cfg)
        setups[name] = (model, ofs.make_optimizer(name, model, LR_GRID[name][1]))
    rng = np.random.default_rng(0)
    times = {n: {"step": [], "optimizer": []} for n in names}
    for r in range(rounds + 1):  # round 0 is warm-up
        for name in names:
            model, opt = setups[name]
            for _ in range(steps_per_round):
                x, y = random_windows(train, args.batch, args.seq_len, rng)
                t0 = time.perf_counter()
                _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                t1 = time.perf_counter()
                opt.step()
                t2 = time.perf_counter()
                if r:
                    times[name]["step"].append(t2 - t0)
                    times[name]["optimizer"].append(t2 - t1)
    return {
        n: {
            "ms_per_step": 1000 * float(np.median(t["step"])),
            "ms_optimizer": 1000 * float(np.median(t["optimizer"])),
            "steps_timed": len(t["step"]),
            "threads": args.threads,
        }
        for n, t in times.items()
    }


def best_lr(records, opt_name, steps):
    """The learning rate with the lowest final val bpb among the seed-0 sweep runs of `steps` steps."""
    sweep = [
        r
        for r in records
        if r["optimizer"] == opt_name and r["seed"] == 0 and not r["diverged"] and r["settings"]["steps"] == steps
    ]
    return min(sweep, key=lambda r: r["final_val_bpb"])["lr"] if sweep else None


def main(argv=None):
    """Sweep learning rates (seed 0), then re-run each optimizer's best LR with more seeds."""
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    data = synthetic_corpus() if args.synthetic else load_corpus()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.bench:
        out = benchmark_step_time(args, data)
        with open(os.path.join(os.path.dirname(args.out_dir), "step_time.json"), "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps(out, indent=1))
        return out
    records = []
    if args.only:
        for spec in args.only:
            opt_name, lr, seed = spec.split(":")
            records.append(run_and_save(opt_name, float(lr), int(seed), data, args))
        return records
    grid = LR_GRID if not os.environ.get("QUICK") else {k: v[1:3] for k, v in LR_GRID.items()}
    for opt_name in args.optimizers:
        for lr in grid[opt_name]:
            records.append(run_and_save(opt_name, lr, 0, data, args))
    for opt_name in args.optimizers:
        lr = best_lr(records, opt_name, args.steps)
        for seed in range(1, args.seeds):
            records.append(run_and_save(opt_name, lr, seed, data, args))
    # Horizon check: does the gap between AdamW and Muon survive 3x more training?
    long_args = argparse.Namespace(**{**vars(args), "steps": LONG_STEPS if not os.environ.get("QUICK") else 2 * args.steps})
    for opt_name in ("adamw", "muon"):
        if opt_name in args.optimizers:
            records.append(run_and_save(opt_name, best_lr(records, opt_name, args.steps), 0, data, long_args))
    return records


if __name__ == "__main__":
    main()
