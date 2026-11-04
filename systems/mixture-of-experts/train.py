"""
Mixture of experts from scratch: does sparse routing buy quality per FLOP, what does the
load-balancing loss do, and what do the experts specialize in?

Every run trains the same 4-layer byte-level GPT on Python source (the standard library), for the
same steps, data order and optimizer. Only the feed-forward layers change:
  * dense           - one MLP with d_ff = 512 (the compute baseline);
  * moe8_top1       - 8 experts of d_ff 512, each token routed to 1: same active parameters as dense,
                      8x the stored MLP parameters;
  * moe8_top2       - the same experts, 2 per token (Mixtral-style): 2x the active MLP parameters;
  * dense_wide      - one MLP with d_ff = 1024: the dense model with moe8_top2's active parameters;
  * moe8_top1_nobal - moe8_top1 without the load-balancing loss (does the router collapse?);
  * moe8_top1_cap1  - moe8_top1 with capacity factor 1.0: overflow tokens skip the layer;
  * moe8_top1_unitgate - moe8_top1 without the Switch gate's down-scaling of expert outputs.
Then dense and moe8_top1_unitgate are re-run for 3x the steps (does the gap change with training?).

Measurements, on held-out files: bits per byte, time per step, expert load per layer (entropy,
max / min share, experts with < 1 % of tokens), and specialization: how well the current byte's
class (letter, digit, space, bracket, ...) and the byte itself predict the router's choice.

    python train.py                 # all configurations x 2 seeds (~1.5 CPU-hours)
    python train.py --only dense:0  # one run: config:seed
    python analyze.py               # figures and results/summary.md
    QUICK=1 python train.py         # smoke test on a synthetic corpus
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
import moe_layer as ml  # noqa: E402

from dl_common.byte_corpus import BYTE_CLASS_TABLE, fixed_windows, load_corpus, random_windows, synthetic_corpus  # noqa: E402
from dl_common.bytelm import MLP, ByteGPT, ByteGPTConfig, evaluate_bpb, lr_schedule  # noqa: E402

RUNS_DIR = os.path.join(HERE, "results", "runs")
CONFIGS = {
    "dense": {"kind": "dense", "d_ff": 512},
    "moe8_top1": {"kind": "moe", "d_ff": 512, "experts": 8, "top_k": 1, "balance": 0.01},
    "moe8_top2": {"kind": "moe", "d_ff": 512, "experts": 8, "top_k": 2, "balance": 0.01},
    "dense_wide": {"kind": "dense", "d_ff": 1024},
    "moe8_top1_nobal": {"kind": "moe", "d_ff": 512, "experts": 8, "top_k": 1, "balance": 0.0},
    "moe8_top1_cap1": {"kind": "moe", "d_ff": 512, "experts": 8, "top_k": 1, "balance": 0.01, "capacity": 1.0},
    "moe8_top1_unitgate": {"kind": "moe", "d_ff": 512, "experts": 8, "top_k": 1, "balance": 0.01, "unit_gate": True},
}
LONG = {"configs": ["dense", "moe8_top1_unitgate"], "steps": 3000}  # horizon check, seed 0


def parse_args(argv=None):
    """Options; QUICK=1 makes everything tiny."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", default=None, help="runs as config:seed")
    p.add_argument("--configs", nargs="*", default=list(CONFIGS))
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_windows", type=int, default=128)
    p.add_argument("--final_eval_windows", type=int, default=1024)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--bench", action="store_true", help="only time a training step per configuration (interleaved)")
    p.add_argument("--out_dir", default=RUNS_DIR)
    args = p.parse_args(argv)
    if os.environ.get("QUICK"):
        args.steps, args.warmup, args.batch, args.seq_len, args.n_layer, args.d_model = 10, 2, 4, 32, 2, 32
        args.eval_every, args.eval_windows, args.final_eval_windows, args.synthetic, args.seeds = 5, 8, 32, True, 1
    return args


def build_model(conf, args):
    """ByteGPT with dense or MoE feed-forward layers."""
    cfg = ByteGPTConfig(seq_len=args.seq_len, n_layer=args.n_layer, d_model=args.d_model, n_head=4, d_ff=conf["d_ff"])
    scale = args.d_model / 128  # QUICK uses a narrower model; keep d_ff / d_model fixed
    d_ff = max(8, int(conf["d_ff"] * scale))
    if conf["kind"] == "dense":
        return ByteGPT(cfg, mlp_factory=lambda layer: MLP(args.d_model, d_ff))
    return ByteGPT(
        cfg,
        mlp_factory=lambda layer: ml.MoELayer(
            args.d_model,
            d_ff,
            conf["experts"],
            conf["top_k"],
            conf["balance"],
            capacity_factor=conf.get("capacity"),
            unit_gate=conf.get("unit_gate", False),
        ),
    )


def moe_layers(model):
    """The MoE layers of a model, in depth order."""
    return [m for m in model.modules() if isinstance(m, ml.MoELayer)]


@torch.no_grad()
def routing_analysis(model, x, n_experts):
    """Per MoE layer: expert load over held-out windows, and how predictable the router's first choice
    is from the current byte's class and from the byte itself (lookup tables fitted on half the
    windows, scored on the other half)."""
    model.eval()
    model(torch.from_numpy(x))
    model.train()
    flat = x.reshape(-1)
    classes = BYTE_CLASS_TABLE[flat]
    half = len(flat) // 2
    out = []
    for layer in moe_layers(model):
        choice = layer.last_stats["first_choice"].numpy()
        load = np.bincount(choice, minlength=n_experts) / len(choice)
        nz = load[load > 0]
        row = {
            "load": load.tolist(),
            "load_entropy_bits": float(-(nz * np.log2(nz)).sum()),
            "max_share": float(load.max()),
            "min_share": float(load.min()),
            "experts_under_1pct": int((load < 0.01).sum()),
            "dropped_frac": layer.last_stats["dropped_frac"],
            "class_by_expert": np.stack(
                [np.bincount(classes[choice == e], minlength=BYTE_CLASS_TABLE.max() + 1) for e in range(n_experts)]
            ).tolist(),
        }
        for key, feature in (("class", classes), ("byte", flat)):
            table = {}
            for f_val, e in zip(feature[:half], choice[:half]):
                table.setdefault(int(f_val), np.zeros(n_experts))[e] += 1
            majority = int(np.bincount(choice[:half], minlength=n_experts).argmax())
            pred = np.array([table[int(f)].argmax() if int(f) in table else majority for f in feature[half:]])
            row[f"predict_from_{key}"] = float((pred == choice[half:]).mean())
        row["predict_majority"] = float(
            (choice[half:] == np.bincount(choice[:half], minlength=n_experts).argmax()).mean()
        )
        out.append(row)
    return out


def benchmark_step_time(args, data, configs=tuple(CONFIGS), rounds=5, steps=10):
    """Median time of a full training step per configuration, interleaving configurations round by round
    so they see the same machine load (the training runs' own timings overlap with other jobs)."""
    train = data[0]
    rng = np.random.default_rng(0)
    setups = {}
    for name in configs:
        torch.manual_seed(0)
        model = build_model(CONFIGS[name], args)
        setups[name] = (model, torch.optim.AdamW(model.parameters(), lr=args.lr))
    times = {name: [] for name in configs}
    for r in range(rounds + 1):  # round 0 is warm-up
        for name in configs:
            model, opt = setups[name]
            for _ in range(steps):
                x, y = random_windows(train, args.batch, args.seq_len, rng)
                t0 = time.perf_counter()
                _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
                (loss + model.aux_loss()).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                if r:
                    times[name].append(time.perf_counter() - t0)
    return {n: {"ms_per_step": 1000 * float(np.median(v)), "threads": args.threads} for n, v in times.items()}


def train_run(name, seed, data, args):
    """Train one configuration; return curves, final loss, timing and routing analysis."""
    conf = CONFIGS[name]
    train, val, info = data
    torch.manual_seed(seed)
    model = build_model(conf, args)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    rng = np.random.default_rng(seed)
    vx, vy = fixed_windows(val, args.eval_windows, args.seq_len, seed=0)
    log, t0 = [], time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            rec = {"step": step, "val_bpb": evaluate_bpb(model, vx, vy), "seconds": time.time() - t0}
            layers = moe_layers(model)
            if layers:
                rec["balance_loss"] = float(np.mean([m.last_stats["balance_loss"] for m in layers]))
            log.append(rec)
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_schedule(step, args.steps, args.warmup)
        x, y = random_windows(train, args.batch, args.seq_len, rng)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        total = loss + model.aux_loss()
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
    train_seconds = time.time() - t0
    fx, fy = fixed_windows(val, args.final_eval_windows, args.seq_len, seed=1)
    mlp0 = model.blocks[0].mlp
    n_mlp_active = n_mlp_total = sum(p.numel() for p in mlp0.parameters())
    if conf["kind"] == "moe":
        expert_ff = mlp0.experts[0].fc.out_features
        n_mlp_active, n_mlp_total = ml.active_and_total_mlp_params(
            args.d_model, expert_ff, conf["experts"], conf["top_k"]
        )
    return {
        "config": name,
        "settings_conf": conf,
        "seed": seed,
        "log": log,
        "final_val_bpb": evaluate_bpb(model, fx, fy),
        "seconds": train_seconds,
        "seconds_per_step": train_seconds / args.steps,
        "params_total": model.num_params(),
        "mlp_params_active_per_layer": n_mlp_active,
        "mlp_params_total_per_layer": n_mlp_total,
        "routing": routing_analysis(model, fx[:256], conf["experts"]) if conf["kind"] == "moe" else None,
        "corpus": info,
        "settings": {k: v for k, v in vars(args).items() if k not in ("only", "out_dir")},
    }


def main(argv=None):
    """Run every configuration and seed (or the --only runs), then the horizon check; one JSON per run."""
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
    if args.only:
        specs = [(name, int(seed), args.steps) for name, seed in (s.split(":") for s in args.only)]
    else:
        specs = [(c, s, args.steps) for s in range(args.seeds) for c in args.configs]
        long_steps = LONG["steps"] if not os.environ.get("QUICK") else 2 * args.steps
        specs += [(c, 0, long_steps) for c in LONG["configs"] if c in args.configs]
    results = []
    for name, seed, steps in specs:
        suffix = f"_steps{steps}" if steps != args.steps else ""  # the horizon-check runs
        path = os.path.join(args.out_dir, f"{name}_s{seed}{suffix}.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
            continue
        run_args = argparse.Namespace(**{**vars(args), "steps": steps})
        rec = train_run(name, seed, data, run_args)
        with open(path, "w") as f:
            json.dump(rec, f)
        print(
            f"{name:>18} s{seed} ({steps} steps): val {rec['final_val_bpb']:.4f} bpb, "
            f"{rec['seconds_per_step'] * 1000:.0f} ms/step",
            flush=True,
        )
        results.append(rec)
    return results


if __name__ == "__main__":
    main()
