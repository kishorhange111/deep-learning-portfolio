"""
Grokking on modular addition: delayed generalization, the Fourier algorithm behind it, and the
progress measures that show it forming long before test accuracy moves.

A replication of Power et al. (2022), "Grokking: Generalization Beyond Overfitting on Small
Algorithmic Datasets", and Nanda et al. (2023), "Progress measures for grokking via mechanistic
interpretability", run end to end on a CPU.

A 1-layer transformer is trained full-batch on 30 % of the 113 x 113 pairs (a, b) -> (a + b) mod 113
with AdamW and strong weight decay. It fits the training set within a few hundred steps, sits at
chance on the held-out pairs for thousands of steps, and then generalizes abruptly. The study asks:
  * is the delay real and does it need weight decay?  (weight decay 1.0 vs 0, same seed)
  * how does it depend on the amount of data?        (training fraction 0.2 - 0.7)
  * what did the model learn?                         (Fourier sparsity of the embedding, the share
                                                       of the logits in a few key frequencies, MLP
                                                       neurons that each read a single frequency)
  * when did it learn it?                             (restricted and excluded loss on saved
                                                       snapshots: the circuit forms during the
                                                       plateau, then weight decay cleans up)

    python train.py                       # the full study (~3 CPU-core-hours; finished runs are skipped)
    python train.py --only main_s0        # one named run
    python analyze.py                     # figures and results/summary.md from results/runs/*.json
    QUICK=1 python train.py               # smoke test on p = 23 in a few seconds
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
import grok_model as gm  # noqa: E402

RUNS_DIR = os.path.join(HERE, "results", "runs")


def study_runs(steps):
    """The named runs of the study: {name: settings}. Sweep runs stop early once they have generalized."""
    runs = {"main_s0": {"wd": 1.0, "frac": 0.3, "seed": 0, "steps": steps, "progress": True}}
    for seed in (1, 2):  # extra seeds for the timing of the phases: stop 4,000 steps after generalizing
        runs[f"main_s{seed}"] = {
            "wd": 1.0,
            "frac": 0.3,
            "seed": seed,
            "steps": steps,
            "progress": True,
            "stop_after_grok": 4000,
        }
    runs["no_weight_decay_s0"] = {"wd": 0.0, "frac": 0.3, "seed": 0, "steps": steps, "progress": True}
    for frac in (0.2, 0.4, 0.5, 0.7):
        runs[f"frac{frac}_s0"] = {"wd": 1.0, "frac": frac, "seed": 0, "steps": steps, "stop_after_grok": 500}
    return runs


def parse_args(argv=None):
    """Options; QUICK=1 makes everything tiny."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", default=None, help="run only these named runs")
    p.add_argument("--p", type=int, default=113)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--snapshot_every", type=int, default=250, help="snapshots for the progress measures")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    p.add_argument("--out_dir", default=RUNS_DIR)
    p.add_argument("--force", action="store_true", help="re-run runs whose results already exist")
    args = p.parse_args(argv)
    if os.environ.get("QUICK"):
        args.p, args.steps, args.eval_every, args.snapshot_every = 23, 60, 10, 20
    return args


def grok_step(log, key="test_acc", threshold=0.99):
    """First logged step at which `key` reaches `threshold`, or None."""
    return next((r["step"] for r in log if r[key] >= threshold), None)


def train_run(name, run, args):
    """Train one model; return its log, final measurements and (if run['progress']) progress measures."""
    torch.manual_seed(run["seed"])
    cfg = gm.GrokConfig(p=args.p)
    if os.environ.get("QUICK"):
        cfg.d_model, cfg.n_head, cfg.d_mlp = 32, 4, 64
    model = gm.ModAddTransformer(cfg)
    x, y, train_idx, test_idx = gm.make_dataset(cfg.p, run["frac"], run["seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=run["wd"], betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    x_train, y_train, x_test, y_test = x[train_idx], y[train_idx], x[test_idx], y[test_idx]

    log, snapshots, t0, stop_at = [], [], time.time(), None
    for step in range(run["steps"] + 1):
        if step % args.eval_every == 0 or step == run["steps"]:
            with torch.no_grad():
                tr, te = model(x_train), model(x_test)
            rec = {
                "step": step,
                "train_loss": float(gm.loss_fn(tr, y_train)),
                "test_loss": float(gm.loss_fn(te, y_test)),
                "train_acc": gm.accuracy(tr, y_train),
                "test_acc": gm.accuracy(te, y_test),
                "weight_norm": gm.weight_norm(model),
            }
            log.append(rec)
            if step % (args.eval_every * 40) == 0:
                print(
                    f"[{name}] step {step:6d}  train {rec['train_loss']:.2e} / {rec['train_acc']:.3f}"
                    f"  test {rec['test_loss']:.2e} / {rec['test_acc']:.3f}  {time.time() - t0:.0f}s",
                    flush=True,
                )
            if stop_at is None and run.get("stop_after_grok") and rec["test_acc"] >= 0.99:
                stop_at = step + run["stop_after_grok"]
        if run.get("progress") and step % args.snapshot_every == 0:
            snapshots.append((step, copy.deepcopy(model.state_dict())))
        if step == run["steps"] or (stop_at is not None and step >= stop_at):
            break
        loss = gm.loss_fn(model(x_train), y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    out = {"name": name, "settings": run, "config": vars(cfg), "lr": args.lr, "log": log}
    out["seconds"] = time.time() - t0
    out["grok_step"] = grok_step(log)
    out["memorize_step"] = grok_step(log, "train_acc", 0.99)
    out["final"] = final_analysis(model, x, y, train_idx)
    if run.get("progress"):
        out["progress"] = progress_measures(model, snapshots, x, y, train_idx, out["final"]["key_frequencies"])
    return out


def final_analysis(model, x, y, train_idx):
    """What the trained model computes: key frequencies, their share of the logits, neuron purity."""
    p = model.cfg.p
    logits = gm.logit_grid(model, x)
    shares = gm.frequency_shares_of_logits(logits)
    keys = gm.key_frequencies(logits)
    best, purity = gm.neuron_frequency_purity(model, x)
    emb = gm.embedding_frequency_norms(model)
    mask = torch.zeros(p * p, dtype=torch.bool)
    mask[train_idx] = True
    restricted, excluded = gm.restricted_and_excluded_loss(logits, y.reshape(p, p), keys, mask)
    alive = np.isfinite(purity)
    active = purity[alive]
    return {
        "key_frequencies": keys,
        "logit_share_by_frequency": shares.tolist(),
        "logit_share_key_frequencies": float(shares[[k - 1 for k in keys]].sum()) if keys else 0.0,
        "embedding_norm_by_frequency": emb.tolist(),
        "embedding_gini": gm.gini(emb),
        "neuron_best_frequency": best[alive].tolist(),
        "neuron_purity": active.tolist(),
        "dead_neurons": int((~alive).sum()),
        "neuron_purity_median": float(np.median(active)),
        "neurons_purity_over_0.8": float((active > 0.8).mean()),
        "neurons_on_key_frequencies": float(np.isin(best[alive], keys).mean()) if keys else 0.0,
        "restricted_loss": restricted,
        "excluded_loss_train": excluded,
    }


def progress_measures(model, snapshots, x, y, train_idx, keys):
    """Restricted / excluded loss and embedding sparsity on each snapshot, using the final model's key frequencies."""
    p = model.cfg.p
    mask = torch.zeros(p * p, dtype=torch.bool)
    mask[train_idx] = True
    y_grid = y.reshape(p, p)
    probe = copy.deepcopy(model)
    rows = []
    for step, state in snapshots:
        probe.load_state_dict(state)
        logits = gm.logit_grid(probe, x)
        restricted, excluded = gm.restricted_and_excluded_loss(logits, y_grid, keys, mask)
        shares = gm.frequency_shares_of_logits(logits)
        rows.append(
            {
                "step": step,
                "restricted_loss": restricted,
                "excluded_loss_train": excluded,
                "embedding_gini": gm.gini(gm.embedding_frequency_norms(probe)),
                "logit_share_key_frequencies": float(shares[[k - 1 for k in keys]].sum()) if keys else 0.0,
            }
        )
    return rows


def main(argv=None):
    """Run the study (or the runs named with --only) and write one JSON per run."""
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    runs = study_runs(args.steps)
    names = args.only if args.only else list(runs)
    if os.environ.get("QUICK") and not args.only:
        names = ["main_s0", "frac0.5_s0"]
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for name in names:
        path = os.path.join(args.out_dir, f"{name}.json")
        if os.path.exists(path) and not args.force:
            print(f"[{name}] already done: {path}")
            with open(path) as f:
                results[name] = json.load(f)
            continue
        results[name] = train_run(name, runs[name], args)
        with open(path, "w") as f:
            json.dump(results[name], f)
        r = results[name]
        print(
            f"[{name}] memorized at {r['memorize_step']}, generalized at {r['grok_step']}, keys {r['final']['key_frequencies']}"
        )
    return results


if __name__ == "__main__":
    main()
