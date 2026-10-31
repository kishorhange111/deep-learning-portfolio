"""
Fit a compute-optimal scaling law from an IsoFLOP sweep (Chinchilla, Hoffmann et al. 2022).

For each compute budget C, train several model sizes so that each one spends exactly C FLOPs
(smaller models see more tokens). Loss against log(model size) is roughly a parabola, and its
minimum is the compute-optimal size N_opt(C) for that budget. Across budgets, log N_opt is linear in
log C, and the slope is the scaling exponent. Chinchilla found ~0.5: grow parameters and tokens
equally. Kaplan et al. (2020) had found ~0.73.

Two estimates of the exponent:
  approach 2: parabola minimum per budget, then a line through the (log C, log N_opt) points
  approach 3: fit L(N, D) = E + A / N^alpha + B / D^beta to every run; N_opt ~ C^(beta / (alpha + beta))

    python scaling.py plan                       # the grid and its estimated GPU time
    python scaling.py run [--nproc 8]            # train every point (resumable)
    python scaling.py fit                        # results/scaling_summary.md + figures
    python scaling.py lr-sweep --preset 25M      # check the per-size learning rates
    python scaling.py bench --nproc 8            # DDP vs FSDP throughput and memory
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from gpt import PRESETS, config_for, flops_per_token, non_embedding_params  # noqa: E402

RESULTS = os.path.join(HERE, "results")
SWEEP_DIR = os.path.join(RESULTS, "isoflop")
BUDGETS = [3e16, 1e17, 3e17]
TOKENS_PER_PARAM_RANGE = (1.0, 300.0)  # sizes per budget: from very over- to very under-trained


# --------------------------------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------------------------------
def grid(budgets=BUDGETS, seq_len=1024):
    """[(budget, preset, tokens)] for every preset whose tokens/parameter at that budget is in range."""
    points = []
    for c in budgets:
        for preset in PRESETS:
            cfg = config_for(preset, seq_len=seq_len)
            tokens = c / flops_per_token(cfg)
            ratio = tokens / non_embedding_params(cfg)
            if TOKENS_PER_PARAM_RANGE[0] <= ratio <= TOKENS_PER_PARAM_RANGE[1]:
                points.append((c, preset, tokens))
    return points


def run_name(budget, preset):
    """Directory name of one sweep point, e.g. 'C1e+17_25M'."""
    return f"C{budget:.0e}_{preset}"


def launch(nproc, train_args):
    """Run train.py, through torchrun when nproc > 1."""
    train_py = os.path.join(HERE, "train.py")
    if nproc > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", train_py]
    else:
        cmd = [sys.executable, train_py]
    cmd += train_args
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def cmd_plan(args):
    """Print the grid with tokens, tokens/param and estimated hours at an assumed MFU."""
    total = 0.0
    print(f"{'budget':>8} {'preset':>6} {'tokens':>10} {'tok/param':>9}")
    for c, preset, tokens in grid(args.budgets):
        ratio = tokens / non_embedding_params(config_for(preset))
        print(f"{c:8.0e} {preset:>6} {tokens / 1e6:9.0f}M {ratio:9.1f}")
        total += c
    hours = total / (args.peak_tflops * 1e12 * args.mfu * args.nproc) / 3600
    print(
        f"\n{len(grid(args.budgets))} runs, {total:.2e} FLOPs total: ~{hours:.1f} h on {args.nproc} GPU(s) "
        f"at {100 * args.mfu:.0f}% MFU of {args.peak_tflops:.0f} TFLOP/s"
    )


def cmd_run(args):
    """Train every grid point that has no metrics.json yet."""
    for c, preset, _ in grid(args.budgets):
        name = run_name(c, preset)
        if os.path.exists(os.path.join(SWEEP_DIR, name, "metrics.json")):
            print(f"[skip] {name}")
            continue
        launch(
            args.nproc,
            [
                "--preset",
                preset,
                "--flops",
                str(c),
                "--run_name",
                name,
                "--out_dir",
                SWEEP_DIR,
                "--parallel",
                args.parallel,
                *(["--compile"] if args.compile else []),
            ],
        )


# --------------------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------------------
def load_points(sweep_dir=SWEEP_DIR):
    """[{budget, N, D, loss}] from finished runs."""
    pts = []
    for path in glob.glob(os.path.join(sweep_dir, "*", "metrics.json")):
        with open(path) as f:
            m = json.load(f)
        budget = float(os.path.basename(os.path.dirname(path)).split("_")[0][1:])
        pts.append(
            {
                "budget": budget,
                "N": m["non_embedding_params"],
                "N_total": m.get("total_params"),
                "D": m["tokens"],
                "loss": m["final_val_loss"],
                "preset": m["preset"],
            }
        )
    return sorted(pts, key=lambda p: (p["budget"], p["N"]))


def parabola_minimum(n, loss):
    """Fit loss = a (ln N)^2 + b ln N + c; returns (N_opt, L_opt, coeffs, inside) - inside is False
    when the fitted minimum lies outside the sizes that were trained (an extrapolation)."""
    x = np.log(np.asarray(n, dtype=float))
    a, b, c = np.polyfit(x, np.asarray(loss, dtype=float), 2)
    if a <= 0:
        i = int(np.argmin(loss))
        return float(n[i]), float(loss[i]), (a, b, c), False
    x_opt = -b / (2 * a)
    return float(np.exp(x_opt)), float(a * x_opt**2 + b * x_opt + c), (a, b, c), bool(x.min() <= x_opt <= x.max())


def fit_exponent(budgets, n_opts):
    """Least-squares slope and intercept of ln N_opt against ln C."""
    slope, intercept = np.polyfit(np.log(budgets), np.log(n_opts), 1)
    return float(slope), float(intercept)


def approach2(points):
    """Per-budget parabola minima, the fitted exponent, and its leave-one-run-out range."""
    by_budget = {}
    for p in points:
        by_budget.setdefault(p["budget"], []).append(p)
    minima = {}
    for c, pts in sorted(by_budget.items()):
        if len(pts) >= 3:
            n_opt, l_opt, coeffs, inside = parabola_minimum([p["N"] for p in pts], [p["loss"] for p in pts])
            minima[c] = {"N_opt": n_opt, "L_opt": l_opt, "coeffs": list(coeffs), "inside_grid": inside}
    if len(minima) < 2:
        return {"minima": minima}
    cs = sorted(minima)
    slope, intercept = fit_exponent(cs, [minima[c]["N_opt"] for c in cs])
    # sensitivity: drop each single run (if its budget keeps >= 3 points) and refit
    loo = []
    for drop in points:
        rest = [p for p in points if p is not drop]
        sub = approach2_minima(rest)
        if len(sub) == len(minima):
            loo.append(fit_exponent(sorted(sub), [sub[c] for c in sorted(sub)])[0])
    return {
        "minima": minima,
        "exponent": slope,
        "intercept": intercept,
        "exponent_loo_range": [min(loo), max(loo)] if loo else None,
    }


def approach2_minima(points):
    """{budget: N_opt} for budgets with >= 3 points (helper for the sensitivity analysis)."""
    by_budget = {}
    for p in points:
        by_budget.setdefault(p["budget"], []).append(p)
    return {
        c: parabola_minimum([p["N"] for p in pts], [p["loss"] for p in pts])[0]
        for c, pts in by_budget.items()
        if len(pts) >= 3
    }


def approach3(points, n_starts=64, seed=0):
    """Fit L = E + A/N^alpha + B/D^beta (Huber loss on log L, as in Chinchilla) by multi-start L-BFGS.

    Parameterised as E = exp(e), A = exp(a), B = exp(b) so all three stay positive."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    n = np.array([p["N"] for p in points], dtype=float)
    d = np.array([p["D"] for p in points], dtype=float)
    log_l = np.log(np.array([p["loss"] for p in points], dtype=float))

    def objective(theta):
        a, b, e, alpha, beta = theta
        pred = logsumexp([a - alpha * np.log(n), b - beta * np.log(d), np.full_like(n, e)], axis=0)
        r = pred - log_l
        delta = 1e-3
        return np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta)).sum()

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        x0 = [rng.uniform(0, 25), rng.uniform(0, 25), rng.uniform(-1, 1.5), rng.uniform(0, 2), rng.uniform(0, 2)]
        res = minimize(objective, x0, method="L-BFGS-B", bounds=[(-5, 40), (-5, 40), (-3, 3), (0.01, 3), (0.01, 3)])
        if best is None or res.fun < best.fun:
            best = res
    a, b, e, alpha, beta = best.x
    return {
        "E": float(np.exp(e)),
        "A": float(np.exp(a)),
        "B": float(np.exp(b)),
        "alpha": float(alpha),
        "beta": float(beta),
        "exponent": float(beta / (alpha + beta)),
        "objective": float(best.fun),
    }


def write_summary(points, a2, a3, path):
    """Markdown table of runs and both exponent estimates."""
    lines = [
        "| Budget (FLOPs) | Model | Non-emb. params | Tokens | Tokens/param | Val loss |",
        "|---|---|---|---|---|---|",
    ]
    for p in points:
        lines.append(
            f"| {p['budget']:.0e} | {p['preset']} | {p['N'] / 1e6:.1f}M | {p['D'] / 1e6:.0f}M | "
            f"{p['D'] / p['N']:.1f} | {p['loss']:.4f} |"
        )
    lines += ["", "| Budget | N_opt (parabola) | min inside grid? | Loss at N_opt |", "|---|---|---|---|"]
    for c, m in a2["minima"].items():
        inside = "yes" if m["inside_grid"] else "no (extrapolated)"
        lines.append(f"| {c:.0e} | {m['N_opt'] / 1e6:.1f}M | {inside} | {m['L_opt']:.4f} |")
    lines.append("")
    if "exponent" in a2:
        rng = a2["exponent_loo_range"]
        lines.append(
            f"**Approach 2:** N_opt ∝ C^{a2['exponent']:.3f}"
            + (f" (leave-one-run-out range {rng[0]:.3f}–{rng[1]:.3f})" if rng else "")
        )
    if "exponent" in a2.get("total_params", {}):
        lines.append(
            f"**Approach 2, total parameters (incl. embeddings):** N_opt ∝ C^{a2['total_params']['exponent']:.3f}"
        )
    if a3:
        lines.append(
            f"**Approach 3:** L = {a3['E']:.3f} + {a3['A']:.3g}/N^{a3['alpha']:.3f} + {a3['B']:.3g}/D^{a3['beta']:.3f}"
            f"  →  N_opt ∝ C^{a3['exponent']:.3f}"
        )
    lines.append(
        "\nReference: Chinchilla ≈ 0.50 (approaches 1–2: 0.50 / 0.49; approach 3: 0.46), Kaplan et al. ≈ 0.73."
    )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def plot_isoflop(points, a2, a3, path_iso, path_frontier):
    """IsoFLOP parabolas (one sequential-blue line per budget) and the N_opt-vs-C frontier."""
    from common.plotting import INK_2, MUTED, apply_style

    plt = apply_style()
    blues = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal ramp: light = small budget
    budgets = sorted(a2["minima"])
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for i, c in enumerate(budgets):
        pts = [p for p in points if p["budget"] == c]
        color = blues[min(i, len(blues) - 1)]
        ax.plot([p["N"] for p in pts], [p["loss"] for p in pts], "o", color=color, label=f"{c:.0e} FLOPs")
        a, b, cc = a2["minima"][c]["coeffs"]
        xs = np.linspace(np.log(min(p["N"] for p in pts)), np.log(max(p["N"] for p in pts)), 100)
        ax.plot(np.exp(xs), a * xs**2 + b * xs + cc, color=color, linewidth=1.5)
        m = a2["minima"][c]
        ax.plot(m["N_opt"], m["L_opt"], marker="*", markersize=13, color=color, markeredgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("non-embedding parameters")
    ax.set_ylabel("validation loss (nats/token)")
    ax.set_title("IsoFLOP curves: loss vs. model size at fixed compute")
    ax.legend(title="budget", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path_iso, dpi=150)
    plt.close(fig)

    if "exponent" not in a2:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    cs = np.array(budgets)
    n_opt = np.array([a2["minima"][c]["N_opt"] for c in budgets])
    ax.plot(cs, n_opt, "o", color="#2a78d6", markersize=8, label="parabola minima")
    xs = np.geomspace(cs.min() / 3, cs.max() * 10, 50)
    ax.plot(
        xs,
        np.exp(a2["intercept"]) * xs ** a2["exponent"],
        color="#2a78d6",
        label=f"fit: N_opt ∝ C^{a2['exponent']:.2f}",
    )
    ref = n_opt[0] * (xs / cs[0]) ** 0.5
    ax.plot(xs, ref, "--", color=MUTED, linewidth=1.5, label="Chinchilla slope 0.5 (anchored at first point)")
    if a3:
        ax.text(
            0.02,
            0.95,
            f"approach 3: exponent {a3['exponent']:.2f}",
            transform=ax.transAxes,
            color=INK_2,
            fontsize=9,
            va="top",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training compute C (FLOPs)")
    ax.set_ylabel("compute-optimal non-embedding params")
    ax.set_title("Compute-optimal model size")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path_frontier, dpi=150)
    plt.close(fig)


def cmd_fit(args, sweep_dir=SWEEP_DIR, out_dir=RESULTS):
    """Fit both approaches, write the summary and figures."""
    points = load_points(sweep_dir)
    if len(points) < 3:
        print(f"only {len(points)} finished runs in {sweep_dir}")
        return None
    a2 = approach2(points)
    a3 = approach3(points) if len(points) >= 6 else None
    # Kaplan counted non-embedding parameters, Chinchilla total parameters; at small scale the choice
    # shifts the exponent (Pearce & Song, 2024), so report both.
    if all(p.get("N_total") for p in points):
        a2["total_params"] = approach2([{**p, "N": p["N_total"]} for p in points])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scaling_fit.json"), "w") as f:
        json.dump({"points": points, "approach2": a2, "approach3": a3}, f, indent=1, default=str)
    print(write_summary(points, a2, a3, os.path.join(out_dir, "scaling_summary.md")))
    plot_isoflop(points, a2, a3, os.path.join(out_dir, "isoflop.png"), os.path.join(out_dir, "frontier.png"))
    return {"approach2": a2, "approach3": a3}


# --------------------------------------------------------------------------------------------------
# Learning-rate check and parallelism benchmark
# --------------------------------------------------------------------------------------------------
def cmd_lr_sweep(args):
    """Same model and budget at several peak LRs; the table's LR should sit near the minimum."""
    out = os.path.join(RESULTS, "lr_sweep")
    rows = []
    for lr in args.lrs:
        name = f"{args.preset}_lr{lr:g}"
        if not os.path.exists(os.path.join(out, name, "metrics.json")):
            launch(
                args.nproc,
                [
                    "--preset",
                    args.preset,
                    "--flops",
                    str(args.flops),
                    "--lr",
                    str(lr),
                    "--run_name",
                    name,
                    "--out_dir",
                    out,
                    "--parallel",
                    args.parallel,
                ],
            )
        with open(os.path.join(out, name, "metrics.json")) as f:
            rows.append((lr, json.load(f)["final_val_loss"]))
    text = "| peak LR | val loss |\n|---|---|\n" + "\n".join(f"| {lr:g} | {loss:.4f} |" for lr, loss in rows)
    with open(os.path.join(RESULTS, f"lr_sweep_{args.preset}.md"), "w") as f:
        f.write(text + "\n")
    print(text)


def cmd_bench(args):
    """Throughput (tokens/s/GPU), MFU and peak memory for each parallelism mode on synthetic data."""
    out = os.path.join(RESULTS, "parallel_bench")
    modes = ["none"] if args.nproc == 1 else ["ddp", "fsdp"]
    rows = []
    for mode in modes:
        name = f"{args.preset}_{mode}_w{args.nproc}"
        launch(
            args.nproc,
            [
                "--preset",
                args.preset,
                "--synthetic",
                "--parallel",
                mode,
                "--max_steps",
                str(args.steps),
                "--micro_batch",
                str(args.micro_batch),
                "--run_name",
                name,
                "--out_dir",
                out,
                "--eval_interval",
                "100000",
                *(["--compile"] if args.compile else []),
            ],
        )
        with open(os.path.join(out, name, "metrics.json")) as f:
            rows.append(json.load(f))
    lines = ["| Mode | GPUs | GPU | Tokens/s/GPU | MFU | Peak memory (GB) |", "|---|---|---|---|---|---|"]
    for m in rows:
        mfu = f"{100 * m['mfu']:.1f} %" if m["mfu"] else "—"
        mem = f"{m['peak_memory_gb']:.1f}" if m["peak_memory_gb"] else "—"
        lines.append(
            f"| {m['parallel']} | {m['world_size']} | {m['gpu']} | {m['tok_per_s_per_gpu']:,.0f} | {mfu} | {mem} |"
        )
    text = "\n".join(lines)
    with open(os.path.join(RESULTS, f"parallel_bench_{args.preset}_w{args.nproc}.md"), "w") as f:
        f.write(text + "\n")
    print(text)


def main(argv=None):
    """Sub-command dispatcher."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "fit", "lr-sweep", "bench"):
        s = sub.add_parser(name)
        s.add_argument("--budgets", nargs="*", type=float, default=BUDGETS)
        s.add_argument("--nproc", type=int, default=1)
        s.add_argument("--parallel", default="ddp")
        s.add_argument("--compile", action="store_true")
        s.add_argument("--peak_tflops", type=float, default=312.0)
        s.add_argument("--mfu", type=float, default=0.35)
        s.add_argument("--preset", default="25M")
        s.add_argument("--flops", type=float, default=3e16)
        s.add_argument("--lrs", nargs="*", type=float, default=[3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3])
        s.add_argument("--steps", type=int, default=30)
        s.add_argument("--micro_batch", type=int, default=16)
    args = p.parse_args(argv)
    if args.nproc == 1 and args.parallel == "ddp":
        args.parallel = "none"
    return {"plan": cmd_plan, "run": cmd_run, "fit": cmd_fit, "lr-sweep": cmd_lr_sweep, "bench": cmd_bench}[args.cmd](
        args
    )


if __name__ == "__main__":
    main()
