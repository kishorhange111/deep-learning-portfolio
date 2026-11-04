"""
Figures and the summary table for the optimizer study, from results/runs/*.json (written by train.py).

    python analyze.py     # -> results/summary.md, results/summary.json and the figures
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # <repo>/src
from dl_common.plotting import MUTED, ColorMap, apply_style  # noqa: E402

RUNS = os.path.join(HERE, "results", "runs")
OUT = os.path.join(HERE, "results")
ORDER = ["adamw", "adamw+clip", "lion", "muon"]


def load_runs(runs_dir=RUNS):
    """List of run records."""
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(path) as f:
            out.append(json.load(f))
    return out


def welch(a, b):
    """Two-sided Welch t-test p-value (NaN if scipy is missing or there are too few seeds)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        from scipy.stats import ttest_ind
    except ImportError:
        return float("nan")
    return float(ttest_ind(a, b, equal_var=False).pvalue)


def first_crossing(steps, values, target):
    """First (linearly interpolated) step at which a decreasing curve reaches `target`, or None."""
    for i in range(1, len(values)):
        if values[i] <= target:
            s0, s1, v0, v1 = steps[i - 1], steps[i], values[i - 1], values[i]
            return float(s0 + (s1 - s0) * (v0 - target) / (v0 - v1)) if v0 != v1 else float(s1)
    return None


def main_steps(runs):
    """The step budget of the main comparison (the most common one)."""
    counts = {}
    for r in runs:
        counts[r["settings"]["steps"]] = counts.get(r["settings"]["steps"], 0) + 1
    return max(counts, key=counts.get)


def summarize(runs, step_time=None):
    """Per optimizer: the sweep, the best LR, its seeds, and efficiency numbers (per-second numbers
    use the interleaved benchmark in step_time.json when it exists)."""
    steps_main = main_steps(runs)
    by_opt = {}
    for r in runs:
        if r["settings"]["steps"] == steps_main:
            by_opt.setdefault(r["optimizer"], []).append(r)
    summary = {}
    for opt in [o for o in ORDER if o in by_opt] + sorted(set(by_opt) - set(ORDER)):
        rs = by_opt[opt]
        sweep = sorted((r for r in rs if r["seed"] == 0), key=lambda r: r["lr"])
        ok = [r for r in sweep if not r["diverged"]]
        if not ok:
            continue
        best = min(ok, key=lambda r: r["final_val_bpb"])["lr"]
        seeds = sorted((r for r in rs if r["lr"] == best), key=lambda r: r["seed"])
        finals = [r["final_val_bpb"] for r in seeds]
        steps = [x["step"] for x in seeds[0]["log"]]
        curves = np.array([[x["val_bpb"] for x in r["log"]] for r in seeds if len(r["log"]) == len(steps)])
        lrs = [r["lr"] for r in sweep]
        summary[opt] = {
            "sweep": {f"{r['lr']:g}": (r["final_val_bpb"] if not r["diverged"] else None) for r in sweep},
            "best_lr": best,
            "best_lr_at_grid_edge": best in (lrs[0], lrs[-1]),
            "final_val_bpb": finals,
            "mean": float(np.mean(finals)),
            "std": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
            "curve_steps": steps,
            "curve_mean": curves.mean(0).tolist(),
            "curve_std": curves.std(0).tolist(),
            "seconds_per_step": (
                step_time[opt.removesuffix("+clip")]["ms_per_step"] / 1000
                if step_time and opt.removesuffix("+clip") in step_time
                else float(np.mean([r["seconds"] for r in seeds]) / seeds[0]["settings"]["steps"])
            ),
            "optimizer_share_of_time": (
                step_time[opt.removesuffix("+clip")]["ms_optimizer"]
                / step_time[opt.removesuffix("+clip")]["ms_per_step"]
                if step_time and opt.removesuffix("+clip") in step_time
                else float(np.mean([r["optimizer_seconds"] / r["seconds"] for r in seeds]))
            ),
            "seeds": len(seeds),
        }
    if "adamw" in summary:
        base = summary["adamw"]
        for opt, s in summary.items():
            s["delta_vs_adamw"] = s["mean"] - base["mean"]
            s["p_vs_adamw"] = welch(s["final_val_bpb"], base["final_val_bpb"]) if opt != "adamw" else None
            # target: AdamW's loss at the end of its own curve (same evaluation windows as every curve)
            reach = first_crossing(s["curve_steps"], s["curve_mean"], base["curve_mean"][-1])
            s["steps_to_adamw_final"] = reach
            s["step_speedup_vs_adamw"] = base["curve_steps"][-1] / reach if reach else None
            s["time_speedup_vs_adamw"] = (
                (base["curve_steps"][-1] * base["seconds_per_step"]) / (reach * s["seconds_per_step"])
                if reach
                else None
            )
    long_runs = [r for r in runs if r["settings"]["steps"] != steps_main]
    if long_runs:
        summary["_horizon"] = {
            f"{r['optimizer']} (lr {r['lr']:g}, {r['settings']['steps']} steps)": {
                "final_val_bpb": r["final_val_bpb"],
                "steps": r["settings"]["steps"],
                "curve": [(x["step"], x["val_bpb"]) for x in r["log"]],
            }
            for r in sorted(long_runs, key=lambda r: r["optimizer"])
        }
        by = {r["optimizer"]: r for r in long_runs}
        if "adamw" in by and "muon" in by:
            target = by["adamw"]["log"][-1]["val_bpb"]
            steps = [x["step"] for x in by["muon"]["log"]]
            reach = first_crossing(steps, [x["val_bpb"] for x in by["muon"]["log"]], target)
            summary["_horizon_speedup"] = {"muon_steps_to_adamw_final": reach, "adamw_steps": steps[-1]}
    return summary


def write_markdown(summary, path):
    """The comparison table."""
    lines = [
        "| Optimizer | Best LR (sweep) | Val bits/byte, mean ± std | Δ vs AdamW | p (Welch) | "
        "Steps to AdamW's final loss | Speedup per step | Speedup per second | Optimizer share of step time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for opt, s in optimizers(summary).items():
        p = f"{s['p_vs_adamw']:.3f}" if s.get("p_vs_adamw") is not None and np.isfinite(s["p_vs_adamw"]) else "—"
        reach = f"{s['steps_to_adamw_final']:.0f}" if s.get("steps_to_adamw_final") else "not reached"
        sp = f"{s['step_speedup_vs_adamw']:.2f}×" if s.get("step_speedup_vs_adamw") else "—"
        tp = f"{s['time_speedup_vs_adamw']:.2f}×" if s.get("time_speedup_vs_adamw") else "—"
        edge = " (grid edge!)" if s["best_lr_at_grid_edge"] else ""
        lines.append(
            f"| {opt} | {s['best_lr']:g}{edge} | {s['mean']:.4f} ± {s['std']:.4f} (n={s['seeds']}) | "
            f"{s.get('delta_vs_adamw', 0):+.4f} | {p} | {reach} | {sp} | {tp} | "
            f"{100 * s['optimizer_share_of_time']:.0f} % |"
        )
    lines += ["", "Learning-rate sweeps (seed 0, final val bits/byte; — = diverged):", ""]
    for opt, s in optimizers(summary).items():
        lines.append(
            f"- **{opt}**: "
            + ", ".join(f"{lr}: {v:.4f}" if v is not None else f"{lr}: —" for lr, v in s["sweep"].items())
        )
    if "_horizon" in summary:
        lines += ["", "Horizon check (seed 0, best LR of the main comparison):", ""]
        for name, h in summary["_horizon"].items():
            lines.append(f"- {name}: final val {h['final_val_bpb']:.4f} bits/byte")
        hs = summary.get("_horizon_speedup")
        if hs and hs["muon_steps_to_adamw_final"]:
            lines.append(
                f"- Muon reaches AdamW's {hs['adamw_steps']}-step loss at step {hs['muon_steps_to_adamw_final']:.0f} "
                f"({hs['adamw_steps'] / hs['muon_steps_to_adamw_final']:.2f}× fewer steps)"
            )
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def optimizers(summary):
    """The per-optimizer entries of a summary (skips extra sections such as '_horizon')."""
    return {k: v for k, v in summary.items() if not k.startswith("_")}


def plot_lr_sweep(summary, path):
    """Final validation loss vs learning rate, per optimizer (each has its own LR scale)."""
    plt = apply_style()
    summary = optimizers(summary)
    colors = ColorMap(list(summary))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for opt, s in summary.items():
        pts = [(float(lr), v) for lr, v in s["sweep"].items() if v is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=colors[opt], label=opt)
        ax.errorbar([s["best_lr"]], [s["mean"]], yerr=[s["std"]], fmt="none", ecolor=colors[opt], capsize=4)
    ax.set_xscale("log")
    ax.set_xlabel("peak learning rate")
    ax.set_ylabel("final validation bits per byte")
    ax.set_title("Learning-rate sweeps (error bar: 3 seeds at the best LR)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(summary, path):
    """Validation loss at each optimizer's best LR, against steps and against wall-clock time."""
    plt = apply_style()
    summary = optimizers(summary)
    colors = ColorMap(list(summary))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for opt, s in summary.items():
        steps, mean, std = np.array(s["curve_steps"]), np.array(s["curve_mean"]), np.array(s["curve_std"])
        for ax, x in ((axes[0], steps), (axes[1], steps * s["seconds_per_step"])):
            ax.plot(x, mean, color=colors[opt], label=f"{opt} (lr {s['best_lr']:g})")
            ax.fill_between(x, mean - std, mean + std, color=colors[opt], alpha=0.15, linewidth=0)
    lo = min(min(s["curve_mean"]) for s in summary.values())
    axes[0].set_ylim(lo - 0.1, lo + 1.6)
    axes[0].set_xlabel("step")
    axes[1].set_xlabel("wall-clock seconds (CPU)")
    axes[0].set_ylabel("validation bits per byte")
    axes[0].set_title("Per step")
    axes[1].set_title("Per second")
    axes[0].legend(fontsize=8)
    fig.text(0.01, 0.005, "Mean of 3 seeds at each optimizer's best LR; bands: ±1 std.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(runs_dir=RUNS, out_dir=OUT):
    """Write the summary and both figures."""
    runs = load_runs(runs_dir)
    if not runs:
        print(f"no runs under {runs_dir} - run train.py first")
        return None
    step_time_path = os.path.join(os.path.dirname(runs_dir.rstrip("/")), "step_time.json")
    step_time = None
    if os.path.exists(step_time_path):
        with open(step_time_path) as f:
            step_time = json.load(f)
    summary = summarize(runs, step_time)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(write_markdown(summary, os.path.join(out_dir, "summary.md")))
    plot_lr_sweep(summary, os.path.join(out_dir, "lr_sweep.png"))
    plot_curves(summary, os.path.join(out_dir, "curves.png"))
    return summary


if __name__ == "__main__":
    main()
