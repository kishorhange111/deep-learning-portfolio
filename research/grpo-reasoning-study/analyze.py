"""
Turn the finished runs into the study's results: accuracy per config with uncertainty, paired tests
against the baseline, training dynamics and pass@k.

Statistics:
  * Seed spread - mean +/- std of test accuracy over seeds (training randomness).
  * Paired bootstrap over test questions - every config is evaluated on the same 1,319 questions, so
    compare configs question by question (per-question correctness averaged over seeds), resample
    questions 10,000 times, and report the 95 % CI of the accuracy difference and a two-sided p-value.
    Pairing removes the variance from questions being hard or easy, so it is much more sensitive
    than comparing two independent accuracy numbers.

    python analyze.py      # writes results/summary.md, results/summary.json and figures
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from common.plotting import INK_2, MUTED, ColorMap, apply_style  # noqa: E402

RUNS = os.path.join(HERE, "results", "runs")
OUT = os.path.join(HERE, "results")


def load_runs(runs_dir=RUNS):
    """{config: {seed: {'eval': ..., 'log': [...], 'passk': ... or None}}}"""
    runs = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "seed*", "eval_final.json"))):
        seed_dir = os.path.dirname(path)
        name, seed = os.path.basename(os.path.dirname(seed_dir)), int(os.path.basename(seed_dir)[4:])
        with open(path) as f:
            rec = {"eval": json.load(f), "log": [], "passk": None}
        log_path = os.path.join(seed_dir, "train_log.jsonl")
        if os.path.exists(log_path):
            with open(log_path) as f:
                rec["log"] = [json.loads(line) for line in f if line.strip()]
        passk_path = os.path.join(seed_dir, "passk_final.json")
        if os.path.exists(passk_path):
            with open(passk_path) as f:
                rec["passk"] = json.load(f)
        runs.setdefault(name, {})[seed] = rec
    return runs


def per_question(seeds, key="correct"):
    """Per-question correctness averaged over seeds -> array (n_questions,)."""
    return np.mean([np.array(r["eval"][key], dtype=float) for r in seeds.values()], axis=0)


def paired_bootstrap(a, b, n_boot=10_000, seed=0):
    """Mean of (a - b) with a 95 % percentile CI and two-sided p-value, resampling questions."""
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        "diff": float(diff.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p": float(min(p, 1.0)),
    }


def summarize(runs, reference="baseline"):
    rows = {}
    ref_q = per_question(runs[reference]) if reference in runs else None
    base_q = per_question(runs["base"]) if "base" in runs else None
    for name, seeds in runs.items():
        accs = [r["eval"]["accuracy"] for r in seeds.values()]
        row = {
            "seeds": len(accs),
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "format_mean": float(np.mean([r["eval"]["format"] for r in seeds.values()])),
            "samples_mean": float(np.mean([r["eval"].get("samples_total", 0) for r in seeds.values()])),
        }
        q = per_question(seeds)
        if ref_q is not None and name not in (reference, "base") and len(q) == len(ref_q):
            row["vs_" + reference] = paired_bootstrap(q, ref_q)
        if base_q is not None and name != "base" and len(q) == len(base_q):
            row["vs_base"] = paired_bootstrap(q, base_q)
        logs = [r["log"] for r in seeds.values() if r["log"]]
        if logs:
            row["zero_std_frac_mean"] = float(np.mean([np.mean([s["zero_std_frac"] for s in log]) for log in logs]))
        rows[name] = row
    return rows


def fmt_pct(x):
    return f"{100 * x:.1f}"


def write_markdown(rows, path, reference="baseline"):
    lines = [
        "| Config | Seeds | Accuracy % (mean ± std) | Format % | Δ vs baseline (95 % CI) | p | "
        "Zero-variance groups | Samples generated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = (["base"] if "base" in rows else []) + sorted(k for k in rows if k != "base")
    for name in order:
        r = rows[name]
        vs = r.get("vs_" + reference)
        delta = f"{100 * vs['diff']:+.1f} ({100 * vs['ci_low']:+.1f}, {100 * vs['ci_high']:+.1f})" if vs else "—"
        p = f"{vs['p']:.3f}" if vs else "—"
        zs = fmt_pct(r["zero_std_frac_mean"]) + " %" if "zero_std_frac_mean" in r else "—"
        lines.append(
            f"| {name} | {r['seeds']} | {fmt_pct(r['accuracy_mean'])} ± {fmt_pct(r['accuracy_std'])} | "
            f"{fmt_pct(r['format_mean'])} | {delta} | {p} | {zs} | {r['samples_mean']:,.0f} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def smooth(x, k=10):
    if len(x) < k:
        return np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(k) / k, mode="valid")


def plot_training(runs, colors, path):
    plt = apply_style()
    panels = [
        ("correct_reward", "step", "Correctness reward (train samples)"),
        ("zero_std_frac", "step", "Groups with zero reward variance"),
        ("completion_len", "step", "Completion length (tokens)"),
        ("correct_reward", "samples_total", "Correctness reward vs. samples generated"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, x_key, title) in zip(axes.flat, panels):
        for name, seeds in runs.items():
            logs = [r["log"] for r in seeds.values() if r["log"]]
            if not logs:
                continue
            n = min(len(log) for log in logs)
            ys = np.array([smooth([s[key] for s in log[:n]]) for log in logs])
            raw_x = np.array([[s[x_key] for s in log[:n]] for log in logs]).mean(0)
            xs = raw_x[len(raw_x) - ys.shape[1] :]  # align with the moving average's 'valid' window
            mean, std = ys.mean(0), ys.std(0)
            ax.plot(xs, mean, color=colors[name], label=name)
            ax.fill_between(xs, mean - std, mean + std, color=colors[name], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("generated samples" if x_key == "samples_total" else "step")
    axes.flat[0].legend(loc="lower right", fontsize=8)
    fig.text(
        0.01,
        0.005,
        "Lines: mean over seeds (10-step moving average); bands: ±1 std over seeds.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_deltas(rows, path, reference="baseline"):
    """Dot-and-whisker: accuracy change vs. the baseline with 95 % paired-bootstrap CIs."""
    plt = apply_style()
    names = [n for n in rows if "vs_" + reference in rows[n]]
    if not names:
        return
    names.sort(key=lambda n: rows[n]["vs_" + reference]["diff"])
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.2))
    for i, n in enumerate(names):
        v = rows[n]["vs_" + reference]
        ax.plot([100 * v["ci_low"], 100 * v["ci_high"]], [i, i], color="#2a78d6", linewidth=2)
        ax.plot(100 * v["diff"], i, "o", color="#2a78d6", markersize=8)
        ax.text(100 * v["ci_high"] + 0.15, i, f"{100 * v['diff']:+.1f}", va="center", color=INK_2, fontsize=9)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel(f"GSM8K test accuracy vs. {reference} (percentage points)")
    ax.set_title("Effect of each intervention (95 % paired-bootstrap CI)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pass_at_k(runs, colors, path):
    plt = apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    plotted = False
    for name, seeds in runs.items():
        rec = seeds.get(0)
        if not rec or not rec["passk"]:
            continue
        pk = {int(k): v for k, v in rec["passk"]["pass_at_k"].items()}
        ks = sorted(pk)
        ax.plot(ks, [100 * pk[k] for k in ks], marker="o", color=colors[name], label=name)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xscale("log", base=2)
    ax.set_xlabel("k (samples per question)")
    ax.set_ylabel("pass@k (%)")
    ax.set_title("pass@k: does RL add solutions or sharpen existing ones?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(runs_dir=RUNS, out_dir=OUT):
    runs = load_runs(runs_dir)
    if not runs:
        print(f"no finished runs under {runs_dir} - run sweep.py first")
        return None
    colors = ColorMap(["base", "baseline"] + sorted(k for k in runs if k not in ("base", "baseline")))
    rows = summarize(runs)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(write_markdown(rows, os.path.join(out_dir, "summary.md")))
    trained = {k: v for k, v in runs.items() if k != "base"}
    plot_training(trained, colors, os.path.join(out_dir, "training_dynamics.png"))
    plot_deltas(rows, os.path.join(out_dir, "accuracy_vs_baseline.png"))
    plot_pass_at_k(runs, colors, os.path.join(out_dir, "pass_at_k.png"))
    return rows


if __name__ == "__main__":
    main()
