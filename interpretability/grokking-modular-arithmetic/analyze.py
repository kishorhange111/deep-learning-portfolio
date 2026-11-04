"""
Figures and the summary table for the grokking study, from results/runs/*.json (written by train.py).

    python analyze.py      # -> results/summary.md, results/summary.json and the figures
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # <repo>/src
from dl_common.plotting import INK_2, MUTED, SERIES, ColorMap, apply_style  # noqa: E402

RUNS = os.path.join(HERE, "results", "runs")
OUT = os.path.join(HERE, "results")


def load_runs(runs_dir=RUNS):
    """{name: run record}."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        runs[r["name"]] = r
    return runs


def first_step(log, key, threshold, below=False):
    """First logged step where `key` crosses `threshold` (from below, or from above if below=True)."""
    for r in log:
        if (r[key] <= threshold) if below else (r[key] >= threshold):
            return r["step"]
    return None


def phases(run):
    """Phase boundaries from the logged curves (definitions in the README).

    * memorization: train accuracy first reaches 100 %;
    * circuit formation: from the minimum of the excluded loss (the point where the training-set fit
      relies least on the key frequencies) until the restricted loss first drops below 0.01 (the
      Fourier algorithm alone now solves the task);
    * cleanup: until the weight norm first comes within 5 % of its final value (weight decay has
      removed the memorizing weights);
    * test accuracy milestones for reference."""
    log = run["log"]
    out = {
        "train_acc_100": first_step(log, "train_acc", 1.0),
        "test_acc_50": first_step(log, "test_acc", 0.5),
        "test_acc_99": first_step(log, "test_acc", 0.99),
        "test_acc_99.9": first_step(log, "test_acc", 0.999),
        "weight_norm_settled": first_step(log, "weight_norm", 1.05 * log[-1]["weight_norm"], below=True),
    }
    prog = run.get("progress")
    out["restricted_below_1e-2"] = first_step(prog, "restricted_loss", 1e-2, below=True) if prog else None
    out["excluded_min"] = min(prog, key=lambda r: r["excluded_loss_train"])["step"] if prog else None
    return out


def summarize(runs):
    """One row per run: settings, phase steps and the final mechanism measurements."""
    rows = {}
    for name, r in runs.items():
        f = r["final"]
        rows[name] = {
            "weight_decay": r["settings"]["wd"],
            "train_frac": r["settings"]["frac"],
            "seed": r["settings"]["seed"],
            "steps_run": r["log"][-1]["step"],
            "final_train_acc": r["log"][-1]["train_acc"],
            "final_test_acc": r["log"][-1]["test_acc"],
            "final_test_loss": r["log"][-1]["test_loss"],
            **phases(r),
            "key_frequencies": f["key_frequencies"],
            "logit_share_key_frequencies": f["logit_share_key_frequencies"],
            "embedding_gini": f["embedding_gini"],
            "neuron_purity_median": f["neuron_purity_median"],
            "neurons_purity_over_0.8": f["neurons_purity_over_0.8"],
            "neurons_on_key_frequencies": f["neurons_on_key_frequencies"],
            "restricted_loss": f["restricted_loss"],
            "excluded_loss_train": f["excluded_loss_train"],
            "seconds": r["seconds"],
        }
    return rows


def fmt_step(x):
    """Step count or a dash."""
    return f"{x:,}" if x is not None else "—"


def write_markdown(rows, path):
    """The summary table as Markdown."""
    lines = [
        "| Run | Weight decay | Train fraction | Train acc 100 % at | Test acc 50 % at | Test acc 99 % at | "
        "Final test acc | Key frequencies | Logit variance in key freqs | Neurons with purity > 0.8 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(rows, key=lambda n: (not n.startswith("main"), n)):
        r = rows[name]
        lines.append(
            f"| {name} | {r['weight_decay']} | {r['train_frac']} | {fmt_step(r['train_acc_100'])} | "
            f"{fmt_step(r['test_acc_50'])} | {fmt_step(r['test_acc_99'])} | {100 * r['final_test_acc']:.1f} % | "
            f"{', '.join(map(str, r['key_frequencies'])) or '—'} | {100 * r['logit_share_key_frequencies']:.0f} % | "
            f"{100 * r['neurons_purity_over_0.8']:.0f} % |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def plot_grokking(runs, path):
    """Train and test accuracy and loss for the main seeds and the no-weight-decay control."""
    plt = apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ColorMap(["train", "test"])
    for name, r in runs.items():
        if not (name.startswith("main") or name.startswith("no_weight_decay")):
            continue
        steps = [x["step"] for x in r["log"]]
        dashed = name.startswith("no_weight_decay")
        alpha = 1.0 if name in ("main_s0", "no_weight_decay_s0") else 0.35
        for split in ("train", "test"):
            ls = "--" if dashed else "-"
            label = None
            if name in ("main_s0", "no_weight_decay_s0"):
                label = f"{split} ({'weight decay 0' if dashed else 'weight decay 1'})"
            axes[0].plot(
                steps, [100 * x[f"{split}_acc"] for x in r["log"]], ls, color=colors[split], alpha=alpha, label=label
            )
            axes[1].plot(
                steps, [x[f"{split}_loss"] for x in r["log"]], ls, color=colors[split], alpha=alpha, label=label
            )
    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("accuracy (%)")
    axes[1].set_title("Loss (log scale)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("training step (full batch)")
    axes[0].legend(fontsize=8, loc="center right")
    fig.text(
        0.01, 0.005, "Faint lines: seeds 1 and 2. Dashed: same seed without weight decay.", color=MUTED, fontsize=8
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_progress(run, path):
    """The progress measures for one run, with the phase boundaries."""
    plt = apply_style()
    prog = run["progress"]
    steps = [x["step"] for x in prog]
    ph = phases(run)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    log_steps = [x["step"] for x in run["log"]]
    panels = [
        (axes[0, 0], "Train and test loss"),
        (axes[0, 1], "Restricted loss (all pairs) and excluded loss (train pairs)"),
        (axes[1, 0], "Share of logit variance in the key frequencies"),
        (axes[1, 1], "Weight norm and Fourier sparsity of the embedding"),
    ]
    ax = panels[0][0]
    ax.plot(log_steps, [x["test_loss"] for x in run["log"]], color=SERIES[1], label="test loss")
    ax.plot(log_steps, [x["train_loss"] for x in run["log"]], color=SERIES[0], label="train loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax = panels[1][0]
    ax.plot(steps, [x["restricted_loss"] for x in prog], color=SERIES[2], label="restricted loss")
    ax.plot(steps, [x["excluded_loss_train"] for x in prog], color=SERIES[3], label="excluded loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax = panels[2][0]
    ax.plot(steps, [100 * x["logit_share_key_frequencies"] for x in prog], color=SERIES[0])
    ax.set_ylabel("%")
    ax = panels[3][0]
    ax.plot(log_steps, [x["weight_norm"] for x in run["log"]], color=SERIES[6], label="sum of squared weights")
    ax.set_ylabel("sum of squared weights", color=SERIES[6])
    twin = ax.twinx()
    twin.plot(steps, [x["embedding_gini"] for x in prog], color=SERIES[4], label="Gini of embedding Fourier norms")
    twin.set_ylabel("Gini (0 = spread, 1 = one frequency)", color=SERIES[4])
    twin.grid(False)
    for ax, title in panels:
        ax.set_title(title, fontsize=9)
        for key, ls in (("train_acc_100", ":"), ("restricted_below_1e-2", "-."), ("weight_norm_settled", "-")):
            if ph[key] is not None:
                ax.axvline(ph[key], color=MUTED, linestyle=ls, linewidth=1)
    for ax in axes[1]:
        ax.set_xlabel("training step")
    fig.text(
        0.01,
        0.005,
        "Vertical lines: memorized (train accuracy 100 %, dotted), circuit formed (restricted loss < 0.01, "
        "dash-dot), cleaned up (weight norm within 5 % of final, solid).",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mechanism(run, path):
    """The final model's Fourier structure: embedding and logit spectra, and neuron purity."""
    plt = apply_style()
    f = run["final"]
    keys = set(f["key_frequencies"])
    k = np.arange(1, len(f["embedding_norm_by_frequency"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in (
        (axes[0], "embedding_norm_by_frequency", "Token embedding: share of norm per frequency"),
        (axes[1], "logit_share_by_frequency", "Logits: share of variance in cos/sin(w(a+b))"),
    ):
        vals = np.array(f[key])
        colors = [SERIES[0] if kk in keys else "#c3c2b7" for kk in k]
        ax.bar(k, 100 * vals, color=colors, width=0.9)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("frequency k (w = 2 pi k / p)")
        ax.set_ylabel("%")
        for kk in keys:
            ax.text(kk, 100 * vals[kk - 1], str(kk), ha="center", va="bottom", fontsize=8, color=INK_2)
    purity = np.array(f["neuron_purity"])
    axes[2].hist(purity, bins=np.linspace(0, 1, 26), color=SERIES[0])
    axes[2].set_title("MLP neurons: variance explained by one frequency", fontsize=9)
    axes[2].set_xlabel("purity")
    axes[2].set_ylabel("neurons")
    fig.text(
        0.01,
        0.005,
        "Blue bars: the key frequencies (each carries >= 2 % of the logit variance).",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_data_fraction(rows, path):
    """Steps until test accuracy reaches 99 %, against the fraction of pairs used for training (every
    weight-decay run; the 0.3 point has three seeds)."""
    plt = apply_style()
    from matplotlib.ticker import FuncFormatter

    runs = [r for r in rows.values() if r["weight_decay"] == 1.0]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    done = [(r["train_frac"], r["test_acc_99"]) for r in runs if r["test_acc_99"] is not None]
    if done:
        ax.plot([d[0] for d in done], [d[1] for d in done], "o", color=SERIES[0], alpha=0.8)
        by_frac = {}
        for frac, step in done:
            by_frac.setdefault(frac, []).append(step)
        fracs = sorted(by_frac)
        ax.plot(fracs, [np.median(by_frac[f]) for f in fracs], "-", color=SERIES[0], linewidth=1.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    for r in runs:
        if r["test_acc_99"] is None:
            ax.axvline(r["train_frac"], color=MUTED, linestyle=":", linewidth=1)
            ax.text(
                r["train_frac"] + 0.01,
                0.5,
                f"no generalization\nin {r['steps_run']:,} steps",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                color=MUTED,
                va="center",
            )
    ax.set_xlabel("fraction of the 113 x 113 pairs used for training")
    ax.set_ylabel("steps until test accuracy >= 99 %")
    ax.set_title("Less data, longer delay (line: median over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(runs_dir=RUNS, out_dir=OUT):
    """Write the summary and every figure whose runs exist."""
    runs = load_runs(runs_dir)
    if not runs:
        print(f"no runs under {runs_dir} - run train.py first")
        return None
    rows = summarize(runs)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(write_markdown(rows, os.path.join(out_dir, "summary.md")))
    plot_grokking(runs, os.path.join(out_dir, "grokking.png"))
    if "main_s0" in runs and runs["main_s0"].get("progress"):
        plot_progress(runs["main_s0"], os.path.join(out_dir, "progress_measures.png"))
        plot_mechanism(runs["main_s0"], os.path.join(out_dir, "fourier_mechanism.png"))
    if "no_weight_decay_s0" in runs and runs["no_weight_decay_s0"].get("progress"):
        plot_progress(runs["no_weight_decay_s0"], os.path.join(out_dir, "progress_measures_no_weight_decay.png"))
    plot_data_fraction(rows, os.path.join(out_dir, "data_fraction.png"))
    return rows


if __name__ == "__main__":
    main()
