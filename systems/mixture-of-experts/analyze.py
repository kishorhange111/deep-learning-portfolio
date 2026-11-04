"""
Figures and the summary table for the mixture-of-experts study, from results/runs/*.json.

    python analyze.py     # -> results/summary.md, results/summary.json and the figures
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # <repo>/src
from dl_common.byte_corpus import BYTE_CLASSES  # noqa: E402
from dl_common.plotting import MUTED, ColorMap, apply_style  # noqa: E402

RUNS = os.path.join(HERE, "results", "runs")
OUT = os.path.join(HERE, "results")
ORDER = ["dense", "moe8_top1", "moe8_top1_unitgate", "moe8_top2", "dense_wide", "moe8_top1_nobal", "moe8_top1_cap1"]


def load_runs(runs_dir=RUNS, horizon=False):
    """{config: [records over seeds]} for the main comparison (or, with horizon=True, the longer runs)."""
    records = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(path) as f:
            records.append(json.load(f))
    if not records:
        return {}
    counts = {}
    for r in records:
        counts[r["settings"]["steps"]] = counts.get(r["settings"]["steps"], 0) + 1
    main_steps = max(counts, key=counts.get)
    out = {}
    for r in records:
        if (r["settings"]["steps"] != main_steps) == horizon:
            out.setdefault(r["config"], []).append(r)
    return {
        k: sorted(v, key=lambda r: r["seed"])
        for k, v in sorted(out.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99)
    }


def mean_std(values):
    """(mean, sample std) of a list (std 0 for one value)."""
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize(runs, step_time=None):
    """Per configuration: loss, parameters, speed (from the interleaved benchmark in step_time.json when
    it exists), and routing statistics averaged over layers and seeds."""
    rows = {}
    base = runs.get("dense")
    for name, rs in runs.items():
        finals = [r["final_val_bpb"] for r in rs]
        m, s = mean_std(finals)
        row = {
            "seeds": len(rs),
            "val_bpb": finals,
            "val_bpb_mean": m,
            "val_bpb_std": s,
            "params_total": rs[0]["params_total"],
            "mlp_active": rs[0]["mlp_params_active_per_layer"],
            "mlp_total": rs[0]["mlp_params_total_per_layer"],
            "ms_per_step": (
                step_time[name]["ms_per_step"]
                if step_time and name in step_time
                else (None if step_time else 1000 * float(np.mean([r["seconds_per_step"] for r in rs])))
            ),
            "ms_per_step_source": "benchmark" if step_time and name in step_time else "training runs",
        }
        if base and name != "dense" and len(base) == len(rs):
            diffs = [r["final_val_bpb"] - b["final_val_bpb"] for r, b in zip(rs, base)]  # same seed = same data order
            row["paired_diff_vs_dense"] = mean_std(diffs)
        if rs[0]["routing"]:
            layers = [layer for r in rs for layer in r["routing"]]
            for key in ("load_entropy_bits", "max_share", "min_share", "experts_under_1pct", "dropped_frac"):
                row[key] = float(np.mean([layer[key] for layer in layers]))
            for key in ("predict_from_class", "predict_from_byte", "predict_majority"):
                row[key] = float(np.mean([layer[key] for layer in layers]))
                row[key + "_by_layer"] = np.mean([[layer[key] for layer in r["routing"]] for r in rs], axis=0).tolist()
            row["n_experts"] = len(layers[0]["load"])
        rows[name] = row
    return rows


def write_markdown(rows, path):
    """Two tables: quality / cost, and routing."""
    lines = [
        "| Config | MLP params per layer: active / stored | Val bits/byte (mean ± std) | "
        "Δ vs dense (paired by seed) | ms / step (CPU) |",
        "|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        d = r.get("paired_diff_vs_dense")
        delta = f"{d[0]:+.4f} ± {d[1]:.4f}" if d else "—"
        ms = f"{r['ms_per_step']:.0f}" if r["ms_per_step"] else "—"
        lines.append(
            f"| {name} | {r['mlp_active'] / 1e3:.0f}k / {r['mlp_total'] / 1e3:.0f}k | "
            f"{r['val_bpb_mean']:.4f} ± {r['val_bpb_std']:.4f} "
            f"(n={r['seeds']}) | {delta} | {ms} |"
        )
    lines += [
        "",
        "Routing on held-out windows (first choice; averaged over layers and seeds):",
        "",
        "| Config | Load entropy (bits; max = log2 E) | Largest / smallest expert share | Experts < 1 % | "
        "Tokens dropped | Choice predicted from byte class | from the byte itself | majority baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        if "load_entropy_bits" not in r:
            continue
        lines.append(
            f"| {name} | {r['load_entropy_bits']:.2f} / {np.log2(r['n_experts']):.0f} | {100 * r['max_share']:.1f} % / "
            f"{100 * r['min_share']:.1f} % | {r['experts_under_1pct']:.1f} | {100 * r['dropped_frac']:.1f} % | "
            f"{100 * r['predict_from_class']:.0f} % | {100 * r['predict_from_byte']:.0f} % | "
            f"{100 * r['predict_majority']:.0f} % |"
        )
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def plot_curves(runs, path):
    """Validation loss curves (mean over seeds) for every configuration."""
    plt = apply_style()
    colors = ColorMap(list(runs))
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for name, rs in runs.items():
        steps = [x["step"] for x in rs[0]["log"]]
        curve = np.mean([[x["val_bpb"] for x in r["log"]] for r in rs], axis=0)
        ax.plot(steps, curve, color=colors[name], label=name)
    lo = min(min(x["val_bpb"] for x in rs[0]["log"]) for rs in runs.values())
    ax.set_ylim(lo - 0.05, lo + 1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("validation bits per byte")
    ax.set_title("Dense vs mixture-of-experts feed-forward layers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_load(runs, path):
    """Expert load per layer for top-1 routing with and without the load-balancing loss (seed 0)."""
    plt = apply_style()
    names = [n for n in ("moe8_top1", "moe8_top1_nobal") if n in runs]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 3.8), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        load = np.array([layer["load"] for layer in runs[name][0]["routing"]])  # (layers, experts)
        im = ax.imshow(100 * load, cmap="Blues", vmin=0, vmax=max(40, 100 * load.max()), aspect="auto")
        for i in range(load.shape[0]):
            for j in range(load.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{100 * load[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#0b0b0b" if load[i, j] < 0.25 else "white",
                )
        ax.set_xticks(range(load.shape[1]))
        ax.set_yticks(range(load.shape[0]), [f"layer {i}" for i in range(load.shape[0])])
        ax.set_xlabel("expert")
        ax.set_title(f"{name}: % of tokens per expert", fontsize=9)
        ax.grid(False)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_specialization(runs, path, name="moe8_top1"):
    """Which byte classes each expert receives, per layer (seed 0): rows normalized per class."""
    if name not in runs:
        return
    plt = apply_style()
    routing = runs[name][0]["routing"]
    fig, axes = plt.subplots(1, len(routing), figsize=(3.2 * len(routing), 3.8), sharey=True, squeeze=False)
    for li, (ax, layer) in enumerate(zip(axes[0], routing)):
        counts = np.array(layer["class_by_expert"], dtype=float).T  # (classes, experts)
        share = counts / np.maximum(counts.sum(1, keepdims=True), 1)
        ax.imshow(100 * share, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(share.shape[1]))
        ax.set_xlabel("expert")
        ax.set_title(f"layer {li}: predicted from byte {100 * layer['predict_from_byte']:.0f} %", fontsize=9)
        ax.grid(False)
    axes[0][0].set_yticks(range(len(BYTE_CLASSES)), BYTE_CLASSES)
    fig.text(0.01, 0.005, "Each row: where tokens of that byte class go (row sums to 100 %).", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_horizon(horizon, path):
    """Validation loss of the longer runs, dense vs MoE."""
    plt = apply_style()
    colors = ColorMap(list(horizon))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, rs in horizon.items():
        log = rs[0]["log"]
        ax.plot([x["step"] for x in log], [x["val_bpb"] for x in log], color=colors[name], label=name)
    lo = min(x["val_bpb"] for rs in horizon.values() for x in rs[0]["log"])
    ax.set_ylim(lo - 0.05, lo + 1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("validation bits per byte")
    ax.set_title("Horizon check: 3x the training steps (seed 0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(runs_dir=RUNS, out_dir=OUT):
    """Write the summary and figures."""
    runs = load_runs(runs_dir)
    if not runs:
        print(f"no runs under {runs_dir} - run train.py first")
        return None
    step_time_path = os.path.join(os.path.dirname(runs_dir.rstrip("/")), "step_time.json")
    step_time = None
    if os.path.exists(step_time_path):
        with open(step_time_path) as f:
            step_time = json.load(f)
    rows = summarize(runs, step_time)
    horizon = load_runs(runs_dir, horizon=True)
    if horizon:
        rows["_horizon"] = {
            name: {"steps": rs[0]["settings"]["steps"], "final_val_bpb": rs[0]["final_val_bpb"], "curve": rs[0]["log"]}
            for name, rs in horizon.items()
        }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(write_markdown({k: v for k, v in rows.items() if not k.startswith("_")}, os.path.join(out_dir, "summary.md")))
    if horizon:
        with open(os.path.join(out_dir, "summary.md"), "a") as f:
            f.write("\nHorizon check (seed 0):\n\n")
            for name, h in rows["_horizon"].items():
                f.write(f"- {name}, {h['steps']} steps: val {h['final_val_bpb']:.4f} bits/byte\n")
        plot_horizon(horizon, os.path.join(out_dir, "horizon.png"))
    plot_curves(runs, os.path.join(out_dir, "curves.png"))
    plot_load(runs, os.path.join(out_dir, "expert_load.png"))
    plot_specialization(runs, os.path.join(out_dir, "specialization.png"))
    return rows


if __name__ == "__main__":
    main()
