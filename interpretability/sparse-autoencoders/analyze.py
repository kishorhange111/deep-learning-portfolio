"""
Figures and tables for the sparse-autoencoder study, from results/study.json (written by train.py).

    python analyze.py     # -> results/summary.md, results/summary.json and the figures
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # <repo>/src
from dl_common.plotting import MUTED, ColorMap, apply_style  # noqa: E402

STUDY = os.path.join(HERE, "results", "study.json")
OUT = os.path.join(HERE, "results")
KIND_LABELS = {"relu": "ReLU + L1", "topk": "TopK", "batchtopk": "BatchTopK"}


def summarize(study):
    """SAE rows sorted by kind and L0, and the purity comparison."""
    rows = sorted(study["saes"], key=lambda r: (list(KIND_LABELS).index(r["kind"]), r["l0"]))
    purity = study["features"]["purity"]
    comp = {}
    for name, p in purity.items():
        cur = np.array(p["current"])
        comp[name] = {
            "n": len(cur),
            "median_current": float(np.median(cur)),
            "share_current_ge_0.9": float((cur >= 0.9).mean()),
            "median_next": float(np.median(p["next"])),
            "median_class": float(np.median(p["class"])),
        }
    return {
        "clean_bpb": study["clean_bpb"],
        "mean_ablated_bpb": study["mean_ablated_bpb"],
        "rows": [{k: v for k, v in r.items() if k != "curve"} for r in rows],
        "purity": comp,
        "alive": study["features"]["alive"],
        "n_latents": study["features"]["n_latents"],
    }


def write_markdown(s, study, path):
    """The SAE comparison and the interpretability check."""
    lines = [
        f"Model: {s['clean_bpb']:.3f} bits/byte clean, {s['mean_ablated_bpb']:.3f} with the layer mean-ablated. "
        f"SAEs: {study['settings']['latents']:,} latents on {study['n_train_tokens']:,} training bytes; metrics on "
        f"{study['n_eval_tokens']:,} held-out bytes.",
        "",
        "| SAE | Setting | L0 (active latents per byte) | FVU | Loss recovered | Spliced bits/byte | Dead latents |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in s["rows"]:
        setting = f"k = {r['k']}" if r["k"] else f"L1 = {r['l1']}"
        lines.append(
            f"| {KIND_LABELS[r['kind']]} | {setting} | {r['l0']:.1f} | {r['fvu']:.3f} | "
            f"{100 * r['loss_recovered']:.1f} % | {r['spliced_bpb']:.3f} | {100 * r['dead_frac']:.1f} % |"
        )
    names = {
        "sae": "SAE latents (TopK)",
        "residual_dims": "residual-stream dimensions (both signs)",
        "random_dirs": "random directions",
    }
    lines += [
        "",
        f"Interpretability check (top {study['settings']['top_n']} activations per direction on held-out text):",
        "",
        "| Directions | Count | Median share on the same current byte | Directions with ≥ 90 % on one byte | "
        "Median share on the same next byte | Median share on the same byte class |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in names.items():
        p = s["purity"][key]
        lines.append(
            f"| {label} | {p['n']:,} | {100 * p['median_current']:.0f} % | {100 * p['share_current_ge_0.9']:.0f} % | "
            f"{100 * p['median_next']:.0f} % | {100 * p['median_class']:.0f} % |"
        )
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def plot_frontier(s, path):
    """Loss recovered and FVU against L0 for each SAE kind."""
    plt = apply_style()
    colors = ColorMap(list(KIND_LABELS))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, label in KIND_LABELS.items():
        rows = sorted((r for r in s["rows"] if r["kind"] == kind), key=lambda r: r["l0"])
        if not rows:
            continue
        l0 = [r["l0"] for r in rows]
        axes[0].plot(l0, [100 * r["loss_recovered"] for r in rows], "o-", color=colors[kind], label=label)
        axes[1].plot(l0, [r["fvu"] for r in rows], "o-", color=colors[kind], label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("L0: active latents per byte (log scale)")
    axes[0].set_ylabel("next-byte loss recovered (%)")
    axes[0].set_title("Better: up and to the left")
    axes[1].set_ylabel("fraction of variance unexplained")
    axes[1].set_title("Better: down and to the left")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_purity(study, path):
    """Distribution of current-byte purity for SAE latents, residual dimensions and random directions."""
    plt = apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    labels = {"sae": "SAE latents", "residual_dims": "residual dimensions", "random_dirs": "random directions"}
    colors = ColorMap(list(labels))
    for key, label in labels.items():
        vals = np.array(study["features"]["purity"][key]["current"])
        ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=colors[key], density=True, label=label)
    ax.set_xlabel(f"share of the top {study['settings']['top_n']} activations on the same current byte")
    ax.set_ylabel("density")
    ax.set_title("Are directions about one thing?")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.005, "Held-out text; each direction's strongest activations.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(study_path=STUDY, out_dir=OUT):
    """Write the summary and figures."""
    with open(study_path) as f:
        study = json.load(f)
    s = summarize(study)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    print(write_markdown(s, study, os.path.join(out_dir, "summary.md")))
    plot_frontier(s, os.path.join(out_dir, "frontier.png"))
    plot_purity(study, os.path.join(out_dir, "purity.png"))
    return s


if __name__ == "__main__":
    main()
