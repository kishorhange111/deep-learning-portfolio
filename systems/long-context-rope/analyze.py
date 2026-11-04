"""
Figures and tables for the long-context study, from results/study.json (written by train.py).

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
LABELS = {"none": "plain RoPE (extrapolate)", "pi": "position interpolation", "ntk": "NTK-aware", "yarn": "YaRN"}


def paired_bootstrap(a, b, n_boot=10_000, seed=0):
    """Mean of a - b over windows with a 95 % percentile interval from resampling windows."""
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(1)
    return {
        "diff": float(diff.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
    }


def summarize(study):
    """Zero-shot and fine-tuned bits/byte per method and length, with paired intervals against the
    plain-RoPE model at the training length (does extra context help?) and between methods."""
    train_len = str(study["settings"]["train_len"])
    zs = study["zero_shot"]
    lengths = list(zs["none"])
    s = {"zero_shot": {}, "fine_tuned": {}, "lengths": lengths}
    for method, by_len in zs.items():
        s["zero_shot"][method] = {n: by_len[n]["bpb"] for n in lengths}
    ft_lengths = None
    for name, r in study["fine_tuned"].items():
        s["fine_tuned"][name] = {m: {n: v["bpb"] for n, v in by_len.items()} for m, by_len in r["results"].items()}
        ft_lengths = list(next(iter(r["results"].values())))
    s["ft_lengths"] = ft_lengths
    s["fine_tuned_fixed"] = {
        name: {n: v["bpb"] for n, v in r["results_fixed"][name].items()}
        for name, r in study["fine_tuned"].items()
        if "results_fixed" in r
    }
    # Pairwise: each fine-tuned method vs plain RoPE fine-tuned the same way, at every length
    s["ft_vs_plain"] = {}
    for key in ("results", "results_fixed"):
        for length in ft_lengths:
            base_ft = study["fine_tuned"]["none"]["results"]["none"][length]["per_window"]
            for name, r in study["fine_tuned"].items():
                if name == "none" or key not in r or name not in r[key]:
                    continue
                diff = paired_bootstrap(r[key][name][length]["per_window"], base_ft)
                s["ft_vs_plain"].setdefault(f"{'dynamic' if key == 'results' else 'fixed'} {length}", {})[name] = diff
    # Zero-shot: each method at the longest length vs the same windows' first train_len bytes
    s["zero_shot_vs_train_len"] = {
        m: paired_bootstrap(zs[m][lengths[-1]]["per_window"], zs["none"][train_len]["per_window"]) for m in zs
    }
    return s


def write_markdown(s, study, path):
    """Zero-shot table, fine-tuned table, paired comparisons."""
    lengths = s["lengths"]
    lines = [
        f"### Zero-shot: trained on {study['settings']['train_len']}-byte windows, evaluated longer (bits/byte, "
        f"{study['settings']['eval_windows']} held-out windows)",
        "",
        "| Method | " + " | ".join(f"{n} bytes" for n in lengths) + " |",
        "|---|" + "---|" * len(lengths),
    ]
    for m, by_len in s["zero_shot"].items():
        lines.append(f"| {LABELS.get(m, m)} | " + " | ".join(f"{by_len[n]:.3f}" for n in lengths) + " |")
    ft = s["ft_lengths"]
    lines += [
        "",
        f"### After {study['settings']['ft_steps']} fine-tuning steps on {study['settings']['ft_len']}-byte windows "
        "(each method active during fine-tuning)",
        "",
        "| Fine-tuned with | " + " | ".join(f"{n} bytes" for n in ft) + " |",
        "|---|" + "---|" * len(ft),
    ]
    for name, by_method in s["fine_tuned"].items():
        if name == "more_short_training":
            for m, by_len in by_method.items():
                lines.append(
                    f"| control: {study['settings']['ft_steps']} more steps at {study['settings']['train_len']} bytes, "
                    f"evaluated with {LABELS.get(m, m)} | " + " | ".join(f"{by_len[n]:.3f}" for n in ft) + " |"
                )
        else:
            by_len = by_method[name]
            lines.append(f"| {LABELS.get(name, name)} | " + " | ".join(f"{by_len[n]:.3f}" for n in ft) + " |")
    if s.get("fine_tuned_fixed"):
        lines += [
            "",
            "Same fine-tuned models with the scale factor kept at the fine-tuning value at every length:",
            "",
            "| Fine-tuned with | " + " | ".join(f"{n} bytes" for n in ft) + " |",
            "|---|" + "---|" * len(ft),
        ]
        for name, by_len in s["fine_tuned_fixed"].items():
            lines.append(f"| {LABELS.get(name, name)} | " + " | ".join(f"{by_len[n]:.3f}" for n in ft) + " |")
    lines += [
        "",
        "Paired differences vs plain RoPE fine-tuned the same way (95 % bootstrap interval over windows):",
        "",
    ]
    for setting, diffs in s["ft_vs_plain"].items():
        parts = [
            f"{LABELS.get(n, n)} {d['diff']:+.3f} ({d['ci_low']:+.3f}, {d['ci_high']:+.3f})" for n, d in diffs.items()
        ]
        lines.append(f"- {setting} bytes: " + "; ".join(parts))
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def smooth(x, k=16):
    """Moving average for per-position curves."""
    x = np.asarray(x)
    return np.convolve(x, np.ones(k) / k, mode="valid") if len(x) > k else x


def plot_positions(study, path):
    """Loss by position at the longest length: zero-shot (left) and after fine-tuning (right)."""
    plt = apply_style()
    colors = ColorMap(list(LABELS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    longest = str(max(study["settings"]["eval_lens"]))
    for m, by_len in study["zero_shot"].items():
        y = smooth(by_len[longest]["per_position"])
        axes[0].plot(np.arange(len(y)) + 8, y, color=colors[m], label=LABELS.get(m, m))
    ft_longest = list(next(iter(study["fine_tuned"].values()))["results"].values())[0]
    ft_longest = list(ft_longest)[-1]
    for name, r in study["fine_tuned"].items():
        if name == "more_short_training" or name not in r["results"]:
            continue
        y = smooth(r["results"][name][ft_longest]["per_position"])
        axes[1].plot(np.arange(len(y)) + 8, y, color=colors[name], label=LABELS.get(name, name))
    for ax, title in (
        (axes[0], f"Zero-shot at {longest} bytes"),
        (axes[1], f"After fine-tuning, at {ft_longest} bytes"),
    ):
        ax.axvline(study["settings"]["train_len"], color=MUTED, linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("position in the window (bytes)")
    axes[1].axvline(study["settings"]["ft_len"], color=MUTED, linestyle="--", linewidth=1)
    axes[0].set_ylabel("loss at that position (bits/byte, 16-position average)")
    lo = min(min(smooth(v[longest]["per_position"])) for v in study["zero_shot"].values())
    axes[0].set_ylim(lo - 0.1, lo + 2.5)
    axes[0].legend(fontsize=8)
    fig.text(0.01, 0.005, "Dotted: training length. Dashed: fine-tuning length.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(study_path=STUDY, out_dir=OUT):
    """Write the summary and figure."""
    with open(study_path) as f:
        study = json.load(f)
    s = summarize(study)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    print(write_markdown(s, study, os.path.join(out_dir, "summary.md")))
    plot_positions(study, os.path.join(out_dir, "loss_by_position.png"))
    return s


if __name__ == "__main__":
    main()
