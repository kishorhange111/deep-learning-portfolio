"""
Figures and tables for the DPO vs RLHF study, from results/study.json (written by train.py).

    python analyze.py     # -> results/summary.md, results/summary.json and the figures
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # <repo>/src
from dl_common.plotting import MUTED, SERIES, apply_style  # noqa: E402

STUDY = os.path.join(HERE, "results", "study.json")
OUT = os.path.join(HERE, "results")
METHODS = {
    "dpo": "DPO (β)",
    "ipo": "IPO (τ)",
    "rlhf_mlp": "RLHF, MLP reward model (KL coef)",
    "rlhf_bigram": "RLHF, bigram reward model (KL coef)",
}


def ms(values, fmt="{:.2f}"):
    """'mean ± std' over seeds."""
    return f"{fmt.format(np.mean(values))} ± {fmt.format(np.std(values))}"


def mean_log(runs, key="log"):
    """Seed-average of logs that share evaluation points: {field: array}."""
    n = min(len(r[key]) for r in runs)
    fields = [f for f, v in runs[0][key][0].items() if isinstance(v, (int, float))]
    return {f: np.mean([[row.get(f, np.nan) for row in r[key][:n]] for r in runs], axis=0) for f in fields}


def all_points(study, method):
    """Every (kl, gold) point seen during training by any strength and seed of a method."""
    return [(row["kl"], row["gold"]) for runs in study[method].values() for r in runs for row in r["log"]]


def summarize(study):
    """Per method and strength: peak and final gold (with KL), across seeds; plus the frontier."""
    s = {"methods": {}, "agreement": {}}
    for method in METHODS:
        s["methods"][method] = {}
        for strength, runs in study[method].items():
            s["methods"][method][strength] = {
                key: [r[key] for r in runs] for key in ("peak_gold", "kl_at_peak", "final_gold", "final_kl")
            }
    if study.get("reward_model_agreement"):
        for name in ("mlp", "bigram"):
            s["agreement"][name] = [a[name] for a in study["reward_model_agreement"]]
    bon = study["bon_mlp"]
    s["bon_peak_gold"] = [max(r["gold"] for r in curve) for curve in bon]
    s["bon_final"] = [curve[-1]["gold"] for curve in bon]
    s["bon_max_n"] = bon[0][-1]["n"]
    s["oracle"] = {c: {"gold": log[-1]["gold"], "kl": log[-1]["kl"]} for c, log in study["oracle"].items()}
    grid = [1, 2, 4, 8, 16]
    s["frontier_grid"] = grid
    s["frontier"] = {}
    for method in METHODS:
        pts = np.array(all_points(study, method))
        s["frontier"][method] = [float(pts[pts[:, 0] <= k, 1].max()) if (pts[:, 0] <= k).any() else None for k in grid]
    displacement = {}
    for beta, runs in study["dpo"].items():
        last = [r["log"][-1] for r in runs]
        displacement[beta] = {
            "chosen_logratio": [x["chosen_logratio"] for x in last],
            "rejected_logratio": [x["rejected_logratio"] for x in last],
            "implicit_accuracy": [x["implicit_accuracy"] for x in last],
        }
    s["dpo_displacement"] = displacement
    s["dpo_data_size"] = {
        n: {key: [r[key] for r in runs] for key in ("peak_gold", "final_gold", "final_kl", "kl_at_peak")}
        for n, runs in study["dpo_data_size"].items()
    }
    return s


def write_markdown(s, study, path):
    """Tables: endpoints per method and strength, frontier, DPO dynamics, data size."""
    seeds = study["settings"]["seeds"]
    lines = [
        f"### Endpoints ({study['settings']['n_pairs']:,} comparisons, {seeds} seeds, mean ± std)",
        "",
        "| Method | Strength | Peak gold | KL at peak | Final gold | Final KL |",
        "|---|---|---|---|---|---|",
    ]
    for method, label in METHODS.items():
        for strength, r in s["methods"][method].items():
            lines.append(
                f"| {label} | {strength} | {ms(r['peak_gold'])} | {ms(r['kl_at_peak'], '{:.1f}')} | "
                f"{ms(r['final_gold'])} | {ms(r['final_kl'], '{:.1f}')} |"
            )
    lines += [
        "",
        f"Best-of-n (MLP reward model): peak gold {ms(s['bon_peak_gold'])}, gold at n = {s['bon_max_n']:,} "
        f"{ms(s['bon_final'])}.",
        "RL on the gold reward itself, by KL coefficient: "
        + ", ".join(f"{c}: {v['gold']:.2f} at KL {v['kl']:.1f}" for c, v in s["oracle"].items())
        + ".",
    ]
    if s["agreement"]:
        lines.append(
            f"Reward-model agreement with gold on held-out pairs: MLP {ms(s['agreement']['mlp'], '{:.3f}')}, "
            f"bigram {ms(s['agreement']['bigram'], '{:.3f}')}."
        )
    lines += [
        "",
        "### Best gold reached at or below each KL budget (any strength, any seed, any point in training)",
        "",
        "| Method | " + " | ".join(f"KL ≤ {k}" for k in s["frontier_grid"]) + " |",
        "|---|" + "---|" * len(s["frontier_grid"]),
    ]
    for method, label in METHODS.items():
        vals = " | ".join(f"{v:.2f}" if v is not None else "—" for v in s["frontier"][method])
        lines.append(f"| {label} | {vals} |")
    lines += [
        "",
        "### DPO at the end of training: likelihood displacement and implicit-reward accuracy",
        "",
        "| β | log π/ref of chosen | log π/ref of rejected | Implicit reward orders held-out pairs like gold |",
        "|---|---|---|---|",
    ]
    for beta, r in s["dpo_displacement"].items():
        lines.append(
            f"| {beta} | {ms(r['chosen_logratio'], '{:.1f}')} | {ms(r['rejected_logratio'], '{:.1f}')} | "
            f"{100 * np.mean(r['implicit_accuracy']):.1f} % |"
        )
    lines += [
        "",
        f"### DPO with more or fewer comparisons (β = {study['settings']['data_size_beta']}, "
        f"{study['settings']['steps']:,} steps)",
        "",
        "| Comparisons | Peak gold | KL at peak | Final gold | Final KL |",
        "|---|---|---|---|---|",
    ]
    for n, r in s["dpo_data_size"].items():
        lines.append(
            f"| {int(n):,} | {ms(r['peak_gold'])} | {ms(r['kl_at_peak'], '{:.1f}')} | {ms(r['final_gold'])} | "
            f"{ms(r['final_kl'], '{:.1f}')} |"
        )
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def plot_frontier(study, path):
    """Gold vs sqrt(KL): training trajectories of each method (seed means), best-of-n, and the ceiling."""
    plt = apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    panels = [
        (axes[0], ("dpo", "ipo"), "Direct alignment (no reward model)"),
        (axes[1], ("rlhf_mlp", "rlhf_bigram"), "RLHF"),
    ]
    oracle = sorted((v[-1]["kl"], v[-1]["gold"]) for v in study["oracle"].values())
    bon = np.mean([[r["gold"] for r in curve] for curve in study["bon_mlp"]], axis=0)
    bon_kl = [r["kl"] for r in study["bon_mlp"][0]]
    for ax, methods, title in panels:
        ax.plot(
            np.sqrt([p[0] for p in oracle]),
            [p[1] for p in oracle],
            ":",
            color=MUTED,
            marker="o",
            markersize=3,
            label="RL on the gold reward (KL-penalized)",
        )
        ax.plot(np.sqrt(bon_kl), bon, color="#52514e", linewidth=1.2, label="best-of-n (MLP RM)")
        for mi, method in enumerate(methods):
            strengths = list(study[method])
            for si, strength in enumerate(strengths):
                m = mean_log(study[method][strength])
                color = SERIES[(4 * mi + si) % len(SERIES)] if mi == 0 else SERIES[(si + 3) % len(SERIES)]
                ls = "-" if mi == 0 else "--"
                ax.plot(
                    np.sqrt(m["kl"]),
                    m["gold"],
                    ls,
                    color=color,
                    linewidth=1.4,
                    label=f"{METHODS[method].split(' (')[0]} {strength}",
                )
        ax.set_title(title)
        ax.set_xlabel("sqrt(KL from the reference)  (sqrt nats)")
        ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    axes[0].set_ylabel("gold reward")
    fig.text(
        0.01,
        0.005,
        "Lines: mean over seeds of each run's training trajectory. Solid: DPO / MLP RM; dashed: IPO / bigram RM.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_displacement(study, path):
    """DPO: log-ratios of chosen and rejected sequences, and gold, over training."""
    plt = apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (beta, runs) in enumerate(study["dpo"].items()):
        m = mean_log(runs)
        axes[0].plot(m["epoch"], m["chosen_logratio"], color=SERIES[i], label=f"β {beta}: chosen")
        axes[0].plot(m["epoch"], m["rejected_logratio"], "--", color=SERIES[i], label=f"β {beta}: rejected")
        axes[1].plot(m["epoch"], m["gold"], color=SERIES[i], label=f"β {beta}")
    axes[0].axhline(0, color=MUTED, linewidth=0.8)
    axes[0].set_title("Both chosen and rejected become less likely")
    axes[0].set_ylabel("mean log π(y) − log ref(y) on training pairs")
    axes[1].set_title("Gold reward of the policy's samples")
    for ax in axes:
        ax.set_xlabel("epochs over the comparisons")
    axes[0].legend(fontsize=6.5, ncol=2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
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
    plot_frontier(study, os.path.join(out_dir, "frontier.png"))
    plot_displacement(study, os.path.join(out_dir, "dpo_dynamics.png"))
    return s


if __name__ == "__main__":
    main()
