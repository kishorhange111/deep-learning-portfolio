"""
Figures and tables for the overoptimization study, from results/study.json (written by train.py).

    python analyze.py     # -> results/summary.md, results/summary.json and the figures
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # <repo>/src
import goodhart as gh  # noqa: E402

from dl_common.plotting import INK_2, MUTED, SERIES, apply_style  # noqa: E402

STUDY = os.path.join(HERE, "results", "study.json")
OUT = os.path.join(HERE, "results")
ORACLE = "#898781"


def mean_curve(runs, key):
    """Average a list of curves (same evaluation points) over seeds: {field: (mean array, std array)}."""
    n = min(len(r[key]) for r in runs)
    fields = [k for k in runs[0][key][0] if isinstance(runs[0][key][0][k], (int, float))]
    out = {}
    for f in fields:
        vals = np.array([[row[f] for row in r[key][:n]] for r in runs], dtype=float)
        out[f] = (vals.mean(0), vals.std(0))
    return out


def ms(values):
    """'mean ± std' of a list."""
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}"


def paired(a, b):
    """Paired differences a - b over seeds: mean, std, and how many seeds have a > b."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {
        "mean": float(d.mean()),
        "std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "wins": int((d > 0).sum()),
        "n": len(d),
    }


def fmt_paired(p):
    """'+0.12 ± 0.05 (4/5)'."""
    return f"{p['mean']:+.2f} ± {p['std']:.2f} ({p['wins']}/{p['n']})"


def gold_at_kl(curve, kl):
    """Gold reward on a (KL-increasing) curve, linearly interpolated at a KL value."""
    kls, golds = np.array([r["kl"] for r in curve]), np.array([r["gold"] for r in curve])
    order = np.argsort(kls)
    return float(np.interp(kl, kls[order], golds[order]))


def summarize(study):
    """All the numbers the README reports."""
    out = {"data_scaling": {}, "kl_penalty": {}, "ensembles": {}}
    for n_pairs, runs in study["data_scaling"].items():
        rl, bon = mean_curve(runs, "rl"), mean_curve(runs, "bon")
        rl_fit = gh.fit_functional_form(np.sqrt(rl["kl"][0]), rl["gold"][0], "rl")
        bon_fit = gh.fit_functional_form(np.sqrt(bon["kl"][0]), bon["gold"][0], "bon")
        out["data_scaling"][n_pairs] = {
            "agreement": [r["info"]["agreement"] for r in runs],
            "rl_peak_gold": [r["rl_stats"]["peak_gold"] for r in runs],
            "rl_kl_at_peak": [r["rl_stats"]["kl_at_peak"] for r in runs],
            "rl_final_gold": [r["rl_stats"]["final_gold"] for r in runs],
            "rl_final_proxy": [r["rl"][-1]["proxy"] for r in runs],
            "rl_final_kl": [r["rl_stats"]["final_kl"] for r in runs],
            "bon_peak_gold": [r["bon_stats"]["peak_gold"] for r in runs],
            "bon_n_at_peak": [max(r["bon"], key=lambda x: x["gold"])["n"] for r in runs],
            "bon_final_gold": [r["bon_stats"]["final_gold"] for r in runs],
            "rl_fit": dict(zip(("alpha", "beta", "r2"), rl_fit)),
            "bon_fit": dict(zip(("alpha", "beta", "r2"), bon_fit)),
        }
    out["oracle"] = {
        "rl_final_gold": study["oracle"]["rl"][-1]["gold"],
        "bon_final_gold": study["oracle"]["bon"][-1]["gold"],
    }
    ref_size = str(study["settings"]["ensemble_data"])
    unpenalized = [r["rl"] for r in study["data_scaling"].get(ref_size, [])]
    for coef, runs in study.get("kl_penalty", {}).items():
        row = {
            "final_gold": [r["rl_stats"]["final_gold"] for r in runs],
            "final_kl": [r["rl_stats"]["final_kl"] for r in runs],
        }
        if len(unpenalized) == len(runs):  # same proxy per seed: compare with stopping that run early
            same = [gold_at_kl(u, r["rl_stats"]["final_kl"]) for u, r in zip(unpenalized, runs)]
            row["unpenalized_gold_at_same_kl"] = same
            row["paired_diff"] = paired([r["rl_stats"]["final_gold"] for r in runs], same)
        out["kl_penalty"][coef] = row
    ens = study.get("ensembles", {})
    for mode, runs in ens.items():
        row = {
            "rl_peak_gold": [r["rl_stats"]["peak_gold"] for r in runs],
            "rl_final_gold": [r["rl_stats"]["final_gold"] for r in runs],
            "rl_final_kl": [r["rl_stats"]["final_kl"] for r in runs],
            "bon_final_gold": [r["bon_stats"]["final_gold"] for r in runs],
            "bon_peak_gold": [r["bon_stats"]["peak_gold"] for r in runs],
        }
        if mode != "single" and "single" in ens:
            row["rl_final_vs_single"] = paired(
                row["rl_final_gold"], [r["rl_stats"]["final_gold"] for r in ens["single"]]
            )
            row["rl_peak_vs_single"] = paired(row["rl_peak_gold"], [r["rl_stats"]["peak_gold"] for r in ens["single"]])
            row["bon_vs_single"] = paired(row["bon_final_gold"], [r["bon_stats"]["final_gold"] for r in ens["single"]])
        out["ensembles"][mode] = row
    return out


def write_markdown(summary, study, path):
    """Three tables: data scaling, KL penalty, ensembles."""
    n_max = study["data_scaling"][next(iter(study["data_scaling"]))][0]["bon"][-1]["n"]
    lines = [
        f"### Proxy data size ({study['settings']['seeds']} seeds each, mean ± std)",
        "",
        f"| Comparisons | Agreement with gold | Best-of-n: peak gold (n at peak) | Best-of-n: gold at n = {n_max} | "
        "RL: peak gold | RL: KL at peak (nats) | RL: final gold | RL: final proxy | Fit R²: best-of-n / RL |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for n_pairs, r in summary["data_scaling"].items():
        lines.append(
            f"| {int(n_pairs):,} | {100 * np.mean(r['agreement']):.1f} % | {ms(r['bon_peak_gold'])} "
            f"({'/'.join(str(n) for n in r['bon_n_at_peak'])}) | {ms(r['bon_final_gold'])} | {ms(r['rl_peak_gold'])} | "
            f"{ms(r['rl_kl_at_peak'])} | {ms(r['rl_final_gold'])} | {ms(r['rl_final_proxy'])} | "
            f"{r['bon_fit']['r2']:.3f} / {r['rl_fit']['r2']:.2f} |"
        )
    o = summary["oracle"]
    lines += [
        "",
        f"Optimizing the gold reward itself reaches {o['rl_final_gold']:.2f} with RL and {o['bon_final_gold']:.2f} "
        f"with best-of-{n_max}.",
        "",
        f"### KL penalty (RL on the same {study['settings']['seeds']} proxies, "
        f"{study['settings']['ensemble_data']:,} comparisons)",
        "",
        "Each penalized run is compared with stopping the unpenalized run on the same proxy at the same KL.",
        "",
        "| KL coefficient | Final KL (nats) | Final gold | Early-stopped unpenalized run, same KL | "
        "Penalty − early stopping (seeds where penalty is better) |",
        "|---|---|---|---|---|",
    ]
    for coef, r in summary["kl_penalty"].items():
        same = ms(r["unpenalized_gold_at_same_kl"]) if "unpenalized_gold_at_same_kl" in r else "—"
        diff = fmt_paired(r["paired_diff"]) if "paired_diff" in r else "—"
        lines.append(f"| {coef} | {ms(r['final_kl'])} | {ms(r['final_gold'])} | {same} | {diff} |")
    if summary["ensembles"]:
        lines += [
            "",
            f"### Ensembles of {study['settings']['ensemble_size']} proxies ({study['settings']['ensemble_data']:,} "
            f"comparisons, {study['settings']['seeds']} seeds)",
            "",
            "Paired differences are against a single member of the same ensemble (seeds where the ensemble is better).",
            "",
            "| Objective | RL: peak gold | RL: final gold | RL: final gold − single | Best-of-n: gold at "
            f"n = {n_max} | Best-of-n − single |",
            "|---|---|---|---|---|---|",
        ]
        for mode, r in summary["ensembles"].items():
            vs = fmt_paired(r["rl_final_vs_single"]) if "rl_final_vs_single" in r else "—"
            bvs = fmt_paired(r["bon_vs_single"]) if "bon_vs_single" in r else "—"
            lines.append(
                f"| {mode} | {ms(r['rl_peak_gold'])} | {ms(r['rl_final_gold'])} | {vs} | "
                f"{ms(r['bon_final_gold'])} | {bvs} |"
            )
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def plot_overoptimization(study, path):
    """Gold reward vs sqrt(KL) for best-of-n and RL, one line per proxy data size, plus the oracle."""
    plt = apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    sizes = list(study["data_scaling"])
    palette = {s: SERIES[i] for i, s in enumerate(sizes)}
    for ax, key, title in ((axes[0], "bon", "Best-of-n"), (axes[1], "rl", "RL (policy gradient, no KL penalty)")):
        o = study["oracle"][key]
        ax.plot(
            np.sqrt([r["kl"] for r in o]),
            [r["gold"] for r in o],
            color=ORACLE,
            linestyle=":",
            label="optimize gold (ceiling)",
        )
        for s in sizes:
            c = mean_curve(study["data_scaling"][s], key)
            d = np.sqrt(c["kl"][0])
            ax.plot(d, c["gold"][0], color=palette[s], label=f"{int(s):,} comparisons")
            ax.fill_between(
                d, c["gold"][0] - c["gold"][1], c["gold"][0] + c["gold"][1], color=palette[s], alpha=0.12, linewidth=0
            )
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("sqrt(KL from the reference policy)  (sqrt nats)")
    axes[0].set_ylabel("gold reward (std of reference samples)")
    axes[1].legend(fontsize=8, loc="lower right")
    seeds, n_max = study["settings"]["seeds"], 2 ** study["settings"]["max_log2_n"]
    note = f"Lines: mean of {seeds} proxy seeds; bands: ±1 std. Best-of-n spans n = 1 to {n_max:,}."
    fig.text(0.01, 0.005, note, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mechanism(study, path):
    """Why the gold reward falls: the proxy keeps rising while the diminishing quality overshoots its peak."""
    plt = apply_style()
    size = str(study["settings"]["ensemble_data"])
    c = mean_curve(study["data_scaling"][size], "rl")
    d = np.sqrt(c["kl"][0])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fields = [
        ("proxy", SERIES[1], "proxy reward (what RL optimizes)"),
        ("gold", SERIES[0], "gold reward"),
        ("diminishing", SERIES[3], "diminishing quality d"),
    ]
    if study["settings"].get("w_steady", 0) > 0:
        fields.append(("steady", SERIES[2], "steady quality s"))
    for field, color, label in fields:
        ax.plot(d, c[field][0], color=color, label=label)
    ax.axhline(study["settings"]["peak"], color=SERIES[3], linestyle=":", linewidth=1)
    ax.text(
        d[0], study["settings"]["peak"], "  d's peak: more d hurts beyond here", color=INK_2, fontsize=8, va="bottom"
    )
    ax.set_xlabel("sqrt(KL)  (sqrt nats)")
    ax.set_ylabel("reference-sample std units")
    title = f"Inside one overoptimization curve ({int(size):,} comparisons, RL)"
    if study["settings"].get("w_steady", 0) > 0:
        title = f"Gold = {study['settings']['w_steady']:g}·s + diminishing d ({int(size):,} comparisons, RL)"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_kl_penalty(study, path):
    """RL trajectories with different KL coefficients on one proxy, in the gold-vs-KL plane."""
    plt = apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    size = str(study["settings"]["ensemble_data"])
    base = study["data_scaling"][size][0]["rl"]
    ax.plot(np.sqrt([r["kl"] for r in base]), [r["gold"] for r in base], color=MUTED, label="no penalty (trajectory)")
    for i, (coef, runs) in enumerate(study["kl_penalty"].items()):
        r = runs[0]
        kl, gold = np.sqrt([x["kl"] for x in r["rl"]]), [x["gold"] for x in r["rl"]]
        ax.plot(kl, gold, color=SERIES[i], label=f"KL coefficient {coef}")
        ax.plot(kl[-1], gold[-1], "o", color=SERIES[i])
    ax.set_xlabel("sqrt(KL)  (sqrt nats)")
    ax.set_ylabel("gold reward")
    ax.set_title("A KL penalty stops the policy early; does it reach a better point? (seed 0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ensembles(study, path):
    """RL gold-vs-KL curves for one proxy and for ensemble objectives (mean of 3 seeds)."""
    plt = apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (mode, runs) in enumerate(study["ensembles"].items()):
        c = mean_curve(runs, "rl")
        ax.plot(np.sqrt(c["kl"][0]), c["gold"][0], color=SERIES[i], label=mode)
    ax.set_xlabel("sqrt(KL)  (sqrt nats)")
    ax.set_ylabel("gold reward")
    ax.set_title("Ensemble objectives under RL")
    ax.legend(fontsize=8, title="objective", title_fontsize=8)
    note = f"uwo_λ: mean − λ·std over members (Coste et al., 2023). Mean of {study['settings']['seeds']} seeds."
    fig.text(0.01, 0.005, note, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(study_path=STUDY, out_dir=OUT):
    """Write the summary and the four figures."""
    with open(study_path) as f:
        study = json.load(f)
    summary = summarize(study)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(write_markdown(summary, study, os.path.join(out_dir, "summary.md")))
    plot_overoptimization(study, os.path.join(out_dir, "overoptimization.png"))
    plot_mechanism(study, os.path.join(out_dir, "mechanism.png"))
    if study.get("kl_penalty"):
        plot_kl_penalty(study, os.path.join(out_dir, "kl_penalty.png"))
    if study.get("ensembles"):
        plot_ensembles(study, os.path.join(out_dir, "ensembles.png"))
    mixed_path = os.path.join(os.path.dirname(study_path), "study_mixed.json")
    if os.path.exists(mixed_path):
        with open(mixed_path) as f:
            mixed = json.load(f)
        summary["mixed"] = summarize(mixed)
        write_markdown(summary["mixed"], mixed, os.path.join(out_dir, "summary_mixed.md"))
        plot_overoptimization(mixed, os.path.join(out_dir, "overoptimization_mixed.png"))
        plot_mechanism(mixed, os.path.join(out_dir, "mechanism_mixed.png"))
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
    return summary


if __name__ == "__main__":
    main()
