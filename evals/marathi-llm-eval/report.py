"""
Turn a run directory into results: tables with confidence intervals, the English-Marathi gap,
robustness checks, language fidelity, judge scores and how far the judges can be trusted.

    python report.py --run main        # writes results/main/summary.md, summary.json and figures
"""

import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import eval_stats as st  # noqa: E402
from providers import cost_usd  # noqa: E402
from run_eval import read_jsonl  # noqa: E402

CONDITION_NAMES = {"en": "English", "mr": "Marathi", "mr_roman": "Marathi (Latin script)"}


def load(run_dir, prefix):
    """All rows from files starting with prefix."""
    rows = []
    for name in sorted(os.listdir(run_dir)):
        if name.startswith(prefix) and name.endswith(".jsonl"):
            rows += read_jsonl(os.path.join(run_dir, name))
    return rows


def mcq_summary(rows, prices=None):
    """Per model: accuracy per condition with CI, gaps, order and cross-lingual consistency, cost."""
    out = {}
    for model in sorted({r["model"] for r in rows}):
        mrows = [r for r in rows if r["model"] == model]
        by = {(r["condition"], r["reversed"], r["id"]): r for r in mrows}
        conds = sorted({r["condition"] for r in mrows}, key=list(CONDITION_NAMES).index)
        res = {
            "conditions": {},
            "tokens": [sum(r["input_tokens"] for r in mrows), sum(r["output_tokens"] for r in mrows)],
        }
        res["cost_usd"] = cost_usd(model, *res["tokens"], prices)
        for c in conds:
            sel = [r for r in mrows if r["condition"] == c and not r["reversed"]]
            mean, lo, hi = st.bootstrap_ci([r["correct"] for r in sel])
            cres = {
                "accuracy": mean,
                "ci": [lo, hi],
                "n": len(sel),
                "unparsed": sum(not r["parsed"] for r in sel),
                "refused": sum(r["refused"] for r in sel),
                "errors": sum(r["error"] is not None for r in sel),
                "mean_input_tokens": float(np.mean([r["input_tokens"] for r in sel])),
            }
            rev = [(r, by.get((c, True, r["id"]))) for r in sel]
            rev = [(a, b) for a, b in rev if b is not None and a["pred"] is not None and b["pred"] is not None]
            if rev:
                cres["order_consistency"] = float(np.mean([a["pred"] == b["pred"] for a, b in rev]))
            res["conditions"][c] = cres
        ids = [r["id"] for r in mrows if r["condition"] == "en" and not r["reversed"]]

        def correct(cond, by=by, ids=ids):
            return [by[(cond, False, i)]["correct"] for i in ids if (cond, False, i) in by]

        if "en" in conds and "mr" in conds:
            res["gap_en_minus_mr"] = st.paired_bootstrap(correct("en"), correct("mr"))
            same = [by[("en", False, i)]["pred"] == by[("mr", False, i)]["pred"] for i in ids if ("mr", False, i) in by]
            res["crosslingual_consistency"] = float(np.mean(same))
        if "en" in conds and "mr" in conds and res["conditions"]["en"]["mean_input_tokens"]:
            # the same question costs this many times more input tokens in Marathi (tokenizer "tax")
            res["token_ratio_mr_en"] = (
                res["conditions"]["mr"]["mean_input_tokens"] / res["conditions"]["en"]["mean_input_tokens"]
            )
        if "mr" in conds and "mr_roman" in conds:
            res["gap_mr_minus_roman"] = st.paired_bootstrap(correct("mr"), correct("mr_roman"))
        out[model] = res
    return out


def generation_summary(gen_rows, judge_rows):
    """Language-label shares per model and mean judge scores per (judge, model)."""
    out = {"language": {}, "judges": defaultdict(dict)}
    for model in sorted({r["model"] for r in gen_rows}):
        rows = [r for r in gen_rows if r["model"] == model and not r["error"]]
        counts = Counter(r["lang_label"] for r in rows)
        out["language"][model] = {
            k: counts.get(k, 0) / max(1, len(rows)) for k in ("marathi", "hindi", "english", "mixed", "empty")
        }
        out["language"][model]["n"] = len(rows)
    for (judge, model), rows in itertools.groupby(
        sorted(judge_rows, key=lambda r: (r["judge"], r["model"])), key=lambda r: (r["judge"], r["model"])
    ):
        rows = [r for r in rows if r.get("judge_parsed")]
        if rows:
            out["judges"][judge][model] = {k: st.bootstrap_ci([r[k] for r in rows]) for k in ("helpfulness", "fluency")}
    out["judges"] = dict(out["judges"])
    return out


def agreement_summary(gen_rows, judge_rows, human_rows):
    """Judge-vs-judge, judge-vs-human and automatic-language-check-vs-human agreement."""
    out = {}
    judged = defaultdict(dict)  # (model, id) -> {judge: row}
    for r in judge_rows:
        if r.get("judge_parsed"):
            judged[(r["model"], r["id"])][r["judge"]] = r
    judges = sorted({j for d in judged.values() for j in d})
    for a, b in itertools.combinations(judges, 2):
        keys = [k for k, d in judged.items() if a in d and b in d]
        out[f"{a} vs {b}"] = {
            m: st.agreement([judged[k][a][m] for k in keys], [judged[k][b][m] for k in keys])
            for m in ("helpfulness", "fluency")
        }
    if human_rows:
        human = {(r["model"], r["id"]): r for r in human_rows}
        for j in judges:
            keys = [k for k in human if j in judged.get(k, {})]
            if keys:
                out[f"{j} vs human"] = {
                    m: st.agreement([judged[k][j][m] for k in keys], [human[k][m] for k in keys])
                    for m in ("helpfulness", "fluency")
                }
        auto = {(r["model"], r["id"]): r["lang_label"] for r in gen_rows}
        keys = [k for k in human if k in auto]
        if keys:
            out["language check vs human"] = {
                "n": len(keys),
                "accuracy": float(np.mean([auto[k] == human[k]["language"] for k in keys])),
                "confusion": Counter(f"{human[k]['language']}->{auto[k]}" for k in keys),
            }
    return out


def contamination_summary(rows):
    """Mean continuation overlap vs. the chance baseline, per model and language, and the share of
    passages reproduced largely verbatim (overlap > 0.5)."""
    out = {}
    for (model, lang), grp in itertools.groupby(
        sorted(rows, key=lambda r: (r["model"], r["lang"])), key=lambda r: (r["model"], r["lang"])
    ):
        grp = [r for r in grp if not r["error"]]
        if grp:
            diff = st.paired_bootstrap([r["overlap"] for r in grp], [r["baseline"] for r in grp])
            out.setdefault(model, {})[lang] = {
                "overlap": float(np.mean([r["overlap"] for r in grp])),
                "baseline": float(np.mean([r["baseline"] for r in grp])),
                "excess": diff,
                "verbatim_share": float(np.mean([r["overlap"] > 0.5 for r in grp])),
                "n": len(grp),
            }
    return out


def fmt_ci(mean, lo, hi, pct=True):
    """'63.2 (60.1–66.3)'."""
    s = 100 if pct else 1
    return f"{s * mean:.1f} ({s * lo:.1f}–{s * hi:.1f})" if pct else f"{mean:.2f} ({lo:.2f}–{hi:.2f})"


def write_markdown(mcq, gen, agree, path, contam=None):
    """summary.md."""
    lines = []
    if mcq:
        lines += [
            "## Reading comprehension (Belebele, accuracy % with 95 % bootstrap CI)",
            "",
            "| Model | English | Marathi | Marathi (Latin script) | Gap EN−MR (pts, 95 % CI) | Same answer EN & MR | "
            "Order consistency (MR) | Tokens MR ÷ EN | Cost |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for model, r in mcq.items():
            cells = [
                fmt_ci(r["conditions"][c]["accuracy"], *r["conditions"][c]["ci"]) if c in r["conditions"] else "—"
                for c in ("en", "mr", "mr_roman")
            ]
            g = r.get("gap_en_minus_mr")
            gap = f"{100 * g['diff']:+.1f} ({100 * g['ci_low']:+.1f}, {100 * g['ci_high']:+.1f})" if g else "—"
            cons = f"{100 * r['crosslingual_consistency']:.0f} %" if "crosslingual_consistency" in r else "—"
            oc = r["conditions"].get("mr", {}).get("order_consistency")
            cost = f"${r['cost_usd']:.2f}" if r["cost_usd"] is not None else "—"
            ratio = f"{r['token_ratio_mr_en']:.1f}×" if "token_ratio_mr_en" in r else "—"
            order = f"{100 * oc:.0f} %" if oc is not None else "—"
            lines.append(f"| {model} | {' | '.join(cells)} | {gap} | {cons} | {order} | {ratio} | {cost} |")
        lines.append("")
    if gen and gen["language"]:
        lines += [
            "## Language fidelity on Marathi prompts (automatic check)",
            "",
            "| Model | Marathi | Hindi | English | Mixed | n |",
            "|---|---|---|---|---|---|",
        ]
        for model, s in gen["language"].items():
            lines.append(
                f"| {model} | {100 * s['marathi']:.0f} % | {100 * s['hindi']:.0f} % | {100 * s['english']:.0f} % | "
                f"{100 * (s['mixed'] + s['empty']):.0f} % | {s['n']} |"
            )
        lines.append("")
    for judge, models in (gen or {}).get("judges", {}).items():
        lines += [f"## Judge: {judge} (mean 1–5, 95 % CI)", "", "| Model | Helpfulness | Fluency |", "|---|---|---|"]
        lines += [
            f"| {m} | {fmt_ci(*s['helpfulness'], pct=False)} | {fmt_ci(*s['fluency'], pct=False)} |"
            for m, s in models.items()
        ]
        lines.append("")
    if contam:
        lines += [
            "## Memorisation probe (continue a Belebele/FLORES passage from its first half)",
            "",
            "| Model | Language | Char 5-gram overlap with true continuation | Chance (other passage) | "
            "Excess (95 % CI) | Passages > 50 % overlap |",
            "|---|---|---|---|---|---|",
        ]
        for model, langs in contam.items():
            for lang, c in langs.items():
                e = c["excess"]
                lines.append(
                    f"| {model} | {lang} | {c['overlap']:.3f} | {c['baseline']:.3f} | {e['diff']:+.3f} "
                    f"({e['ci_low']:+.3f}, {e['ci_high']:+.3f}) | {100 * c['verbatim_share']:.0f} % |"
                )
        lines.append("")
    if agree:
        lines += [
            "## Can the judges be trusted?",
            "",
            "| Pair | Metric | n | Spearman | Quadratic κ | Exact | Within 1 |",
            "|---|---|---|---|---|---|---|",
        ]
        for pair, metrics in agree.items():
            if pair == "language check vs human":
                lines.append(
                    f"| {pair} | language label | {metrics['n']} | — | — | {100 * metrics['accuracy']:.0f} % | — |"
                )
                continue
            for m, a in metrics.items():
                lines.append(
                    f"| {pair} | {m} | {a['n']} | {a['spearman']:.2f} | {a['qwk']:.2f} | {100 * a['exact']:.0f} % | "
                    f"{100 * a['within_1']:.0f} % |"
                )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def plot(mcq, gen, run_dir):
    """mcq_accuracy.png (dot-and-whisker per model and condition) and language_fidelity.png (stacked bars)."""
    from common.plotting import ColorMap, apply_style

    plt = apply_style()
    if mcq:
        colors = ColorMap(list(CONDITION_NAMES))
        models = list(mcq)
        fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(models) + 1.5))
        for i, m in enumerate(models):
            for j, c in enumerate(CONDITION_NAMES):
                if c in mcq[m]["conditions"]:
                    r = mcq[m]["conditions"][c]
                    y = i + (j - 1) * 0.22
                    ax.plot([100 * r["ci"][0], 100 * r["ci"][1]], [y, y], color=colors[c], linewidth=2)
                    ax.plot(100 * r["accuracy"], y, "o", color=colors[c], label=CONDITION_NAMES[c] if i == 0 else None)
        ax.set_yticks(range(len(models)), models)
        ax.axvline(25, color="#898781", linewidth=1, linestyle="--")
        ax.set_xlabel("Belebele accuracy (%)  ·  dashed line = chance")
        ax.set_title("Same questions, three scripts/languages")
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "mcq_accuracy.png"), dpi=150)
        plt.close(fig)
    if gen and gen["language"]:
        cats = ["marathi", "hindi", "english", "mixed"]
        colors = ColorMap(cats)
        models = list(gen["language"])
        fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(models) + 1.4))
        left = np.zeros(len(models))
        for c in cats:
            vals = np.array(
                [100 * (gen["language"][m][c] + (gen["language"][m]["empty"] if c == "mixed" else 0)) for m in models]
            )
            ax.barh(models, vals, left=left, color=colors[c], label=c, edgecolor="#fcfcfb", linewidth=2, height=0.6)
            left += vals
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of answers to Marathi prompts")
        ax.set_title("Which language does the model answer in?")
        ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25))
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "language_fidelity.png"), dpi=150)
        plt.close(fig)


def report(run_dir, prices=None):
    """Compute everything for one run directory."""
    mcq_rows, gen_rows, judge_rows = load(run_dir, "mcq_"), load(run_dir, "gen_"), load(run_dir, "judge_")
    contam = contamination_summary(load(run_dir, "contam_"))
    human_path = os.path.join(run_dir, "human_labels.jsonl")
    human = read_jsonl(human_path) if os.path.exists(human_path) else []
    mcq = mcq_summary(mcq_rows, prices) if mcq_rows else {}
    gen = generation_summary(gen_rows, judge_rows) if gen_rows else {}
    agree = agreement_summary(gen_rows, judge_rows, human) if judge_rows else {}
    summary = {"mcq": mcq, "generation": gen, "agreement": agree, "contamination": contam}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=dict)
    print(write_markdown(mcq, gen, agree, os.path.join(run_dir, "summary.md"), contam))
    plot(mcq, gen, run_dir)
    return summary


def main(argv=None):
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="main")
    p.add_argument("--prices", default=None, help='JSON {"model-id": [usd_per_M_input, usd_per_M_output]}')
    p.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    args = p.parse_args(argv)
    prices = None
    if args.prices:
        with open(args.prices) as f:
            prices = {k: tuple(v) for k, v in json.load(f).items()}
    return report(os.path.join(args.out_dir, args.run), prices)


if __name__ == "__main__":
    main()
