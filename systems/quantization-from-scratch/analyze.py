"""
Figures and tables for the quantization study, from results/quantization.json (written by train.py).

    python analyze.py     # -> results/summary.md, results/summary.json and the figures
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # <repo>/src
from dl_common.plotting import MUTED, SERIES, apply_style  # noqa: E402

STUDY = os.path.join(HERE, "results", "quantization.json")
OUT = os.path.join(HERE, "results")


def summarize(q):
    """Rows of (bits, group, effective bits, RTN loss, GPTQ mean/std, deltas) plus the ablations."""
    base = q["float_val_bpb"]
    rows = []
    for r in q["grid"]:
        g = [x["val_bpb"] for x in r["gptq"]]
        rows.append(
            {
                "bits": r["bits"],
                "group": r["group"],
                "effective_bits": r["effective_bits"],
                "rtn": r["rtn"]["val_bpb"],
                "gptq_mean": float(np.mean(g)),
                "gptq_std": float(np.std(g, ddof=1)) if len(g) > 1 else 0.0,
                "rtn_delta": r["rtn"]["val_bpb"] - base,
                "gptq_delta": float(np.mean(g)) - base,
                "gptq_recovers": (r["rtn"]["val_bpb"] - float(np.mean(g))) / (r["rtn"]["val_bpb"] - base)
                if r["rtn"]["val_bpb"] - base > 5e-3  # too small to divide by below this
                else None,
                "rtn_weight_error": float(np.mean(list(r["rtn"]["weight_rel_error"].values()))),
                "gptq_weight_error": float(np.mean([np.mean(list(x["weight_rel_error"].values())) for x in r["gptq"]])),
                "gptq_seconds": float(np.mean([x["quantize_seconds"] for x in r["gptq"]])),
            }
        )
    sens = {k: v - base for k, v in q["sensitivity"]["layers"].items()}
    calib = {
        int(k): (float(np.mean(v)) - base, float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        for k, v in q["calibration"]["sizes"].items()
    }
    return {"float_val_bpb": base, "rows": rows, "sensitivity": sens, "calibration": calib}


def write_markdown(s, q, path):
    """The main table and the two ablations."""
    lines = [
        f"Float model: **{s['float_val_bpb']:.4f}** bits per byte on held-out files "
        f"({q['quantized_params']:,} weights quantized; {q['float_params_kept']:,} embedding / head / "
        "LayerNorm parameters kept in float).",
        "",
        "| Bits | Groups | Effective bits / weight | RTN: bits/byte (Δ) | GPTQ: bits/byte (Δ), mean ± std of "
        f"{q['settings']['calib_seeds']} calibration sets | "
        "Share of RTN's damage GPTQ removes | Weight error RTN / GPTQ |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in s["rows"]:
        rec = f"{100 * r['gptq_recovers']:.0f} %" if r["gptq_recovers"] is not None else "—"
        lines.append(
            f"| {r['bits']} | {r['group']} | {r['effective_bits']:.2f} | {r['rtn']:.4f} ({r['rtn_delta']:+.4f}) | "
            f"{r['gptq_mean']:.4f} ± {r['gptq_std']:.4f} ({r['gptq_delta']:+.4f}) | {rec} | "
            f"{r['rtn_weight_error']:.3f} / {r['gptq_weight_error']:.3f} |"
        )
    lines += [
        "",
        f"Layer sensitivity (RTN, {q['sensitivity']['bits']} bits, per channel, one layer type at a time; "
        "Δ bits/byte):",
        "",
        "| Layer type | Δ bits/byte |",
        "|---|---|",
    ]
    for k, v in sorted(s["sensitivity"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v:+.4f} |")
    lines += [
        "",
        f"GPTQ calibration size ({q['calibration']['bits']} bits, group {q['calibration']['group']}; "
        f"windows of {q['settings']['seq_len']} bytes; Δ bits/byte, mean ± std of "
        f"{q['settings']['calib_seeds']} sets):",
        "",
        "| Calibration windows | Δ bits/byte |",
        "|---|---|",
    ]
    for n, (m, sd) in sorted(s["calibration"].items()):
        lines.append(f"| {n} | {m:+.4f} ± {sd:.4f} |")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def plot_tradeoff(s, path):
    """Loss increase over the float model against effective bits per weight, RTN vs GPTQ (log scale)."""
    plt = apply_style()
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(7, 4.3))
    for method, color, key in (("RTN", SERIES[1], "rtn_delta"), ("GPTQ", SERIES[0], "gptq_delta")):
        for group, marker in (("channel", "o"), ("64", "s"), ("32", "^")):
            pts = sorted((r["effective_bits"], r[key]) for r in s["rows"] if r["group"] == group and r[key] > 1e-3)
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker=marker,
                    color=color,
                    linestyle="-" if group == "channel" else ":",
                    label=f"{method}, {'per channel' if group == 'channel' else 'group ' + group}",
                )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("effective bits per weight (scales and zero points included)")
    ax.set_ylabel("held-out loss increase over float (bits/byte)")
    ax.set_title("Quality vs storage: plain rounding against GPTQ")
    ax.legend(fontsize=7, ncol=2)
    note = f"Float model: {s['float_val_bpb']:.3f} bits/byte. 8-bit settings are within 0.0002 of float (not shown)."
    fig.text(0.01, 0.005, note, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(study_path=STUDY, out_dir=OUT):
    """Write the summary and the figure."""
    with open(study_path) as f:
        q = json.load(f)
    s = summarize(q)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    print(write_markdown(s, q, os.path.join(out_dir, "summary.md")))
    plot_tradeoff(s, os.path.join(out_dir, "tradeoff.png"))
    return s


if __name__ == "__main__":
    main()
