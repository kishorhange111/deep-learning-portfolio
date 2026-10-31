"""
Interpret a trained crosscoder: which latents belong to which model, and what do they respond to?

  * relative decoder norm histogram: shared vs. A-only vs. B-only latents
  * decoder cosine of shared latents: is a "shared" latent really the same direction in both models?
  * for the most frequently active B-only and A-only latents, the text contexts where they fire
    most strongly (the activating token is shown in «guillemets»). These are what you read to name
    what fine-tuning added or removed.

    python analyze.py --run_dir results/base_vs_instruct_batchtopk
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from activations import ActivationData  # noqa: E402
from crosscoder import Crosscoder, classify, decoder_cosines, relative_norms  # noqa: E402


def load_run(run_dir, device):
    """Crosscoder + the arguments and data metadata it was trained with."""
    ckpt = torch.load(os.path.join(run_dir, "crosscoder.pt"), map_location=device)
    a = ckpt["args"]
    model = Crosscoder(ckpt["meta"]["d_model"], a["n_latents"], a["sparsity"], a["k"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), a, ckpt["meta"]


@torch.no_grad()
def latent_statistics(model, data, n_samples, top_examples, device, chunk=8192):
    """Activation frequency and the top-activating rows for every latent over the first n_samples rows."""
    n = min(n_samples, data.meta["n"])
    F = model.n_latents
    freq = torch.zeros(F, device=device)
    top_vals = torch.full((F, top_examples), -1.0, device=device)
    top_rows = torch.zeros((F, top_examples), dtype=torch.long, device=device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        x = (torch.from_numpy(np.asarray(data.acts[start:stop], dtype=np.float32)) * data.scale).to(device)
        f = model.encode(x, train_mode=False)  # (b, F)
        freq += (f > 0).sum(0)
        vals = torch.cat([top_vals, f.T], dim=1)
        rows = torch.cat([top_rows, torch.arange(start, stop, device=device).expand(F, -1)], dim=1)
        top_vals, idx = vals.topk(top_examples, dim=1)
        top_rows = rows.gather(1, idx)
    return freq / n, top_vals, top_rows


def context_strings(data_dir, meta, rows, width=12, tokenizer=None):
    """Decoded text around each activation row: '... preceding text «token» next'."""
    index = np.memmap(os.path.join(data_dir, "index.bin"), dtype=np.int32, mode="r", shape=(meta["n"], 2))
    n_seqs = os.path.getsize(os.path.join(data_dir, "tokens.bin")) // (4 * meta["seq_len"])
    tokens = np.memmap(os.path.join(data_dir, "tokens.bin"), dtype=np.int32, mode="r", shape=(n_seqs, meta["seq_len"]))
    out = []
    for row in rows:
        seq, pos = index[row]
        before = tokenizer.decode(tokens[seq, max(0, pos - width) : pos].tolist())
        tok = tokenizer.decode([int(tokens[seq, pos])])
        after = tokenizer.decode(tokens[seq, pos + 1 : pos + 3].tolist())
        out.append(f"…{before}«{tok}»{after}".replace("\n", "⏎").replace("|", "\\|"))
    return out


def plot_histograms(r, cos, alive, shared, out_dir):
    """Relative-norm histogram (all live latents) and decoder-cosine histogram (shared latents)."""
    from common.plotting import MUTED, apply_style

    plt = apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(r[alive].cpu().numpy(), bins=50, range=(0, 1), color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.5)
    axes[0].set_yscale("log")
    for x in (0.1, 0.9):
        axes[0].axvline(x, color=MUTED, linestyle="--", linewidth=1)
    axes[0].set_xlabel("relative decoder norm  ||d_B|| / (||d_A|| + ||d_B||)")
    axes[0].set_ylabel("live latents (log scale)")
    axes[0].set_title("A-only ← shared → B-only")
    if shared.any():
        axes[1].hist(
            cos[shared].cpu().numpy(), bins=40, range=(-1, 1), color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.5
        )
    axes[1].set_xlabel("cosine(d_A, d_B)")
    axes[1].set_title("Shared latents: same direction in both models?")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "relative_norms.png"), dpi=150)
    plt.close(fig)


def analyze(args):
    """Write analysis.json, features.md and relative_norms.png into the run directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_args, meta = load_run(args.run_dir, device)
    data_dir = args.data_dir or (
        os.path.join(args.run_dir, "planted_data") if meta.get("planted") else train_args["data_dir"]
    )
    data = ActivationData(data_dir, 0)
    freq, top_vals, top_rows = latent_statistics(model, data, args.samples, args.examples, device)
    r, cos = relative_norms(model).detach(), decoder_cosines(model).detach()
    alive = freq > 0
    masks = {k: v & alive for k, v in classify(r).items()}
    summary = {
        "alive": int(alive.sum()),
        **{k: int(v.sum()) for k, v in masks.items()},
        "shared_cosine_median": float(cos[masks["shared"]].median()) if masks["shared"].any() else None,
    }
    plot_histograms(r, cos, alive, masks["shared"], args.run_dir)

    tokenizer = None
    if not meta.get("planted"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(meta["model_a"])
    lines = [
        f"# Crosscoder latents: {meta['model_a']} (A) vs {meta['model_b']} (B), layer {meta.get('layer')}",
        "",
        f"Live latents: {summary['alive']} · shared: {summary['shared']} · A-only: {summary['A-only']} · "
        f"B-only: {summary['B-only']} · median decoder cosine of shared latents: {summary['shared_cosine_median']}",
        "",
    ]
    latents = {}
    for cls in ("B-only", "A-only", "shared"):
        ids = torch.nonzero(masks[cls]).flatten()
        ids = ids[freq[ids].argsort(descending=True)][: args.top_latents if cls != "shared" else 5].tolist()
        lines += [
            f"## {cls} latents (most frequent first)",
            "",
            "| Latent | rel. norm | freq | top contexts |",
            "|---|---|---|---|",
        ]
        for i in ids:
            contexts = (
                context_strings(data_dir, meta, top_rows[i].tolist()[:3], tokenizer=tokenizer) if tokenizer else []
            )
            latents[i] = {
                "class": cls,
                "relative_norm": float(r[i]),
                "frequency": float(freq[i]),
                "max_activation": float(top_vals[i, 0]),
                "contexts": contexts,
            }
            lines.append(f"| {i} | {r[i]:.2f} | {freq[i]:.2e} | {'<br>'.join(contexts) or '—'} |")
        lines.append("")
    with open(os.path.join(args.run_dir, "features.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(args.run_dir, "analysis.json"), "w") as f:
        json.dump({"summary": summary, "latents": latents}, f, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


def main(argv=None):
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--samples", type=int, default=500_000)
    p.add_argument("--examples", type=int, default=10)
    p.add_argument("--top_latents", type=int, default=15)
    return analyze(p.parse_args(argv))


if __name__ == "__main__":
    main()
