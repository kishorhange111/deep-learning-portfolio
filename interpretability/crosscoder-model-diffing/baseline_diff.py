"""
Cheap model-diffing baselines to read before the crosscoder.

  * weights: relative change ||W_B - W_A|| / ||W_A|| for every matrix, grouped by layer and type
    (attention q/k/v/o, MLP gate/up/down). This shows where fine-tuning moved the parameters.
  * activations: cosine similarity between the two models' residual streams at every layer, on the
    same tokens. This shows where the representations diverge.

These say *where* the models differ, not *what* differs; the crosscoder answers the second question.

    python baseline_diff.py --model_a Qwen/Qwen2.5-0.5B --model_b Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))

GROUPS = {
    "q_proj": "attn q",
    "k_proj": "attn k",
    "v_proj": "attn v",
    "o_proj": "attn o",
    "gate_proj": "mlp gate",
    "up_proj": "mlp up",
    "down_proj": "mlp down",
}


def weight_diff(model_a, model_b):
    """{group: [relative change per layer]} for every decoder-layer weight matrix."""
    params_b = dict(model_b.named_parameters())
    out = {}
    for name, wa in model_a.named_parameters():
        m = re.search(r"layers\.(\d+)\..*\.(\w+_proj)\.weight$", name)
        if not m or m.group(2) not in GROUPS:
            continue
        layer, group = int(m.group(1)), GROUPS[m.group(2)]
        rel = ((params_b[name].float() - wa.float()).norm() / wa.float().norm()).item()
        out.setdefault(group, {})[layer] = rel
    return {g: [v[i] for i in sorted(v)] for g, v in out.items()}


@torch.no_grad()
def activation_cosine(model_a, model_b, batches):
    """Mean per-token cosine similarity of the residual streams after each layer."""
    sums, count = None, 0
    for ids, mask in batches:
        ha = model_a(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        hb = model_b(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        m = mask.bool()
        cos = torch.stack(
            [torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)[m].sum() for a, b in zip(ha, hb)]
        )
        sums = cos if sums is None else sums + cos
        count += m.sum().item()
    return (sums / count).tolist()


def plot(wd, cos, out_path):
    """Two panels: weight change per layer and type; residual-stream cosine per layer."""
    from common.plotting import ColorMap, apply_style

    plt = apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    groups = ["attn q", "attn k", "attn v", "attn o", "mlp gate", "mlp up", "mlp down"]
    colors = ColorMap(groups)
    for g in groups:
        if g in wd:
            axes[0].plot(range(len(wd[g])), wd[g], marker="o", markersize=3, color=colors[g], label=g)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("||W_B − W_A|| / ||W_A||")
    axes[0].set_title("Where did fine-tuning change the weights?")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].plot(range(len(cos)), cos, marker="o", markersize=3, color="#2a78d6")
    axes[1].set_xlabel("residual stream after layer (0 = embeddings)")
    axes[1].set_ylabel("mean cosine similarity A vs B")
    axes[1].set_title("Where do the representations diverge?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    """Load both models, compute both diffs, write JSON and a figure."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_a", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--model_b", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--texts_file", default=None, help="jsonl with a 'text' field; default: collect.py's chat+web mix")
    p.add_argument("--n_texts", type=int, default=64)
    p.add_argument("--out", default=os.path.join(HERE, "results", "baseline_diff"))
    args = p.parse_args(argv)

    import collect

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model_a)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    ma, mb = (
        AutoModelForCausalLM.from_pretrained(m, dtype=dtype).to(device).eval() for m in (args.model_a, args.model_b)
    )
    stream = collect.text_stream(argparse.Namespace(texts_file=args.texts_file, sources=["chat", "web"]), tok)
    texts = [next(stream) for _ in range(args.n_texts)]
    batches = []
    for i in range(0, len(texts), 8):
        enc = tok(texts[i : i + 8], return_tensors="pt", padding=True, truncation=True, max_length=256)
        batches.append((enc["input_ids"].to(device), enc["attention_mask"].to(device)))
    wd, cos = weight_diff(ma, mb), activation_cosine(ma, mb, batches)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "baseline_diff.json"), "w") as f:
        json.dump(
            {"model_a": args.model_a, "model_b": args.model_b, "weight_relative_change": wd, "residual_cosine": cos},
            f,
            indent=1,
        )
    plot(wd, cos, os.path.join(args.out, "baseline_diff.png"))
    return wd, cos


if __name__ == "__main__":
    main()
