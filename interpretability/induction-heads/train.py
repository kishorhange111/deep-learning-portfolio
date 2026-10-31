"""
Induction heads: watch them form, find them, and show they cause in-context learning.

A replication of Olsson et al. (2022), "In-context Learning and Induction Heads", at single-GPU
scale. A 1-layer and a 2-layer attention-only transformer are trained on FineWeb-Edu, and every
`eval_every` steps the script measures:
  * in-context learning score: loss late in the context minus loss early in the context
  * per-head induction and previous-token scores on repeated random token sequences
  * loss on the repeated half of those sequences (only predictable by copying from context)

Expected result: the 2-layer model shows a phase change. Within a short window of training, an
induction head appears, the in-context learning score drops sharply, and repeated-sequence loss
collapses. The 1-layer model shows none of this. The 2-layer model is then probed causally:
  * ablating the top induction head should destroy copying;
  * ablating layer 0's previous-token head should too, because the induction head reads its output
    through K-composition, which the composition scores check from the weights alone;
  * ablating random heads should barely matter.

    python train.py                               # uses the pretraining project's FineWeb-Edu shards
    python train.py --synthetic                   # no data needed (repeating-span synthetic text)
    QUICK=1 python train.py --synthetic           # CPU smoke test
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import attn_only as ao  # noqa: E402

DEFAULT_DATA = os.path.join(HERE, "..", "..", "systems", "gpt-pretraining-scaling-laws", "data", "fineweb_edu")


def parse_args(argv=None):
    """Options; QUICK=1 makes everything tiny."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=DEFAULT_DATA)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--layers", nargs="*", type=int, default=[1, 2])
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--vocab", type=int, default=50304)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--rep_len", type=int, default=50, help="length of the random sequence that gets repeated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    args = p.parse_args(argv)
    if os.environ.get("QUICK"):
        args.steps, args.batch, args.seq_len, args.d_model, args.n_head = 30, 8, 64, 32, 4
        args.vocab, args.eval_every, args.rep_len = 512, 10, 12
    return args


class ShardWindows:
    """Random windows from the uint16 token shards written by the pretraining project's prepare_data.py."""

    def __init__(self, data_dir, split, seq_len, seed):
        with open(os.path.join(data_dir, "meta.json")) as f:
            meta = json.load(f)
        self.shards = [np.memmap(os.path.join(data_dir, s["path"]), dtype=np.uint16, mode="r") for s in meta[split]]
        self.seq_len, self.rng = seq_len, np.random.default_rng(seed)

    def batch(self, n):
        """(n, seq_len + 1) int64 tensor."""
        out = []
        for _ in range(n):
            s = self.shards[self.rng.integers(len(self.shards))]
            i = self.rng.integers(0, len(s) - self.seq_len - 1)
            out.append(np.asarray(s[i : i + self.seq_len + 1], dtype=np.int64))
        return torch.from_numpy(np.stack(out))


class SyntheticText:
    """LZ77-style random text: at each point either emit a fresh random token or copy a span of 4-12
    tokens from earlier in the same sequence. About half of all tokens are copies, which are
    predictable only by looking back at the context (the job induction heads do)."""

    def __init__(self, vocab, seq_len, seed, copy_prob=0.5):
        self.vocab, self.seq_len, self.copy_prob, self.rng = vocab, seq_len, copy_prob, random.Random(seed)

    def batch(self, n):
        """(n, seq_len + 1) int64 tensor."""
        rows = []
        for _ in range(n):
            seq = [self.rng.randrange(self.vocab) for _ in range(4)]
            while len(seq) < self.seq_len + 1:
                if len(seq) > 8 and self.rng.random() < self.copy_prob:
                    start = self.rng.randrange(0, len(seq) - 4)
                    span = self.rng.randrange(4, 13)
                    seq += seq[start : start + span]
                else:
                    seq.append(self.rng.randrange(self.vocab))
            rows.append(seq[: self.seq_len + 1])
        return torch.tensor(rows)


@torch.no_grad()
def measure(model, val, rep_tokens, rep_len, ablate=()):
    """In-context learning score, repeated-sequence loss and head scores (with optional ablation)."""
    model.eval()
    icl = ao.in_context_learning_score(ao.per_position_loss(model, val, ablate))
    rep_loss = ao.repeated_sequence_loss(model, rep_tokens, rep_len, ablate)
    _, patterns = model(rep_tokens, ablate=ablate, return_patterns=True)
    induction, previous = ao.head_scores(patterns, rep_len)
    model.train()
    return {"icl_score": icl, "rep_loss": rep_loss, "induction": induction.tolist(), "previous": previous.tolist()}


def train_one(n_layer, args, data, val, rep_tokens, device):
    """Train one model, measuring every eval_every steps; returns (model, log)."""
    torch.manual_seed(args.seed)
    model = ao.AttnOnlyTransformer(args.vocab, n_layer, args.n_head, args.d_model, args.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
    log, t0 = [], time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            m = measure(model, val, rep_tokens, args.rep_len)
            m.update({"step": step, "tokens": step * args.batch * args.seq_len})
            log.append(m)
            best = max(max(row) for row in m["induction"])
            print(
                f"[{n_layer}L] step {step:5d}  icl {m['icl_score']:+.3f}  rep-loss {m['rep_loss']:.3f}  "
                f"max induction {best:.3f}  ({time.time() - t0:.0f}s)"
            )
        if step == args.steps:
            break
        x = data.batch(args.batch).to(device)
        logits = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model, log


def ablation_study(model, val, rep_tokens, rep_len, seed=0):
    """Causal tests on the 2-layer model: knock out heads and measure copying and in-context learning."""
    base = measure(model, val, rep_tokens, rep_len)
    ind, prev = torch.tensor(base["induction"]), torch.tensor(base["previous"])
    top_ind = ind[1].argsort(descending=True).tolist()
    top_prev = int(prev[0].argmax())
    rng = random.Random(seed)
    others1 = [h for h in range(model.n_head) if h not in top_ind[:2]]
    others0 = [h for h in range(model.n_head) if h != top_prev]
    conditions = {
        "none": [],
        "top induction head (L1)": [(1, top_ind[0])],
        "top 2 induction heads (L1)": [(1, top_ind[0]), (1, top_ind[1])],
        "2 random other L1 heads": [(1, h) for h in rng.sample(others1, 2)],
        "top previous-token head (L0)": [(0, top_prev)],
        "1 random other L0 head": [(0, rng.choice(others0))],
        # heads are often redundant (several induction heads), so also knock out each whole group
        "all L1 heads with induction score > 0.3": [(1, h) for h in range(model.n_head) if ind[1, h] > 0.3],
        "all L0 heads with previous-token score > 0.2": [(0, h) for h in range(model.n_head) if prev[0, h] > 0.2],
    }
    rows = {}
    for name, heads in conditions.items():
        m = measure(model, val, rep_tokens, rep_len, ablate=set(heads))
        rows[name] = {"heads": heads, "rep_loss": m["rep_loss"], "icl_score": m["icl_score"]}
    return rows


def heatmap(ax, data, title, cmap, vmin=None, vmax=None, fmt="{:.2f}", ylabels=None, xlabel="head"):
    """Annotated grid; text colour flips for contrast on dark cells."""
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    lo, hi = (vmin if vmin is not None else data.min()), (vmax if vmax is not None else data.max())
    for (r, c), v in np.ndenumerate(data):
        dark = (v - lo) / (hi - lo + 1e-9) > 0.6
        ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=7, color="white" if dark else "#0b0b0b")
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]), ylabels or [f"L{i}" for i in range(data.shape[0])])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(False)
    return im


def make_figures(logs, final, composition, out_dir):
    """phase_change.png, head_scores.png, composition.png."""
    from matplotlib.colors import LinearSegmentedColormap

    from common.plotting import ColorMap, apply_style

    plt = apply_style()
    colors = ColorMap([f"{n}-layer" for n in sorted(logs)])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for n, log in sorted(logs.items()):
        toks = [r["tokens"] / 1e6 for r in log]
        c = colors[f"{n}-layer"]
        axes[0].plot(toks, [r["icl_score"] for r in log], color=c, label=f"{n}-layer")
        axes[1].plot(toks, [max(map(max, r["induction"])) for r in log], color=c, label=f"{n}-layer")
        axes[2].plot(toks, [r["rep_loss"] for r in log], color=c, label=f"{n}-layer")
    for ax, t in zip(
        axes,
        [
            "In-context learning score (lower = more)",
            "Max induction score (any head)",
            "Loss on repeated random tokens",
        ],
    ):
        ax.set_title(t)
        ax.set_xlabel("training tokens (M)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_change.png"), dpi=150)
    plt.close(fig)

    blues = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.6))
    heatmap(axes[0], np.array(final["previous"]), "Previous-token score", blues, 0, 1)
    heatmap(axes[1], np.array(final["induction"]), "Induction score", blues, 0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "head_scores.png"), dpi=150)
    plt.close(fig)

    if composition is not None:
        diverging = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
        lim = float(np.abs(composition).max())
        fig, ax = plt.subplots(figsize=(6.5, 5))
        heatmap(
            ax,
            composition,
            "K-composition: layer-0 head (row) -> layer-1 head (column)",
            diverging,
            -lim,
            lim,
            ylabels=[f"L0.{h}" for h in range(composition.shape[0])],
            xlabel="layer-1 head",
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "composition.png"), dpi=150)
        plt.close(fig)


def main(argv=None):
    """Train the 1- and 2-layer models, then analyse the 2-layer one."""
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.synthetic:
        data, val_src = SyntheticText(args.vocab, args.seq_len, args.seed), SyntheticText(args.vocab, args.seq_len, 999)
    else:
        data = ShardWindows(args.data_dir, "train", args.seq_len, args.seed)
        val_src = ShardWindows(args.data_dir, "val", args.seq_len, 999)
    val = val_src.batch(64 if not os.environ.get("QUICK") else 8).to(device)
    gen = torch.Generator().manual_seed(1)
    rep_tokens = ao.repeated_random_tokens(
        64 if not os.environ.get("QUICK") else 8, args.rep_len, args.vocab, gen, device
    )

    logs, models = {}, {}
    for n in args.layers:
        models[n], logs[n] = train_one(n, args, data, val, rep_tokens, device)
        with open(os.path.join(args.out_dir, f"train_log_{n}L.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in logs[n])

    results = {"config": vars(args), "final": {n: logs[n][-1] for n in logs}}
    composition = None
    if 2 in models:
        model = models[2]
        results["ablation"] = ablation_study(model, val, rep_tokens, args.rep_len, args.seed)
        comps = {k: ao.composition_scores(model, k) for k in ("K", "Q", "V")}
        composition = comps["K"].numpy()
        results["composition"] = {k: v.tolist() for k, v in comps.items()}
        ind = torch.tensor(logs[2][-1]["induction"])
        top = int(ind[1].argmax())
        results["top_induction_head"] = [1, top]
        results["strongest_K_composition_into_top_head"] = int(comps["K"][:, top].argmax())
        lines = ["| Ablated | Repeated-sequence loss | In-context learning score |", "|---|---|---|"]
        for k, v in results["ablation"].items():
            heads = ", ".join(f"L{lyr}.{h}" for lyr, h in v["heads"])
            lines.append(f"| {k}{f' ({heads})' if heads else ''} | {v['rep_loss']:.3f} | {v['icl_score']:+.3f} |")
        with open(os.path.join(args.out_dir, "ablation.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1)
    make_figures(logs, logs[max(logs)][-1], composition, args.out_dir)
    return results


if __name__ == "__main__":
    main()
