"""
Sparse autoencoders on a byte-level language model: which sparsity mechanism gives the best
reconstruction per active latent, and are the latents more interpretable than the model's own
dimensions or random directions?

Steps:
  1. Train a 4-layer byte-level GPT on Python source (the standard library; ~8M bytes), or load it.
  2. Collect the residual stream after layer 2 of 4 on 524k training bytes and 32k held-out bytes.
  3. Train SAEs with 2,048 latents (16x the width) using relu + L1, topk and batchtopk at four
     sparsity levels each, and measure on held-out text:
       * L0, the average number of active latents per byte;
       * fraction of variance unexplained (FVU);
       * loss recovered: splice the reconstruction into the model in place of the real activations
         and measure next-byte loss, as a fraction of the gap between mean-ablating the layer (0)
         and the clean model (1);
       * dead latents: never active on the held-out set.
  4. Interpretability check on the topk (k = 32) SAE: for every live latent, its 20 strongest
     activations on held-out text; "purity" = share of them on the same current byte. Compared with
     the same statistic for the model's 128 residual dimensions and for 2,048 random directions.
     A random sample of latents is written out with their top contexts (results/features.md).

    python train.py      # ~35 min on 2 CPU cores -> results/study.json, results/features.md
    python analyze.py    # figures and results/summary.md
    QUICK=1 python train.py
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # <repo>/src
import sae_lib  # noqa: E402

from dl_common.byte_corpus import (  # noqa: E402
    BYTE_CLASS_TABLE,
    BYTE_CLASSES,
    fixed_windows,
    load_corpus,
    random_windows,
    synthetic_corpus,
)
from dl_common.bytelm import ByteGPT, ByteGPTConfig, lr_schedule  # noqa: E402


def parse_args(argv=None):
    """Options; QUICK=1 makes everything tiny."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lm_steps", type=int, default=2000)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--layer", type=int, default=1, help="residual stream after this block (0-based)")
    p.add_argument("--train_windows", type=int, default=4096)
    p.add_argument("--eval_windows", type=int, default=256)
    p.add_argument("--latents", type=int, default=2048)
    p.add_argument("--sae_steps", type=int, default=600)
    p.add_argument("--ks", nargs="*", type=int, default=[8, 16, 32, 64])
    p.add_argument("--l1s", nargs="*", type=float, default=[0.3, 1.0, 3.0, 10.0])
    p.add_argument("--feature_k", type=int, default=32, help="which topk SAE to analyze feature by feature")
    p.add_argument("--top_n", type=int, default=20)
    p.add_argument("--sample_features", type=int, default=30)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--features_only", action="store_true", help="redo only the feature analysis (updates study.json)")
    p.add_argument("--checkpoint", default=os.path.join(HERE, "checkpoints", "byte_lm.pt"))
    p.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    args = p.parse_args(argv)
    if os.environ.get("QUICK"):
        args.lm_steps, args.seq_len, args.train_windows, args.eval_windows = 30, 32, 64, 16
        args.latents, args.sae_steps, args.ks, args.l1s, args.feature_k = 128, 20, [4, 8], [1.0], 8
        args.top_n, args.sample_features, args.synthetic = 5, 4, True
    return args


def train_lm(args, data):
    """The language model under study: trained here (or loaded from checkpoints/)."""
    train, _, _ = data
    quick = os.environ.get("QUICK")
    cfg = ByteGPTConfig(seq_len=args.seq_len, n_layer=4 if not quick else 2, d_model=128 if not quick else 32, n_head=4)
    cfg.d_ff = 4 * cfg.d_model
    torch.manual_seed(0)
    model = ByteGPT(cfg)
    if os.path.exists(args.checkpoint) and not quick:
        model.load_state_dict(torch.load(args.checkpoint))
        return model.eval()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}], lr=3e-3, betas=(0.9, 0.95)
    )
    rng = np.random.default_rng(0)
    for step in range(args.lm_steps):
        for g in opt.param_groups:
            g["lr"] = 3e-3 * lr_schedule(step, args.lm_steps, max(1, args.lm_steps // 20))
        x, y = random_windows(train, 32, args.seq_len, rng)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"LM step {step}: {loss.item() / math.log(2):.3f} bits/byte", flush=True)
    if not quick:
        os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
    return model.eval()


@torch.no_grad()
def collect(model, layer, x, batch=64):
    """Residual-stream activations after block `layer` for windows x: (windows * seq_len, d)."""
    store = []
    hook = model.blocks[layer].register_forward_hook(lambda m, i, o: store.append(o.reshape(-1, o.shape[-1])))
    try:
        for i in range(0, len(x), batch):
            model(torch.from_numpy(x[i : i + batch]))
    finally:
        hook.remove()
    return torch.cat(store)


@torch.no_grad()
def spliced_loss(model, layer, x, y, replace, batch=64):
    """Mean next-byte loss (bits) when block `layer`'s output is passed through `replace` (or kept)."""
    hook = None
    if replace is not None:
        hook = model.blocks[layer].register_forward_hook(lambda m, i, o: replace(o.reshape(-1, o.shape[-1])).view_as(o))
    try:
        total, count = 0.0, 0
        for i in range(0, len(x), batch):
            logits, _ = model(torch.from_numpy(x[i : i + batch]))
            yb = torch.from_numpy(y[i : i + batch])
            total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1), reduction="sum"))
            count += yb.numel()
    finally:
        if hook is not None:
            hook.remove()
    return total / count / math.log(2)


def purity_stats(scores, cur_bytes, next_bytes, top_n):
    """For each direction (column of scores), the share of its top_n activations on the most common
    current byte, on the most common next byte, and on the most common byte class."""
    top = torch.topk(scores, top_n, dim=0).indices.numpy()  # (top_n, directions)
    out = {"current": [], "next": [], "class": [], "mode_current": [], "mode_next": []}
    for j in range(top.shape[1]):
        cur, nxt = cur_bytes[top[:, j]], next_bytes[top[:, j]]
        cls = BYTE_CLASS_TABLE[cur]
        for key, vals in (("current", cur), ("next", nxt), ("class", cls)):
            counts = np.bincount(vals)
            out[key].append(float(counts.max() / top_n))
        out["mode_current"].append(int(np.bincount(cur).argmax()))
        out["mode_next"].append(int(np.bincount(nxt).argmax()))
    return out


def show(b):
    """Printable form of a byte for the feature table."""
    c = chr(b)
    return {"\n": "⏎", " ": "␣", "\t": "⇥", "|": "\\|", "`": "\\`"}.get(
        c, c if c.isprintable() and b < 128 else f"\\x{b:02x}"
    )


def context(flat, pos, seq_len, before=24, after=4):
    """The bytes around a position, with the activating byte bracketed. Stays inside the position's own
    window (the model never sees bytes from other windows)."""
    start = pos - pos % seq_len
    lo, hi = max(start, pos - before), min(start + seq_len, pos + after + 1)
    return (
        "".join(show(int(b)) for b in flat[lo:pos])
        + "["
        + show(int(flat[pos]))
        + "]"
        + "".join(show(int(b)) for b in flat[pos + 1 : hi])
    )


def analyze_features(sae, eval_acts, vx, vy, args):
    """Interpretability check on one SAE: current- / next-byte purity of each live latent's strongest
    activations, against the residual dimensions and random directions; writes results/features.md
    with a random sample of live latents. Returns the purity statistics."""
    flat = vx.reshape(-1)
    nxt = vy.reshape(-1)
    with torch.no_grad():
        codes = sae.encode(eval_acts)
    alive = (codes > 0).sum(0) >= args.top_n
    centered = eval_acts - eval_acts.mean(0)
    g = torch.Generator().manual_seed(0)
    random_dirs = torch.randn(eval_acts.shape[1], args.latents, generator=g)
    random_dirs = random_dirs / random_dirs.norm(dim=0, keepdim=True)
    sign = torch.where(torch.rand(args.latents, generator=g) < 0.5, -1.0, 1.0)  # either end of a direction
    purity = {
        "sae": purity_stats(codes[:, alive], flat, nxt, args.top_n),
        "residual_dims": purity_stats(torch.cat([centered, -centered], dim=1), flat, nxt, args.top_n),
        "random_dirs": purity_stats((centered @ random_dirs) * sign, flat, nxt, args.top_n),
    }
    features = {"alive": int(alive.sum()), "n_latents": args.latents, "purity": purity}

    # a random sample of live latents, with their strongest contexts
    alive_idx = torch.nonzero(alive).squeeze(1).numpy()
    rng = np.random.default_rng(0)
    sample = sorted(rng.choice(alive_idx, size=min(args.sample_features, len(alive_idx)), replace=False).tolist())
    freq = (codes > 0).float().mean(0)
    lines = [
        f"# A random sample of {len(sample)} live latents (topk SAE, k = {args.feature_k})",
        "",
        "Each row: the latent's firing rate on held-out text, the most common current byte among its "
        f"{args.top_n} strongest activations (with purity), the most common next byte, and its three "
        "strongest contexts (the activating byte in brackets; ⏎ = newline, ␣ = space).",
        "",
        "| Latent | Fires on | Current byte (purity) | Next byte (purity) | Top contexts |",
        "|---|---|---|---|---|",
    ]
    pos_in_alive = {int(j): i for i, j in enumerate(alive_idx)}
    for j in sample:
        i = pos_in_alive[j]
        top = torch.topk(codes[:, j], 3).indices.numpy()
        ctx = "<br>".join(f"`{context(flat, int(t), args.seq_len)}`" for t in top)
        lines.append(
            f"| {j} | {100 * float(freq[j]):.2f} % | {show(purity['sae']['mode_current'][i])} "
            f"({100 * purity['sae']['current'][i]:.0f} %) | {show(purity['sae']['mode_next'][i])} "
            f"({100 * purity['sae']['next'][i]:.0f} %) | {ctx} |"
        )
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "features.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return features


def main(argv=None):
    """Train the LM and the SAEs, measure them, analyze features; write results/."""
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    t0 = time.time()
    data = synthetic_corpus() if args.synthetic else load_corpus()
    model = train_lm(args, data)
    for p in model.parameters():
        p.requires_grad_(False)
    tx, _ = fixed_windows(data[0], args.train_windows, args.seq_len, seed=7)
    vx, vy = fixed_windows(data[1], args.eval_windows, args.seq_len, seed=1)
    train_acts, eval_acts = collect(model, args.layer, tx), collect(model, args.layer, vx)
    scale = float(math.sqrt(train_acts.shape[1] / train_acts.pow(2).sum(1).mean()))  # mean ||x||^2 = d
    train_acts, eval_acts = train_acts * scale, eval_acts * scale
    mean_act = eval_acts.mean(0)
    clean = spliced_loss(model, args.layer, vx, vy, None)
    ablated = spliced_loss(model, args.layer, vx, vy, lambda h: (mean_act / scale).expand_as(h))
    out = {
        "settings": {k: v for k, v in vars(args).items() if k not in ("checkpoint", "out_dir")},
        "corpus": data[2] if len(data) > 2 else None,
        "clean_bpb": clean,
        "mean_ablated_bpb": ablated,
        "activation_scale": scale,
        "n_train_tokens": len(train_acts),
        "n_eval_tokens": len(eval_acts),
        "saes": [],
    }
    print(
        f"activations collected; clean {clean:.3f}, mean-ablated {ablated:.3f} bits/byte ({time.time() - t0:.0f}s)",
        flush=True,
    )
    if args.features_only:
        sae = sae_lib.SparseAutoencoder(train_acts.shape[1], args.latents, kind="topk", k=args.feature_k, seed=0)
        sae_lib.train_sae(sae, train_acts, args.sae_steps)
        path = os.path.join(args.out_dir, "study.json")
        with open(path) as f:
            previous = json.load(f)
        previous["features"] = analyze_features(sae, eval_acts, vx, vy, args)
        with open(path, "w") as f:
            json.dump(previous, f)
        print(f"updated the feature analysis in {path} ({time.time() - t0:.0f}s)")
        return previous

    configs = [("relu", None, l1) for l1 in args.l1s] + [
        (kind, k, 0.0) for kind in ("topk", "batchtopk") for k in args.ks
    ]
    saes = {}
    for kind, k, l1 in configs:
        sae = sae_lib.SparseAutoencoder(train_acts.shape[1], args.latents, kind=kind, k=k or 0, l1=l1, seed=0)
        curve = sae_lib.train_sae(sae, train_acts, args.sae_steps)
        stats = sae_lib.reconstruction_stats(sae, eval_acts)
        with torch.no_grad():
            spliced = spliced_loss(model, args.layer, vx, vy, lambda h, s=sae: s.decode(s.encode(h * scale)) / scale)
        rec = {
            "kind": kind,
            "k": k,
            "l1": l1,
            "l0": stats["l0"],
            "fvu": stats["fvu"],
            "dead_frac": float((stats["fires"] == 0).float().mean()),
            "spliced_bpb": spliced,
            "loss_recovered": (ablated - spliced) / (ablated - clean),
            "curve": curve,
        }
        out["saes"].append(rec)
        saes[(kind, k, l1)] = sae
        print(
            f"{kind:>9} k={k} l1={l1}: L0 {rec['l0']:.1f}  FVU {rec['fvu']:.3f}  recovered {rec['loss_recovered']:.3f}  "
            f"dead {rec['dead_frac']:.2f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    out["features"] = analyze_features(saes[("topk", args.feature_k, 0.0)], eval_acts, vx, vy, args)
    out["byte_classes"] = list(BYTE_CLASSES)
    out["seconds"] = time.time() - t0
    with open(os.path.join(args.out_dir, "study.json"), "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out_dir} in {out['seconds']:.0f}s")
    return out


if __name__ == "__main__":
    main()
