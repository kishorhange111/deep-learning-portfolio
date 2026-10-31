"""
Train a sparse crosscoder on paired activations from two models, to find what fine-tuning changed.

Input: the memmaps written by collect.py (activations from model A and model B at the same tokens).
Output: the crosscoder, training curves and the summary analyze.py starts from. That summary covers
reconstruction quality per model (fraction of variance explained), sparsity (L0), dead latents,
and how many latents are shared, A-only or B-only by relative decoder norm.

--planted writes a synthetic dataset with known shared, A-only and B-only features first and
trains on it. This checks the method itself: a working crosscoder must give the planted B-only
features relative norm ~1 and the A-only ones ~0. The tests rely on it.

    python train.py --data_dir data/base_vs_instruct                    # BatchTopK (default)
    python train.py --data_dir data/base_vs_instruct --sparsity l1      # Anthropic's L1 objective
    python train.py --planted                                           # sanity check, CPU, ~15 s
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crosscoder import Crosscoder, classify, decoder_cosines, fraction_variance_explained, relative_norms  # noqa: E402


def parse_args(argv=None):
    """Options; defaults suit Qwen2.5-0.5B (d_model 896) on one A100."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=os.path.join(HERE, "data", "base_vs_instruct"))
    p.add_argument("--planted", action="store_true", help="train on synthetic data with known features")
    p.add_argument("--n_latents", type=int, default=16384)
    p.add_argument("--sparsity", choices=["batchtopk", "l1"], default="batchtopk")
    p.add_argument("--k", type=int, default=100, help="BatchTopK: average active latents per token")
    p.add_argument("--l1_coeff", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=None, help="default 1e-4 (batchtopk) / 5e-5 (l1)")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", default=None)
    p.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    args = p.parse_args(argv)
    if args.planted:
        args.n_latents, args.k, args.steps, args.batch = 128, 4, 3000, 512
        args.l1_coeff, args.lr, args.log_every = 0.3, args.lr or 3e-3, 500
    if os.environ.get("QUICK"):
        args.steps, args.log_every = min(args.steps, 300), 100
    args.lr = args.lr or (1e-4 if args.sparsity == "batchtopk" else 5e-5)
    return args


def write_planted_dataset(out_dir, n=60000, d=32, n_shared=24, n_only=6, p=0.04, seed=0):
    """Synthetic paired activations in collect.py's format, plus the true feature directions.

    Shared features appear in both models with the same coefficient but *different* directions (each
    model has its own basis). A-only and B-only features appear in one model only."""
    g = torch.Generator().manual_seed(seed)

    def directions(k):
        v = torch.randn(k, d, generator=g)
        return v / v.norm(dim=-1, keepdim=True)

    def coefficients(k):
        return (torch.rand(n, k, generator=g) < p).float() * (0.5 + torch.rand(n, k, generator=g))

    shared_a, shared_b, only_a, only_b = (
        directions(n_shared),
        directions(n_shared),
        directions(n_only),
        directions(n_only),
    )
    cs = coefficients(n_shared)
    acts = torch.stack(
        [cs @ shared_a + coefficients(n_only) @ only_a, cs @ shared_b + coefficients(n_only) @ only_b], 1
    )
    os.makedirs(out_dir, exist_ok=True)
    acts.numpy().astype(np.float16).tofile(os.path.join(out_dir, "acts.bin"))
    meta = {"model_a": "planted-A", "model_b": "planted-B", "d_model": d, "n": n, "scale": [1.0, 1.0], "planted": True}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    truth = {"shared_a": shared_a, "shared_b": shared_b, "only_a": only_a, "only_b": only_b}
    torch.save(truth, os.path.join(out_dir, "truth.pt"))
    return truth


class ActivationData:
    """Random minibatches from the (N, 2, d) float16 memmap, scaled so mean ||x|| ~ sqrt(d) per model."""

    def __init__(self, data_dir, seed):
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)
        n, d = self.meta["n"], self.meta["d_model"]
        self.acts = np.memmap(os.path.join(data_dir, "acts.bin"), dtype=np.float16, mode="r", shape=(n, 2, d))
        self.scale = torch.tensor(self.meta["scale"], dtype=torch.float32).view(1, 2, 1)
        self.rng = np.random.default_rng(seed)

    def batch(self, size, device):
        """Sorted random indices read faster from a memmap; order within a batch does not matter."""
        idx = np.sort(self.rng.integers(0, len(self.acts), size=size))  # with replacement: duplicates are negligible
        return (torch.from_numpy(np.asarray(self.acts[idx], dtype=np.float32)) * self.scale).to(device)

    def first(self, size, device):
        """A fixed evaluation slice."""
        return (torch.from_numpy(np.asarray(self.acts[:size], dtype=np.float32)) * self.scale).to(device)


def planted_recovery(model, truth):
    """For each planted direction, the best-matching live latent's cosine and relative norm."""
    dec = model.W_dec / model.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    r = relative_norms(model)
    out = {}
    for name, dirs, m in [
        ("only_a", truth["only_a"], 0),
        ("only_b", truth["only_b"], 1),
        ("shared", truth["shared_b"], 1),
    ]:
        cos = dirs.to(dec.device) @ dec[:, m].T
        best = cos.argmax(1)
        out[name] = {"cosine": cos.max(1).values.tolist(), "relative_norm": r[best].tolist()}
    return out


def train(args):
    """Train, evaluate and write results/<run>/{crosscoder.pt, train_log.jsonl, metrics.json}."""
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = args.run_name or (
        "planted" if args.planted else f"{os.path.basename(os.path.normpath(args.data_dir))}_{args.sparsity}"
    )
    run_dir = os.path.join(args.out_dir, run)
    os.makedirs(run_dir, exist_ok=True)
    data_dir = os.path.join(run_dir, "planted_data") if args.planted else args.data_dir
    truth = write_planted_dataset(data_dir) if args.planted else None
    data = ActivationData(data_dir, args.seed)
    d = data.meta["d_model"]
    model = Crosscoder(d, args.n_latents, args.sparsity, args.k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    warmup_l1 = max(1, int(0.05 * args.steps))  # ramp the L1 penalty in, as in the crosscoder paper
    last_active = torch.zeros(args.n_latents, device=device)
    log, t0 = [], time.time()
    with open(os.path.join(run_dir, "train_log.jsonl"), "w") as log_file:
        for step in range(args.steps):
            x = data.batch(args.batch, device)
            l1 = args.l1_coeff * min(1.0, step / warmup_l1) if args.sparsity == "l1" else 0.0
            loss, f, stats = model.loss(x, l1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_active = torch.where((f > 0).any(0), torch.zeros_like(last_active), last_active + 1)
            if step % args.log_every == 0 or step == args.steps - 1:
                rec = {
                    "step": step,
                    "loss": loss.item(),
                    **stats,
                    "dead_frac": (last_active > 1000).float().mean().item(),
                }
                log.append(rec)
                log_file.write(json.dumps(rec) + "\n")
                print(
                    f"step {step:6d}  mse {stats['mse']:.4f}  L0 {stats['l0']:.1f}  dead {rec['dead_frac']:.3f}  "
                    f"({time.time() - t0:.0f}s)"
                )

    model.eval()
    with torch.no_grad():
        x = data.first(min(20000, data.meta["n"]), device)
        x_hat, f = model(x)
        fve = fraction_variance_explained(x, x_hat).tolist()
        r = relative_norms(model)
        alive = (f > 0).any(0)
        classes = {k: int((v & alive).sum()) for k, v in classify(r).items()}
        cos = decoder_cosines(model)
        shared_mask = classify(r)["shared"] & alive
        metrics = {
            "run": run,
            "sparsity": args.sparsity,
            "n_latents": args.n_latents,
            "d_model": d,
            "fve_model_a": fve[0],
            "fve_model_b": fve[1],
            "l0": (f > 0).float().sum(-1).mean().item(),
            "alive_latents": int(alive.sum()),
            "latent_classes": classes,
            "shared_decoder_cosine_median": float(cos[shared_mask].median()) if shared_mask.any() else None,
            "model_a": data.meta["model_a"],
            "model_b": data.meta["model_b"],
            "layer": data.meta.get("layer"),
            "steps": args.steps,
            "minutes": (time.time() - t0) / 60,
        }
        if truth is not None:
            metrics["planted_recovery"] = planted_recovery(model, truth)
    torch.save(
        {"state_dict": model.state_dict(), "args": vars(args), "meta": data.meta},
        os.path.join(run_dir, "crosscoder.pt"),
    )
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "planted_recovery"}, indent=1))
    return model, metrics


def main(argv=None):
    """Entry point."""
    return train(parse_args(argv))[1]


if __name__ == "__main__":
    main()
