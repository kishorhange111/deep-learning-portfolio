"""
Reward-model overoptimization: when does optimizing a learned reward stop helping, and what delays it?

A CPU-scale replication of the synthetic setup in Gao, Schulman and Hilton (2022), "Scaling Laws
for Reward Model Overoptimization", plus the ensemble mitigation of Coste et al. (2023), "Reward
Model Ensembles Help Mitigate Overoptimization". A gold reward stands in for human judgement.
Proxy reward models are trained on comparisons that the gold reward labels, with Bradley-Terry
noise, between samples from a reference policy. Then a policy is optimized against each proxy in
two ways, and the gold reward is measured as a function of KL(policy || reference):
  * best-of-n: sample n from the reference and keep the proxy's favourite; KL = log n - (n - 1) / n;
  * RL: policy gradient (RLOO) on the proxy, optionally with a KL penalty.

Questions (the statistics are fixed in advance: 5 seeds of proxy training per setting, mean +/- std,
and paired differences where two methods share a seed's proxies):
  1. Overoptimization: does gold reward peak and then fall while the proxy keeps rising?
  2. Data: how do the peak and where it happens change with the number of comparisons (1k - 100k)?
  3. Functional form: do Gao et al.'s fits hold here, d (alpha - beta d) for best-of-n and
     d (alpha - beta log d) for RL, with d = sqrt(KL)?
  4. KL penalty: is it better than stopping early, i.e. does it move the gold-vs-KL frontier?
  5. Ensembles: do mean, worst-case and uncertainty-weighted ensembles of 5 proxies raise the
     peak or remove the decline?

    python train.py        # ~15 min on one CPU core -> results/study.json
    python train.py --w_steady 0.5 --no_extras --out results/study_mixed.json   # gold with a steady part too
    python analyze.py      # figures and results/summary.md
    QUICK=1 python train.py
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
import goodhart as gh  # noqa: E402


def parse_args(argv=None):
    """Options; QUICK=1 makes everything tiny."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--vocab", type=int, default=8)
    p.add_argument("--peak", type=float, default=2.5, help="where the diminishing quality turns negative (std units)")
    p.add_argument(
        "--w_steady", type=float, default=0.0, help="weight of the steady-returns quality in the gold reward"
    )
    p.add_argument(
        "--no_extras", action="store_true", help="only the data-size experiment (no KL penalty, no ensembles)"
    )
    p.add_argument("--data_sizes", nargs="*", type=int, default=[1000, 3000, 10000, 30000, 100000])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--pool", type=int, default=2**19, help="reference samples for the best-of-n estimator")
    p.add_argument("--max_log2_n", type=int, default=12)
    p.add_argument("--rl_steps", type=int, default=800)
    p.add_argument("--kl_coefs", nargs="*", type=float, default=[0.01, 0.03, 0.1, 0.3])
    p.add_argument("--ensemble_size", type=int, default=5)
    p.add_argument("--ensemble_data", type=int, default=10000)
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads (the tensors are small)")
    p.add_argument("--out", default=os.path.join(HERE, "results", "study.json"))
    args = p.parse_args(argv)
    if os.environ.get("QUICK"):
        args.data_sizes, args.seeds, args.pool, args.max_log2_n = [300, 1000], 2, 4096, 6
        args.rl_steps, args.kl_coefs, args.ensemble_size, args.ensemble_data = 40, [0.1], 2, 300
    return args


class Setup:
    """The reference policy, the gold reward, a best-of-n pool and held-out pairs for agreement."""

    def __init__(self, args):
        self.args = args
        self.ref = gh.make_reference(args.length, args.vocab, seed=0)
        self.gold = gh.GoldReward(args.length, args.vocab, seed=1, peak=args.peak, w_steady=args.w_steady)
        g = torch.Generator().manual_seed(12345)
        self.gold.normalize(self.ref.sample(200_000, g))
        self.pool = self.ref.sample(args.pool, g)
        self.pool_gold = self.score(self.gold)
        s, d = self.gold.parts(self.pool)
        self.pool_parts = {"steady": s.numpy(), "diminishing": d.numpy()}
        self.eval_a, self.eval_b = self.ref.sample(20_000, g), self.ref.sample(20_000, g)
        self.ns = [2**k for k in range(args.max_log2_n + 1)]

    @torch.no_grad()
    def score(self, fn, chunk=65536):
        """Score the whole pool with a reward function, in chunks."""
        return np.concatenate([fn(self.pool[i : i + chunk]).numpy() for i in range(0, len(self.pool), chunk)])

    def train_proxy(self, n_pairs, data_seed, init_seed):
        """A proxy trained on `n_pairs` gold-labelled comparisons, normalized on reference samples."""
        data = gh.preference_data(self.ref, self.gold, n_pairs, seed=data_seed)
        proxy, info = gh.train_proxy(data, self.args.length, self.args.vocab, seed=init_seed)
        proxy.normalize(self.pool[:100_000])
        info["agreement"] = gh.preference_accuracy(proxy, self.gold, self.eval_a, self.eval_b)
        return proxy, info

    def evaluate(self, reward_fn, kl_coef=0.0, seed=0):
        """Best-of-n and RL curves against one reward function."""
        bon = gh.best_of_n_curve(self.score(reward_fn), self.pool_gold, self.ns, extra=self.pool_parts)
        rl = gh.rl_optimize(self.ref, reward_fn, self.gold, steps=self.args.rl_steps, kl_coef=kl_coef, seed=seed)
        return {"bon": bon, "rl": rl}


def peak_stats(curve):
    """Peak gold reward on a curve, the KL where it happens, and the gold reward at the curve's end."""
    best = max(curve, key=lambda r: r["gold"])
    return {
        "peak_gold": best["gold"],
        "kl_at_peak": best["kl"],
        "final_gold": curve[-1]["gold"],
        "final_kl": curve[-1]["kl"],
    }


def main(argv=None):
    """Run every experiment and write results/study.json."""
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    t0 = time.time()
    setup = Setup(args)
    out = {"settings": vars(args), "gold_peak_share": float((setup.pool_parts["diminishing"] > args.peak).mean())}

    # 0. the ceiling: optimize the gold reward itself
    out["oracle"] = setup.evaluate(setup.gold)
    print(f"oracle done ({time.time() - t0:.0f}s)", flush=True)

    # 1-3. proxies trained on more and more comparisons
    out["data_scaling"], kl_proxies = {}, []
    for n_pairs in args.data_sizes:
        runs = []
        for seed in range(args.seeds):
            proxy, info = setup.train_proxy(n_pairs, data_seed=1000 * seed + n_pairs, init_seed=seed)
            res = setup.evaluate(proxy, seed=seed)
            res.update(info=info, bon_stats=peak_stats(res["bon"]), rl_stats=peak_stats(res["rl"]))
            runs.append(res)
            if n_pairs == args.ensemble_data:
                kl_proxies.append(proxy)
        out["data_scaling"][str(n_pairs)] = runs
        agree = np.mean([r["info"]["agreement"] for r in runs])
        peak = np.mean([r["rl_stats"]["peak_gold"] for r in runs])
        print(
            f"{n_pairs:>7} pairs: agreement {agree:.3f}, RL peak gold {peak:.2f} ({time.time() - t0:.0f}s)", flush=True
        )

    if not args.no_extras:
        # 4. KL penalty, on the same proxies (one per seed) as the unpenalized runs at this data size
        if not kl_proxies:
            kl_proxies = [
                setup.train_proxy(args.ensemble_data, data_seed=1000 * s + args.ensemble_data, init_seed=s)[0]
                for s in range(args.seeds)
            ]
        out["kl_penalty"] = {}
        for coef in args.kl_coefs:
            runs = []
            for seed, proxy in enumerate(kl_proxies):
                rl = gh.rl_optimize(setup.ref, proxy, setup.gold, steps=args.rl_steps, kl_coef=coef, seed=seed)
                runs.append({"rl": rl, "rl_stats": peak_stats(rl)})
            out["kl_penalty"][str(coef)] = runs
        print(f"KL penalty done ({time.time() - t0:.0f}s)", flush=True)

        # 5. ensembles: members share the data and differ in initialization (as in Coste et al.)
        out["ensembles"] = {}
        for seed in range(args.seeds):
            data_seed = 1000 * seed + args.ensemble_data
            members = [
                setup.train_proxy(args.ensemble_data, data_seed, init_seed=100 + seed * 10 + k)[0]
                for k in range(args.ensemble_size)
            ]
            for mode, lam in (("single", None), ("mean", None), ("min", None), ("uwo", 0.5), ("uwo", 1.0)):
                fn = members[0] if mode == "single" else gh.Ensemble(members, mode, lam if lam is not None else 1.0)
                name = mode if lam is None else f"{mode}_{lam}"
                res = setup.evaluate(fn, seed=seed)
                res.update(bon_stats=peak_stats(res["bon"]), rl_stats=peak_stats(res["rl"]))
                out["ensembles"].setdefault(name, []).append(res)
            print(f"ensemble seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    out["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out} in {out['seconds']:.0f}s")
    return out


if __name__ == "__main__":
    main()
