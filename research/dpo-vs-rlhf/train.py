"""
DPO vs RLHF: does skipping the reward model also skip reward-model overoptimization?

Uses the exact-KL setup of research/reward-model-overoptimization (same reference policy, same gold
reward with diminishing returns, same Bradley-Terry-labelled comparisons), so every method is
measured as gold reward against KL(policy || reference). Methods, all trained on the same pairs:
  * DPO at beta in {0.01, 0.03, 0.1, 0.3, 1} and IPO at tau in {0.003, 0.01, 0.03, 0.1, 0.3};
  * RLHF: an MLP reward model (as in the overoptimization study), then RLOO with a KL penalty;
  * RLHF with a *bigram* reward model, which has exactly the function class of DPO's implicit reward
    for this policy - the control that separates "DPO vs RL" from "which reward function";
  * best-of-n against the MLP reward model;
  * the ceiling: RL against the gold reward itself with a KL penalty.

Statistics (fixed before running): 3 seeds per setting (new comparisons and initialization each),
mean +/- std; DPO and RLHF on the same seed share the same comparisons.

    python train.py        # ~25 min on one CPU core -> results/study.json
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
import direct_alignment as da  # noqa: E402

gh = da.gh


def parse_args(argv=None):
    """Options; QUICK=1 makes everything tiny."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--vocab", type=int, default=8)
    p.add_argument("--peak", type=float, default=2.5)
    p.add_argument("--n_pairs", type=int, default=10000)
    p.add_argument("--data_sizes", nargs="*", type=int, default=[1000, 3000, 10000, 30000, 100000])
    p.add_argument("--dpo_betas", nargs="*", type=float, default=[0.01, 0.03, 0.1, 0.3, 1.0])
    p.add_argument("--ipo_taus", nargs="*", type=float, default=[0.003, 0.01, 0.03, 0.1, 0.3])
    p.add_argument("--kl_coefs", nargs="*", type=float, default=[0.0, 0.01, 0.03, 0.1, 0.3])
    p.add_argument("--oracle_kl_coefs", nargs="*", type=float, default=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    p.add_argument("--data_size_beta", type=float, default=0.03)
    p.add_argument("--steps", type=int, default=1200, help="DPO / IPO optimizer steps (~60 epochs of 10k pairs)")
    p.add_argument("--rl_steps", type=int, default=800)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--pool", type=int, default=2**19)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", default=os.path.join(HERE, "results", "study.json"))
    args = p.parse_args(argv)
    if os.environ.get("QUICK"):
        args.n_pairs, args.data_sizes, args.dpo_betas, args.ipo_taus = 300, [300, 600], [0.1], [0.1]
        args.kl_coefs, args.oracle_kl_coefs, args.steps, args.rl_steps = [0.0], [0.1], 20, 20
        args.seeds, args.pool = 1, 4096
    return args


def main(argv=None):
    """Run every method and write results/study.json."""
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    t0 = time.time()
    ref = gh.make_reference(args.length, args.vocab, seed=0)
    gold = gh.GoldReward(args.length, args.vocab, seed=1, peak=args.peak)
    g = torch.Generator().manual_seed(12345)
    gold.normalize(ref.sample(200_000, g))
    pool = ref.sample(args.pool, g)
    held_out = (ref.sample(10_000, g), ref.sample(10_000, g))
    with torch.no_grad():
        pool_gold = gold(pool).numpy()
    ns = [2**k for k in range(int(np.log2(args.pool)) - 6)]
    out = {"settings": vars(args), "dpo": {}, "ipo": {}, "rlhf_mlp": {}, "rlhf_bigram": {}, "bon_mlp": [], "oracle": {}}

    def log_time(msg):
        print(f"{msg} ({time.time() - t0:.0f}s)", flush=True)

    # the ceiling: KL-regularized RL on the gold reward itself
    for coef in args.oracle_kl_coefs:
        out["oracle"][str(coef)] = gh.rl_optimize(ref, gold, gold, steps=args.rl_steps, kl_coef=coef, seed=0)
    log_time("oracle")

    data_by_seed = {}
    for seed in range(args.seeds):
        raw = gh.preference_data(ref, gold, args.n_pairs, seed=1000 * seed + args.n_pairs)
        pairs = da.as_pairs(raw)
        data_by_seed[seed] = (raw, pairs)
        for beta in args.dpo_betas:
            log = da.train_direct(ref, gold, pairs, "dpo", beta, args.steps, seed=seed, held_out=held_out)
            out["dpo"].setdefault(str(beta), []).append({"log": log, **da.endpoint(log)})
        for tau in args.ipo_taus:
            log = da.train_direct(ref, gold, pairs, "ipo", tau, args.steps, seed=seed, held_out=held_out)
            out["ipo"].setdefault(str(tau), []).append({"log": log, **da.endpoint(log)})
        log_time(f"seed {seed}: DPO and IPO")

        mlp, info = gh.train_proxy(raw, args.length, args.vocab, seed=seed)
        mlp.normalize(pool[:100_000])
        bigram, binfo = da.train_bigram_proxy(pairs, args.length, args.vocab, seed=seed)
        bigram.normalize(pool[:100_000])
        agreement = {
            "mlp": gh.preference_accuracy(mlp, gold, *held_out),
            "bigram": gh.preference_accuracy(bigram, gold, *held_out),
        }
        out.setdefault("reward_model_agreement", []).append(agreement)
        for coef in args.kl_coefs:
            for name, rm in (("rlhf_mlp", mlp), ("rlhf_bigram", bigram)):
                log = gh.rl_optimize(ref, rm, gold, steps=args.rl_steps, kl_coef=coef, seed=seed)
                out[name].setdefault(str(coef), []).append({"log": log, **da.endpoint(log)})
        with torch.no_grad():
            chunks = [mlp(pool[i : i + 65536]).numpy() for i in range(0, len(pool), 65536)]
        out["bon_mlp"].append(gh.best_of_n_curve(np.concatenate(chunks), pool_gold, ns))
        log_time(
            f"seed {seed}: RLHF and best-of-n (agreement mlp {agreement['mlp']:.3f}, bigram {agreement['bigram']:.3f})"
        )

    # DPO with more or fewer comparisons, at a fixed beta and step budget
    out["dpo_data_size"] = {}
    for n_pairs in args.data_sizes:
        for seed in range(args.seeds):
            raw = gh.preference_data(ref, gold, n_pairs, seed=1000 * seed + n_pairs)
            log = da.train_direct(
                ref, gold, da.as_pairs(raw), "dpo", args.data_size_beta, args.steps, seed=seed, held_out=held_out
            )
            out["dpo_data_size"].setdefault(str(n_pairs), []).append({"log": log, **da.endpoint(log)})
        log_time(f"DPO with {n_pairs} comparisons")

    out["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out} in {out['seconds']:.0f}s")
    return out


if __name__ == "__main__":
    main()
