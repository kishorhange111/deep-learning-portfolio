"""
Run the whole study: every config x every seed, sequentially, skipping runs that already finished.

Colab runtimes get reclaimed. Re-running `python sweep.py` resumes where it stopped, because a run
counts as finished once its eval_final.json exists.

All configs are compute-matched at 64 sampled completions per step (prompts_per_step x group_size).
Dynamic sampling is the exception: it generates extra groups to replace uninformative ones, so
analyze.py also compares configs per generated sample.

    python sweep.py --dry_run            # list the runs and the estimated GPU time
    python sweep.py                      # base eval + curriculum + all runs
    python sweep.py --only baseline dyn  # a subset
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PASS_RATES = os.path.join(HERE, "results", "pass_rates.json")

# name -> extra train.py arguments (everything else uses train.py's defaults)
CONFIGS = {
    "baseline": [],  # GRPO as in the TRL run
    "dyn": ["--dynamic_sampling"],  # H1
    "curriculum": ["--curriculum", PASS_RATES],  # H2
    "g16": ["--group_size", "16", "--prompts_per_step", "4"],  # H3 (same 64 samples/step)
    "fmt0.1": ["--format_weight", "0.1"],  # H4
    "fmt0": ["--format_weight", "0.0"],  # H4
    "dyn+cur": ["--dynamic_sampling", "--curriculum", PASS_RATES],  # interventions combined
}
SEEDS = [0, 1, 2]
MIN_PER_RUN = {"default": 25, "dyn": 40, "dyn+cur": 35}  # rough A100 estimates incl. evaluation


def run(cmd, dry_run):
    print("$", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--passk_n", type=int, default=64)
    p.add_argument("extra", nargs=argparse.REMAINDER, help="passed through to train.py after '--'")
    args = p.parse_args()
    extra = [a for a in args.extra if a != "--"]
    runs_dir = os.path.join(HERE, "results", "runs")
    py = sys.executable

    names = args.only or list(CONFIGS)
    total = sum(MIN_PER_RUN.get(n, MIN_PER_RUN["default"]) for n in names) * len(args.seeds)
    print(
        f"{len(names)} configs x {len(args.seeds)} seeds = {len(names) * len(args.seeds)} runs, "
        f"~{total / 60:.1f} A100-hours (+ ~20 min for the base eval and curriculum)\n"
    )

    # 0. the starting model, evaluated once (greedy decoding is deterministic): accuracy + pass@k
    if not os.path.exists(os.path.join(runs_dir, "base", "seed0", "eval_final.json")):
        run(
            [
                py,
                os.path.join(HERE, "train.py"),
                "--run_name",
                "base",
                "--steps",
                "0",
                "--passk_n",
                str(args.passk_n),
                *extra,
            ],
            args.dry_run,
        )
    # 1. pass rates of the starting model on the training set, for the curriculum configs
    if any("cur" in n for n in names) and not os.path.exists(PASS_RATES):
        run([py, os.path.join(HERE, "curriculum.py")], args.dry_run)
    # 2. the grid
    for name in names:
        for seed in args.seeds:
            if os.path.exists(os.path.join(runs_dir, name, f"seed{seed}", "eval_final.json")):
                print(f"[skip] {name} seed {seed} already finished")
                continue
            passk = ["--passk_n", str(args.passk_n)] if seed == 0 else []
            run(
                [
                    py,
                    os.path.join(HERE, "train.py"),
                    "--run_name",
                    name,
                    "--seed",
                    str(seed),
                    *CONFIGS[name],
                    *passk,
                    *extra,
                ],
                args.dry_run,
            )
    print("\nDone. Next: python analyze.py")


if __name__ == "__main__":
    main()
