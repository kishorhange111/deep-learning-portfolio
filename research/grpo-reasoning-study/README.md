# Where does the RL signal go? A controlled study of GRPO on GSM8K

**Status: code complete and tested on CPU. The GPU sweep has not been run yet,** so there are no results below. Every number this README will show comes from `results/summary.md`, which `analyze.py` writes after the sweep.

## The question
In [an earlier project](https://github.com/kishorhange111/generative-ai-engineering/tree/main/reinforcement-learning/grpo-math-reasoning), 307 steps of TRL's GRPO on Qwen2.5-1.5B-Instruct took format compliance on GSM8K from 55 % to 98 %, but answer accuracy stayed at **67.6 %**. TRL reported that 10–30 % of the 8-sample groups had identical rewards. Identical rewards mean zero advantage, so those groups produce zero gradient. This study takes that null result and asks why. Each intervention below is one hypothesis about where the learning signal is lost.

| Config | Hypothesis | Change |
|---|---|---|
| `baseline` | — | GRPO as in the TRL run: G = 8, format reward 0.5, KL β = 0.04 |
| `dyn` | **H1** zero-variance groups waste the batch | drop groups whose rewards are all equal and refill with new questions ([DAPO](https://arxiv.org/abs/2503.14476) dynamic sampling) |
| `curriculum` | **H2** most questions are too easy or too hard | train only on questions the starting model solves sometimes (0 < pass@8 < 1) |
| `g16` | **H3** 8 samples are too few to see a rare success | G = 16 with half as many questions, so still 64 samples per step |
| `fmt0.1`, `fmt0` | **H4** the cheap format reward drowns out correctness | format reward 0.1 or 0 |
| `dyn+cur` | H1 + H2 together | both |

Each config runs with **3 seeds** on Qwen2.5-0.5B-Instruct. The 0.5B model starts lower than the 1.5B model, so more questions have mixed success, and 21 runs stay affordable.

The study also measures **pass@k for k up to 64**, before and after RL. [Yue et al. (2025)](https://arxiv.org/abs/2504.13837) report that RL with verifiable rewards raises pass@1 without raising pass@k at large k. In other words, it makes the model pick solutions it could already find more reliably, rather than teaching it new ones. This study tests whether that holds here.

## Analysis plan (fixed before running)
- **Primary metric:** greedy-decoding accuracy on all 1,319 GSM8K test questions. Format compliance is reported separately, so a format gain cannot pass as better reasoning.
- **Comparison:** each config vs. `baseline` with a **paired bootstrap over test questions**. Per-question correctness is averaged over seeds, questions are resampled 10,000 times, and the result is a 95 % CI of the accuracy difference plus a two-sided p-value. The seed standard deviation is also reported.
- **Compute fairness:** dynamic sampling generates extra completions to replace uninformative groups. Curves are therefore plotted per *generated sample* as well as per step, and the summary table lists the samples each config used.
- **Mechanism:** the fraction of zero-variance groups is logged every step, split into all-correct and all-wrong groups. This checks whether an intervention changed the thing it was supposed to change.

## Implementation (no TRL)
| File | What it does |
|---|---|
| [`grpo.py`](grpo.py) | rewards, group-relative advantages, token log-probs, clipped surrogate with k3 KL, token- or sequence-level aggregation, Dr. GRPO option, unbiased pass@k |
| [`sampling.py`](sampling.py) | Hugging Face `generate()` and vLLM samplers behind one interface; vLLM weight sync after each step |
| [`train.py`](train.py) | the training loop: dynamic sampling, curriculum filter, gradient accumulation that matches one big batch, per-step logging, evaluation |
| [`curriculum.py`](curriculum.py) | pass@8 of the starting model on all 7,473 training questions |
| [`sweep.py`](sweep.py) | all configs × seeds, resumable after a Colab disconnect |
| [`analyze.py`](analyze.py) | summary table, paired bootstrap, figures |

Design decisions worth defending:
- **fp32 master weights + bf16 autocast.** At lr 1e-6, a weight changes by about 1e-6 of its size per step. That is far below bf16's ~4e-3 relative resolution, so a pure-bf16 policy would silently never move.
- **Samples come from the actual policy.** Qwen's `generation_config` sets top-k 20, top-p 0.8 and a repetition penalty. Sampling with those would make the training data off-policy, while GRPO's importance ratio assumes samples come from π_old. Both samplers use temperature only.
- **Left-padded prompts, right-padded completions, explicit `position_ids`, `logits_to_keep`.** Logits are computed only for completion positions, so the model never materialises (batch × prompt length × 151k vocab) logits.
- **Gradient accumulation is weighted** by tokens (token aggregation) or by sequences (sequence aggregation), so micro-batching gives the same gradient as one large batch.
- **Sampling switches off gradient checkpointing and train mode.** With them on, `generate()` silently drops the KV cache, a bug found in mini-chatgpt's DPO run.

## Tests
`pytest test_grpo.py` runs 19 tests on CPU in about 15 s:
- answer parsing, the reward function, and advantages, including the zero-variance case;
- the loss gradient equals the policy gradient −A·∇log π when the ratio is 1;
- clipping zeroes the gradient outside the trust region;
- token-level vs. sequence-level weighting;
- log-probs from the padded batch match an unpadded forward pass;
- two complete training runs on a tiny random model, followed by the analysis script.

## Run it (Colab A100)
```bash
pip install torch transformers datasets vllm matplotlib
python sweep.py --dry_run        # 21 runs, ~10 A100-hours
python sweep.py                  # base eval -> curriculum pass rates -> all runs (resumable)
python analyze.py                # results/summary.md + figures
```
The vLLM path uses vLLM internals to sync weights, and those internals change between releases (see `sampling.py`). It has not been run in the CPU-only environment where this was written. Before the full sweep, do one short run and check that `correct_reward` in `train_log.jsonl` moves. Falling back to `--backend hf` always works, at about 5× the step time.

## Results
*Pending the GPU sweep.* `analyze.py` will write `results/summary.md` (accuracy ± seed std, Δ vs. baseline with 95 % CI and p-value, zero-variance fraction, samples used), `results/accuracy_vs_baseline.png`, `results/training_dynamics.png` and `results/pass_at_k.png`.

## References
DeepSeekMath / GRPO (Shao et al., 2024) · DeepSeek-R1 (2025) · DAPO (Yu et al., 2025) · Dr. GRPO, *Understanding R1-Zero-Like Training* (Liu et al., 2025) · Yue et al., *Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?* (2025) · Schulman, *Approximating KL Divergence* (2020) · GSM8K (Cobbe et al., 2021).
