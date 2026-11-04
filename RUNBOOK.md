# Runbook: the pending GPU and API runs

The research, systems and applied-AI projects are written and tested on CPU. Five of them are CPU studies whose results are already measured and committed (grokking, reward-model overoptimization, the optimizer study, mixture of experts, quantization; section 8 has the commands to reproduce them). The others need a GPU or API keys for their headline numbers. This is the order to run them in, cheapest and most informative first. Times are estimates for one A100 unless stated. API costs are rough estimates, not measured: start with the `--n` / `--limit` subsets shown and look at the real per-request cost before running everything.

After each run:
1. Copy the generated summary (`results/.../summary.md` or similar) into that project's README **Results** section.
2. Commit the small result files (JSON, Markdown, PNG). Data shards, activations and checkpoints are git-ignored.

## 0. Colab / GPU machine setup (once per session)
```bash
git clone https://github.com/kishorhange111/deep-learning-portfolio && cd deep-learning-portfolio
pip install -r requirements.txt vllm tiktoken datasets indic-transliteration anthropic openai fastapi uvicorn "mcp>=2"
pytest -q        # ~2 min: confirms the environment before spending GPU time
```
Colab runtimes are reclaimed without warning. Every long script here resumes when re-run (`sweep.py`, `scaling.py run`, the eval caches), so re-run the same command after a disconnect. Keep `results/` on Google Drive or commit it often.

## 1. Shared data: FineWeb-Edu tokens (CPU, ~30–60 min)
Used by the pretraining, induction-head and inference projects.
```bash
cd systems/gpt-pretraining-scaling-laws && python prepare_data.py --train_tokens 2e9 && cd -
```

## 2. Quick GPU results (any GPU, ~40 min total)
| Run | Command | Output |
|---|---|---|
| Triton kernel benchmark | `python systems/triton-kernels/bench.py` | `results/benchmark.md`, softmax and attention plots |
| KV cache and batching | `python systems/inference-from-scratch/bench.py kv batch` | `results/kv.md`, `results/batch.md` |
| Induction heads on FineWeb-Edu | `python interpretability/induction-heads/train.py` | phase change, head scores, ablations, composition |

## 3. GRPO study (~10 A100-hours, the flagship research result)
```bash
cd research/grpo-reasoning-study
python train.py --run_name smoke --steps 5 --eval_n 50   # check the vLLM weight sync: correct_reward should move
python sweep.py --dry_run && python sweep.py && python analyze.py
```
If the vLLM path fails on your vLLM version, add `--backend hf` to the sweep (`python sweep.py -- --backend hf`). It is about 5× slower but always works.

## 4. Crosscoder model diffing (~1.5 A100-hours)
```bash
cd interpretability/crosscoder-model-diffing
python baseline_diff.py
python collect.py --tokens 8e6 && python train.py --data_dir data/base_vs_instruct
python analyze.py --run_dir results/base_vs_instruct_batchtopk
# then mini-chatgpt's SFT vs DPO checkpoints:
python collect.py --model_a /path/to/sft --model_b /path/to/dpo --out_dir data/sft_vs_dpo
python train.py --data_dir data/sft_vs_dpo && python analyze.py --run_dir results/sft_vs_dpo_batchtopk
```
Then read `features.md` and name the top B-only latents. That interpretation is the result.

## 5. Pretraining and scaling laws (~5 A100-hours, or ~40 min on 8 GPUs)
```bash
cd systems/gpt-pretraining-scaling-laws
python scaling.py lr-sweep --preset 25M      # check the learning-rate table first
python scaling.py run --compile && python scaling.py fit
# on a multi-GPU node:
python scaling.py bench --nproc 8 --preset 302M
# draft and target checkpoints for speculative decoding:
python train.py --preset 302M --tokens_per_param 20 --run_name target --save_checkpoint
python train.py --preset 11M  --tokens_per_param 20 --run_name draft  --save_checkpoint
cd ../inference-from-scratch && python bench.py spec \
  --target_ckpt ../gpt-pretraining-scaling-laws/results/runs/target/ckpt.pt \
  --draft_ckpt  ../gpt-pretraining-scaling-laws/results/runs/draft/ckpt.pt
```

## 6. API-based evals
Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` and `XAI_API_KEY` as needed, using each provider's current model IDs.

**Marathi eval**: start small, then remove `--n`:
```bash
cd evals/marathi-llm-eval
python run_eval.py mcq --models anthropic:claude-opus-5-5 openai:<m> gemini:<m> xai:<m> --reverse --n 300
python run_eval.py generate --models ... && python run_eval.py judge --judges anthropic:claude-opus-5-5 openai:<m>
python run_eval.py contamination --models ...
python label.py --n 100        # needs a fluent Marathi reader, ~1 hour
python report.py
```
Rough cost: $20–50 per frontier model for the full MCQ track with reversed options; a few dollars each for generation and judging.

**SQL analyst agent**: or use the manual GitHub workflow *Agent eval (real models)*:
```bash
cd applied-ai/sql-analyst-agent
python eval_agent.py --models anthropic:claude-opus-5-5 openai:<m> gemini:<m> xai:<m> --limit 10   # first pass
python eval_agent.py --models ... --benign_defenses none spotlight                                 # everything
```
Rough cost: $5–15 per frontier model for all 35 tasks × 4 defenses.

## 7. Optional: deploy the agent's web app
```bash
cd applied-ai/sql-analyst-agent && docker build -t shopco-agent .
```
Then deploy to a container host with `ANTHROPIC_API_KEY` and `APP_TOKEN` as secrets. Put a spend limit on the API key first.

## 8. Reproduce the CPU studies (no GPU, no downloads)
These are already run; their results are committed. To regenerate them (approximate times on a 4-core cloud VM; the original runs shared its cores):
| Study | Command | Time |
|---|---|---|
| Grokking | `cd interpretability/grokking-modular-arithmetic && python train.py && python analyze.py` | ~3 hours on one core (runs are independent: `--only` splits them across cores) |
| Reward-model overoptimization | `cd research/reward-model-overoptimization && python train.py && python train.py --w_steady 0.5 --no_extras --out results/study_mixed.json && python analyze.py` | ~20 min on one core |
| Optimizer study | `cd research/optimizer-study && python train.py && python train.py --bench && python analyze.py` | ~1 hour on 2 cores |
| Mixture of experts | `cd systems/mixture-of-experts && python train.py && python train.py --bench && python analyze.py` | ~1.5 hours on 2 cores |
| Quantization | `cd systems/quantization-from-scratch && python train.py && python analyze.py` | ~15 min on one core |

Or all of them: `make cpu-studies`. The byte-level studies read the Python standard library's source as their corpus, so their exact numbers depend on the Python version (recorded in each result file with a SHA-256 of the corpus).
