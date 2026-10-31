# LLM inference from scratch: KV cache, batching and speculative decoding

**Status: code complete and tested on CPU. GPU benchmarks have not been run yet** (`results/` will hold them). The models are the GPTs trained in [`../gpt-pretraining-scaling-laws`](../gpt-pretraining-scaling-laws), whose checkpoints load directly.

## What's implemented
| | File | Idea |
|---|---|---|
| **KV cache** | [`kv_model.py`](kv_model.py) | Pre-allocated per-layer keys and values. Prefill processes the prompt in one parallel pass, then each new token is a single-position forward that attends to the cache. Rolling back is a pointer move (`truncate`). |
| **Decoding** | [`decoding.py`](decoding.py) | Naive (re-run everything), cached, temperature / top-k / top-p sampling |
| **Speculative decoding** | [`decoding.py`](decoding.py) | A small draft model proposes k tokens. The target scores all k in one forward pass and accepts each with probability min(1, p/q). At the first rejection it resamples from norm(max(0, p − q)). If all k are accepted, it adds a free bonus token. |
| **Benchmarks** | [`bench.py`](bench.py) | per-token latency vs. position, throughput vs. batch size, speculative speed-up vs. k at temperature 0 and 1 |

## Why decoding is memory-bound
One decoding step of an 85M-parameter model in bf16 reads ~250 MB of weights (plus the cache) from HBM to do ~0.25 GFLOP of math, far below what the GPU can compute in that time. The step time is the time to read the weights. Two ways to get more tokens out of each read:
- **Batching:** many sequences share one read of the weights, so throughput rises almost linearly until the step becomes compute-bound.
- **Speculative decoding:** one sequence gets several tokens verified per read. This lowers single-request latency, which batching cannot do.

## Correctness (`pytest tests/test_inference.py`, 13 tests, ~17 s on CPU)
- **Exact KV cache:** incremental decoding in mixed chunk sizes gives the same logits as one full forward, and rolling back and continuing matches a fresh run.
- **Checkpoint compatibility:** a pretraining-project `GPT` saved to disk reloads here with identical logits.
- **Greedy speculative decoding is token-for-token identical** to the target's own greedy output, for k = 1, 3 and 6, even with a random draft that is almost always rejected.
  - A model drafting for itself accepts 100 %, giving k+1 tokens per target call.
- **Sampling is distribution-preserving:**
  - the acceptance rule's output distribution equals p analytically;
  - a 20,000-sample Monte-Carlo run of `verify_draft` follows p rather than q (TV distance < 0.015);
  - full speculative sampling's next token follows the target's softmax (TV < 0.04).

## Run the benchmarks
```bash
python bench.py kv batch                         # random weights: speed only depends on the shapes
# speculative decoding needs a trained target and draft (same tokenizer and data):
cd ../gpt-pretraining-scaling-laws
python train.py --preset 302M --tokens_per_param 20 --run_name target --save_checkpoint
python train.py --preset 11M  --tokens_per_param 20 --run_name draft  --save_checkpoint
cd ../inference-from-scratch
python bench.py spec --target_ckpt ../gpt-pretraining-scaling-laws/results/runs/target/ckpt.pt \
                     --draft_ckpt  ../gpt-pretraining-scaling-laws/results/runs/draft/ckpt.pt
```
`results/spec.md` reports, for each k and temperature:
- the measured acceptance rate α;
- tokens per target call next to the theoretical (1 − α^(k+1))/(1 − α);
- the real wall-clock speed-up, which includes the draft's own cost. The speed-up is the number that matters.

**What to expect:** the speed-up first rises with k and then falls once rejected draft work dominates. It is lower at temperature 1 than at greedy, because sampled tokens are harder to predict.

## CPU smoke run (not the GPU result)
On this repository's CPU-only development machine, with an 85M model and random weights, the KV cache was **5.9× faster per token at position 64 and 40× faster at position 1,000**. Naive cost grows linearly with position while cached cost stays flat. The GPU numbers will differ and go in `results/kv.md`.

## Production equivalents
vLLM's PagedAttention (KV cache in fixed-size blocks, so memory isn't fragmented and many requests can be batched continuously) · Hugging Face assisted generation and vLLM speculative decoding · Medusa / EAGLE (draft heads instead of a separate draft model).

## References
Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding* (2023) · Chen et al., *Accelerating LLM Decoding with Speculative Sampling* (2023) · Pope et al., *Efficiently Scaling Transformer Inference* (2022) · Kwon et al., *PagedAttention / vLLM* (2023).
