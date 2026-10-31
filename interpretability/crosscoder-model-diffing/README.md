# What did fine-tuning change? Model diffing with sparse crosscoders

The question: what does chat fine-tuning (or DPO) change inside a model? This project uses Anthropic's **sparse crosscoder** ([Lindsey et al., 2024](https://transformer-circuits.pub/2024/crosscoders/index.html)), a sparse autoencoder that reads the same layer of *two* models at once. It learns one dictionary of latents with a separate decoder direction per model. A latent's **relative decoder norm** says whether it is shared, exists only in the base model, or exists only in the fine-tuned model. Its top-activating contexts suggest what it represents.

**Model pairs:**
- **Default:** `Qwen/Qwen2.5-0.5B` (base) vs. `Qwen/Qwen2.5-0.5B-Instruct`, which anyone can run.
- **Intended:** [mini-chatgpt](https://github.com/kishorhange111/mini-chatgpt)'s own **SFT vs. SFT+DPO** checkpoints (same base, same tokenizer). The question there is what DPO added that SFT didn't have.

**Status:** the method is **validated on synthetic data with known answers (run on CPU, results below)**. The real-model runs need a GPU and have not been done yet.

## Pipeline
| Step | File | What it does |
|---|---|---|
| 0 | [`baseline_diff.py`](baseline_diff.py) | Where do the models differ? It reports relative weight change per layer and matrix type, and residual-stream cosine per layer. |
| 1 | [`collect.py`](collect.py) | Runs both models on identical tokens (half UltraFeedback chat, half FineWeb-Edu web text) and stores paired middle-layer residual activations. Position 0 (the attention sink) is dropped and each model is normalised to mean norm √d. |
| 2 | [`train.py`](train.py) | Trains the crosscoder, with either Anthropic's L1-weighted-by-decoder-norm loss or BatchTopK (see below). |
| 3 | [`analyze.py`](analyze.py) | Classifies latents by relative norm and writes `features.md` with the top contexts of the most frequent B-only and A-only latents, plus histograms. |

The model ([`crosscoder.py`](crosscoder.py)):
```
f   = sparsify( W_enc^A x^A + W_enc^B x^B + b_enc )          one latent vector for both models
x̂^m = W_dec^m f + b_dec^m                                     separate decoder per model
r_i = ||d_i^B|| / (||d_i^A|| + ||d_i^B||)                    0 = A-only · 0.5 = shared · 1 = B-only
```

## Validating the method first (measured, CPU)
Before trusting what a crosscoder says about real models, check it on data where the answer is known. `python train.py --planted` builds paired 32-dim "activations" from three kinds of features:
- **24 shared features:** same coefficients in both models, but different directions, because each "model" has its own basis;
- **6 A-only features**;
- **6 B-only features**.

It then trains a 128-latent crosscoder (about 15 s on CPU). Results are in [`results/planted_batchtopk`](results/planted_batchtopk) and [`results/planted_l1`](results/planted_l1).

| Planted feature | Best-matching latent: median cosine | median relative norm (BatchTopK) | (L1) |
|---|---|---|---|
| B-only (should be ≈ 1) | 0.98 / 1.00 | **0.89** (range 0.77–1.00) | **1.00** |
| A-only (should be ≈ 0) | 1.00 / 1.00 | **0.00** (0.00–0.30) | **0.00** |
| shared (should be ≈ 0.5) | 0.98 / 1.00 | **0.51** (0.39–0.63) | **0.62** (0.42–0.89) |

| | BatchTopK (k = 4) | L1 |
|---|---|---|
| Variance explained (A / B) | 1.000 / 1.000 | 0.977 / 0.976 |
| Live latents classified A-only / B-only (truth: 6 / 6) | 4 / 3 | **20 / 15** |

**What this shows:** both objectives find every planted direction (cosine ≈ 1) and attribute it to the right model, with different failure modes.
- **L1** tags the planted features perfectly, but produces **more than twice as many "model-specific" latents as exist**: duplicated or shrunk copies of real features. It also spreads shared features up to r = 0.89.
- **BatchTopK** reconstructs exactly and keeps shared features tight around 0.5, but leaves some genuine B-only features at r ≈ 0.77, below the 0.9 cut-off.

This matches Minder et al. (2025), who show L1 crosscoders can manufacture fine-tune-specific latents. It is why BatchTopK is the default here, and why a latent's class should be checked rather than read off one threshold.

## Run on real models (GPU)
```bash
pip install torch transformers datasets matplotlib
python baseline_diff.py                                   # minutes
python collect.py --tokens 8e6                            # ~10 min of forward passes, ~29 GB on disk
python train.py --data_dir data/base_vs_instruct          # BatchTopK, 16k latents, ~1 h on an A100
python analyze.py --run_dir results/base_vs_instruct_batchtopk
# mini-chatgpt SFT vs DPO:
python collect.py --model_a /path/to/sft --model_b /path/to/dpo --out_dir data/sft_vs_dpo
```
**Outputs per run:**
- `metrics.json`: variance explained per model, L0, live latents, class counts, decoder cosine of shared latents;
- `relative_norms.png`;
- `features.md`: top contexts per latent, with the activating token in «guillemets».

## Interpreting the output responsibly
- **Contexts suggest, they don't prove.** A list of top contexts is a hypothesis about what a latent means. Confirm it causally: steer with the latent's decoder direction or ablate it, then check the behaviour changes.
- **Treat r as a continuum.** The planted check shows both objectives misclassify near the thresholds. Inspect latents around r ≈ 0.7–0.9 by hand.
- **Shared latents should keep their direction.** Both real models come from the same base, so their decoder cosine should be high. A low cosine means the latent changed what it does.
- **This is one layer.** Changes in other layers aren't seen. `baseline_diff.py` shows where to look next.

## Tests (`pytest tests/test_crosscoder.py`, 7 tests, ~30 s on CPU)
- **BatchTopK:** activates exactly k latents per token across the batch.
- **L1:** the penalty equals the decoder-norm-weighted sum.
- **Relative norms:** computed correctly on hand-set decoders.
- **Planted recovery:** works for both objectives.
- **Pipeline:** the full collect → train → analyze run works on two tiny random Qwen2 models.
- **Baseline diff:** the weight-diff baseline localises a planted 10 % change to the right layer and matrix.

## References
Lindsey et al., *Sparse Crosscoders for Cross-Layer Features and Model Diffing* (Anthropic, 2024) · Minder et al., *Robustly identifying concepts introduced during chat fine-tuning using crosscoders* (2025) · Bussmann et al., *BatchTopK Sparse Autoencoders* (2024) · Bricken et al., *Towards Monosemanticity* (2023).
