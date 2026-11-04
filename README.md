# Deep Learning Portfolio

[![CI](https://github.com/kishorhange111/deep-learning-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/deep-learning-portfolio/actions/workflows/ci.yml)

Two kinds of work:
- **Research, systems and applied-AI projects:** controlled experiments with their statistics fixed before running, on the problems frontier labs work on:
  - training dynamics and mechanistic interpretability;
  - reward modelling and RL;
  - optimizers and scaling laws;
  - distributed training, kernels, inference and quantization;
  - multilingual evals and a production-style agent.
- **Foundations:** fourteen end-to-end deep learning projects across computer vision, NLP and generative models, each trained on cloud GPUs and documented with real results, trade-offs and failure cases.

Every number in a README comes from a file in that project's `results/` folder. Experiments that still need a GPU or API keys are marked **pending**, with exact commands in [RUNBOOK.md](RUNBOOK.md); nothing is filled in before it is run.

## Highlights (measured)
- **Grokking, reverse-engineered:** the generalizing circuit forms *during* the plateau. While test accuracy is still below 50 %:
  - the model's fit to its own training data moves onto 3 Fourier frequencies (excluded loss 0.003 → 11);
  - those 3 frequencies end up carrying 99.2 % of the logit variance.
- **Reward-model overoptimization:** RL against learned reward models drives the proxy up 2.5–4.5 standard deviations while the true reward peaks and halves. *More* preference data made the overshoot worse (final true reward 1.13 → 0.48–0.69), and ensembles did not help: the reward models share one extrapolation error.
- **Muon vs AdamW:** Muon reaches AdamW's loss in **2.2× fewer steps** (1.7× wall-clock; p < 0.001, 3 seeds, tuned learning rates). It still needs 1.9× fewer steps at 3× the training, and the result holds against gradient-clipped AdamW.
- **Mixture of experts:** top-2-of-8 routing beats a dense model with the same active compute by 0.13 bits/byte. Switch-style top-1 loses at this scale, and an ablation traces most of that loss to its gate down-scaling. Without the balancing loss, the router leaves 2–5 of 8 experts per layer unused.
- **Quantization:** GPTQ, written from the paper, removes 64–74 % of round-to-nearest's damage at every bit width. 4-bit weights cost +0.7 % loss, and GPTQ's weights are *further* from the originals than RTN's while its outputs are closer.
- **Distributed training:** DDP and FSDP2 give identical losses in a real 2-process run.
- **Kernels:** a Triton FlashAttention-2 forward and backward pass matches PyTorch, including a float64 gradcheck.

## Research, systems and applied-AI projects
Measured on CPU and checked in CI unless marked pending.

### Research: training dynamics, reward models, optimization
| Project | What it shows | Measured now | Pending |
|---|---|---|---|
| [Grokking, reverse-engineered](interpretability/grokking-modular-arithmetic) | delayed generalization on (a+b) mod 113; the Fourier algorithm read off the weights; progress measures that show the circuit forming before test accuracy moves; weight-decay and data-fraction controls | **full study**: 3 frequencies carry 99 % of the logits; the circuit forms while test accuracy is < 50 %; no grokking in 20k steps without weight decay or with 20 % of the data | — |
| [Reward-model overoptimization](research/reward-model-overoptimization) | Goodhart curves for best-of-n and RL against Bradley-Terry reward models (exact KL, 5 seeds); data size, KL penalty and ensembles as mitigations | **full study**: more data and mean ensembles overshoot *harder*; KL penalty ≈ early stopping with a stable end point | — |
| [AdamW vs Muon vs Lion](research/optimizer-study) | Muon (Newton-Schulz) and Lion from their papers; tuned LRs, 3 seeds, Welch tests, clipped-AdamW and 3×-horizon controls, interleaved timing | **full study**: Muon 2.2× fewer steps, 1.7× wall-clock; Lion behind at small batch | — |
| [Where does the RL signal go? GRPO study](research/grpo-reasoning-study) | GRPO written from scratch; four hypotheses for why GRPO left accuracy flat (dynamic sampling, curriculum, group size, format reward) × 3 seeds; paired bootstrap; pass@k up to 64 | loss gradient = policy gradient; clipping; two end-to-end training runs on a tiny model | ~10 A100-hours sweep |

### Interpretability
| Project | What it shows | Measured now | Pending |
|---|---|---|---|
| [Induction heads](interpretability/induction-heads) | phase change, head scores, causal ablations, K-composition from weights | **CPU demo:** 2-layer induction score 0.78 vs. 1-layer 0.03; ablating induction or previous-token heads removes copying | FineWeb-Edu run (~20 min) |
| [Model diffing with crosscoders](interpretability/crosscoder-model-diffing) | what chat tuning / DPO changed inside Qwen2.5-0.5B | **planted-feature check:** both objectives attribute features to the right model; L1 creates 2–3× too many model-specific latents | ~1.5 A100-hours |

### Systems: training, kernels, inference, efficiency
| Project | What it shows | Measured now | Pending |
|---|---|---|---|
| [Mixture of experts from scratch](systems/mixture-of-experts) | top-k routing (Switch and Mixtral gates), load-balancing and z losses, capacity and token dropping, vs dense at matched compute; gate-scale ablation; what experts specialize in | **full study**: top-2 beats compute-matched dense by 0.13 bits/byte; no balancing loss → 2–5 of 8 experts dead per layer | — |
| [Quantization from scratch: RTN vs GPTQ](systems/quantization-from-scratch) | GPTQ (inverse-Hessian error compensation) and round-to-nearest at 8/4/3/2 bits, per channel and grouped, storage counted in effective bits; layer sensitivity; calibration size | **full study**: GPTQ removes 64–74 % of RTN's damage everywhere; 4-bit GPTQ costs +0.7 % | — |
| [GPT pretraining + scaling laws](systems/gpt-pretraining-scaling-laws) | DDP and FSDP2 under `torchrun`, MFU, a 13-run Chinchilla IsoFLOP sweep fitted two ways | **DDP and FSDP give identical losses** in a real 2-process run; both fits recover a known exponent | ~5 A100-hours, or a multi-GPU node |
| [Triton kernels](systems/triton-kernels) | fused and online softmax; **FlashAttention-2 forward and backward** as an autograd op | outputs and gradients match PyTorch, plus a float64 gradcheck (Triton interpreter) | GPU benchmark vs. PyTorch SDPA |
| [LLM inference from scratch](systems/inference-from-scratch) | KV cache, batching, speculative decoding | greedy speculative decoding is token-identical to the target model; sampled output follows its distribution | GPU benchmarks |

### Evals and applied AI
| Project | What it shows | Measured now | Pending |
|---|---|---|---|
| [Marathi LLM eval](evals/marathi-llm-eval) | English vs. Marathi gap on identical questions, romanized Marathi, language fidelity, LLM judges validated against humans, memorisation probe, for Claude, OpenAI, Gemini, Grok and open models | full pipeline with mock models; real SDKs over a mock transport | API keys |
| [SQL analyst agent](applied-ai/sql-analyst-agent) | tool use, prompt caching, MCP server, four prompt-injection defenses, FastAPI app + Docker | execution-graded eval with scripted models; both backends through the real SDKs; app run locally | API keys |

### What each project is evidence of
| Skill | Projects |
|---|---|
| Designing experiments: baselines, tuned controls, seeds, paired statistics fixed in advance | optimizer study, reward-model overoptimization, GRPO study, mixture of experts |
| Mechanistic interpretability: circuits, progress measures, ablations, dictionary learning | grokking, induction heads, crosscoders |
| RL and alignment for LLMs: reward models, KL control, GRPO | reward-model overoptimization, GRPO study |
| Pretraining at scale: distributed training, scaling laws, optimizers | pretraining + scaling laws, optimizer study |
| Efficient models and serving: kernels, KV cache, speculative decoding, MoE, quantization | Triton kernels, inference, mixture of experts, quantization |
| Evaluating and shipping LLM products | Marathi eval, SQL analyst agent |

## Foundations: computer vision, NLP and generative models
Each project is a single, heavily commented `train.py` written to be read top to bottom (*what problem, which idea, why this design, how it is measured*), plus a README and a `results/` folder with the actual metrics and figures.

### Computer vision
| Project | Key techniques | Result |
|---|---|---|
| [Vision Transformer from scratch](computer-vision/vision-transformer-cifar100) | patch embeddings, multi-head self-attention, no convolutions | **48.8 %** top-1 · 77.0 % top-5 on CIFAR-100 |
| [ShiftViT — a ViT without attention](computer-vision/shiftvit-attention-free-transformer) | zero-parameter shift token mixing, hierarchical stages, stochastic depth | **77.9 %** on CIFAR-10 (reference 76.4 %) |
| [U-Net semantic segmentation](computer-vision/unet-pet-segmentation) | encoder–decoder, separable convs, residual skips | **83.0 %** pixel acc · **0.60** mIoU |
| [Image captioning](computer-vision/image-captioning-cnn-transformer) | frozen EfficientNet + Transformer decoder, cross-attention, BLEU | **BLEU-1 0.59** · BLEU-4 0.18 |
| [Siamese network — contrastive loss](computer-vision/siamese-network-contrastive-loss) | metric learning, shared weights | **97.3 %** pair accuracy |
| [Siamese network — triplet loss](computer-vision/siamese-network-triplet-loss) | ResNet50 transfer learning, custom training step | triplet accuracy **63 % → 83.5 %** |

### Natural language processing
| Project | Key techniques | Result |
|---|---|---|
| [BERT sentiment analysis](nlp/bert-sentiment-analysis) | Transformer fine-tuning with a measured TF-IDF baseline | **92.4 %** accuracy (baseline 90.4 %) |
| [BERT optimisation: distillation + ONNX + int8](nlp/bert-distillation-onnx-quantization) | knowledge distillation, ONNX Runtime, dynamic quantisation, CPU latency benchmark | **4.8× faster**, **85 % smaller**, −0.6 pts accuracy |
| [GPT-2: LoRA vs. full fine-tuning](nlp/gpt2-lora-vs-full-finetuning) | parameter-efficient fine-tuning, memory / speed / perplexity comparison | LoRA: **0.73 %** of params, **−69 % GPU memory** |
| [Seq2seq LSTM arithmetic](nlp/seq2seq-lstm-arithmetic) | encoder–decoder RNN, input reversal | **98.1 %** exact-match |

### Generative models
| Project | Key techniques | Result |
|---|---|---|
| [Neural style transfer (Gatys)](generative-models/neural-style-transfer) | VGG-19 Gram matrices, L-BFGS pixel optimisation, PyTorch | 4 public-domain styles, ~25 s / image |
| [AdaIN real-time style transfer](generative-models/adain-style-transfer) | adaptive instance normalisation, learned decoder, PyTorch | **any unseen style in 31 ms** |
| [Music Transformer](generative-models/music-transformer) | performance-event tokenizer, GPT from scratch, nucleus sampling, PyTorch | test perplexity **16.9** / 388 tokens |
| [Molecule generation with a Transformer](generative-models/molecule-generation-transformer) | SMILES GPT from scratch, RDKit validity / uniqueness / novelty, PyTorch | **73 %** valid · **~100 %** unique & novel |

## Layout
```
deep-learning-portfolio/
├── research/  systems/  interpretability/  evals/  applied-ai/
│                                 # research, systems and applied-AI projects (see the table above)
├── computer-vision/<project>/    # train.py · README.md · results/
├── nlp/<project>/                # train.py · README.md · results/
├── generative-models/<project>/  # train.py · README.md · results/
├── src/dl_common/                # shared helpers: seeding, GPU time budgets, metrics + plots (utils.py),
│                                 #   caption parsing + BLEU, triplet metrics, distillation loss, latency benchmark,
│                                 #   one plotting style (plotting.py), tiny offline Hugging Face models for tests,
│                                 #   a byte-level corpus from the Python stdlib + a small GPT (byte_corpus.py, bytelm.py)
├── tests/                        # CPU-only pytest suite: no GPU, no downloads, no API keys
├── .github/workflows/            # CI: ruff lint + format check, byte-compile every script, tests;
│                                 #   plus a manual real-model eval for the SQL agent
├── RUNBOOK.md                    # the pending GPU / API runs, in order, with commands and expected cost
├── Makefile                      # make install / install-dev / lint / format / test / cpu-studies
├── pyproject.toml                # package metadata (src/dl_common), ruff + pytest config
├── requirements.txt              # runtime stack for the training scripts
└── requirements-dev.txt          # lint + test tooling
```

## Engineering practices
- **Statistics decided before running:** seeds, paired bootstrap confidence intervals over test items, and compute-matched comparisons are fixed before results exist.
- **Fair comparisons:** every method gets its own learning-rate sweep with the optimum inside the grid, baselines get the standard tricks (e.g. gradient clipping), and speed is timed in interleaved benchmarks, not read from runs that shared a machine.
- **Reproducible:** fixed seeds and splits; every number in a README comes from that project's `results/`.
- **Cost-controlled:** wall-clock budgets on every run; resumable sweeps; LLM responses cached so reruns are free.
- **Verified without hardware:**
  - Triton kernels run through Triton's CPU interpreter;
  - DDP and FSDP are compared in a real 2-process CPU run;
  - LLM clients are checked through the real SDKs over a mock HTTP transport;
  - methods are checked on synthetic data with known answers (scaling exponents, planted crosscoder features, prompt-injection victims).
- **Honest evaluation:** held-out test sets, baselines, failure cases shown, and pending results labelled as pending.

## Running a project
```bash
pip install -r requirements.txt
python computer-vision/unet-pet-segmentation/train.py
QUICK=1 RESULTS_DIR=/tmp/smoke python nlp/bert-sentiment-analysis/train.py   # quick smoke test
```
Trained on Google Colab GPUs (NVIDIA A100, L4, T4) with Keras 3 / TensorFlow and PyTorch. Each project README has a **Run on Colab** section with the exact GPU type and commands used. For the research, systems and applied-AI projects, see [RUNBOOK.md](RUNBOOK.md).

## Development
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build is enough for the tests
make install-dev    # = pip install -r requirements-dev.txt -e .  (installs the shared helpers package)
make check          # ruff lint + format check, then pytest (~270 tests, CPU only, no downloads)
make cpu-studies    # re-run the five CPU research studies end to end (several CPU-hours)
```
CI (GitHub Actions) runs the same checks, plus a byte-compile of every `train.py`, on every push and pull request. A second workflow ([agent-eval.yml](.github/workflows/agent-eval.yml)) runs the SQL agent eval against real model APIs on demand, using repository secrets. Optional: `pip install pre-commit && pre-commit install` runs ruff before each commit.

## Credits
Several projects re-implement or extend [Keras code examples](https://keras.io/examples/) (Apache-2.0) — some rewritten in PyTorch, all extended with additional evaluation, baselines and documentation. Original authors are credited in each project.

Related: [**generative-ai-engineering**](https://github.com/kishorhange111/generative-ai-engineering) (LLM fine-tuning, RAG, agents, multimodal) · [**mini-chatgpt**](https://github.com/kishorhange111/mini-chatgpt) (base LLM → SFT → DPO → benchmarks).
