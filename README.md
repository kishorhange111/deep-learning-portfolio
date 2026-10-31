# Deep Learning Portfolio

[![CI](https://github.com/kishorhange111/deep-learning-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/deep-learning-portfolio/actions/workflows/ci.yml)

Two kinds of work:
- **Research, systems and applied-AI projects:** controlled experiments with their statistics decided before running, multi-GPU pretraining and scaling laws, GPU kernels, LLM inference, mechanistic interpretability, multilingual evals and a production-style agent.
- **Foundations:** fourteen end-to-end deep learning projects across computer vision, NLP and generative models, each trained on cloud GPUs and documented with real results, trade-offs and failure cases.

Every number in a README comes from a file in that project's `results/` folder. Experiments that still need a GPU or API keys are marked **pending**, with exact commands in [RUNBOOK.md](RUNBOOK.md); nothing is filled in before it is run.

## Research, systems and applied-AI projects
| Project | What it shows | Verified now (CPU, in CI) | Pending |
|---|---|---|---|
| [Where does the RL signal go? GRPO study](research/grpo-reasoning-study) | GRPO written from scratch; four hypotheses for why GRPO left accuracy flat (dynamic sampling, curriculum, group size, format reward) × 3 seeds; paired bootstrap; pass@k up to 64 | loss gradient = policy gradient; clipping; two end-to-end training runs on a tiny model | ~10 A100-hours sweep |
| [GPT pretraining + scaling laws](systems/gpt-pretraining-scaling-laws) | DDP and FSDP2 under `torchrun`, MFU, a 13-run Chinchilla IsoFLOP sweep fitted two ways | **DDP and FSDP give identical losses** in a real 2-process run; both fits recover a known exponent | ~5 A100-hours, or a multi-GPU node |
| [Triton kernels](systems/triton-kernels) | fused and online softmax; **FlashAttention-2 forward and backward** as an autograd op | outputs and gradients match PyTorch, plus a float64 gradcheck (Triton interpreter) | GPU benchmark vs. PyTorch SDPA |
| [LLM inference from scratch](systems/inference-from-scratch) | KV cache, batching, speculative decoding | greedy speculative decoding is token-identical to the target model; sampled output follows its distribution | GPU benchmarks |
| [Induction heads](interpretability/induction-heads) | phase change, head scores, causal ablations, K-composition from weights | **measured CPU demo:** 2-layer induction score 0.78 vs. 1-layer 0.03; ablating induction or previous-token heads removes copying | FineWeb-Edu run (~20 min) |
| [Model diffing with crosscoders](interpretability/crosscoder-model-diffing) | what chat tuning / DPO changed inside Qwen2.5-0.5B | **measured planted-feature check:** both objectives attribute features to the right model; L1 creates 2–3× too many model-specific latents | ~1.5 A100-hours |
| [Marathi LLM eval](evals/marathi-llm-eval) | English vs. Marathi gap on identical questions, romanized Marathi, language fidelity, LLM judges validated against humans, memorisation probe, for Claude, OpenAI, Gemini, Grok and open models | full pipeline with mock models; real SDKs over a mock transport | API keys |
| [SQL analyst agent](applied-ai/sql-analyst-agent) | tool use, prompt caching, MCP server, four prompt-injection defenses, FastAPI app + Docker | execution-graded eval with scripted models; both backends through the real SDKs; app run locally | API keys |

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
research/  systems/  interpretability/  evals/  applied-ai/    the research, systems and applied-AI projects
computer-vision/  nlp/  generative-models/                    foundations: <project>/ train.py · README.md · results/
common/      shared helpers: seeding, time budgets, metrics, plotting style, tiny Hugging Face models for tests
tests/       CPU-only pytest suite: no GPU, no downloads, no API keys
RUNBOOK.md   the pending GPU / API runs, in order, with commands and expected cost
```

## Engineering practices
- **Statistics decided before running:** seeds, paired bootstrap confidence intervals over test items, and compute-matched comparisons are fixed before results exist.
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
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
ruff check .        # lint
pytest              # ~210 tests, CPU only, ~2 minutes
```
CI (GitHub Actions) runs `ruff check`, byte-compiles every `train.py` and runs the test suite on every push. A second workflow ([agent-eval.yml](.github/workflows/agent-eval.yml)) runs the SQL agent eval against real model APIs on demand, using repository secrets.

## Credits
Several projects re-implement or extend [Keras code examples](https://keras.io/examples/) (Apache-2.0) — some rewritten in PyTorch, all extended with additional evaluation, baselines and documentation. Original authors are credited in each project.

Related: [**generative-ai-engineering**](https://github.com/kishorhange111/generative-ai-engineering) (LLM fine-tuning, RAG, agents, multimodal) · [**mini-chatgpt**](https://github.com/kishorhange111/mini-chatgpt) (base LLM → SFT → DPO → benchmarks).
