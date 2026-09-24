# Deep Learning Portfolio

Fourteen end-to-end deep learning projects across **computer vision, NLP and generative models** — each trained on cloud GPUs, evaluated on held-out data and documented with its real results, trade-offs and failure cases.

Every project is a single, heavily commented `train.py` written to be read top to bottom — *what problem, which idea, why this design, how it is measured* — plus a README and a `results/` folder with the actual metrics and figures.

## Computer vision
| Project | Key techniques | Result |
|---|---|---|
| [Vision Transformer from scratch](computer-vision/vision-transformer-cifar100) | patch embeddings, multi-head self-attention, no convolutions | **48.8 %** top-1 · 77.0 % top-5 on CIFAR-100 |
| [ShiftViT — a ViT without attention](computer-vision/shiftvit-attention-free-transformer) | zero-parameter shift token mixing, hierarchical stages, stochastic depth | **77.9 %** on CIFAR-10 (reference 76.4 %) |
| [U-Net semantic segmentation](computer-vision/unet-pet-segmentation) | encoder–decoder, separable convs, residual skips | **83.0 %** pixel acc · **0.60** mIoU |
| [Image captioning](computer-vision/image-captioning-cnn-transformer) | frozen EfficientNet + Transformer decoder, cross-attention, BLEU | **BLEU-1 0.59** · BLEU-4 0.18 |
| [Siamese network — contrastive loss](computer-vision/siamese-network-contrastive-loss) | metric learning, shared weights | **97.3 %** pair accuracy |
| [Siamese network — triplet loss](computer-vision/siamese-network-triplet-loss) | ResNet50 transfer learning, custom training step | triplet accuracy **63 % → 83.5 %** |

## Natural language processing
| Project | Key techniques | Result |
|---|---|---|
| [BERT sentiment analysis](nlp/bert-sentiment-analysis) | Transformer fine-tuning with a measured TF-IDF baseline | **92.4 %** accuracy (baseline 90.4 %) |
| [BERT optimisation: distillation + ONNX + int8](nlp/bert-distillation-onnx-quantization) | knowledge distillation, ONNX Runtime, dynamic quantisation, CPU latency benchmark | **4.8× faster**, **85 % smaller**, −0.6 pts accuracy |
| [GPT-2: LoRA vs. full fine-tuning](nlp/gpt2-lora-vs-full-finetuning) | parameter-efficient fine-tuning, memory / speed / perplexity comparison | LoRA: **0.73 %** of params, **−69 % GPU memory** |
| [Seq2seq LSTM arithmetic](nlp/seq2seq-lstm-arithmetic) | encoder–decoder RNN, input reversal | **98.1 %** exact-match |

## Generative models
| Project | Key techniques | Result |
|---|---|---|
| [Neural style transfer (Gatys)](generative-models/neural-style-transfer) | VGG-19 Gram matrices, L-BFGS pixel optimisation, PyTorch | 4 public-domain styles, ~25 s / image |
| [AdaIN real-time style transfer](generative-models/adain-style-transfer) | adaptive instance normalisation, learned decoder, PyTorch | **any unseen style in 31 ms** |
| [Music Transformer](generative-models/music-transformer) | performance-event tokenizer, GPT from scratch, nucleus sampling, PyTorch | test perplexity **16.9** / 388 tokens |
| [Molecule generation with a Transformer](generative-models/molecule-generation-transformer) | SMILES GPT from scratch, RDKit validity / uniqueness / novelty, PyTorch | **73 %** valid · **~100 %** unique & novel |

## Layout
```
computer-vision/<project>/   train.py · README.md · results/
nlp/<project>/               train.py · README.md · results/
generative-models/<project>/ train.py · README.md · results/
common/utils.py              shared helpers: seeding, GPU time budgets, metrics + plots
```

## Engineering practices
- **Reproducible:** fixed seeds and splits; every number in a README comes from that project's `results/metrics.json`.
- **Cost-controlled:** every run has a wall-clock budget, so cloud-GPU spend is capped.
- **Smoke-tested:** `QUICK=1` runs each full pipeline on tiny data before spending GPU hours.
- **Honest evaluation:** held-out test sets, baselines where they matter, before/after comparisons, and failure cases shown — not hidden.

## Running a project
```bash
pip install -r requirements.txt
python computer-vision/unet-pet-segmentation/train.py
QUICK=1 RESULTS_DIR=/tmp/smoke python nlp/bert-sentiment-analysis/train.py   # quick smoke test
```
Trained on Google Colab GPUs (NVIDIA A100, L4, T4) with Keras 3 / TensorFlow and PyTorch.

## Credits
Several projects re-implement or extend [Keras code examples](https://keras.io/examples/) (Apache-2.0) — some rewritten in PyTorch, all extended with additional evaluation, baselines and documentation. Original authors are credited in each project.

Related: [**generative-ai-engineering**](https://github.com/kishorhange111/generative-ai-engineering) (LLM fine-tuning, RAG, agents, multimodal) · [**mini-chatgpt**](https://github.com/kishorhange111/mini-chatgpt) (base LLM → SFT → DPO → benchmarks).
