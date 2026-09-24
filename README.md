# Deep Learning Projects

End-to-end deep learning projects across **computer vision, NLP and representation learning**. Each one is trained on cloud GPUs, evaluated on held-out data and documented with its real results.

Every project is a single, heavily commented `train.py` written to be read top-to-bottom: *what problem, which idea, why this design, how it is measured*.

## Projects

| # | Project | Key techniques | Result |
|---|---|---|---|
| 01 | [Vision Transformer from scratch](01_vision_transformer_cifar100) | patch embeddings, multi-head self-attention, no convolutions | **48.8 %** top-1 · 77.0 % top-5 on CIFAR-100 |
| 02 | [ShiftViT — a ViT without attention](02_shiftvit_cifar10) | zero-parameter shift token-mixing, hierarchical stages, stochastic depth, warm-up cosine | **77.9 %** top-1 on CIFAR-10 (Keras reference 76.4 %) |
| 03 | [U-Net pet segmentation](03_unet_pet_segmentation) | encoder–decoder, separable convs, residual skips | **83.0 %** pixel acc · **0.60** mIoU |
| 04 | [Image captioning](04_image_captioning_flickr8k) | frozen EfficientNet + Transformer decoder, cross-attention, BLEU | **BLEU-1 0.59** · BLEU-4 0.18 |
| 05 | [Siamese network – contrastive loss](05_siamese_contrastive_mnist) | metric learning, shared weights | **97.3 %** pair accuracy |
| 06 | [Siamese network – triplet loss](06_siamese_triplet_resnet) | ResNet50 transfer learning, custom train step | triplet acc **63 % → 83.5 %** |
| 07 | [LSTM seq2seq addition](07_seq2seq_addition_lstm) | encoder–decoder RNN, input reversal | **98.1 %** exact-match |
| 08 | [BERT sentiment analysis](08_bert_sentiment_imdb) | Transformer fine-tuning, measured TF-IDF baseline | **92.4 %** acc (baseline 90.4 %) |
| 09 | [GPT-2: LoRA vs. full fine-tuning](09_gpt2_lora_finetuning) | parameter-efficient fine-tuning, KerasHub, memory/speed/perplexity comparison | LoRA: **0.73 %** of params, **−69 % GPU memory**, 70 % of the perplexity gain |
| 10 | [BERT optimization: distillation + ONNX + int8](10_bert_distillation_onnx_quantization) | knowledge distillation, ONNX Runtime, dynamic quantization, CPU latency benchmark | **4.8× faster**, **85 % smaller**, −0.6 pts accuracy |
| 12 | [AdaIN real-time style transfer](12_adain_style_transfer) | PyTorch, adaptive instance normalization, VGG encoder + learned decoder | **any unseen style in 31 ms** per 512² image |
| 13 | [Music Transformer on MAESTRO](13_music_transformer_midi) | performance-event tokenizer, GPT from scratch, nucleus sampling | test perplexity **16.9** / 388 tokens; realistic density & dynamics |
| 14 | [Molecule generation with a Transformer](14_molecule_generation_gpt) | GPT from scratch on SMILES, RDKit validity / uniqueness / novelty | **73 %** valid · **~100 %** unique & novel · property distributions matched |

*Coming next: neural style transfer (Gatys, L-BFGS) — the optimisation-based counterpart of project 12.*

## Repository layout
```
common/utils.py      shared helpers: seeding, GPU time limits, metrics + plots
NN_<project>/        train.py (the whole project) + README.md (approach, results, discussion)
results/NN_<...>/    metrics.json, history.json, training curves, sample predictions
run_all.py           run every (or selected) project, one process each, with logs
ops/                 scripts to run the projects on Google Colab GPUs via the Colab CLI
```

## Engineering practices
- **Reproducible:** fixed seeds, pinned splits, every number in the READMEs comes from `results/*/metrics.json`.
- **Cost-controlled:** every run has a wall-clock budget (`TIME_LIMIT_MIN`) so cloud-GPU spend is capped.
- **Smoke-tested:** `QUICK=1` runs each full pipeline on tiny data in about a minute before spending GPU hours.
- **Honest evaluation:** held-out test sets, baselines where they matter (BERT vs. TF-IDF), before/after comparisons (triplet fine-tuning), failure cases shown, not hidden.

## Running
```bash
pip install -r requirements.txt
python run_all.py               # everything
python run_all.py 05 07         # selected projects
QUICK=1 python run_all.py       # ~5-minute smoke test of all pipelines
```
Training was done on Google Colab (NVIDIA A100, L4 and T4) driven from the terminal with the [Colab CLI](https://github.com/googlecolab/google-colab-cli) — see [`ops/`](ops).

## Credits
Several projects are re-implementations of [Keras code examples](https://keras.io/examples/) (Apache-2.0), extended with extra evaluation (BLEU, mIoU, before/after metrics, baselines), cached feature extraction and documentation. Original authors are credited in each project.
