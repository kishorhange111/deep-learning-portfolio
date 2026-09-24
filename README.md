# Deep Learning Projects

End-to-end deep learning projects across **computer vision, NLP and representation learning**. Each one is trained on cloud GPUs, evaluated on held-out data and documented with its real results.

Every project is a single, heavily commented `train.py` written to be read top-to-bottom: *what problem, which idea, why this design, how it is measured*.

## Projects

| # | Project | Key techniques | Result |
|---|---|---|---|
| 03 | [U-Net pet segmentation](03_unet_pet_segmentation) | encoder–decoder, separable convs, residual skips | **83.0 %** pixel acc · **0.60** mIoU |
| 04 | [Image captioning](04_image_captioning_flickr8k) | frozen EfficientNet + Transformer decoder, cross-attention, BLEU | **BLEU-1 0.59** · BLEU-4 0.18 |
| 05 | [Siamese network – contrastive loss](05_siamese_contrastive_mnist) | metric learning, shared weights | **97.3 %** pair accuracy |
| 06 | [Siamese network – triplet loss](06_siamese_triplet_resnet) | ResNet50 transfer learning, custom train step | triplet acc **63 % → 83.5 %** |
| 07 | [LSTM seq2seq addition](07_seq2seq_addition_lstm) | encoder–decoder RNN, input reversal | **98.1 %** exact-match |
| 08 | [BERT sentiment analysis](08_bert_sentiment_imdb) | Transformer fine-tuning, measured TF-IDF baseline | **92.4 %** acc (baseline 90.4 %) |

*More projects (Vision Transformer, ShiftViT, GPT-2 LoRA vs. full fine-tuning) are being added.*

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
Training was done on Google Colab (NVIDIA L4 and T4) driven from the terminal with the [Colab CLI](https://github.com/googlecolab/google-colab-cli) — see [`ops/`](ops).

## Credits
Several projects are re-implementations of [Keras code examples](https://keras.io/examples/) (Apache-2.0), extended with extra evaluation (BLEU, mIoU, before/after metrics, baselines), cached feature extraction and documentation. Original authors are credited in each project.
