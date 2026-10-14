"""
Shared helpers for every project in this repository.

utils         seeding, results folder, GPU time-budget callback, metrics.json + training-curve plots
captioning    caption cleaning / parsing, token-id decoding and BLEU (image captioning)
embeddings    triplet accuracy + cosine similarities (Siamese triplet network)
distillation  knowledge-distillation loss (BERT -> DistilBERT)
benchmark     p50 / p95 latency measurement (CPU inference benchmark)

Everything here is importable without a GPU and is covered by `tests/`.
"""
