CPU: 12 threads · batch 1 · 128 tokens · accuracy on 2000 IMDB test reviews

| variant | p50 latency (ms) | p95 (ms) | size (MB) | accuracy | speed-up |
|---|---|---|---|---|---|
| BERT-base · PyTorch fp32 (teacher) | 64.8 | 159.0 | 418 | 91.8 % | 1.0x |
| BERT-base · ONNX Runtime fp32 | 48.2 | 108.9 | 418 | 91.8 % | 1.3x |
| BERT-base · ONNX Runtime int8 | 27.7 | 57.8 | 105 | 91.7 % | 2.3x |
| DistilBERT (distilled) · PyTorch fp32 | 32.9 | 71.7 | 255 | 91.1 % | 2.0x |
| DistilBERT (distilled) · ONNX Runtime fp32 | 24.3 | 59.5 | 256 | 91.1 % | 2.7x |
| DistilBERT (distilled) · ONNX Runtime int8 | 13.6 | 15.0 | 64 | 91.2 % | 4.8x |

Full test set (25k, GPU): teacher 92.26 % · DistilBERT without distillation 91.43 % · DistilBERT with distillation 91.49 %
