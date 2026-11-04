Model: 1.495 bits/byte clean, 5.643 with the layer mean-ablated. SAEs: 2,048 latents on 524,288 training bytes; metrics on 32,768 held-out bytes.

| SAE | Setting | L0 (active latents per byte) | FVU | Loss recovered | Spliced bits/byte | Dead latents |
|---|---|---|---|---|---|---|
| ReLU + L1 | L1 = 10.0 | 2.1 | 1.029 | -1.9 % | 5.722 | 0.0 % |
| ReLU + L1 | L1 = 3.0 | 25.4 | 0.546 | 55.0 % | 3.360 | 0.0 % |
| ReLU + L1 | L1 = 1.0 | 116.0 | 0.178 | 89.7 % | 1.922 | 0.0 % |
| ReLU + L1 | L1 = 0.3 | 226.5 | 0.083 | 97.1 % | 1.615 | 0.0 % |
| TopK | k = 8 | 8.0 | 0.089 | 94.9 % | 1.706 | 8.3 % |
| TopK | k = 16 | 16.0 | 0.067 | 96.7 % | 1.632 | 0.2 % |
| TopK | k = 32 | 32.0 | 0.058 | 97.7 % | 1.592 | 0.0 % |
| TopK | k = 64 | 64.0 | 0.054 | 98.2 % | 1.569 | 0.0 % |
| BatchTopK | k = 8 | 6.9 | 0.113 | 93.2 % | 1.779 | 14.5 % |
| BatchTopK | k = 16 | 13.5 | 0.095 | 95.1 % | 1.700 | 2.1 % |
| BatchTopK | k = 32 | 27.3 | 0.082 | 96.5 % | 1.639 | 0.2 % |
| BatchTopK | k = 64 | 56.5 | 0.069 | 97.6 % | 1.594 | 0.0 % |

Interpretability check (top 20 activations per direction on held-out text):

| Directions | Count | Median share on the same current byte | Directions with ≥ 90 % on one byte | Median share on the same next byte | Median share on the same byte class |
|---|---|---|---|---|---|
| SAE latents (TopK) | 1,964 | 70 % | 36 % | 40 % | 95 % |
| residual-stream dimensions (both signs) | 256 | 50 % | 11 % | 35 % | 75 % |
| random directions | 2,048 | 45 % | 10 % | 35 % | 75 % |
