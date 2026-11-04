### Endpoints (10,000 comparisons, 3 seeds, mean ± std)

| Method | Strength | Peak gold | KL at peak | Final gold | Final KL |
|---|---|---|---|---|---|
| DPO (β) | 0.01 | 1.32 ± 0.01 | 7.8 ± 0.2 | 1.24 ± 0.04 | 14.9 ± 0.3 |
| DPO (β) | 0.03 | 1.32 ± 0.01 | 7.2 ± 0.1 | 1.16 ± 0.07 | 18.8 ± 0.7 |
| DPO (β) | 0.1 | 1.31 ± 0.01 | 5.9 ± 0.5 | 1.19 ± 0.10 | 15.6 ± 0.9 |
| DPO (β) | 0.3 | 1.26 ± 0.01 | 5.4 ± 0.5 | 1.26 ± 0.01 | 6.3 ± 0.5 |
| DPO (β) | 1.0 | 0.72 ± 0.02 | 0.6 ± 0.1 | 0.68 ± 0.02 | 0.7 ± 0.0 |
| IPO (τ) | 0.003 | 1.32 ± 0.01 | 7.5 ± 0.3 | 1.19 ± 0.06 | 17.8 ± 0.4 |
| IPO (τ) | 0.01 | 1.32 ± 0.01 | 6.8 ± 0.3 | 1.15 ± 0.08 | 19.1 ± 1.0 |
| IPO (τ) | 0.03 | 1.30 ± 0.01 | 5.8 ± 0.5 | 1.24 ± 0.07 | 11.7 ± 0.6 |
| IPO (τ) | 0.1 | 1.03 ± 0.02 | 1.8 ± 0.2 | 1.02 ± 0.02 | 2.0 ± 0.1 |
| IPO (τ) | 0.3 | 0.48 ± 0.01 | 0.2 ± 0.0 | 0.43 ± 0.01 | 0.2 ± 0.0 |
| RLHF, MLP reward model (KL coef) | 0.0 | 1.22 ± 0.01 | 4.7 ± 0.1 | 0.54 ± 0.07 | 25.9 ± 1.8 |
| RLHF, MLP reward model (KL coef) | 0.01 | 1.22 ± 0.01 | 4.9 ± 0.4 | 0.64 ± 0.12 | 24.4 ± 2.3 |
| RLHF, MLP reward model (KL coef) | 0.03 | 1.22 ± 0.01 | 4.9 ± 0.3 | 0.84 ± 0.05 | 16.9 ± 2.2 |
| RLHF, MLP reward model (KL coef) | 0.1 | 1.23 ± 0.01 | 4.8 ± 0.1 | 1.20 ± 0.02 | 7.2 ± 0.7 |
| RLHF, MLP reward model (KL coef) | 0.3 | 1.18 ± 0.00 | 2.2 ± 0.1 | 1.16 ± 0.01 | 2.2 ± 0.1 |
| RLHF, bigram reward model (KL coef) | 0.0 | 1.23 ± 0.02 | 5.8 ± 0.1 | 0.68 ± 0.32 | 23.8 ± 3.1 |
| RLHF, bigram reward model (KL coef) | 0.01 | 1.23 ± 0.02 | 6.1 ± 0.6 | 0.65 ± 0.34 | 23.8 ± 3.6 |
| RLHF, bigram reward model (KL coef) | 0.03 | 1.23 ± 0.02 | 5.8 ± 0.6 | 0.68 ± 0.28 | 22.5 ± 4.2 |
| RLHF, bigram reward model (KL coef) | 0.1 | 1.23 ± 0.02 | 5.7 ± 1.0 | 0.96 ± 0.13 | 13.4 ± 1.8 |
| RLHF, bigram reward model (KL coef) | 0.3 | 1.27 ± 0.01 | 4.2 ± 0.2 | 1.26 ± 0.01 | 4.2 ± 0.2 |

Best-of-n (MLP reward model): peak gold 1.36 ± 0.00, gold at n = 4,096 1.29 ± 0.02.
RL on the gold reward itself, by KL coefficient: 0.003: 1.39 at KL 9.2, 0.01: 1.37 at KL 6.5, 0.03: 1.33 at KL 4.0, 0.1: 1.23 at KL 2.4, 0.3: 0.99 at KL 1.1, 1.0: 0.56 at KL 0.2.
Reward-model agreement with gold on held-out pairs: MLP 0.860 ± 0.001, bigram 0.824 ± 0.002.

### Best gold reached at or below each KL budget (any strength, any seed, any point in training)

| Method | KL ≤ 1 | KL ≤ 2 | KL ≤ 4 | KL ≤ 8 | KL ≤ 16 |
|---|---|---|---|---|---|
| DPO (β) | 0.86 | 1.12 | 1.26 | 1.33 | 1.33 |
| IPO (τ) | 0.89 | 1.05 | 1.28 | 1.33 | 1.33 |
| RLHF, MLP reward model (KL coef) | 0.91 | 1.14 | 1.23 | 1.24 | 1.24 |
| RLHF, bigram reward model (KL coef) | 0.84 | 1.07 | 1.27 | 1.28 | 1.28 |

### DPO at the end of training: likelihood displacement and implicit-reward accuracy

| β | log π/ref of chosen | log π/ref of rejected | Implicit reward orders held-out pairs like gold |
|---|---|---|---|
| 0.01 | -147.3 ± 2.1 | -190.2 ± 3.5 | 82.3 % |
| 0.03 | -73.2 ± 2.3 | -93.3 ± 3.0 | 82.4 % |
| 0.1 | -23.5 ± 1.6 | -30.7 ± 1.8 | 81.5 % |
| 0.3 | -4.5 ± 0.4 | -7.1 ± 0.5 | 80.6 % |
| 1.0 | -0.2 ± 0.0 | -1.0 ± 0.0 | 80.2 % |

### DPO with more or fewer comparisons (β = 0.03, 1,200 steps)

| Comparisons | Peak gold | KL at peak | Final gold | Final KL |
|---|---|---|---|---|
| 1,000 | 1.26 ± 0.02 | 17.5 ± 4.6 | 1.09 ± 0.07 | 33.7 ± 2.4 |
| 3,000 | 1.31 ± 0.01 | 12.5 ± 3.4 | 1.20 ± 0.09 | 28.5 ± 4.5 |
| 10,000 | 1.32 ± 0.01 | 7.2 ± 0.1 | 1.16 ± 0.07 | 18.8 ± 0.7 |
| 30,000 | 1.32 ± 0.00 | 6.2 ± 0.2 | 0.93 ± 0.01 | 14.6 ± 0.5 |
| 100,000 | 1.32 ± 0.00 | 6.2 ± 0.2 | 0.89 ± 0.06 | 13.6 ± 0.3 |
