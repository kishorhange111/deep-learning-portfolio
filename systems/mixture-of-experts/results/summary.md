| Config | MLP params per layer: active / stored | Val bits/byte (mean ± std) | Δ vs dense (paired by seed) | ms / step (CPU) |
|---|---|---|---|---|
| dense | 131k / 131k | 1.9963 ± 0.0057 (n=2) | — | 73 |
| moe8_top1 | 132k / 1050k | 2.1681 ± 0.0694 (n=2) | +0.1718 ± 0.0637 | 119 |
| moe8_top1_unitgate | 132k / 1050k | 2.0712 ± 0.0236 (n=2) | +0.0748 ± 0.0293 | 125 |
| moe8_top2 | 263k / 1050k | 1.8413 ± 0.0166 (n=2) | -0.1550 ± 0.0109 | 172 |
| dense_wide | 262k / 262k | 1.9681 ± 0.0062 (n=2) | -0.0283 ± 0.0119 | 137 |
| moe8_top1_nobal | 132k / 1050k | 2.1390 ± 0.1303 (n=2) | +0.1426 ± 0.1246 | 106 |
| moe8_top1_cap1 | 132k / 1050k | 2.2329 ± 0.0444 (n=2) | +0.2366 ± 0.0387 | 99 |

Routing on held-out windows (first choice; averaged over layers and seeds):

| Config | Load entropy (bits; max = log2 E) | Largest / smallest expert share | Experts < 1 % | Tokens dropped | Choice predicted from byte class | from the byte itself | majority baseline |
|---|---|---|---|---|---|---|---|
| moe8_top1 | 2.98 / 3 | 15.4 % / 9.6 % | 0.0 | 0.0 % | 37 % | 58 % | 15 % |
| moe8_top1_unitgate | 2.98 / 3 | 16.4 % / 9.2 % | 0.0 | 0.0 % | 32 % | 46 % | 16 % |
| moe8_top2 | 2.98 / 3 | 15.8 % / 9.5 % | 0.0 | 0.0 % | 38 % | 59 % | 15 % |
| moe8_top1_nobal | 1.86 / 3 | 45.4 % / 0.0 % | 3.4 | 0.0 % | 63 % | 75 % | 45 % |
| moe8_top1_cap1 | 2.99 / 3 | 15.3 % / 9.7 % | 0.0 | 5.6 % | 37 % | 61 % | 15 % |

Horizon check (seed 0):

- dense, 3000 steps: val 1.5672 bits/byte
- moe8_top1_unitgate, 3000 steps: val 1.6075 bits/byte
