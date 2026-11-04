### Proxy data size (5 seeds each, mean ± std)

| Comparisons | Agreement with gold | Best-of-n: peak gold (n at peak) | Best-of-n: gold at n = 4096 | RL: peak gold | RL: KL at peak (nats) | RL: final gold | RL: final proxy | Fit R²: best-of-n / RL |
|---|---|---|---|---|---|---|---|---|
| 1,000 | 75.8 % | 1.32 ± 0.02 (4096/2048/4096/4096/4096) | 1.32 ± 0.02 | 1.31 ± 0.08 | 11.55 ± 5.04 | 1.13 ± 0.24 | 4.47 ± 0.59 | 1.000 / 0.95 |
| 3,000 | 81.7 % | 1.34 ± 0.01 (512/512/256/512/512) | 1.29 ± 0.01 | 1.25 ± 0.06 | 10.33 ± 8.84 | 0.90 ± 0.34 | 4.30 ± 0.19 | 1.000 / 0.88 |
| 10,000 | 85.9 % | 1.36 ± 0.00 (256/256/256/256/256) | 1.29 ± 0.02 | 1.22 ± 0.01 | 4.56 ± 0.38 | 0.58 ± 0.13 | 3.16 ± 0.18 | 0.999 / 0.94 |
| 30,000 | 86.9 % | 1.36 ± 0.00 (256/256/256/256/256) | 1.27 ± 0.03 | 1.21 ± 0.01 | 4.39 ± 0.11 | 0.48 ± 0.31 | 2.54 ± 0.20 | 0.999 / 0.95 |
| 100,000 | 87.7 % | 1.36 ± 0.00 (256/256/256/256/256) | 1.30 ± 0.03 | 1.21 ± 0.01 | 4.48 ± 0.35 | 0.69 ± 0.37 | 2.59 ± 0.12 | 0.998 / 0.95 |

Optimizing the gold reward itself reaches 1.40 with RL and 1.42 with best-of-4096.

### KL penalty (RL on the same 5 proxies, 10,000 comparisons)

Each penalized run is compared with stopping the unpenalized run on the same proxy at the same KL.

| KL coefficient | Final KL (nats) | Final gold | Early-stopped unpenalized run, same KL | Penalty − early stopping (seeds where penalty is better) |
|---|---|---|---|---|
| 0.01 | 24.00 ± 1.87 | 0.68 ± 0.14 | 0.58 ± 0.13 | +0.10 ± 0.07 (5/5) |
| 0.03 | 17.15 ± 1.73 | 0.86 ± 0.06 | 0.75 ± 0.08 | +0.12 ± 0.07 (5/5) |
| 0.1 | 7.12 ± 0.57 | 1.20 ± 0.02 | 1.17 ± 0.04 | +0.03 ± 0.03 (4/5) |
| 0.3 | 2.15 ± 0.10 | 1.16 ± 0.01 | 1.10 ± 0.01 | +0.06 ± 0.01 (5/5) |

### Ensembles of 5 proxies (10,000 comparisons, 5 seeds)

Paired differences are against a single member of the same ensemble (seeds where the ensemble is better).

| Objective | RL: peak gold | RL: final gold | RL: final gold − single | Best-of-n: gold at n = 4096 | Best-of-n − single |
|---|---|---|---|---|---|
| single | 1.25 ± 0.06 | 0.66 ± 0.29 | — | 1.27 ± 0.04 | — |
| mean | 1.22 ± 0.01 | 0.38 ± 0.24 | -0.28 ± 0.16 (0/5) | 1.27 ± 0.02 | -0.00 ± 0.04 (3/5) |
| min | 1.22 ± 0.01 | 0.71 ± 0.30 | +0.05 ± 0.30 (3/5) | 1.30 ± 0.03 | +0.03 ± 0.04 (4/5) |
| uwo_0.5 | 1.22 ± 0.01 | 0.56 ± 0.26 | -0.10 ± 0.19 (3/5) | 1.28 ± 0.02 | +0.01 ± 0.04 (4/5) |
| uwo_1.0 | 1.22 ± 0.01 | 0.61 ± 0.23 | -0.04 ± 0.26 (3/5) | 1.29 ± 0.02 | +0.02 ± 0.04 (4/5) |
