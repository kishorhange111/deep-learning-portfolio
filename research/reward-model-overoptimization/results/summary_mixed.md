### Proxy data size (5 seeds each, mean ± std)

| Comparisons | Agreement with gold | Best-of-n: peak gold (n at peak) | Best-of-n: gold at n = 4096 | RL: peak gold | RL: KL at peak (nats) | RL: final gold | RL: final proxy | Fit R²: best-of-n / RL |
|---|---|---|---|---|---|---|---|---|
| 1,000 | 75.9 % | 1.50 ± 0.04 (4096/4096/4096/4096/4096) | 1.50 ± 0.04 | 1.54 ± 0.14 | 17.54 ± 3.30 | 1.51 ± 0.14 | 4.10 ± 0.08 | 1.000 / 0.97 |
| 3,000 | 82.0 % | 1.80 ± 0.09 (4096/1024/4096/4096/4096) | 1.80 ± 0.10 | 1.92 ± 0.29 | 23.21 ± 6.48 | 1.84 ± 0.37 | 4.84 ± 0.15 | 1.000 / 0.98 |
| 10,000 | 85.2 % | 1.90 ± 0.04 (4096/4096/4096/4096/4096) | 1.90 ± 0.04 | 2.17 ± 0.33 | 22.16 ± 6.42 | 2.06 ± 0.50 | 3.84 ± 0.16 | 1.000 / 0.99 |
| 30,000 | 86.2 % | 1.93 ± 0.07 (4096/4096/4096/4096/4096) | 1.93 ± 0.07 | 2.13 ± 0.23 | 20.21 ± 3.84 | 1.98 ± 0.25 | 3.45 ± 0.45 | 1.000 / 0.98 |
| 100,000 | 87.0 % | 2.02 ± 0.06 (4096/4096/4096/4096/4096) | 2.02 ± 0.06 | 2.42 ± 0.30 | 23.44 ± 5.25 | 2.37 ± 0.36 | 3.25 ± 0.26 | 1.000 / 1.00 |

Optimizing the gold reward itself reaches 3.04 with RL and 2.42 with best-of-4096.

### KL penalty (RL on the same 5 proxies, 10,000 comparisons)

Each penalized run is compared with stopping the unpenalized run on the same proxy at the same KL.

| KL coefficient | Final KL (nats) | Final gold | Early-stopped unpenalized run, same KL | Penalty − early stopping (seeds where penalty is better) |
|---|---|---|---|---|
