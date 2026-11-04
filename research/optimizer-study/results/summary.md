| Optimizer | Best LR (sweep) | Val bits/byte, mean ± std | Δ vs AdamW | p (Welch) | Steps to AdamW's final loss | Speedup per step | Speedup per second | Optimizer share of step time |
|---|---|---|---|---|---|---|---|---|
| adamw | 0.003 | 1.9975 ± 0.0045 (n=3) | +0.0000 | — | 1000 | 1.00× | 1.00× | 5 % |
| adamw+clip | 0.003 | 1.9794 ± 0.0041 (n=3) | -0.0180 | 0.007 | 955 | 1.05× | 1.05× | 5 % |
| lion | 0.0003 | 2.5184 ± 0.0568 (n=3) | +0.5209 | 0.004 | not reached | — | — | 4 % |
| muon | 0.01 | 1.6250 ± 0.0054 (n=3) | -0.3724 | 0.000 | 464 | 2.16× | 1.71× | 24 % |

Learning-rate sweeps (seed 0, final val bits/byte; — = diverged):

- **adamw**: 0.001: 2.2057, 0.003: 2.0004, 0.01: 2.8732, 0.03: 3.5438
- **adamw+clip**: 0.001: 2.2141, 0.003: 1.9826, 0.01: 2.4248
- **lion**: 0.0001: 2.9915, 0.0003: 2.5597, 0.001: 3.4155, 0.003: 3.7337
- **muon**: 0.003: 1.8099, 0.01: 1.6197, 0.03: 1.6278, 0.1: 2.9902

Horizon check (seed 0, best LR of the main comparison):

- adamw (lr 0.003, 3000 steps): final val 1.5769 bits/byte
- muon (lr 0.01, 3000 steps): final val 1.4093 bits/byte
- Muon reaches AdamW's 3000-step loss at step 1595 (1.88× fewer steps)
