| Ablated | Repeated-sequence loss | In-context learning score |
|---|---|---|
| none | 1.447 | -3.090 |
| top induction head (L1) (L1.0) | 1.775 | -2.988 |
| top 2 induction heads (L1) (L1.0, L1.6) | 2.239 | -2.771 |
| 2 random other L1 heads (L1.4, L1.7) | 2.016 | -2.813 |
| top previous-token head (L0) (L0.0) | 3.467 | -1.981 |
| 1 random other L0 head (L0.1) | 1.523 | -3.074 |
| all L1 heads with induction score > 0.3 (L1.0, L1.1, L1.2, L1.3, L1.4, L1.5, L1.6, L1.7) | 6.957 | -1.263 |
| all L0 heads with previous-token score > 0.2 (L0.0, L0.2, L0.6) | 6.694 | -0.612 |
