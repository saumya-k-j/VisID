# Results: beauty_s3rec

Mean ± std over seeds (n = number of seeds). Leave-one-out test set, full-catalog ranking.

## Warm test cases

| Model | seeds | R@5 | R@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| sasrec (n_cases=21167) | 1 | 0.0498 ± 0.0000 | 0.0761 ± 0.0000 | 0.0294 ± 0.0000 | 0.0378 ± 0.0000 |
| twotower_attr (n_cases=21167) | 1 | 0.0499 ± 0.0000 | 0.0790 ± 0.0000 | 0.0290 ± 0.0000 | 0.0383 ± 0.0000 |
| tiger_attr (n_cases=21167) | 1 | 0.0319 ± 0.0000 | 0.0539 ± 0.0000 | 0.0204 ± 0.0000 | 0.0274 ± 0.0000 |
| tiger_attr (eps0.1) (n_cases=21167) | 1 | 0.0293 ± 0.0000 | 0.0498 ± 0.0000 | 0.0189 ± 0.0000 | 0.0255 ± 0.0000 |

## Cold test cases

| Model | seeds | R@5 | R@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| sasrec (n_cases=1193) | 1 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| twotower_attr (n_cases=1193) | 1 | 0.0017 ± 0.0000 | 0.0034 ± 0.0000 | 0.0007 ± 0.0000 | 0.0012 ± 0.0000 |
| tiger_attr (n_cases=1193) | 1 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| tiger_attr (eps0.1) (n_cases=1193) | 1 | 0.0218 ± 0.0000 | 0.0293 ± 0.0000 | 0.0120 ± 0.0000 | 0.0144 ± 0.0000 |

## All test cases

| Model | seeds | R@5 | R@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| sasrec (n_cases=22360) | 1 | 0.0472 ± 0.0000 | 0.0720 ± 0.0000 | 0.0278 ± 0.0000 | 0.0358 ± 0.0000 |
| twotower_attr (n_cases=22360) | 1 | 0.0474 ± 0.0000 | 0.0750 ± 0.0000 | 0.0275 ± 0.0000 | 0.0363 ± 0.0000 |
| tiger_attr (n_cases=22360) | 1 | 0.0302 ± 0.0000 | 0.0510 ± 0.0000 | 0.0193 ± 0.0000 | 0.0260 ± 0.0000 |
| tiger_attr (eps0.1) (n_cases=22360) | 1 | 0.0289 ± 0.0000 | 0.0487 ± 0.0000 | 0.0186 ± 0.0000 | 0.0249 ± 0.0000 |

## Diagnostics

| Model | coverage@10 | cold slot share | invalid-ID rate (unconstrained) |
|---|---|---|---|
| sasrec | 0.846 | 0.0000 | n/a |
| twotower_attr | 0.740 | 0.0017 | n/a |
| tiger_attr | 0.218 | 0.0000 | 0.000 |
