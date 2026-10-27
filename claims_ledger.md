# Claims ledger
Every number used outside this repo must have a row here. See SCOPE.md section 11.

| claim id | statement | value (mean, 95% CI, n) | run ids | commit | date (IST) | verified-by | status |
|---|---|---|---|---|---|---|---|
| C1 | DI nominal learned-model safe set produces nonzero False-Safe Volume (FSV) under coverage gap (N=100) | 0.0030 [0.0005, 0.0055] (n=5) | 20261027_161358 | b77b20e | 2026-10-27 | automated pipeline (pytest + runner) | pilot (n=5, superseded) |
| C2 | DI epistemic inflation with k=2.0 reduces mean FSV by 72.7% vs nominal (H2 >=50% target met) | 0.0008 [0.0000, 0.0020] (n=5) | 20261027_161358 | b77b20e | 2026-10-27 | automated pipeline (pytest + runner) | pilot (n=5, superseded) |
| C3 | DI epistemic inflation with k=2.0 increases conservatism cost (FUV rises from 0.0005 to 0.0042, confirming H3) | 0.0042 [0.0006, 0.0078] (n=5) | 20261027_161358 | b77b20e | 2026-10-27 | automated pipeline (pytest + runner) | pilot (n=5, superseded) |
| P1 | Ratio-of-means BCa FSV reduction at k=2.0 vs k=0.0 under hidden physics gap (seeds 1000-1029, N=100, 201x201 grid) | 40.34% [36.47%, 44.73%] (n=30) (Nominal FSV 0.0096 -> k=2.0 FSV 0.0057; >=50% bar not met) | 20261027_184119 | e0be6be (prereg-v1) | 2026-10-28 | automated pipeline (run_pipeline + compute_p1) | partially supported (confirmatory) |
| S1 | Epistemic inflation full k-curve FSV and Gap-FSV reduction across k in [0.0, 3.0] | k=0.0: FSV 0.0096, Gap-FSV 0.0534 -> k=3.0: FSV 0.0043, Gap-FSV 0.0274 (55.2% reduction at k=3.0) | 20261027_184119 | e0be6be (prereg-v1) | 2026-10-28 | automated pipeline | confirmatory (secondary) |
| S2 | Asymmetric safety cost break-even criticality ratio r* under hidden physics gap | r* = 0.4532 (Delta FUV = +0.0018, Delta FSV = 0.0039; inflation is cost-reducing for all r >= 1) | 20261027_184119 | e0be6be (prereg-v1) | 2026-10-28 | automated pipeline | confirmatory (secondary) |

