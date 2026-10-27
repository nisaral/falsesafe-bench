# Claims ledger
Every number used outside this repo must have a row here. See SCOPE.md section 11.

| claim id | statement | value (mean, 95% CI, n) | run ids | commit | date (IST) | verified-by | status |
|---|---|---|---|---|---|---|---|
| C1 | DI nominal learned-model safe set produces nonzero False-Safe Volume (FSV) under coverage gap (N=100) | 0.0030 [0.0005, 0.0055] (n=5) | 20261027_161358 | b77b20e | 2026-10-27 | automated pipeline (pytest + runner) | verified |
| C2 | DI epistemic inflation with k=2.0 reduces mean FSV by 72.7% vs nominal (H2 >=50% target met) | 0.0008 [-0.0003, 0.0020] (n=5) | 20261027_161358 | b77b20e | 2026-10-27 | automated pipeline (pytest + runner) | verified |
| C3 | DI epistemic inflation with k=2.0 increases conservatism cost (FUV rises from 0.0005 to 0.0042, confirming H3) | 0.0042 [0.0006, 0.0078] (n=5) | 20261027_161358 | b77b20e | 2026-10-27 | automated pipeline (pytest + runner) | verified |
