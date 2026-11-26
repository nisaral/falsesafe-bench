# One-page summary

## Problem

When a robot learns its dynamics from finite data, it can compute a safe set
that includes states which are actually unrecoverable under the true physics.
Those wrongly-certified states are the **false-safe volume (FSV)** — the most
dangerous error in a safety certificate, because the model confidently says
"safe" where a crash is already inevitable.

## Method

`falsesafe-bench` computes two Hamilton-Jacobi backward reachable tubes on the
same grid: the true BRT from the exact dynamics (`V_true`) and the learned BRT
from an ensemble model (`V_hat`). It then compares the two learned tubes under
epistemic **disturbance inflation** `k * sigma_ep(x)`, where `sigma_ep` is the
across-member standard deviation, and reports three protected metrics:

- **FSV** — fraction of evaluated cells the learned certificate calls safe but
  the true dynamics do not (false-safe).
- **FUV** — fraction the learned certificate calls unsafe but the true dynamics
  do not (false-unsafe / conservatism cost).
- **IoU** — overlap between the learned and true safe sets.

## Headline result

Claim P1 in [`claims_ledger.md`](../claims_ledger.md): on the double integrator
with a hidden-physics coverage gap, inflation at `k=2.0` cuts FSV by
**40.34%** with a 95% BCa interval of **[36.47%, 44.73%]** over **n=30**
replicates (nominal FSV 0.0096 -> 0.0057). The pre-registered bar was a 50%
reduction, so H2 is **partially supported**: the reduction is real but below the
bar. The auxiliary break-even ratio is `r* = 0.4532`.

## Limits

- FSV is a grid-cell count, so small disturbance changes can be invisible.
- The 30 replicates share a fixed data split (`split_seed=42`), so the interval
  understates data-sampling variability.
- The committed results used a 2-cell evaluation border; SCOPE specifies 3.
- Results cover the double integrator only; Dubins E1–E3 and the latent world
  model are unrun. See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) and
  [`ROADMAP.md`](ROADMAP.md).

## What was fixed in this branch

- Repaired the P1 aggregation path (missing imports, dataclass subscripting,
  wrong cost-argument binding) and extracted it into `aggregate_p1`.
- Added `predict_mean_std` to the ensemble and GP models and made the certifiers
  raise on unsupported models instead of silently substituting a constant.
- Made the C4 control status derive from its pass criterion, and made
  `border_cells` a config parameter (default 3) with legacy configs pinned to 2.
- Corrected the stale report (test count, H2 status, 72.7% pilot number) and
  added a Corrections section to the ledger for the C2/C3/C4 control rows.
- Added a read-only runlog validator, `docs/KNOWN_ISSUES.md`, a changelog,
  citations, a roadmap, and a figure regenerated from stored results.

See [`../CHANGELOG.md`](../CHANGELOG.md) for the commit-level list.
