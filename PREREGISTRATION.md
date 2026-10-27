# falsesafe-bench Pre-Registration v1

**Status:** REGISTERED & FROZEN (Tagged `prereg-v1`).
**Date drafted:** 2026-10-27
**Author(s):** Nisar & Keyush (falsesafe-bench contributors)
**Code commit this registration refers to:** 143c50b

> **Rule of this document:** everything below is decided *before* any confirmatory (n=30) result exists. We report whatever those runs produce. A null or negative result replaces the old claim in `claims_ledger.md`. It is not re-framed.

---

## 0. Pilot disclosure (what we already know, so it cannot be hidden)

- A pilot with n=5 seeds on the *unmodified* double integrator (no hidden physics) was already run (Run `20261027_161358`, commit `b77b20e`). It gave nominal FSV 0.0030, FSV at k=2.0 of 0.0008, and FUV rising from 0.0005 to 0.0042 across the k sweep.
- The value **k = 2.0** and the 50% reduction bar (H2) come from that pilot and earlier planning. They are therefore *pilot-informed*, not blind. We use them as the headline comparison only because they are already public in the report. The full k-curve is reported alongside (Section 4).
- The new dynamics (Section 2.2) and the new certifiers have **not** been run at scale. No result from them informs this document.
- Seeds 0-9 are **development seeds** and may be inspected freely while debugging. Seeds **1000-1029** are **confirmatory seeds** and must not be run, plotted, or inspected before this file is tagged.

---

## 1. Frozen definitions

All volumes are fractions of the evaluation grid cells in $\mathcal{X}_{\text{eval}}$.

- True safe set $\mathcal{S}^*$: from the ground-truth solver (analytic solution cross-checked against the JAX grid BRT) under the *true* dynamics $f^*$, including the hidden physics of Section 2.2.
- Certified safe set $\widehat{\mathcal{S}}$: from the same solver under the certifier's model and disturbance set.
- $\text{FSV} = |\widehat{\mathcal{S}} \setminus \mathcal{S}^*| / |\mathcal{X}_{\text{eval}}|$
- $\text{FUV} = |\mathcal{S}^* \setminus \widehat{\mathcal{S}}| / |\mathcal{X}_{\text{eval}}|$
- **Gap-FSV**: FSV restricted to cells whose position lies inside the withheld window.
- **Cost**: $C(r) = r \cdot \text{FSV} + \text{FUV}$, reported for all $r \in [1, 50]$.
- **Break-even ratio**: $r^* = \Delta\text{FUV} / \Delta\text{FSV}$, where $\Delta\text{FUV} = \text{FUV}_{\text{treat}} - \text{FUV}_{\text{base}}$ and $\Delta\text{FSV} = \text{FSV}_{\text{base}} - \text{FSV}_{\text{treat}}$ (means over seeds). Defined only when $\Delta\text{FSV} > 0$. Otherwise it is reported as "no break-even (no FSV reduction)".
- **Containment violation rate** $V$: fraction of $(x,u)$ evaluation points where $\|\hat f(x,u) - f^*(x,u)\|_2 > d(x)$, with $d(x)$ the certifier's disturbance bound. Reported overall and inside the gap.
- **Visitation-weighted FSV** $\text{FSV}_\pi$: FSV with each cell weighted by the unshielded state-visitation frequency of policy $\pi$ under $f^*$, normalized to sum to 1.
- **Evaluation grid**: 201 x 201 (position x velocity) for all reported numbers. 51 and 101 are used for convergence checks only.

---

## 2. Experimental setup (frozen at the tagged commit)

### 2.1 What is inherited from code
Every hyperparameter not listed here (ensemble size and architecture, optimizer, epochs, learning rate, horizon $\tau$, failure function $l(x)$, control bounds) is taken from `configs/e1_di_sweep.yaml` and `src/lcert/systems.py` **at the tagged commit**. The tag is the specification.

### 2.2 Hidden-physics gap (the dynamics that actually differ from the learned model)
True dynamics: $\dot p = v,\quad \dot v = (1 - \eta\,\mathbf{1}_G(p))\,u + b\,s\,\mathbf{1}_G(p)$, where $G = [p_c - w/2,\; p_c + w/2]$.

| Parameter | Primary value | Meaning |
|---|---|---|
| $w$ (gap width) | **0.6** | Position window withheld from training and containing the hidden physics |
| $p_c$ (gap center) | **0.7** | Position center on trajectories that approach failure boundary $p=0$ (window $[0.4, 1.0]$) |
| $\eta$ | **0.5** | Loss of control authority inside $G$ |
| $b$ | **0.3** | Magnitude of an additive push inside $G$ |
| $s \in \{+1,-1\}$ | **-1** | Negative sign pushes vehicle toward failure boundary at $p \le 0$ |
| $N$ (training transitions) | **100** | As in the pilot |

**Fill rule:** $p_c = 0.7$ and $s = -1$ were selected directly from the domain geometry (approaching $p=0$ at $v < 0$ with stopping distance $\sim 0.5\,\text{m}$) *prior to running any confirmatory sweep*. Frozen.

**Why not drag?** A drag term $-\gamma v|v|$ opposes motion and *assists braking*. Inside the gap it would make reality safer than the model (more false-unsafe, not false-safe). Reduced authority and a push toward the hazard make reality *worse* than the model, which is the false-safe direction. Drag is kept as a control (Section 5, C4).

**Training data rule:** every transition whose position lies in $G$ is excluded from training. Calibration data (for conformal methods) follows Section 3.

**Ensemble blind-spot variant (B):** all members share one architecture with no inter-member diversity beyond initialization (ReLU MLP, linear extrapolation outside data). Registered as a secondary condition (Section 4, S5).

### 2.3 Seeds and randomness
- Seed controls: training-point sampling, network init, bootstrap RNG base.
- Confirmatory seeds: **1000-1029** (n=30). Bootstrap RNG seed: **20261027**.
- Same seed means identical training points and eval grid across all certifiers and all knob values (this is what makes comparisons paired).

---

## 3. Certifiers and their knobs (frozen)

| ID | Certifier | Disturbance bound $d(x)$ | Knob grid |
|---|---|---|---|
| E | Ensemble-$k$ (per-channel) | $k\,\sigma_{\text{ep}}(x)$ on the learned channel only | $k \in \{0.0, 0.2, \dots, 3.0\}$ |
| L | Distance-aware | $\varepsilon_0 + \lambda\,\hat L\,\text{dist}((x,u),\mathcal{D}_{\text{train}})$ | $\lambda \in \{0.0, 0.2, \dots, 3.0\}$ |
| C0 | Naive split conformal | $\hat q_{1-\alpha}$ from i.i.d. held-out residuals | $\alpha \in \{0.5, 0.3, 0.2, 0.1, 0.05, 0.01\}$ |
| C1 | Normalized conformal | scores divided by $\sigma_{\text{ep}}(x)$ (C1a) or by $\text{dist}$ (C1b) | same $\alpha$ grid |
| C2 | Leave-region-out conformal | calibrate on a region withheld from *that* certifier's training split | same $\alpha$ grid |
| O | Oracle (true $f^*$) | 0 | none (solver floor) |

- $\text{dist}$ is Euclidean in the joint $(p, v, u)$ space, each axis divided by its range over $\mathcal{X}_{\text{eval}} \times \mathcal{U}$.
- $\hat L = \max_{i \ne j} \|y_i - y_j\| / \|z_i - z_j\|$ from training pairs. Known failure mode, registered in advance: $\hat L$ from clustered data underestimates the Lipschitz constant across the gap.
- $\varepsilon_0$ = the largest training residual of the learned model, fixed before any gap evaluation.
- C2's held-out calibration region is a window of width $w$ located *away from* $G$, so it is not an oracle for the true gap.

---

## 4. Hypotheses and analysis plan

### 4.1 Primary (confirmatory, one test)

**P1.** At the primary condition (Section 2.2), Ensemble-$k$ with $k=2.0$ reduces FSV relative to $k=0$.

- **Estimand:** $R = \dfrac{\overline{\text{FSV}}_{k=0} - \overline{\text{FSV}}_{k=2}}{\overline{\text{FSV}}_{k=0}} \times 100\%$ (ratio of means over seeds).
- **Interval:** BCa bootstrap, 10,000 resamples of seeds (paired), 95%. No clipping.
- **Decision rule, fixed now:**
  - **Supported:** $R \ge 50\%$ and lower bound $> 0$.
  - **Partially supported:** lower bound $> 0$ but $R < 50\%$.
  - **Not supported:** lower bound $\le 0$.
- **Always reported with P1:** $\Delta$FUV with its own BCa interval, $r^*$ with its BCa interval, and the cost curve $C(r)$.

### 4.2 Secondary (pre-specified, reported with 95% BCa, no multiplicity correction, labelled as secondary)

- **S1. Full k-curve.** FSV, FUV, Gap-FSV, $V$ versus $k$ at the primary condition. Test of monotonic FUV growth (original H3): per-seed Spearman $\rho(k, \text{FUV})$, report the median and BCa interval across seeds.
- **S2. Certifier comparison.** AUPC for E, L, C0, C1a, C1b, C2 at the primary condition.
  - *AUPC definition:* with $\rho = \text{FSV} / \text{FSV}_{\text{nominal}}$ (nominal = E at $k=0$, same seed), let $\text{FUV}^*(\rho)$ be the lowest FUV among that certifier's knob settings with FSV-ratio $\le \rho$. AUPC $= \int_0^1 \text{FUV}^*(\rho)\,d\rho$ (trapezoid). Where no setting reaches $\rho$, $\text{FUV}^*$ is set to the largest FUV observed across all certifiers and settings in that condition. Lower is better.
  - Also report FUV at $\rho \in \{0.5, 0.25, 0.1\}$.
- **S3. Coverage-gap sweeps (pre-specified, not primary).** Gap width $w \in \{0.1, 0.2, \dots, 1.2\}$ at $N=100$; training size $N \in \{50, 100, 200, 500\}$ at $w=0.6$. Plot FSV, FUV, Gap-FSV, $V$ per certifier. n=30 seeds per cell. If compute-limited, the ladder may be thinned to $w \in \{0.2, 0.6, 1.0\}$; this is a registered fallback, not a deviation.
- **S4. Directional predictions about bounds** (reported either way):
  - (a) C0 has gap containment rate below $1-\alpha$ at the nominal $\alpha$.
  - (b) $L$ has $\hat L$ underestimated in the gap, giving gap violations unless $\lambda > 1$.
  - (c) C2 has better gap containment than C0.
- **S5. Ensemble blind spot.** In variant B, E's $V$ inside the gap is higher than in the standard ensemble, and E's FUV@$\rho$ is worse than L's.
- **S6. Closed-loop validation.**
  - Goal placed behind the gap. Policy family: random, nominal goal-directed, shortcut-seeking (maximum-speed path through the gap).
  - Initial states: 100 per seed, drawn uniformly from $\mathcal{S}^*$ (so a crash is avoidable in principle). Fixed horizon $T$ taken from the config.
  - Crash: true-dynamics trajectory enters the failure set while the certifier's shield is active.
  - Unit of analysis: condition means (certifier x knob x $w$). Report Spearman $\rho_s$ and Kendall $\tau$ of FSV and of $\text{FSV}_\pi$ against crash rate, with seed-bootstrap intervals.
  - Interpretation labels fixed in advance: $\rho_s \ge 0.7$ strong, $0.4$ to $0.7$ moderate, below $0.4$ weak. Conditions are not independent, so this is reported as descriptive evidence, not a significance test.
- **S7. Dubins per-channel separation (H4).** Registered now, run after DI. Primary comparison: FUV at matched FSV, per-channel Cartesian disturbance set versus isotropic quadrature blend $\sqrt{d_{\text{al}}^2 + (k\sigma_{\text{ep}})^2}$. The Dubins hidden-physics parameters must be added as an amendment (Section 10) **before** the Dubins confirmatory run.

### 4.3 Exploratory
Anything not listed above (additional policies, other $k$ ranges, new certifiers, UI-driven observations) is exploratory and labelled as such in any writeup.

---

## 5. Controls and sanity checks (must pass or be reported as failed)

- **C1 Oracle.** O must give FSV = 0. Its FUV is the **solver floor**. Any FSV reported at the primary condition below 3x the floor is flagged "not distinguishable from numerical error".
- **C2 No-gap control.** Full-coverage training, no hidden physics. FSV should sit near the floor.
- **C3 Corrupted model.** An inverted-dynamics model must fail verification (existing H5 test).
- **C4 Benign-drag control.** Drag-only patch (Section 2.2 note). Prediction: FUV rises, FSV does not. If FSV rises instead, the geometry assumption is wrong and gets reported.
- **C5 Grid convergence.** FSV at the primary condition on 51, 101, 201 grids. Pass if $|\text{FSV}_{101} - \text{FSV}_{201}| \le 25\%$ of $\text{FSV}_{201}$. Otherwise all headline numbers are reported on the finest grid with a convergence warning.

---

## 6. Exclusions and failed runs

- No seed is excluded post hoc.
- A run that diverges (NaN loss, solver failure) counts as a failed run. Failures are counted and reported. The primary analysis is run once with failed seeds dropped and once with them included at worst-case FSV, and both are shown.
- No re-running a confirmatory seed after seeing its result. A rerun is allowed only for infrastructure failure (process killed, disk full), documented in the run log.

---

## 7. Multiple comparisons

One primary test (P1). Secondary results use 95% BCa intervals with no correction and are described as such. No result from Section 4.2 or 4.3 is promoted to "confirmed" in the paper unless P1-style criteria were fixed here in advance.

---

## 8. Software and provenance

- Dependency versions per `requirements.txt` at the tagged commit.
- Every run writes: run ID, git commit, `prereg-v1` tag hash, seed list, config hash, grid size.
- The confirmatory pipeline must **refuse to run** seeds 1000-1029 unless (a) the `prereg-v1` tag exists and (b) the working tree is clean. This is an automated guard in `scripts/run_pipeline.py`, not a convention.
- The viewer's audit banner reads these fields from the run output. The test `test_viewer_fsv_matches_ledger` compares displayed FSV with the ledger.

---

## 9. Ledger entry format

Each claim in `claims_ledger.md` carries: claim ID, estimand, estimate with BCa interval, n, grid size, run ID, commit, prereg tag, and a **status** of `confirmatory`, `secondary`, or `exploratory`. The old C1-C3 are relabelled `pilot (n=5, superseded)`.

---

## 10. Amendments and deviations

After tagging, this file is append-only. Each amendment has a date, a reason, a statement of whether any confirmatory result had been seen when it was made, and a git tag `prereg-v1.N`. Anything changed after seeing a confirmatory result is labelled **post hoc** wherever the result appears.

### Amendment log
_(empty)_

---

## Before you commit (checklist)

- [x] Fill author, commit hash, $p_c$ and $s$ (Section 2.2). Do not look at FSV while choosing them.
- [x] Confirm the horizon, failure function, and control bounds in `systems.py` match what this text assumes.
- [x] Add the confirmatory guard to `scripts/run_pipeline.py` (Section 8).
- [ ] Commit this file, then `git tag -a prereg-v1 -m "Pre-registration v1"` and push the tag.
- [ ] Only then run seeds 1000-1029.
