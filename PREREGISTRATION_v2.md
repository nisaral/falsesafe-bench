# falsesafe-bench-latent Pre-Registration v2 (Phase 4)

**Status:** DRAFT until committed and tagged `prereg-v2`. After tagging this file is frozen; changes follow Section 12.
**Date drafted:** 2026-10-28
**Author(s):** <<FILL>>
**Code commit this registration refers to:** <<FILL>>
**Relation to v1:** `PREREGISTRATION.md` (tag `prereg-v1`) stays in force and is not altered by this file. All v2 ledger IDs carry the prefix `v2-` (for example `v2-P2`, `v2-S1`) so they can never collide with v1 IDs.

> **Rule of this document:** everything below is decided before any confirmatory latent result exists. We report whatever the confirmatory runs produce. A negative or null result goes into the ledger as it is.

---

## 0. Disclosure of what is already known

- **v1 results are known.** Under the hidden-physics patch, P1 was *partially supported*: FSV reduction at k=2.0 was 40.3% (BCa 95% [36.5, 44.7]), below the 50% bar. Two further observations were made *after* seeing that result and are **post hoc**: (a) roughly three quarters of nominal FSV sits inside the gap, and inflation removed a smaller share of it than of FSV outside the gap; (b) k=3.0 reached 55.2%. The hypotheses in Section 6 are informed by these observations. They are not confirmed by them.
- **Literature informed the hypotheses.** Recent work reports that ensemble disagreement can stay small even when a world model's predictions deviate from observations. That is the motivation for P2. Our "ACI-style" certifier is a re-implementation in that spirit and does not speak for the original authors' implementation.
- **No latent result exists.** No world model has been trained at scale and no latent FSV has been computed on any seed set, apart from whatever development seeds are used while building.
- **Seed sets.**
  - Development seeds: **0-9** (may be inspected freely).
  - v1 confirmatory seeds: 1000-1029 (**not reused here**).
  - Seeds 2000-2029 are reserved for the v1.1 amendment (**not reused here**).
  - **v2 confirmatory seeds: 3000-3029.** They must not be run, plotted or inspected before `prereg-v2` is tagged.

---

## 1. What "world model" means in this document

The robot never receives its state $x = (p, v)$. It receives an image $o = R(x)$. A world model is learned from images and actions only:

1. **Encoder** $E_\phi$: image $\to$ latent $z \in \mathbb{R}^{d_z}$.
2. **Latent dynamics** $f_{\theta_m}(z,u) \to z'$, an ensemble of $M$ networks.
3. **Margin head** $\ell_\psi(z)$: a learned signed margin to failure ($<0$ means failed).
4. **Decoder** $D_\omega$: $z \to$ image. Auxiliary only, used as a regularizer.

A **latent certifier** computes a Hamilton-Jacobi safety value $V_{\text{cert}}(z)$ entirely inside the learned latent space, using $f_\theta$, $\ell_\psi$ and the certifier's disturbance bound. It never sees $x$.

---

## 2. Frozen definitions

### 2.1 Ground truth
- True dynamics $f^*$ are the v1 double integrator with the hidden-physics patch (parameters in Section 3).
- $\mathcal{S}^*$ is computed with the v1 JAX solver on the 201 x 201 $(p,v)$ evaluation grid.
- **Worst-case-mode convention.** In hidden-mode conditions the patch is active in only some episodes. Ground truth is always the **patch-active** safe set $\mathcal{S}^*(m{=}1)$, i.e. "safe even in the worst mode". This is valid only if the patch makes the system weakly worse (gate G2). FUV in hidden conditions therefore counts states safe under the worst mode that the certifier calls unsafe. It does not count states that are unsafe for some mode-agnostic controller but safe for a mode-aware one. This limitation is intentional and will be stated wherever hidden-condition FUV is reported.
- The ground truth is **identical across all four conditions** (Section 3). Only what the robot sees and learns differs.

### 2.2 Rendering
- $R(p,v;\text{vis}) \to$ 32 x 32 float image: a Gaussian blob at the column for $p$, a motion-blur tail whose length and direction follow $v$, a static marker for the failure boundary, and (if `vis=1`) a filled band over the columns of the gap $G$. Constants are in the config at the tagged commit.
- $o$ is a **deterministic function of $(p, v)$ and the visibility flag**. There is no history stack. This is deliberate: it lets us render *every* evaluation-grid state exactly.

### 2.3 Latent certified set and volumes
- For each evaluation cell $x$: $z_x = E_\phi(R(x))$. The cell is **certified safe** iff $V_{\text{cert}}(z_x) \ge 0$ (bilinear interpolation on the latent grid).
- $\text{FSV}_z = |\{x : \text{certified safe},\ x \notin \mathcal{S}^*\}| / |\mathcal{X}_{\text{eval}}|$, and $\text{FUV}_z$ symmetrically. Gap-FSV is restricted to cells with $p \in G$.
- **Cost, break-even ratio, BCa bootstrap**: exactly as v1 (Sections 1 and 4 of `PREREGISTRATION.md`).
- **Containment violation** $V_z$: fraction of evaluation $(x,u)$ with $\|f_\theta(z_x,u) - E_\phi(R(x'))\|_2 > d(z_x,u)$, where $x'$ is the true next state under the **patch-active** dynamics. Reported overall and inside $G$.
- **Latent penalty**: $\text{FSV}_z - \text{FSV}_{\text{priv}}$, where $\text{FSV}_{\text{priv}}$ comes from the v1 state-based pipeline trained on the same transitions and seeds.

### 2.4 Warning margin (monitor mode)
- Roll out a task policy **without** any shield in the true patch-active dynamics from 100 initial states per seed, drawn uniformly from $\mathcal{S}^*$.
- $t_{\text{pnr}}$ = first step with $x_t \notin \mathcal{S}^*$ (the true **point of no return**). A rollout with such a step is **doomed**. A rollout that never leaves $\mathcal{S}^*$ and never fails is **safe**.
- $t_{\text{flag}}$ = first step at which a certifier's monitor flags unsafe.
  - Static certifiers: $V_{\text{cert}}(z_t) < 0$.
  - ACI-style: $\min_{\|z'-z_t\|\le r_t} V_0(z') < 0$, where $V_0$ is the $k=0$ value function (definition in Section 4.4).
- **Margin** $= (t_{\text{pnr}} - t_{\text{flag}})\,\Delta t$. Positive means flagged before the point of no return.
- **Detection rate**: fraction of doomed rollouts flagged at or before $t_{\text{pnr}}$. **Miss rate** $= 1 -$ detection rate.
- **False-alarm rate (FAR)**: fraction of safe rollouts with any flag.
- Margin is always reported *together with* FAR, because a certifier that flags everything gets large margins for free.

---

## 3. Conditions (2 x 2)

All conditions use the v1-frozen patch: $p_c = 0.7$, $w = 0.6$, $\eta = 0.5$, $b = 0.3$, $s = -1$.

| | **Visible patch** (band drawn; patch active in every episode) | **Hidden patch** (band not drawn; patch active per episode with prob. $q=0.5$, unobserved) |
|---|---|---|
| **Covered** (transitions with $p \in G$ are in training data) | **QA**: visible + covered (control) | **QC**: hidden + covered |
| **Withheld** (no training transitions with $p \in G$) | **QB**: visible + withheld | **QD**: hidden + withheld |

**Factor 2 means one thing:** *is the patch's effect a deterministic function of what the camera sees?* The drawn band and the per-episode randomness change together on purpose.

**Training data (all conditions):** 200 trajectories x 50 steps = 10,000 transitions, collected with the behavior-policy mixture in the config. Split by trajectory 80/10/10 into train, validation and calibration. Withheld conditions remove every transition with $p \in G$ from all three splits. Hidden conditions draw $m \sim \text{Bernoulli}(q)$ per trajectory.

---

## 4. World model and certifiers (frozen at the tagged commit)

### 4.1 Inherited settings
Encoder and decoder architecture, latent dimension, ensemble size $M$, optimizer, epochs, loss weights, behavior policy, horizon $\tau$, failure function, and control bounds are taken from `configs/latent_di.yaml` and `src/lcert/systems.py` **at the tagged commit**. The tag is the specification.

### 4.2 Fixed choices
- **$d_z = 2$** (the intrinsic dimension of the double integrator), so the latent HJ problem can be solved with the exact grid solver used in v1. This isolates world-model error from value-solver error.
- Latent vector field: $\dot z = (\bar f_\theta(z,u) - z)/\Delta t$, with $\bar f_\theta$ the ensemble mean. Target set: $\{z : \ell_\psi(z) < 0\}$. Latent grid: bounding box of $E_\phi(R(\mathcal{X}_{\text{eval}}))$ plus 10% margin, at **101 x 101** for confirmatory runs. 51 and 201 are used for convergence only.
- Margin head $\ell_\psi$ is trained by regression on the privileged margin $l(x)$ over all training frames. The failure region is assumed to lie outside $G$ (checked by a unit test).

### 4.3 Static certifiers (disturbance inside the latent HJ problem)
Latent axes are arbitrary, so the bound is an isotropic ball in latent space.

| ID | Certifier | Bound $d(z,u)$ | Knob grid |
|---|---|---|---|
| E | Ensemble-$k$ | $k\,\sigma_{\text{ep}}(z,u)$ (ensemble std of predicted next latent) | $k \in \{0, 0.5, 1, 1.5, 2, 2.5, 3\}$ |
| L | Distance-aware | $\varepsilon_0 + \lambda \hat L\,\text{dist}((z,u), \mathcal{D}_{\text{train}})$ | $\lambda \in \{0, 0.5, \dots, 3\}$ |
| C0 | Naive split conformal | $\hat q_{1-\alpha}$ of calibration residuals | $\alpha \in \{0.3, 0.1, 0.05, 0.01\}$ |
| C1a | Normalized conformal (by $\sigma_{\text{ep}}$) | as v1 | same $\alpha$ grid |
| C1b | Normalized conformal (by dist) | as v1 | same $\alpha$ grid |
| C2 | Leave-region-out conformal | calibrate on a window of width $w$ away from $G$ | same $\alpha$ grid |

- Residual: $\|f_\theta(z_t,u_t) - E_\phi(o_{t+1})\|_2$, the gap between the model's predicted next latent and the latent inferred from the next observation.
- $\text{dist}$: Euclidean in the joint $(z,u)$ space, each axis divided by its range. $\hat L$ and $\varepsilon_0$ as in v1 Section 3.
- The coarser knob grid than v1 is a compute decision made now, not after seeing results. **Operating points used in tables are fixed in advance:** $k=2$, $\lambda=2$, $\alpha=0.05$.

### 4.4 Online certifier (closed-loop metrics only)
- **ACI-style (O-ACI).** Score $s_t = \|f_\theta(z_{t-1},u_{t-1}) - z_t\|_2$. Radius $r_t$ = empirical $(1-\alpha_t)$-quantile of the last $W=50$ scores, where $\alpha_{t+1} = \alpha_t + \gamma(\alpha - \text{err}_t)$ with target $\alpha = 0.1$, step $\gamma = 0.01$, and $\text{err}_t = \mathbf{1}[s_t > r_t]$. Initial window = calibration scores.
- Volume metrics (FSV, FUV) are **not** defined for O-ACI because its radius depends on the rollout history. It is evaluated by warning margin, detection rate, FAR and crash rate only.

---

## 5. Gates (checked on development seeds 0-9 only, before confirmatory runs)

- **G1 (pipeline works).** In QA at $k=0$, mean $\text{FSV}_z \le 0.03$ **and** mean $\text{FUV}_z \le 0.03$ over development seeds. If not, D1 stops. An amendment changing the architecture or solver may be tested **only** on development seeds.
- **G2 (patch only makes things worse).** Numerically verify $\mathcal{S}^*(m{=}1) \subseteq \mathcal{S}^*(m{=}0)$ on the 201 x 201 grid. If it fails, ground truth becomes $\mathcal{S}^*(0) \cap \mathcal{S}^*(1)$ by amendment.
- **G3 (the gap bites).** In QB and QC at $k=0$, mean $\text{FSV}_z \ge 3 \times \text{floor}_z$, where $\text{floor}_z$ = mean $\text{FSV}_z$ of QA at $k=0$ on the same seeds. If it fails, patch parameters may be changed by amendment using development seeds only.
- **Rule:** the gates are the only sanctioned reasons to modify patch, architecture or solver. Changes are logged with date and tag `prereg-v2.N` **before** confirmatory seeds are touched.

---

## 6. Hypotheses and analysis plan

### 6.1 Primary (confirmatory, one test)

**v2-P2.** Ensemble-$k$ inflation helps less when the model is *confidently wrong inside data it has seen* (QC) than when it is *in unseen territory* (QB).

- **Estimand.** $R_E(Q) = \dfrac{\overline{\text{FSV}}_{z,k=0}(Q) - \overline{\text{FSV}}_{z,k=2}(Q)}{\overline{\text{FSV}}_{z,k=0}(Q)} \times 100\%$ for each quadrant $Q$ (ratio of means over seeds). Contrast: $\Delta_R = R_E(\text{QC}) - R_E(\text{QB})$.
- **Interval.** BCa bootstrap, 10,000 resamples, seed indices resampled jointly across quadrants, 95%. RNG seed **20261028**.
- **Evaluability rule.** P2 is evaluated only if G3 holds on the confirmatory seeds for both QB and QC. Otherwise it is recorded as "not evaluable".
- **Decision rule, fixed now:**
  - **Supported:** upper bound of the $\Delta_R$ interval $< 0$.
  - **Inconclusive:** point estimate $< 0$ but upper bound $\ge 0$.
  - **Not supported:** point estimate $\ge 0$.
- **Always reported with P2:** $R_E$ in all four quadrants with intervals, and the full $k$-curve for each.

### 6.2 Secondary (pre-specified, 95% BCa, no multiplicity correction, labelled as secondary)

- **v2-S1. The 2 x 2 table.** For each quadrant and certifier at the operating points of Section 4.3: FSV$_z$, FUV$_z$, Gap-FSV$_z$, $V_z$ (overall and in $G$), AUPC (v1 definition, nominal = E at $k=0$ in the same quadrant and seed), and FUV at $\rho \in \{0.5, 0.25, 0.1\}$.
- **v2-S2. Directional predictions (reported either way).**
  - (a) QA: every certifier's FSV$_z$ is within 2x $\text{floor}_z$. This is a control.
  - (b) QB: C0's containment violation inside $G$ exceeds its nominal $\alpha$, and C0's AUPC is worse than C2's and L's.
  - (c) *Mechanism check for P2:* mean $\sigma_{\text{ep}}$ inside $G$ is smaller in QC than in QB.
  - (d) QC: C0 at $\alpha \in \{0.05, 0.01\}$ gives lower FSV$_z$ than E at $k=2$.
- **v2-S3. Warning margin.** On the shortcut-seeking policy (goal behind the gap) in each quadrant: detection rate, miss rate, FAR, and median margin among detections, for E, L, C0, C1a, C1b, C2, O-ACI. Summary: detection rate and median margin **at FAR = 5%**, interpolated from the operating curves traced by each certifier's knob. Prediction: in QC, O-ACI's detection rate at FAR = 5% exceeds E's.
- **v2-S4. Closed-loop crash rates.** v1 S6 protocol (goal behind gap, three-policy family, shield from each certifier) repeated in latent space. Report Spearman $\rho_s$ and Kendall $\tau$ between FSV$_z$ (and visitation-weighted FSV$_z$) and crash rate, with seed-bootstrap intervals. Labels as v1: $\ge 0.7$ strong, $0.4$-$0.7$ moderate, $<0.4$ weak. Descriptive, not a significance test.
- **v2-S5. Latent penalty** (Section 2.3) per quadrant and certifier.
- **v2-S6. Latent geometry diagnostic.** Spearman correlation between latent distance and true-state distance over 5,000 random evaluation pairs, and FSV$_z$ by distortion quartile.

### 6.3 Exploratory (must be labelled as such in any writeup)
UNISafe-style certifier, pixel Dubins (D3), $d_z > 2$ with an RL-style solver (D4), training-set-size and gap-width sweeps in latent, additional policies. Any of these needs its parameters registered as an amendment **before** the run that reports it.

### 6.4 Stopping rule
If P2 is "not supported" **and** v2-S1 shows no visible certifier-by-quadrant pattern (no certifier changes rank between quadrants), D3 and D4 are not pursued under this registration. The writeup reports the negative result.

---

## 7. Controls (must pass or be reported as failed)

- **C2.1 Floor.** $\text{floor}_z$ (Section 5) is reported with every latent FSV. Any FSV$_z$ below $3\times \text{floor}_z$ is flagged "not distinguishable from pipeline error".
- **C2.2 Corrupted world model.** Latent dynamics with shuffled training pairs must give FSV$_z$ at least 10x $\text{floor}_z$. If not, the verifier cannot detect a broken model.
- **C2.3 Latent grid convergence.** FSV$_z$ at 51, 101, 201 for the primary quadrants. Pass if $|\text{FSV}_{101} - \text{FSV}_{201}| \le 25\%$ of $\text{FSV}_{201}$. Otherwise report on the 201 grid with a warning.
- **C2.4 Benign-drag control.** In QB geometry, replace the patch with pure drag. Prediction: FUV rises and FSV does not.
- **C2.5 Encoder folding.** Fraction of evaluation cells whose nearest latent neighbor is not within 3 grid steps in $(p,v)$ must be $\le 5\%$. Otherwise G1 is treated as failed.

---

## 8. Exclusions and failed runs

- No seed is excluded post hoc.
- **Failed run** = NaN loss, solver failure, or encoder collapse (latent variance below the config threshold). Failed runs are counted and reported. Each primary analysis is shown twice: failed seeds dropped, and failed seeds included at worst-case FSV.
- A confirmatory seed is never rerun after its result is seen, except for infrastructure failure (killed process, full disk), documented in the run log.

---

## 9. Multiple comparisons

One primary test (P2). Everything in Section 6.2 is secondary and described as such. No exploratory result is called "confirmed".

---

## 10. Software and provenance

- Dependency versions per `requirements.txt` at the tagged commit. Every run writes: run ID, git commit, `prereg-v2` tag hash, seed list, config hash, latent grid size, and condition.
- **The confirmatory guard is a single shared function** called by **every** script that accepts a seeds argument, including benchmark, closed-loop, and analysis scripts. It refuses seeds 3000-3029 unless the `prereg-v2` tag exists and the working tree is clean. A test asserts that every script accepting seeds calls it. (In v1 one script touched confirmatory seeds outside the guard path; this closes that hole.)
- The viewer and the ledger read provenance fields from run outputs. A regression test recomputes FSV$_z$ from exported grids and compares it to the ledger.

---

## 11. Ledger entry format

Each claim carries: ID (prefix `v2-`), estimand, estimate with BCa interval, n, grid sizes, run ID, commit, tag, condition, and **status** in {`confirmatory`, `secondary`, `exploratory`, `not evaluable`, `failed`}.

---

## 12. Amendments and deviations

After tagging this file is append-only. Each amendment records its date, reason, whether any confirmatory result had been seen, and a tag `prereg-v2.N`. Anything changed after seeing a confirmatory result is labelled **post hoc** wherever the result appears.

### Amendment log
_(empty)_

---

## Before you commit (checklist)

- [ ] Fill author and commit hash.
- [ ] Create `configs/latent_di.yaml` and check that every "inherited" setting in Section 4.1 exists there.
- [ ] Implement the shared confirmatory guard (Section 10) and its coverage test.
- [ ] Confirm the failure region lies outside $G$ (unit test) and that $R$ matches Section 2.2.
- [ ] Run G1-G3 on development seeds 0-9 **before tagging** only if you want the gates settled first. If any gate forces a change, make it, then tag. Do not look at seeds 3000-3029.
- [ ] Commit, then `git tag -a prereg-v2 -m "Pre-registration v2"`, push the tag, and only then run seeds 3000-3029.
