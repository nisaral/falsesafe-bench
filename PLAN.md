# PLAN: Learning and safety auditing for simulated robots, policies, and populations

Author: Keyush Nisar. Plan written 2026-10-25 (IST). Horizon: 6 weeks (Oct 26 to Dec 6, 2026), ahead of the Dec 15 deadlines (UVic, MBZUAI, McGill).
Status markers: [READ] = abstract read on arxiv.org on 2026-10-25. [TITLE] = only title seen. [UNVERIFIED] = not checked.

Scope expanded with human approval on 2026-10-25 (IST). The six-week horizon remains the scientific core, not a commitment to finish every extension. The subsequent human request authorises starter implementation and GitHub setup/push: executable diagnostics and numerical primitives now exist (see START_HERE.md). Full training, safety-filter integration, protected evaluation, and reported extension studies remain gated; this is not a completed research result.

## 1. Motivation
HJ reachability gives a safe set only if the dynamics model is right. In practice the model is learned from finite, unevenly covered data. A learned model that is confidently wrong in an unvisited region can make the computed safe set larger than the true one. That is the dangerous error ("false-safe"). The question is how that error grows with data size and with coverage gaps, and whether the model's own uncertainty, used as an HJ disturbance bound, closes it.

This links to my prior result ("one statistic, two unknowns"): a single ensemble-disagreement statistic mixes epistemic and aleatoric uncertainty. For reachability the two should be treated differently. Epistemic error is reducible and depends on where we have data. Aleatoric noise is a true disturbance. Conflating them should give either false-safe regions or excess conservatism, and this can be measured exactly on systems with known ground truth.

The broader agenda is a shared simulated-robot lab: robots collect trajectories, learn a dynamics model and a task policy, and act with or without an HJ safety filter. A fleet supports population-flow and incentive experiments, while symmetry/physics constraints and emergent behaviours provide explicit ablations or exploratory demos. Each topic needs its own question and baseline; a polished animation is not a substitute for a result.

## 2. Thesis (one paragraph)
On low-dimensional systems with known dynamics, we can compute the exact false-safe volume of an HJ safe set built from a learned model. We hypothesise that (a) a nominal learned model (no uncertainty) produces false-safe volume concentrated in coverage gaps; (b) inflating the HJ disturbance by ensemble or GP epistemic uncertainty removes most of it at a measurable cost in conservatism; and (c) using a single blended uncertainty statistic, instead of a separated epistemic bound plus aleatoric disturbance, is measurably worse on at least one of the two axes. The secondary piece asks the same "what is identifiable from behaviour" question for incentives (QRE inverse games), where scale ambiguity is the analogue of the blended-statistic problem.

## 3. Falsifiable hypotheses (fixed; also in SCOPE.md)
| ID | Hypothesis | Refuted if |
|---|---|---|
| H1 | Nominal learned-model safe sets have nonzero false-safe volume (FSV) that falls with data size N, and is larger when a coverage gap exists | FSV is statistically indistinguishable from 0 (95% CI includes 0) at every N for both systems, or gap vs no-gap CIs overlap at every N |
| H2 | Epistemic-inflated HJ (disturbance = k * ensemble std or GP std) reduces FSV by at least 50% vs nominal at matched N | Reduction < 50% on mean FSV, or CIs overlap, for both Dubins and double integrator |
| H3 | Inflation has a conservatism cost: false-unsafe volume (FUV) rises vs nominal | FUV does not rise (then H2 gains are free and that itself is the result) |
| H4 | Separated epistemic+aleatoric bound beats a blended single-statistic bound on FSV at equal FUV (or vice versa) when observation noise is heteroscedastic | No difference beyond CI on either metric under heteroscedastic noise |
| H5 (stretch, games) | In a 2-player QRE game, the feasible-reward set estimated without knowing temperature is a ray (scale ambiguity) and collapses to a point under the normalisation of 2511.05640 | Estimated sets do not show the predicted shape in simulation |

## 4. Week-by-week milestones
| Week | Dates (IST) | Deliverable | Done criteria |
|---|---|---|---|
| W1 | Oct 26 to Nov 1 | Ground-truth pipeline: double integrator + Dubins car BRTs with hj_reachability; analytic DI test; FSV/FUV metric code; data collection with controllable coverage gap | `make test` green; DI matches analytic (already passing on box, see results/); Dubins converges under grid refinement (101 vs 151 cells change < 1% set volume) |
| W2 | Nov 2 to Nov 8 | Learned models: 5-member MLP ensemble (PyTorch), exact GP (gpytorch) on residual dynamics; recovery test on known system | Ensemble/GP one-step RMSE on held-out in-coverage data within 2x noise std; shuffled-label control gives near-chance fit |
| W3 | Nov 9 to Nov 15 | H1: nominal FSV vs N (5 sizes) x gap/no-gap, 5+ seeds | Plot results/figs/fsv_vs_N.png, every point in claims ledger |
| W4 | Nov 16 to Nov 22 | **MVR**: H2/H3 epistemic-inflated HJ, FSV and FUV tradeoff curve over k. Short arXiv-style note + README results | MVR checklist in SCOPE.md met; numbers only from runlog |
| W5 | Nov 23 to Nov 29 | H4: heteroscedastic noise; blended vs separated uncertainty. After MVR sign-off, build a logged robot replay and begin the bounded E6 single-robot RL pilot | E3 plots + ledger or negative result; no RL work that bypasses the core gates |
| W6 | Nov 30 to Dec 6 | Freeze the reproducible scientific core and workshop note. Deliver the E6 learning-robot demo if its gates pass; start E4/E7 fleet work only if E6 is verified | Core reproduces from a clean clone with `make reproduce`; demo traces to real checkpoints/runs, or its deferral is documented |

After W6: fleet scaling, OT/bridges/MFG, inverse games, geometric/PINN ablations, emergent swarms, and DeepReach remain in scope with separately reviewed schedules and budgets. Do not present this queue as completed six-week work.

## 5. Experiments table
| Exp | Environment | Learned model | Metric | Baseline(s) | Expected plots |
|---|---|---|---|---|---|
| E0 | Double integrator, failure p<0 | none (true) | max abs value error, IoU vs analytic | analytic formula | value slice, error heatmap |
| E1 | DI + Dubins car (obstacle avoid), N in {100,300,1k,3k,10k}, gap / no gap | MLP ensemble mean, GP mean | FSV, FUV (fractions of grid volume) | true-dynamics BRT; nominal model | FSV vs N, gap vs no-gap |
| E2 | same | ensemble std, GP std as disturbance bound, k in {0.5,1,2,3} | FSV, FUV | nominal; constant global disturbance tuned to match FUV | FSV-FUV Pareto curve |
| E3 | DI + Dubins with heteroscedastic noise | blended single std vs separated epistemic + aleatoric | FSV at matched FUV | E2 inflated | bar chart with CIs |
| E4 (fleet extension) | 3 Dubins agents pairwise | ensemble | FSV per pair, compute time | nominal | per-pair FSV; pairwise results are not a joint fleet guarantee |
| E5 (inverse-game extension) | 2x2 / 3x3 QRE game (numpy) | NLS estimator | set Hausdorff error vs samples | known-temperature estimator | feasible set plot; do not assume PPO behaviour is QRE |

Metric definitions are fixed in SCOPE.md section 3.

## 6. Minimum-viable result (end of W4) and staged extensions
- MVR: "On double integrator and Dubins car, the nominal learned-model HJ safe set has false-safe volume X (CI) that concentrates in the coverage gap; epistemic inflation reduces it by Y at a cost Z in false-unsafe volume." With whatever X, Y, Z the runs give, including "no reduction". Citable as a GitHub repo + short note.
- First showcase: an actual trained navigation policy, evaluated with no shield, a nominal learned-model shield, an uncertainty-aware shield, and a labelled true-dynamics oracle shield. Compare the same frozen policy before considering separate shielded-training ablations.
- Next: the 3-agent pairwise safety benchmark and cooperative/competitive fleet policies; then the population, incentive, modelling-bias, and emergent-behaviour studies in section 13. H4 and the original H5 remain unchanged. DeepReach is optional, not a replacement for the verified grid core.
- Public framing: auditing the reliability of safety certification under learned dynamics. Do not claim that uncertainty or a successful robot video establishes unconditional safety.

## 7. Related work
| ID | Title | Status | What it covers | Gap left for us |
|---|---|---|---|---|
| 1705.01292 | A General Safety Framework for Learning-Based Control in Uncertain Robotic Systems (Fisac et al.) | [READ] | HJ reachability with Bayesian (GP) disturbance-bound updates; quadrotor | Not a systematic false-safe-volume vs N / coverage study; not ensemble; not epistemic/aleatoric split |
| 2505.07594 | Finite-Sample-Based Reachability for Safe Control with GP Dynamics | [READ] | Sample-complexity result: reachable set from finitely many GP posterior samples; GP-MPC with guarantees | GP-specific, MPC not HJ grid; no neural ensemble |
| 2407.21029 | Data-Driven Abstractions via Binary-Tree GPs for Formal Verification | [READ] | BTGP gives interval Markov chain abstraction, bounds on reachability probability, robust to aleatoric+epistemic | Abstraction-based, stochastic temporal logic; not HJ, not ensembles |
| 2609.34300 | When World Models Lie: Adaptive Safety Analysis Under Wrong Imaginations | [READ] | Latent HJ safety filters; states ensemble disagreement is already used as an adaptation signal and can stay small when the model is wrong; uses adaptive conformal inference | **Closest. Ensemble disagreement for HJ is NOT new.** Our angle must be exact ground-truth FSV measurement in low-dim, coverage gaps, and epistemic/aleatoric separation |
| 2312.08604 | Verification of Neural Reachable Tubes via Scenario Optimization and Conformal Prediction | [READ] | Probabilistic guarantees for DeepReach tubes | Assumes dynamics known; error is from the neural value, not the learned model |
| 2203.12407 | Verification of safety critical control policies using kernel methods | [READ] | GP model of value-function error in HJ | Models value error, not dynamics |
| 2011.02082 | DeepReach | [READ earlier pass] | Neural HJ solver for high-dim | Known dynamics |
| 2511.05640 | Blind Inverse Game Theory | [READ] | QRE with unknown temperature: scale ambiguity, normalisation for identification, O(N^-1/2) NLS, Markov games | Our H5 only reproduces this in simulation; no novelty claim there |
| 2602.04397 | Optimal Rates for Feasible Payoff Set Estimation in Games | [READ] | Minimax rates for feasible payoff set (Hausdorff) under Nash play | Nash, not QRE; we would only use as a baseline concept |
| 2610.28106 | TD Learning of Probabilistic Reach-Avoid Value Functions | [TITLE + first lines] | probabilistic reach-avoid via TD | not checked in depth |
| 2510.18195 | Ensemble based closed-loop optimal control using PINNs | [TITLE + first lines] | ensemble PINN for HJB | control design, not safety under learned dynamics |

Search notes (arXiv abstract search, loose matching, counts rough): "reachability" "ensemble" "learned dynamics" = 0; "Hamilton-Jacobi reachability" "epistemic" = 0; "Hamilton-Jacobi" ensemble learned dynamics = 2. Semantic Scholar was rate-limited (429), so no citation-graph check. Older robust/learning HJ work (for example Berkenkamp safe-RL with GPs, Akametalu et al. 2014) is [UNVERIFIED] here and must be checked in W1.

## 8. Reading list (ordered; links returned the page on 2026-10-25 unless marked)
1. hj_reachability README and examples: https://github.com/StanfordASL/hj_reachability
2. Bansal, Chen, Herbert, Tomlin, "Hamilton-Jacobi Reachability: A Brief Overview and Recent Advances": https://arxiv.org/abs/1709.07523 [UNVERIFIED link this session]
3. Fisac et al. 1705.01292: https://arxiv.org/abs/1705.01292
4. 2609.34300: https://arxiv.org/abs/2609.34300
5. 2505.07594: https://arxiv.org/abs/2505.07594
6. DeepReach 2011.02082: https://arxiv.org/abs/2011.02082 and https://github.com/smlbansal/deepreach
7. 2312.08604: https://arxiv.org/abs/2312.08604
8. Rasmussen and Williams, GPML ch. 2: https://gaussianprocess.org/gpml/ [UNVERIFIED link this session]
9. Lakshminarayanan et al., Deep Ensembles, 1612.01474 [UNVERIFIED link this session]
10. 2511.05640 and 2602.04397 (for the games stretch)
11. McKelvey and Palfrey 1995, QRE (journal paper; no link verified)

## 9. Math primer
- **Dynamics:** x' = f(x,u,d), u in U (control), d in D (disturbance).
- **Failure set and l(x):** choose l with l(x) < 0 exactly on the failure set (for DI, l = p).
- **Value function (BRT, avoid):** V(x,t) = max over control strategies, min over disturbances, of min over s in [t,0] of l(x(s)). The state is safe on the horizon iff V >= 0.
- **HJ PDE:** dV/dt + min{0, H(x, dV/dx)} = 0, V(x,0)=l(x), with H = max_u min_d dV/dx . f(x,u,d). The min{0,.} (hj.solver.backwards_reachable_tube) makes it a tube (failure at any time) rather than a set at exactly time T.
- **Worked DI check:** p''=u, |u|<=1, l=p. Best u = +1. p(t)=p+vt+t^2/2 is minimised at t*=clip(-v,0,T), so V = p + v t* + t*^2/2. Our test compares the solver to this; see results/.
- **Learned model in HJ:** replace f with mean model mu(x,u) and add disturbance |d_i| <= k * sigma_i(x). Note: hj_reachability Box disturbance sets are state-independent, so a state-dependent bound needs either a custom Hamiltonian or a d in [-1,1] with a state-dependent disturbance_jacobian = diag(k*sigma(x)). The second fits the API.
- **GP basics:** prior f ~ GP(0,k). With data (X,y), noise s^2: mean k*^T(K+s^2 I)^-1 y, variance k** - k*^T(K+s^2 I)^-1 k*. Posterior variance is epistemic only; s^2 is aleatoric.
- **Ensembles:** M networks; spread of means approximates epistemic, mean predicted variance approximates aleatoric (if heteroscedastic heads). The sum is the blended total.
- **QRE:** player i plays pi_i(a) proportional to exp(Q_i(a)/tau). Scaling reward by c and tau by c gives the same behaviour, hence the scale ambiguity in 2511.05640.

## 10. Risk register
| Risk | Likelihood | Fallback pivot |
|---|---|---|
| JAX learning curve slows W1 | Medium | Use hj_reachability only as a solver; all learning stays in PyTorch, pass numpy arrays |
| State-dependent disturbance not easy in hj_reachability | Medium | Use disturbance_jacobian = diag(k*sigma(x)) trick; else evaluate sigma on a grid lookup inside dynamics |
| FSV is ~0 everywhere (H1 refuted) | Low-Med | Report it; make the gap larger or system harder (Dubins with fast turn), clearly labelled as a new experiment |
| Prior work already covers the full study | Medium (2609.34300 is close) | Reframe as exact-ground-truth benchmark + epistemic/aleatoric split; cite honestly |
| Dubins grid too slow on CPU | Low | 3D grid 101^3 is fine on CPU; else GPU/Colab |
| Games stretch eats time | Medium | Drop it; it is optional |
| Broader scope crowds out the scientific core | High unless staged | Keep E0 to E3 first; prioritise one robot demo, and explicitly defer unrun extensions |
| RL or fleet training exceeds the current budget | Unmeasured | Run only an approved bounded pilot; report runtime and request a new budget rather than silently running longer |
| A shield changes the commanded action | Methodological | Log policy-commanded and executed actions separately; never compute PPO likelihoods as though shield-modified actions were sampled directly from the raw policy |
| Pairwise controls conflict or leave the certified domain | Methodological | Detect infeasibility/domain exits and report them; no automatic global fleet-safety claim |
| OT, bridges, games, and physics constraints become decorative labels | Methodological | Require a distinct numerical formulation, baseline, held-out check, and explicit limitations for each |

## 11. Write-up plan
- Workshop paper outline (4 pages): 1 Intro (false-safe sets from learned models); 2 Setup and metrics (FSV, FUV, exact ground truth); 3 Methods (nominal, inflated, separated); 4 Results (E1 to E3); 5 Limits (low dim, grid solver, simulated noise); 6 Related work (table above).
- Candidate venues: robotics/safe-learning workshops (check deadlines; [UNVERIFIED]).
- README: one-paragraph result, reproduce command, figure, link to claims ledger.
- Honest citing in outreach:
  - Before W3: "I am building a benchmark that measures false-safe volume of HJ safe sets built from learned dynamics" (no results claimed).
  - After MVR: "Preliminary result (5 seeds, 2 low-dim systems): ..." with repo link, numbers copied from claims_ledger.md only.
  - After draft: "Workshop draft under preparation" only if it is actually drafted; never "accepted" or "submitted" until true.
  - Always state that ensemble disagreement for HJ safety is already used (2609.34300) and what is different here.

## 12. Compute budget
- DI 2D grid 101^2 and Dubins 3D grid 101^3 retain the SCOPE.md per-solve limits. Actual starter solve timings belong in logged snapshot manifests; earlier setup-box timings are not a portable runtime guarantee.
- E1 to E3 require the full reviewed system/data/gap/seed/inflation schedule. Estimate the whole-study runtime from verified runs before launching it; do not silently lower grid sizes or seed counts to fit an estimate.
- Ensembles/GPs: small (thousands of points); exact GP up to N=10k needs GPU or sparse GP; budget 2 GPU-hours.
- DeepReach stretch: several GPU-hours per system on the 24 GB card [UNVERIFIED].

Extension runtimes are not yet measured. Keep the existing SCOPE.md budgets for initial pilots; obtain human approval for any larger experiment, new package, GPU service, or high-dimensional neural solver. Animation rendering does not justify reducing experimental seeds or hiding failed runs.

## 13. Expanded research tracks and proposed experiments
These are future experiments, not new results or changes to H1 to H5. Proposed metric names below are specification targets only: their exact aggregation, horizons, seed schedule, and edge cases require human review before evaluation code is written or final runs start. Core FSV/FUV/IoU/Gap-FSV definitions stay fixed.

| Exp | Question and method | Required comparison/check | Proposed outputs |
|---|---|---|---|
| E6: learning robot | Can a small navigation policy use a learned-dynamics HJ shield without losing all task progress? Start with a PPO baseline and one fixed-speed Dubins robot; broader on/off-policy, offline, preference, and grouped comparisons are specified in ALGORITHMS.md | Same frozen policy: no shield, nominal shield, uncertainty-aware shield, true-dynamics oracle shield; scripted controller baseline. Separate shielded-training and algorithm comparisons only at reviewed matched budgets | Collision and goal-reach fractions over a fixed horizon, interventions, time to goal, model error, and logged replay |
| E7: learning fleet | How do MAPPO/shared and independent PPO policies and local shields behave for cooperative goal assignment or competitive crossing? Begin with the E4 three-agent setting; critic/observation contracts require review | Scripted agents; unshielded and shielded matched policies; shared vs independent learners. Include deliberate pairwise-control conflicts; evaluate equilibrium claims separately | Fleet and per-agent collision/goal statistics, deadlock and intervention counts, pairwise FSV, and trajectories |
| E8: population OT | Can entropic OT between observed fleet-position distributions explain held-out intermediate population snapshots? Use POT and simulator-generated data | Static/linear interpolation and known-simulator propagation; mass conservation and marginal residuals. Snapshot matching alone does not identify individual trajectories or incentives | Population discrepancy on held-out times, transport cost, residuals, and density/flow replay |
| E9: Schrodinger bridge | Infer a finite-horizon stochastic path distribution between population marginals relative to an explicitly declared reference dynamics | Known-reference propagation and E8; check endpoint constraints, intermediate marginals, process-noise assumptions, and path-space objective. An OT interpolation alone is not a bridge | Endpoint/intermediate errors, numerical residuals, and sampled path replay |
| E10: mean-field game | Solve a separate small discrete congestion game with a representative agent coupled to the population distribution | Independent best response to a frozen population and a cooperative planner; check best-response/distribution consistency and compare against a finite-agent simulation | Equilibrium consistency residuals, congestion/cost summaries, and density evolution; no equilibrium claim from a crowd animation alone |
| E11: physics/geometric learning | Do correct body-frame features, equivariance, or physics-informed/PINN constraints reduce the data needed for useful learned-dynamics safety sets? | Same data/model budget, unconstrained learner, augmentation-only baseline, and a deliberately incorrect constraint. Transform goals/obstacles as well as states; do not leak unknown true residual dynamics into the learner | Held-out model error and core FSV/FUV vs data size, with a constraint-mismatch negative control |
| E12: emergent swarm | Can the same robot arena exhibit flocking/foraging patterns and expose local-shield coordination failures? | Scripted flocking/foraging vs learned coordination; retain collision/domain/conflict logs. Exploratory behaviour is not a new artificial-life engine or a safety theorem | Labelled swarm replay; reviewed cohesion/task/safety summaries only if promoted to a reported experiment |

E4 supplies the initial fleet safety study and E5 supplies the original QRE incentive-identifiability study. RL behaviour and QRE-generated behaviour must remain distinct datasets unless an explicitly reviewed model links them.

Starter implementations now live in src/lcert/envs/, rl/, population/, games/, constraints/, and visualization/, with entry points under scripts/. Learned-model safety filters and full study/training runners are not yet implemented. Use existing PyTorch/NumPy/POT/Matplotlib pins; external RL/simulator/graphics frameworks require separate dependency approval.

The starter runlog schema reserves E0 to E12 plus sanity/simulation/data identifiers. Schema acceptance is not scientific approval: E1 to E12 remain blocked by the CLI until their protocol/evaluator/training gates are implemented and reviewed. Scripted robot and data diagnostics are not relabelled E6/E7 studies. Additional learned checkpoint/rollout provenance still needs review.

Broader algorithm exploration is approved as a direction, not a result: REINFORCE/A2C/PPO, robot-action DPO/GRPO-style objectives, behaviour cloning, SAC/TD3, IQL/CQL, and MAPPO/independent PPO. Objective functions, replay, and policy-network primitives exist where explicitly marked in ALGORITHMS.md; complete trainers and matched comparisons do not. DPO/GRPO are proposed robot-action adaptations, not new language-model training or established robot-safety methods.

## 14. Flagship demonstration and delivery gates
Working title: Learn-and-Shield Robot Lab. The full visual/artifact specification is in SIMULATION_DEMOS.md.

1. Ground-truth gate: original sanity checks, reviewed metrics, and Dubins convergence.
2. Model gate: recovery, shuffled-label control, fixed data splits, and E1/E2 results with the mandatory baselines.
3. MVR gate: human review of the scientific core; no robot RL training before this gate.
4. Single-robot gate: verified environment and actual PPO checkpoints, frozen-policy shield comparisons, logged failures, and a replay generated from the same runs.
5. Fleet gate: independent replay and constraint-conflict checks; larger populations do not increase the HJ grid state dimension.
6. Research-extension gate: select one of population/incentive/modelling tracks, review its formulation, metrics, and budget, and report negative/null outcomes too.

The presentation should show robots really learning and moving: early/intermediate/final checkpoint replays, trajectory trails, true vs learned safe-set overlays, uncertainty heatmaps, visible shield interventions, and a coverage-gap failure case. Freeze checkpoint-selection and showcase-seed rules using training/validation data before final test access; publish all evaluated runs, not only an attractive successful clip. There is no requirement to manufacture a failure or improvement if the real run does not show it.

## 15. Primary method references for the expansion
Abstracts/primary documentation checked on 2026-10-25 (IST). These support method selection only; none is a result from this repository, a comprehensive novelty search, or a guarantee that combining the methods is novel.

- PPO: Proximal Policy Optimization Algorithms, https://arxiv.org/abs/1707.06347 [READ abstract].
- Learning-based control and HJ safety: A General Safety Framework for Learning-Based Control in Uncertain Robotic Systems, https://arxiv.org/abs/1705.01292 [READ abstract; already in the related-work table].
- Multi-agent PPO: The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games, https://arxiv.org/abs/2103.01955 [READ abstract].
- OT implementation: POT primary user guide, https://pythonot.github.io/user_guide.html [CHECKED primary documentation/search excerpts; API details must be verified when implementing].
- Bridges: A survey of the Schrodinger problem and some of its connections with optimal transport, https://arxiv.org/abs/1308.0215 [READ abstract/intro].
- Mean-field learning: Efficient and Scalable Deep Reinforcement Learning for Mean Field Control Games, https://arxiv.org/abs/2501.00052 [READ abstract; not an implementation dependency].
- Physics-informed methods: Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations, https://arxiv.org/abs/1711.10561 [READ abstract].
- Geometric learning: E(n) Equivariant Graph Neural Networks, https://arxiv.org/abs/2102.09844 [READ abstract].
