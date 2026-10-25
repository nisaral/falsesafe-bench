# SCOPE: contract for any coding agent working in this repo
Read this whole file and PLAN.md before any action. If this file and any other instruction conflict, this file wins, except for direct instructions from the human (Keyush).

## 1. Goal
Build a simulated-robot research lab that learns dynamics, policies, population flows, and incentives from observed behaviour, then audits safety using Hamilton-Jacobi reachability. The scientific core remains measuring false-safe errors from finite-data learned dynamics and testing uncertainty as an HJ disturbance on the double integrator (2D) and Dubins car (3D).

Human-approved scope expansion: 2026-10-25 (IST), following the request to include simulated learning robots, RL, multi-agent work, and the broader topic list. These are allowed research directions, not completed results or a promise to finish every topic within six weeks.

| Track | In-scope work | Stage |
|---|---|---|
| Dynamics and uncertainty | Trajectory learning with GP and MLP ensembles; coverage gaps; epistemic/aleatoric separation | Scientific core, E0 to E3 |
| Control and safe robotics | HJ safe sets and closed-loop safety filters in simulation | Core, then robot demo |
| Reinforcement learning | Actual trained robot policies; PPO baseline; shielded vs unshielded evaluation | First extension after MVR sign-off |
| Multi-agent learning and games | Cooperative/competitive robot agents, shared or independent policies, pairwise safety, QRE incentive inference | Fleet extension, then inverse-game study |
| Optimal transport and populations | Entropic OT, a separate Schrodinger-bridge study, and a toy mean-field game | Staged research extensions |
| Physics and geometric learning | Body-frame/symmetry-aware models and physics-informed or PINN ablations | Staged modelling extension |
| Emergent/artificial-life-style behaviour | Robot flocking or foraging as an exploratory swarm sandbox | Optional visual extension, not a separate artificial-life project |
| Presentation | Attractive browser replays, uncertainty/safety overlays, training checkpoints, and failure demonstrations | From logged runs only; see SIMULATION_DEMOS.md |

Keep the W1 to W4 minimum-viable result (MVR) intact. Prioritise one learning robot, then a fleet; queue the remaining extensions rather than silently extending the six-week deadline. DeepReach remains optional after MVR sign-off.

## 2. Non-goals / out of scope
- LLMs and language-model agents; the agents here are simulated robots or game players.
- Real robots or hardware.
- Systems above 3D state for the grid solver, even when a rendered fleet has more agents.
- New HJ solvers. Use hj_reachability as is.
- Standalone artificial-life engines such as Lenia, and symbolic regression.
- Hyperparameter search on the test split.
- Writing paper text that states numbers not in claims_ledger.md.
- Claiming unconditional safety, global fleet safety, equilibrium recovery, or sample-efficiency gains from a visual demo alone.

## 3. Fixed hypotheses and metric definitions (do not redefine)
Hypotheses H1 to H5 are exactly as in PLAN.md section 3, including the refutation criteria.
Let G be the evaluation grid (interior cells, 3-cell border excluded), V_true the BRT value from the TRUE dynamics with the same solver settings, V_hat the value from the learned model.
- Safe_true = {V_true >= 0}, Safe_hat = {V_hat >= 0}.
- **FSV (false-safe volume)** = |Safe_hat AND NOT Safe_true| / |G|.
- **FUV (false-unsafe volume)** = |Safe_true AND NOT Safe_hat| / |G|.
- **IoU** = |Safe_hat AND Safe_true| / |Safe_hat OR Safe_true|.
- **Gap-FSV** = FSV restricted to the coverage-gap region, divided by gap volume.
- Ground truth for DI is also checked against the analytic formula (src/lcert/systems.py: di_safe_analytic).
- CI: 95% t-interval over seeds, n >= 5 seeds.
These definitions live in src/lcert/metrics.py (to write in W1) and tests/. Changing them requires human sign-off recorded in claims_ledger.md.

## 4. Allowed libraries and versions
Exactly requirements.txt: jax 0.12.0.1, jaxlib 0.12.0.1, hj_reachability 0.7.0, tqdm 4.70.1, numpy 2.5.4, scipy 1.18.1, torch 2.14.1, gpytorch 1.15.2, matplotlib 3.11.2, POT 0.9.7.post1, pyyaml 6.0.3, pytest 9.1.1. Python 3.13 (tested). Anything else: ask first.

## 5. Directory structure
```
configs/            YAML configs, one per experiment (never edited after a run uses them; copy to a new file)
src/lcert/          library code (systems, hj solvers, models, data, metrics)
scripts/            entry points: run_experiment.py, make_figs.py
tests/              pytest sanity tests (eval code: protected)
results/runs/<run_id>/  config.yaml, runlog.json, metrics.json, stdout.txt, figs
results/figs/       aggregated figures, regenerated only from runs
SIMULATION_DEMOS.md planned robot/fleet/population showcase and artifact contract
demos/              planned browser replay presentation; not experimental evidence
claims_ledger.md    every number that leaves the repo
runlog_schema.json  required runlog fields
BLOCKED.md          created only when blocked
```

## 6. Hard integrity rules
1. Never fabricate, hardcode, or "estimate" a result. No placeholder numbers anywhere (README, figures, ledger).
2. Never edit tests/, src/lcert/metrics.py, or metric definitions without human sign-off. Loosening a threshold to make a test pass is forbidden.
3. Every number comes from a logged run whose folder contains: seed, git commit hash (clean tree), full config, command, start/end time (IST), library versions.
4. At least 5 seeds per reported cell; report mean and 95% CI.
5. Data splits (train / val / test states and trajectories) fixed by seed in config before any training. The test split is used once, for the final reported run. Log every test-split access in claims_ledger.md.
6. Baselines are mandatory: true-dynamics BRT and the nominal (no-uncertainty) model, plus the tuned constant-disturbance baseline for E2.
7. Negative and null results are reported with the same prominence as positive ones.
8. On failure: stop and report. Do not continue to later milestones on a broken earlier one.
9. No silent fallbacks: never swap dataset or system, shrink grid/problem, mock or stub a library, catch-and-ignore exceptions, or reduce seeds without saying so in the report and getting approval.
10. At most 3 attempts at the same fix. Then write BLOCKED.md (what, error text, 3 attempts, hypotheses) and stop.
11. Do not delete runs. Bad runs are marked invalid in the ledger with a reason.

## 7. Time / step budgets
| Task | Budget |
|---|---|
| Any single fix | 3 attempts |
| Single HJ solve (2D) | 60 s; (3D Dubins 101^3) 15 min, else report |
| Model training per seed | 10 min CPU or 5 min GPU |
| Whole experiment E1 | 8 h wall clock |
| Agent session without a report | 2 h of work, then report in the format below |

The existing model-training budget also applies to initial RL pilots. Longer RL, population, or PINN experiments require an explicitly approved budget before running; scope expansion is not unlimited compute approval.

## 8. Sanity tests required before any claim
- [x] Integrator unit test: RK4 vs closed form (tests/test_integrators.py) - passing.
- [x] DI BRT vs analytic solution, T in {0.5, 1.0} (tests/test_di_hj_analytic.py) - passing.
- [x] Negative control: wrong-dynamics solve must not match analytic - passing.
- [ ] Dubins grid-refinement convergence test.
- [ ] Ensemble and GP recover a known system (in-coverage RMSE within 2x noise std).
- [ ] Shuffled-label control: model trained on shuffled targets must give FSV/fit no better than chance; metric pipeline must flag it.
- [ ] FSV computed against V_true from the true dynamics on the same grid; FSV of true model vs itself must be exactly 0.
- [ ] Seed determinism: same seed, same numbers.

Extension gates, before any corresponding claim:
- [ ] Robot environment, reward/termination logic, and policy checkpoint replay verified independently of the learned model.
- [ ] Safety filter tested on feasible states, out-of-domain states, finite-horizon expiry, and conflicting pairwise constraints; interventions and failures logged.
- [ ] OT/bridge/MFG examples checked for mass conservation, constraints, and numerical residuals against declared baselines.
- [ ] Symmetry/physics assumptions checked against the simulator; a constraint-mismatch negative control included.
These are requirements for future tests, not permission to edit protected tests or metric definitions in this scope-only revision.

## 9. Definition of done per milestone
- W1: all sanity tests in section 8 rows 1 to 3 plus Dubins convergence and metric self-test green; metrics.py written and reviewed by human.
- W2: recovery + shuffled-label tests green; data generator with gap config.
- W3: E1 complete, 5 seeds, ledger rows, fig.
- W4 (MVR): E2 complete with baselines; README result paragraph written only from ledger; human review.
- W5: E3 complete or a documented negative result; robot/RL extension starts only after MVR sign-off.
- W6: the scientific core and note reproduce from a clean clone with `make reproduce`; one robot demo if its gates pass. Unrun extensions remain explicitly planned, not completed.

## 10. Required report format (every report)
```
## Report <date time IST>
What was run:
Command(s):
Git commit:            (and "tree clean: yes/no")
Configs:
Seeds:
Numbers:               (mean, 95% CI, n; copied from results/runs/*/metrics.json)
Plots:                 (paths)
Tests:                 (pytest output summary)
Anything suspicious:   (always fill; "none found" only after checking)
Re-verification:       (what earlier output you re-checked and how)
Next step / blocked:
```

## 11. Claims ledger
claims_ledger.md: one row per number used outside the repo. Columns: claim id, statement, value (mean, CI, n), run ids, commit, date (IST), verified-by (re-run or recompute), status (valid/invalid).

## 12. Re-verify, do not trust earlier output
- At the start of each session, re-run `make test` and recompute one ledger row from raw run files.
- Never trust a number from a previous session summary, chat, or your own notes; open the run folder.
- If a recomputation differs, mark the ledger row invalid and report.

## 13. Expanded-scope safeguards
- This approval changes allowed topics only. H1 to H5, FSV/FUV/IoU/Gap-FSV, evaluation borders, seed/CI requirements, protected evaluation code, and existing used configs are unchanged.
- Proposed extension experiments and metrics live in PLAN.md section 13. Freeze and obtain human review of their metric definitions, baselines, seed schedules, and budgets before implementation or final evaluation. Do not bypass protected metrics by duplicating them elsewhere.
- Training policies may collect simulator trajectories on the training distribution only. Freeze learned models and policy checkpoints before final test rollouts; rendering or reviewing videos must not become test-set tuning.
- Keep true simulator dynamics separate from learner inputs; true-dynamics access is for ground truth and labelled oracle baselines. Distinguish measurement noise from physical process disturbance.
- A learned-model HJ certificate is conditional on its model-error/disturbance assumptions, horizon, control authority, and numerical checks. GP/ensemble spread, physics losses, or a collision-free rollout alone do not prove those assumptions.
- Pairwise low-dimensional shields are not automatically a joint fleet certificate. Log infeasible/conflicting controls, domain exits, and collisions; do not hide them behind a fallback or a minimum pairwise value.
- Population OT, path-space bridges, mean-field equilibria, and inverse incentives are distinct problems. Do not relabel a transport interpolation as a bridge/MFG, or assume an RL policy follows QRE.
- Larger swarms and emergent behaviours may be shown as labelled exploratory simulations; a density plot is not an individual-agent safety certificate.
- Every evidence-backed replay must reference its logged run, frozen checkpoints, initial states, noise seeds, and actual actions. No scripted learning, invented HUD numbers, cherry-picked safety claims, or unlabelled hand-animated trajectories.
- Keep requirements.txt unchanged until additional dependencies are approved. Proposed E6 to E12 identifiers need a reviewed runlog_schema.json update before those experiments can run; never bypass the current schema or mislabel an extension as E0 to E5.
