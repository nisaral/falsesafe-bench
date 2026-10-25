# Simulation showcase: Learn-and-Shield Robot Lab

Status: the subsequent human starter request authorises implementation. A seeded single/fleet robot arena and standalone browser replay now exist, with scripted-controller trajectories, controls, events, and provenance. They are explicitly untrained and uncertified. The learned D0-D6 showcase, safety overlays, actual trained policy checkpoints, and reported population/fleet comparisons below remain planned. SCOPE.md is binding and PLAN.md defines the delivery gates; START_HERE.md documents the runnable base.

## 1. Flagship experience
A top-down robot arena that makes the scientific story visible: collect trajectories, learn dynamics with uncertainty, train a task policy, and compare its behaviour with and without a safety filter. Start with a single fixed-speed Dubins robot; reuse the arena for the initial three-agent fleet before considering larger configurable populations.

Use a polished but readable presentation: dark navy background, vector robot bodies with heading arrows, glowing trajectory trails, clearly outlined obstacles/goals, an explicit colour legend, and smooth replay controls. Appearance must never change the underlying recorded states, actions, or event times.

The main comparison uses the same frozen policy, obstacle layout, initial state, horizon, and declared noise-seed pairing across four panels:
- Unshielded RL policy.
- Policy with the nominal learned-model HJ shield.
- Policy with the uncertainty-aware learned-model HJ shield.
- Policy with a labelled true-dynamics oracle shield, kept separate from what the learner can observe.

Include the scripted-controller baseline in the run catalogue. Shielded training is a separate ablation, not quietly substituted for a frozen-policy comparison.

## 2. Visual features
- Real learning timeline: replay predetermined early, intermediate, and final training checkpoints; display the checkpoint and training seed. Do not animate a hand-scripted path and call it learning.
- Safety overlays: show true and learned safe-set slices, with false-safe cells visibly highlighted against ground truth. For Dubins, label the heading slice and time-to-go; a 2D slice is not the full 3D FSV.
- Uncertainty view: distinguish epistemic uncertainty, observation noise, and declared process disturbance. Do not label ensemble/GP spread as a validated bound without the corresponding check.
- Shield intervention: draw the policy-commanded action separately from the executed action; flash an intervention marker and show the logged reason.
- Coverage-gap view: display where training trajectories were absent and what the actual model/policy did there. Show failure or lack of improvement when that is what happened; do not force a dramatic result.
- Timeline controls: pause, scrub, playback speed, seed/checkpoint selection, and toggles between recorded overlays. A k slider must load an actual precomputed run, not invent a new result.
- Evidence HUD: run ID, policy/model checkpoint, simulation time, horizon, and event counts loaded from recorded artifacts. Separate per-episode counters from aggregate mean/CI values loaded from metrics.json.
- Failure visibility: retain collisions, domain exits, horizon expiry, deadlocks, and pairwise-control conflicts in the replay and run catalogue, even if they make the demo less flattering.

Use an accessible legend as well as colour. Playback interpolation may improve visual smoothness but must not hide a collision between rendered frames or alter numerical evaluation.

## 3. Demo catalogue
| Demo | What viewers see | Scientific backing and stage |
|---|---|---|
| D0: model vs reality | True vs learned reachable-set overlays and coverage-gap errors | Reviewed E0/E1/E2 results; no RL required |
| D1: robot learns to navigate | Actual checkpoint replays and the four shield comparisons | E6, after the core MVR is reviewed |
| D2: learning fleet | Cooperative goal assignment or competitive crossing, coloured agents, and visible local-shield conflicts | E4/E7 after D1 and fleet verification; pairwise analysis is not a global certificate |
| D3: population flow | Particle trajectories, population densities, OT flow, stochastic bridge samples, and congestion dynamics | Separate E8/E9/E10 formulations and residual checks; do not conflate their methods |
| D4: incentives from play | Behaviour samples and feasible-reward sets before/after normalisation | E5 using an explicitly QRE-generated game, not arbitrary RL behaviour |
| D5: physics/symmetry ablation | Matched-data constrained vs unconstrained models and the resulting safety-set differences | E11 including an incorrect-constraint negative control |
| D6: emergent swarm sandbox | Flocking/foraging patterns and coordination failures in the same robot arena | Optional E12 exploratory simulation; no new artificial-life engine or theorem |

An optional interactive playground must be visibly labelled exploratory and kept apart from frozen evaluation runs. Changing a scene interactively does not produce a reportable metric or permission to tune on the test split.

## 4. Artifact and provenance contract
Each evidence-backed demo comes from results/runs/<run_id>/, retaining the required config.yaml, runlog.json, metrics.json, and stdout.txt. Additional artifacts for the planned learned demonstrations:
- checkpoints/: frozen policy and learned-model checkpoints, with hashes and training seeds.
- rollouts.npz: raw state, observation, commanded/executed action, uncertainty, safety value, intervention reason, and collision/domain/horizon events, with timestamps and agent IDs.
- replay.json: browser-consumable export derived from the same raw rollout, not a second simulation.
- demo_manifest.json: source run IDs, artifact hashes, checkpoint hashes, seeds, initial conditions, horizon, noise/obstacle configuration, heading-slice convention, and exporter version/command.
- demo.html: a replay generated from those artifacts; any aggregated number must come from the run metrics and claims ledger.

A comparison manifest links all matched runs rather than pretending that different shields followed the same trajectory. Freeze the checkpoint and showcase-seed selection rule on training/validation data before the final test split is accessed. Publish all evaluated runs and the catalogue, not only a successful highlight clip.

Do not synthesize missing checkpoints, metrics, events, or run metadata. Keep true simulator parameters out of learner inputs except explicitly labelled oracle baselines. Record observation noise and process disturbance separately.

## 5. Implementation and dependency boundaries
- Use the existing pinned NumPy/PyTorch stack for the small environment and PPO baseline; existing hj_reachability for the low-dimensional safety computations.
- Prefer vanilla HTML/Canvas/SVG for browser presentation and Matplotlib for scientific plots, without adding a frontend framework solely for visual polish.
- Starter code is under src/lcert/envs/, rl/, population/, games/, constraints/, and visualization/, with run/export entry points under scripts/. The learned safety filter and complete trainers/study evaluators remain absent.
- Larger rendered populations do not authorise a larger HJ grid. Any pairwise relative-state formulation must be reviewed, and infeasible joint control must be detected rather than hidden.
- GIF/MP4 encoding, external simulators, RL frameworks, graphics libraries, and DeepReach dependencies require separate approval if they are not already in requirements.txt.
- The schema reserves E6 to E12, but these studies remain gated pending reviewed protocols/metrics and their actual runners. Scripted simulation/data diagnostics use their own identifiers, not E6/E7. A browser exporter cannot bypass those prerequisites.

The original 2026-10-25 scope revision was documentation-only. The current authorised starter adds code and synthetic/numerical diagnostics, but no new pinned dependency, protected test/metric change, training run, or completed scientific experiment.

## Current starter replay

Run `python scripts/run_experiment.py simulate --config configs/starters/single_robot.json` or use `configs/starters/fleet.json` in the pinned environment with a clean project-local Git tree. Open the printed folder's `demo.html`; no web server is required.

`replay.json` stores all recorded states/actions/events. `demo_manifest.json` links the run, seed, replay hash, template hash, and absence of checkpoints/certification. The replay template does not invent uncertainty, safe-set overlays, learning progress, or benchmark HUD numbers. Raw figures for successful E0 ground-truth runs can separately be generated by `scripts/make_figs.py`; these are not learned-model safety comparisons.

## 6. Acceptance and order
1. Finish the core ground-truth/model/metric gates and retain E1/E2 baselines and reproducibility.
2. Human reviews the MVR; then the bounded single-robot RL pilot and safety-filter tests may start.
3. Validate frozen-checkpoint replay and publish D1 with failures and matched baselines, or document its failure/deferral.
4. Add D2 only after detecting pairwise-control conflicts and verifying independent fleet replay.
5. Select subsequent population/incentive/physics/swarm tracks one at a time with reviewed metrics and budgets.

Experimental claims still require the fixed core metrics where applicable, at least five independent seeds, 95% CIs, logged final test access, and ledger-backed numbers. A visually convincing demo is communication, not an additional safety guarantee.
