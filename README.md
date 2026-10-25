# learning_certify_project
Learning dynamics, policies, population flows, and incentives from simulated behaviour, then auditing safety with Hamilton-Jacobi reachability (hj_reachability, JAX). The scientific core measures false-safe errors caused by learned dynamics; it does not assume that a learned model or an attractive demo is automatically safe.

Status (2026-10-25 IST): scaffold only. No experimental results yet. The only passing checks are sanity tests (integrator, double-integrator BRT vs analytic solution, a negative control); output in results/sanity_test_output_*.txt.

## Expanded scope (planned, not implemented)
- Core: double integrator and Dubins dynamics learned with GP/MLP ensembles, uncertainty-aware reachability, and ground-truth safety-error benchmarks.
- Learning robots: actual PPO navigation policies, shielded vs unshielded rollouts, then cooperative/competitive multi-agent learning and pairwise fleet safety.
- Research extensions: population OT, Schrodinger bridges, a toy mean-field game, QRE inverse incentives, and physics-informed/geometric model ablations.
- Optional exploration: emergent flocking/foraging in the robot arena, not a standalone artificial-life engine. Real hardware and language-model agents remain out of scope.

## Planned flagship: Learn-and-Shield Robot Lab
Polished browser replays of robots learning and moving, with checkpoint timelines, uncertainty heatmaps, true vs learned safe-set overlays, visible shield interventions, and fleet/population views. Every evidence-backed demo must come from real logged rollouts and checkpoints, including failures. See SIMULATION_DEMOS.md.

The W1 to W4 scientific MVR stays first. Single-robot RL comes after human review, then the fleet; remaining extensions have separately reviewed schedules and budgets. This is an expanded agenda, not a claim that all tracks will be completed within six weeks.

## Project documents and commands

- Plan: PLAN.md. Agent contract: SCOPE.md (copies: AGENTS.md, CLAUDE.md).
- Demo design and evidence requirements: SIMULATION_DEMOS.md.
- Install: `python -m venv .venv && . .venv/bin/activate && make install`
- Test: `make test`
- Reproduce: `make reproduce` (fails until experiment scripts exist, by design).
- Every reported number must appear in claims_ledger.md.
