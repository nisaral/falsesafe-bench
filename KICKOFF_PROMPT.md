# Week 1 kickoff prompt (expanded scope, scientific core first)
This is an implementation kickoff, not permission to execute setup or training during a read-only or scope-only request. Follow the human's current request first.

You are working in learning_certify_project. Before doing anything:
1. Read SCOPE.md fully, then PLAN.md and SIMULATION_DEMOS.md fully. SCOPE.md is binding. Robot/game agents in those documents are experimental agents, not a requirement to launch coding subagents.
2. Create a venv, `make install`, then `make test`. Report the pytest output verbatim. If anything fails, stop and report; do not edit tests.
3. Confirm that `git rev-parse --show-toplevel` identifies the project, not an ancestor/home repository. If project-local Git is absent, request approval for setup and the initial scaffold commit; do not initialise or commit an ancestor repository. No Git setup/commit is part of a scope-only update.

Week 1 tasks (PLAN.md section 4, W1):
a. Obtain human authorisation before creating/modifying protected src/lcert/metrics.py or tests. Then implement FSV, FUV, IoU, Gap-FSV exactly as defined in SCOPE.md section 3, plus true-vs-itself FSV == 0 and a hand-built 2x2 mask check. Stop and ask for human review before using the implementation in an experiment.
b. Implement Dubins car (state x, y, theta; fixed speed; turn-rate control in a box; circular obstacle failure set) in src/lcert/hj_dubins.py and a grid-refinement test (101 vs 151 per dim, or the largest that fits the 15 min budget; report what you used).
c. Implement data generation with a configurable coverage gap (src/lcert/data.py), splits fixed by split_seed.
d. Write scripts/run_experiment.py that writes results/runs/<run_id>/ with a runlog matching runlog_schema.json.
Do not train any dynamics model, RL policy, or multi-agent learner this week, or report any FSV number except metric self-tests. Do not build a flashy animation with fabricated trajectories or statistics.

Expanded-scope follow-through, only after the corresponding gates:
- W2 to W4: learned dynamics, recovery/shuffled-label checks, E1/E2, and human MVR review.
- After MVR sign-off: E6 actual PPO robot learning, frozen-policy shield comparisons, and a replay from real run artifacts.
- After single-robot verification: E4/E7 cooperative/competitive fleet work with explicit pairwise-conflict checks.
- Later reviewed stages: E8/E9/E10 population OT, bridges, and mean-field games; E5 QRE incentives; E11 physics/geometric ablations; optional E12 emergent swarms.
- Before an extension runs: approve its metrics/budget/dependencies and extend the runlog schema correctly; never bypass protected evaluation or claim a video proves global safety.

End with a report in the SCOPE.md section 10 format. Respect the 3-attempt rule and BLOCKED.md.
