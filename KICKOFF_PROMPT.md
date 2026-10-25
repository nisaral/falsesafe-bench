# Week 1 kickoff prompt (expanded scope, scientific core first)
This is an implementation kickoff, not permission to execute setup or training during a read-only or scope-only request. Follow the human's current request first.

You are working in learning_certify_project. Before doing anything:
1. Read SCOPE.md fully, then PLAN.md and SIMULATION_DEMOS.md fully. SCOPE.md is binding. Robot/game agents in those documents are experimental agents, not a requirement to launch coding subagents.
2. Use the approved Python 3.13 environment (`scripts/bootstrap.ps1` on Windows when setup is authorised), run the environment doctor, then `make test` or the documented direct pytest command. Report the pytest output verbatim. If anything fails, stop and report; do not edit tests.
3. Confirm that `git rev-parse --show-toplevel` identifies the project, not an ancestor/home repository. If project-local Git is absent, request approval for setup and the initial scaffold commit; do not initialise or commit an ancestor repository. No Git setup/commit is part of a scope-only update.

Current starter status: repository setup/push and starter code are authorised by the subsequent human request. The robot arena, scripted browser replay, fixed trajectory generator, true-dynamics Dubins wrapper, run logger/CLI, learning objectives, and population/game/constraint primitives exist. See START_HERE.md and ALGORITHMS.md before replacing or duplicating them. They are not trained policies, reviewed benchmark results, or completed extensions.

Week 1 tasks still to finish (PLAN.md section 4, W1):
a. Obtain human authorisation before creating/modifying protected src/lcert/metrics.py or tests. Then implement FSV, FUV, IoU, Gap-FSV exactly as defined in SCOPE.md section 3, plus true-vs-itself FSV == 0 and a hand-built 2x2 mask check. Stop and ask for human review before using the implementation in an experiment.
b. Verify the existing Dubins wrapper (state x, y, theta; fixed speed; bounded turn rate; circular failure set). Obtain protected-test authorisation before adding the grid-refinement test (101 vs 151 per dim, or a separately approved alternative within the 15 min budget; report what you used).
c. Review the existing coverage-gap trajectory generator and fixed split_seed. Its starter noise is synthetic derivative-observation noise, not physical process noise or a finite-difference estimator.
d. Review the existing scripts/run_experiment.py/run logger and the reserved E0-E12/diagnostic schema. Complete the reviewed metric evaluator and study runners before treating E1-E12 as executable reported experiments.
Do not train any dynamics model, RL policy, or multi-agent learner this week, or report any FSV number except metric self-tests. Do not build a flashy animation with fabricated trajectories or statistics.

Expanded-scope follow-through, only after the corresponding gates:
- W2 to W4: learned dynamics, recovery/shuffled-label checks, E1/E2, and human MVR review.
- After MVR sign-off: E6 actual robot learning, starting with PPO and extending to the approved algorithm/data contracts in ALGORITHMS.md; frozen-policy shield comparisons and real checkpoint replays.
- After single-robot verification: E4/E7 cooperative/competitive fleet work with explicit pairwise-conflict checks.
- Later reviewed stages: E8/E9/E10 population OT, bridges, and mean-field games; E5 QRE incentives; E11 physics/geometric ablations; optional E12 emergent swarms.
- Before an extension runs: approve its metrics/budget/dependencies and extend the runlog schema correctly; never bypass protected evaluation or claim a video proves global safety.

End with a report in the SCOPE.md section 10 format. Respect the 3-attempt rule and BLOCKED.md.
