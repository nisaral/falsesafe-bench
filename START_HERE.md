# Starting the research lab

This repository is an executable foundation for the broader programme, not a claim that all experiments or learning algorithms are finished. The current human request authorises repository setup/push and starter implementation. It does not waive the protected evaluation, reviewed-protocol, or MVR gates in SCOPE.md.

## 1. Environment and repository

Use the actual project-local repository, not an ancestor/home Git repository. In the original Windows workspace it is the inner `learning_certify_project` directory; a GitHub clone puts this README at its repository root.

```powershell
git rev-parse --show-toplevel
.\scripts\bootstrap.ps1
.\.venv\Scripts\python.exe scripts/run_experiment.py doctor
```

`uv` and Git must already be available. The bootstrap uses `.tools/python`, `.venv`, and `.cache/uv`, installs only the existing pinned requirements, then verifies Python 3.13 and installed distribution versions. It does not train anything, configure GitHub credentials, or alter global Python. GitHub publishing additionally needs your authenticated GitHub CLI/Git credentials; never store tokens in this repo.

Linux/macOS equivalent, with `uv` installed:

```sh
uv python install 3.13 --install-dir .tools/python --no-bin --cache-dir .cache/uv
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
uv venv --python 3.13 --managed-python --cache-dir .cache/uv .venv
uv pip install --python .venv/bin/python --cache-dir .cache/uv -r requirements.txt
. .venv/bin/activate
python scripts/run_experiment.py doctor
```

All commands below assume the pinned environment. On Windows use `.\.venv\Scripts\python.exe` instead of `python` unless it is activated. No editable package installation is required; entry-point scripts put `src` on the import path.

## 2. Verify the base

```sh
python -m pytest -v -s -p no:cacheprovider
python scripts/check_scaffold.py
python scripts/run_experiment.py list
python scripts/run_experiment.py algorithms
python scripts/run_experiment.py validate --config configs/starters/single_robot.json
```

The protected original pytest suite checks the DI analytic comparison, wrong-dynamics control, and integrator. `check_scaffold.py` separately checks synthetic configuration/robot/objective/replay/probability/geometry contracts and unfitted-model rejection. It does not replace the scientific evaluator, train models, or produce research metrics. Its generated test-data fixture is not model evaluation on the final test split.

The config command validates common seed/serialization fields; operation-specific validation happens in the corresponding simulator, data generator, or solver. Errors stop execution rather than selecting a smaller grid, fewer samples, or a different algorithm. `doctor` uses installed distribution versions, while a runtime such as Torch may additionally show a build suffix.

## 3. Generate actual robot replays

Run recording requires a committed, clean project tree at both start and completion. The tracked starter configs are ready to use; commit your own new config before running it.

```sh
python scripts/run_experiment.py simulate --config configs/starters/single_robot.json
python scripts/run_experiment.py simulate --config configs/starters/fleet.json
```

Each command prints `results/runs/<run_id>/`. Open that folder's `demo.html` manually in a browser. The replay is standalone and includes actual recorded positions, observations, commanded/executed actions, realized noisy turn rates, and events. The visual robot body is a heading glyph; collision geometry uses the configured disk radius. Terminal robots are removed from active dynamics, and collision checking is numerical/substepped, not a continuous-time certificate.

The starter controller is scripted, untrained, and unshielded. Playback, restart, scrubbing, speed, trails, goal markers, states, events, and provenance are available now. Learned overlays/checkpoint timelines must wait for real trained artifacts. Neither polishing a replay nor reaching a goal licenses a safety or aggregate-performance claim.

Optional browser verification uses an already-installed Node 22+ and explicitly selected Chrome/Chromium executable, without any npm dependency. It opens an isolated headless profile, checks the replay controls/mobile layout, and saves a midpoint screenshot plus a derived browser-check manifest in the run folder. It does not use your existing browser profile or alter the rollout. Browser launch may require your sandbox approval.

```powershell
node scripts/check_demo_browser.mjs --browser 'C:\Program Files\Google\Chrome\Application\chrome.exe' --run-dir results/runs/<actual-simulation-run-id>
```

## 4. Generate fixed dynamics data

```sh
python scripts/run_experiment.py generate-data --config configs/starters/trajectory_data.json
```

Outputs: `train.npz`, `validation.npz`, `test.npz`, and `data_manifest.json`. Each split has `states`, `actions`, trajectory IDs, regression `features` (state plus action), and noisy derivative `targets`. Noise is synthetic derivative-observation noise along simulated trajectories, not finite-difference estimation or physical process disturbance. The configured gap excludes train/validation trajectory states; test artifacts are generated but not evaluated.

Do not load the final test split for model selection. Freeze split seeds before training, keep true dynamics separate from learner inputs, and obtain evaluator approval plus ledger logging before final test access. The generic model classes do not themselves enforce the study protocol: future runners must load only the approved splits and preserve provenance.

## 5. Ground-truth snapshots and figures

```sh
python scripts/run_experiment.py run --config configs/starters/e0_ground_truth.json
python scripts/run_experiment.py run --config configs/starters/e0_dubins_ground_truth.json
python scripts/make_figs.py --run-dir results/runs/<actual-E0-run-id>
```

Replace the last path with a real successful E0 run. The configured grids are 101 squared for DI and 101 cubed for Dubins, with periodic heading for Dubins. Each snapshot runs in a worker process with an enforced SCOPE.md timeout, conservatively including worker startup and export; timeout/failure is logged, not retried on a smaller grid. E0 writes actual NPZ values/states/spacings and a snapshot manifest, not FSV/FUV/IoU results. A single Dubins snapshot is not its required grid-refinement test. `solve_snapshot.py` is an internal worker, not an alternative experiment runner.

Plotting verifies source hashes, successful status, config, and clean plotting source, and writes derived figures plus their own manifest under the run's `figs/`. It never edits the original runlog. The Dubins figure labels the actual nearest heading slice; `--heading` selects a different slice of the same recorded 3D field. This does not compute full-state safety volume or learned-model error.

## 6. Implementation map and next work

| Location | Foundation available | Next reviewed implementation |
|---|---|---|
| `src/lcert/artifacts.py`, `config.py`, `experiments.py`, `cli.py` | Logging, hashes, configuration, E0-E12 registry and gates | Reviewed study runners and final-test logging |
| `src/lcert/envs/`, `simulation.py`, `visualization/`, `demos/` | Single/fleet arena and real-artifact browser replays | Reward/termination review, frozen-checkpoint and shield comparisons |
| `src/lcert/data.py`, `models/` | Trajectory splits, bootstrapped ensemble and exact GP classes | Recovery, shuffled-label controls, validated noise separation |
| `src/lcert/hj_di.py`, `hj_dubins.py` | Existing solver and true-dynamics wrappers | Protected metrics review, Dubins convergence, learned-model disturbances |
| `src/lcert/rl/` | Objective functions, untrained actor-critic, replay and proposed navigation reward | Actual on/off-policy, offline/preference/grouped and multi-agent trainers |
| `src/lcert/population/`, `games/` | OT, finite-state bridge/MFG, QRE forward solve/play | Separate baselines/held-out studies and inverse estimator |
| `src/lcert/constraints/` | Geometry and public kinematic prior | Correct-vs-mismatched prior and geometric/PINN ablations |

Algorithm sequencing is intentionally broader than PPO: REINFORCE/A2C/PPO; robot GRPO-style grouped learning; BC/DPO-style preferences; SAC/TD3; IQL/CQL; MAPPO/independent PPO. The available objective primitives are not complete trainers. Their different on-policy, replay, paired-context, grouped-context, and reference-checkpoint contracts are documented in ALGORITHMS.md. Full PINNs, emergent swarms, joint fleet certification, inverse games, and learned safety filters are not implemented.

Next: request explicit human authorisation for protected metrics and Dubins/refinement tests; complete W1-W4 and the MVR review. Only then start bounded E6 learning and its matched shield comparisons, followed by E4/E7 and separately approved research extensions. Do not launch training just because a class exposes `fit` or an experiment ID appears in the catalogue.

## 7. Provenance and publishing

Every run retains config copy/source hashes, seed/split seed, commit, clean-tree checks, timestamps in IST, versions/hardware, stdout, artifact hashes, and failure status. `metrics.json` deliberately has `reportable: false` and empty values until the protected evaluator is reviewed. Failed runs are retained; do not delete them or relabel them as success.

The source repository does not ship generated runs, environments, checkpoints, or secrets. Regenerate starter diagnostics locally using the committed configs. New configs are copied rather than editing a config already used by a run. `make starter` is convenience orchestration, not the scientific MVR; `make reproduce` refuses to pretend the unfinished benchmark is reproducible.
