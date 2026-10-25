## Report 2026-10-26 01:15 IST (2026-10-25 19:45 UTC)

What was run:
- Authenticated GitHub account/repository verification, project-local Git commits, and pinned-environment bootstrap/doctor.
- Original protected sanity tests; synthetic infrastructure, robot, objective, replay, population/game, and geometry checks. No model or policy training.
- Actual seeded single/fleet scripted simulations, fixed trajectory-data generation, DI and Dubins true-dynamics snapshots, and raw-artifact plots.
- Isolated headless browser checks for both replays, full replay re-simulation, artifact/config/figure hash audits, and a final Dubins interface re-run.
- CLI catalogue/config checks and the expected refusal of incomplete scientific reproduction.

Command(s):
```powershell
.\scripts\bootstrap.ps1
.\.venv\Scripts\python.exe scripts/run_experiment.py doctor
.\.venv\Scripts\python.exe scripts/check_scaffold.py --output .cache/scaffold_check.json
.\.venv\Scripts\python.exe -m pytest -v -s -p no:cacheprovider
.\.venv\Scripts\python.exe -m compileall -q src scripts
node --check scripts/check_demo_browser.mjs
.\.venv\Scripts\python.exe scripts/run_experiment.py list
.\.venv\Scripts\python.exe scripts/run_experiment.py algorithms
.\.venv\Scripts\python.exe scripts/run_experiment.py validate --config configs/starters/single_robot.json
.\.venv\Scripts\python.exe scripts/run_experiment.py reproduce
.\.venv\Scripts\python.exe scripts/run_experiment.py simulate --config configs/starters/single_robot.json
.\.venv\Scripts\python.exe scripts/run_experiment.py simulate --config configs/starters/fleet.json
.\.venv\Scripts\python.exe scripts/run_experiment.py generate-data --config configs/starters/trajectory_data.json
.\.venv\Scripts\python.exe scripts/run_experiment.py run --config configs/starters/e0_ground_truth.json
.\.venv\Scripts\python.exe scripts/run_experiment.py run --config configs/starters/e0_dubins_ground_truth.json
.\.venv\Scripts\python.exe scripts/make_figs.py --run-dir results/runs/20261026_005801_E0_s0_6d51a7b1
.\.venv\Scripts\python.exe scripts/make_figs.py --run-dir results/runs/20261026_005834_E0_s0_8d041406
.\.venv\Scripts\python.exe scripts/make_figs.py --run-dir results/runs/20261026_010930_E0_s0_81270280
node scripts/check_demo_browser.mjs --browser 'C:\Program Files\Google\Chrome\Application\chrome.exe' --run-dir results/runs/20261026_005758_simulation_s0_f26f03d6
node scripts/check_demo_browser.mjs --browser 'C:\Program Files\Google\Chrome\Application\chrome.exe' --run-dir results/runs/20261026_005759_simulation_s1_8ea95648
```
`reproduce` returns the expected nonzero gate, not a successful benchmark reproduction. The Dubins command ran on both recorded source commits below; no configured grid, horizon, or seed was reduced. Bootstrap verification reused the already-installed approved environment without modifying requirements.

Git commit:
- `6c6b80a5027e0a755d15b21f2328bcf702edcd27`: initial diagnostic evidence; tree clean: yes at each logged run's start and completion.
- `91b020548ba246e76fb26c24bb08a5b7c53c8106`: verified preview plus descriptive Dubins grid-parameter cleanup; tree clean: yes at the final logged solver run's start and completion.
- This report is committed after those runs. The GitHub target was verified as private `nisaral/learning_certify_project`, authenticated as `nisaral`; publishing follows the report commit and is verified in the final terminal handoff.

Configs:

| Config | Logged run | Role / diagnostic |
|---|---|---|
| `configs/starters/single_robot.json` | `20261026_005758_simulation_s0_f26f03d6` | Validation, scripted single robot |
| `configs/starters/fleet.json` | `20261026_005759_simulation_s1_8ea95648` | Validation, scripted fleet |
| `configs/starters/trajectory_data.json` | `20261026_005800_data_s0_4eb31535` | Training-data generation; test artifact written, not evaluated |
| `configs/starters/e0_ground_truth.json` | `20261026_005801_E0_s0_6d51a7b1` | DI snapshots on configured 101 squared grid |
| `configs/starters/e0_dubins_ground_truth.json` | `20261026_005834_E0_s0_8d041406` | Dubins snapshot on configured 101 cubed grid |
| `configs/starters/e0_dubins_ground_truth.json` | `20261026_010930_E0_s0_81270280` | Final interface re-verification on the same grid/config |

Seeds:
The logged source configs explicitly use seed 0, except the fleet's seed 1; all use split_seed 42. Config copies and source hashes are retained in their run folders. Synthetic checker fixtures use their declared local fixture seeds, not a scientific seed schedule. No five-seed reported study or final-test model/policy evaluation was performed.

Numbers:
No research mean, 95% CI, or reported sample count is available. Every audited run's `metrics.json` contains `reportable: false` and empty `values`. Solver timings are recorded in raw manifests but are not published as runtime/performance benchmarks here. Sanity-test debug values in the verbatim transcript below are not learned-model or multi-seed results. `claims_ledger.md` remains empty; no research number was added or invented.

Plots:
- `results/runs/20261026_005801_E0_s0_6d51a7b1/figs/20261026_005851_7f8451/ground_truth_T0p5.png`
- `results/runs/20261026_005801_E0_s0_6d51a7b1/figs/20261026_005851_7f8451/ground_truth_T1p0.png`
- `results/runs/20261026_005834_E0_s0_8d041406/figs/20261026_010141_da5e3c/ground_truth_T1p0.png`
- `results/runs/20261026_010930_E0_s0_81270280/figs/20261026_010943_e1b442/ground_truth_T1p0.png`
- `demos/preview.png`: unmodified midpoint browser screenshot of the fleet diagnostic, including a recorded collision. Its hashes and source/config/selection provenance are in `demos/preview_manifest.json`; full browser checks/previews remain in both simulation run folders.

Tests:
- Original pytest: all protected tests pass; exact captured output below. The cache provider is disabled to avoid Windows cache-directory permission warnings, not to change a test or threshold.
- All synthetic scaffold-check groups pass, including finite objective gradients, equal-reward grouped advantages, terminal events, seeded replay, unsupported bridge endpoints, probability/mass checks, QRE scaling, incorrect-speed prior, and unfitted-model rejection. No optimizer/model/policy training is part of these checks.
- Both actual browser replays pass syntax/runtime, state-panel, playback/speed/scrub/restart/keyboard, and mobile-layout checks, with no recorded runtime error.
- Python compilation, browser-helper syntax, PowerShell bootstrap parsing, bootstrap/doctor, and CLI contracts pass. `reproduce` deliberately returns `ExperimentUnavailable` instead of pretending E1-E12 are complete.
- Test/metric/SCOPE/AGENTS/CLAUDE/requirements files remain unchanged from the initial project scaffold. Verification summaries are locally retained in `.cache/scaffold_check.json`, `.cache/starter_audit.json`, `.cache/final_solver_check.json`, and `.cache/cli_checks.json`.

Anything suspicious:
- The scripted single robot ends in a collision. The fleet ends with domain exits and a collision. These are retained, reproducible episode outcomes, not hidden or re-labelled success. Run status `ok` means the diagnostic completed and logged correctly, not that navigation or safety succeeded.
- No model/policy checkpoint exists, no learned uncertainty is displayed, no safety shield is attached, and no recovery/shuffled-label, Dubins-refinement, or protected metric study is completed. Population/game/physics components are numerical primitives, not reviewed benchmark results.
- Pinned POT imports trigger a Torch JIT deprecation warning; its first import also showed an invalid-escape SyntaxWarning. Dependencies were not patched, swapped, or added. The original pytest cache permission warning is avoided with `-p no:cacheprovider`; denied temporary cache directories were not deleted.
- GitHub network verification required sandbox escalation; the isolated headless browser used approved escalation and a fresh project-local profile, not the user's existing browser profile.
- Full generated runs/data/figures/checkpoints are Git-ignored and remain locally available. Only the source, documentation, and verified small preview are published. A preview is not scientific evidence by itself or a global fleet certificate.

Re-verification:
- Opened all runlogs/config copies; verified source/copy config hashes, seeds, start/end source-clean fields, registered artifact hashes, non-reportable metrics, and absence of final-test evaluation.
- Re-simulated the exact single/fleet configs and compared all logged frames/actions/events and attached run metadata for exact equality, including failures.
- Checked training/validation data for finite features/targets; the final test file was hashed for transfer integrity only, not loaded for model selection or evaluation.
- Verified E0 NPZ shapes/finite values and figure/runlog/source hashes. The final Dubins re-run exactly matches the earlier value, states, spacings, and horizon arrays despite the interface-name cleanup.
- Inspected the actual fleet screenshot and Dubins ground-truth figure. Copied the screenshot without modifying its bytes, verified its hash, and recorded the fixed midpoint selection rule.
- Re-ran the original protected tests, verified the installed pinned stack, and checked the explicit incomplete-reproduction gate. No existing ledger row could be recomputed because the ledger has no claims.

Next step / blocked:
The requested starter foundation is ready for GitHub publishing. Research training/evaluation remains intentionally gated: obtain explicit human approval for protected metrics and Dubins-refinement tests, finish W1/model recovery and shuffled-label controls, then reviewed E1/E2 and MVR sign-off. Only afterward begin bounded actual E6 learning and matched shield comparisons, then E4/E7 and separately reviewed population/game/geometric/swarm extensions. Complete SAC/TD3/CQL/IQL/MAPPO trainers and DPO/GRPO robot adaptations require their own data/critic/reference/protocol checks; registry entries are not implementations.

### Verbatim protected pytest output

Captured from the pinned environment; raw transcript also remains at `.cache/final_pytest_stdout.txt`. These are sanity checks, not reportable research comparisons.

```text
============================= test session starts =============================
platform win32 -- Python 3.13.11, pytest-9.1.1, pluggy-1.7.0 -- C:\Users\nisar\OneDrive\Desktop\learning_certify_project\learning_certify_project\.venv\Scripts\python.exe
rootdir: C:\Users\nisar\OneDrive\Desktop\learning_certify_project\learning_certify_project
configfile: pyproject.toml
testpaths: tests
collecting ... collected 4 items

tests/test_di_hj_analytic.py::test_di_value_and_set_match_analytic[0.5] 
  0%|          |  0.0000/0.5 [00:00<?, ?sim_s/s]
 98%|#########7|  0.4900/0.5 [00:00<00:00,  4.83sim_s/s]
100%|##########|  0.5000/0.5 [00:00<00:00,  4.62sim_s/s]
T=0.5 max|V-Va|=0.0006657 mean=6.427e-05 IoU=1.00000 false_safe_frac=0.00000 cell=0.0400
PASSED
tests/test_di_hj_analytic.py::test_di_value_and_set_match_analytic[1.0] 
  0%|          |  0.0000/1.0 [00:00<?, ?sim_s/s]
 45%|####4     |  0.4500/1.0 [00:00<00:00,  4.44sim_s/s]
 91%|######### |  0.9100/1.0 [00:00<00:00,  4.49sim_s/s]
100%|##########|  1.0000/1.0 [00:00<00:00,  4.42sim_s/s]
T=1.0 max|V-Va|=0.0009511 mean=0.0001771 IoU=1.00000 false_safe_frac=0.00000 cell=0.0400
PASSED
tests/test_di_hj_analytic.py::test_negative_control_wrong_dynamics_is_detected 
  0%|          |  0.0000/1.0 [00:00<?, ?sim_s/s]
 70%|######9   |  0.6960/1.0 [00:00<00:00,  6.83sim_s/s]
100%|##########|  1.0000/1.0 [00:00<00:00,  6.73sim_s/s]
negative control max|V-Va|=0.25
PASSED
tests/test_integrators.py::test_rk4_matches_closed_form PASSED

============================== 4 passed in 3.28s ==============================
```
