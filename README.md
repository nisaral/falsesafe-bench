<div align="center">

# 🛡️ falsesafe-bench

[![Live demo](https://img.shields.io/badge/live%20demo-nisaral.github.io%2Ffalsesafe--bench-2ea44f)](https://nisaral.github.io/falsesafe-bench/) [![3D viewer](https://img.shields.io/badge/3D-viewer-blue)](https://nisaral.github.io/falsesafe-bench/viewer3d/) [![Gym](https://img.shields.io/badge/interactive-gym-orange)](https://nisaral.github.io/falsesafe-bench/gym/)


### Measuring How Wrong "Safe" Can Be — When Robots Learn Their Own Physics

[![CI](https://github.com/nisaral/falsesafe-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/nisaral/falsesafe-bench/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Status: Research Starter](https://img.shields.io/badge/status-research_starter-orange.svg)](#status)

**A benchmark for measuring false-safe errors in Hamilton-Jacobi reachability when dynamics are learned from finite data.**

[Quick Start](#quick-start) · [What This Does](#what-this-does) · [Experiments](#experiments) · [Contributing](#contributing) · [Roadmap](#roadmap)

</div>

---

## What This Does

When a robot learns a model of how it moves from data, it can compute a **safe set** — states where it's guaranteed not to crash. But what happens when the learned model is wrong?

**falsesafe-bench** measures exactly that:

```
                    True Safe Set          Learned Safe Set
                   ┌─────────────┐       ┌──────────────────┐
                   │             │       │  FALSE-SAFE ⚠️    │
                   │   ACTUALLY  │       │  (Model says safe │
                   │    SAFE     │       │   but it's NOT)   │
                   │             │       │                   │
                   └─────────────┘       └──────────────────┘
```

The **false-safe volume** is the most dangerous error in safety-critical systems: states where your learned model confidently says "go ahead, it's safe" — but it isn't.

### Key Questions

| # | Hypothesis | Status |
|---|---|---|
| H1 | Learned-model safe sets have false-safe volume that grows in coverage gaps | 🔬 Testing |
| H2 | Epistemic uncertainty inflation reduces false-safe volume by ≥50% | 🔬 Testing |
| H3 | Inflation has a conservatism cost (false-unsafe volume rises) | 🔬 Testing |
| H4 | Separating epistemic and aleatoric uncertainty beats a blended statistic | 🔬 Testing |
| H5 | QRE inverse games exhibit the predicted scale ambiguity | 📋 Planned |

## Status

**Research in progress.** What exists today (2026-10-30):

- **Double integrator (DI), done as pilot/primary runs:** MLP-ensemble dynamics learned from data with a
  coverage gap, learned-model HJ backward reachable tube (`hj_learned.py`), FSV/FUV metrics with
  bootstrap CIs (`metrics.py`), and six certifiers compared (`certifiers/`). Every number is in
  [`claims_ledger.md`](claims_ledger.md) with run ids and commits. Headline (claim P1, n=30 seeds):
  epistemic inflation at k=2 cut false-safe volume by 40.3% [36.5%, 44.7%], which **does not meet**
  the pre-registered ≥50% bar for H2.
- **Dubins car:** ground-truth HJ solver and learned-model code exist; the Dubins experiments (incl. H4) are not run yet.
- **Simulator demos:** seeded scripted-controller replays only. No RL policy has been trained and
  no learned safety filter drives the browser demos.

![Recorded scripted fleet replay with a visible collision; not learned or certified](demos/preview.png)

*Midpoint screenshot of the seeded fleet diagnostic, including its recorded failure. This is not a successful-learning or safety result.*

## Architecture

```
falsesafe-bench/
├── src/lcert/              # Core library
│   ├── systems.py          # Double integrator & Dubins dynamics
│   ├── hj_di.py            # HJ reachability for double integrator
│   ├── hj_dubins.py        # HJ reachability for Dubins car
│   ├── hj_learned.py       # HJ reachability on a learned (ensemble) DI model
│   ├── metrics.py          # FSV / FUV / IoU, bootstrap CIs
│   ├── certifiers/         # Conformal, distance-aware, Pareto (AUPC) comparison
│   ├── data.py             # Trajectory data collection with coverage gaps
│   ├── models/             # MLP ensembles & Gaussian Processes
│   ├── envs/               # Dubins arena & fleet simulation
│   ├── rl/                 # RL loss functions (REINFORCE, A2C, PPO, DPO, GRPO) + small actor-critic; no training loop yet
│   ├── population/         # Toy entropic OT, discrete bridge, mean-field congestion game
│   ├── games/              # Logit QRE solver (inverse games not implemented)
│   ├── constraints/        # Physics & geometry constraints
│   └── visualization/      # Browser replay generation
├── configs/                # Experiment configurations (immutable after use)
├── scripts/                # Entry points & utilities
├── tests/                  # Sanity tests (protected)
├── demos/                  # Browser-based replay viewer
└── results/                # Logged runs with full provenance
```

## Tech Stack

| Component | Technology |
|---|---|
| HJ Solver | [hj_reachability](https://github.com/StanfordASL/hj_reachability) (JAX) |
| Dynamics Learning | PyTorch (MLP ensembles) + GPyTorch (Gaussian Processes) |
| RL | PyTorch loss functions only (PPO, A2C, REINFORCE, DPO/GRPO) |
| Optimal Transport | [POT](https://pythonot.github.io/) |
| Visualization | Matplotlib + standalone HTML5 Canvas |
| Numerics | NumPy, SciPy, JAX |
| Runtime | Python 3.13, managed by [uv](https://github.com/astral-sh/uv) |

## Quick Start

### Windows

```powershell
git clone https://github.com/nisaral/falsesafe-bench.git
cd falsesafe-bench
.\scripts\bootstrap.ps1

# Verify the installation
.\.venv\Scripts\python.exe scripts/run_experiment.py doctor
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider

# Run a single robot simulation
.\.venv\Scripts\python.exe scripts/run_experiment.py simulate --config configs/starters/single_robot.json

# Run a fleet simulation
.\.venv\Scripts\python.exe scripts/run_experiment.py simulate --config configs/starters/fleet.json
```

### Linux / macOS

```bash
git clone https://github.com/nisaral/falsesafe-bench.git
cd falsesafe-bench

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Bootstrap
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt

# Verify & run
python scripts/run_experiment.py doctor
python -m pytest -q -p no:cacheprovider
python scripts/run_experiment.py simulate --config configs/starters/single_robot.json
```

Open the printed run folder's `demo.html` in a browser. It works without a server or frontend dependencies.

## Experiments

| ID | What | System | Status |
|---|---|---|---|
| E0 | Ground-truth BRT vs analytic | Double Integrator | ✅ Passing |
| E1 | FSV vs data size, gap vs no-gap | DI + Dubins | 🔬 DI pilot + primary cell run (`results/`); size ladder and Dubins not run |
| E2 | Epistemic inflation tradeoff | DI + Dubins | 🔬 DI run (claims P1, S1); Dubins not run |
| E3 | Separated vs blended uncertainty | DI + Dubins | 🔬 Planned |
| E4 | Fleet pairwise safety | 3 Dubins | 📋 Extension |
| E5 | QRE inverse games | 2×2 / 3×3 | 📋 Extension |
| E6 | Learning robot with HJ shield | Dubins | 📋 Extension |
| E7 | Multi-agent fleet learning | 3 Dubins | 📋 Extension |
| E8-E12 | Population, physics, swarms | Various | 📋 Future |

## Integrity

Every number in this repo is traceable:
- **No fabricated results.** If a metric isn't computed, it says so.
- **All runs are logged** with seeds, git commit, configs, and artifact hashes.
- **Negative results** are reported with the same prominence as positive ones.
- **claims_ledger.md** tracks every number that leaves this repo.

See [SCOPE.md](SCOPE.md) for the full integrity contract and [PLAN.md](PLAN.md) for the research plan.

## Contributing

We welcome contributions! Areas where help is especially valuable:

- 🤖 **New robot environments** (quadrotors, car-like vehicles, manipulators)
- 🧠 **Dynamics learning methods** (transformers, neural ODEs, conformal prediction)
- 🛡️ **Safety methods** (CBFs, MPC, alternative reachability solvers)
- 📊 **Benchmarking** (running experiments on new systems, comparing methods)
- 📖 **Documentation** (tutorials, notebooks, examples)

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting PRs.

## Roadmap

- [x] Simulation engine with deterministic replay
- [x] HJ reachability on double integrator (analytic match)
- [x] Browser-based fleet replay viewer
- [x] RL loss-function primitives
- [x] `metrics.py` — FSV/FUV/IoU computation pipeline
- [x] Learned dynamics models (MLP ensembles used in DI runs; GP code present, no GP result in claims_ledger.md yet)
- [ ] Core experiments E1-E3 (DI partly done; Dubins pending)
- [ ] Safety filter integration
- [ ] Robot RL with shielding
- [ ] Gymnasium-compatible environments
- [ ] Interactive Colab notebooks
- [ ] Documentation site

## Citation

If you use this work, please cite:

```bibtex
@software{falsesafe_bench,
  author = {Nisar, Keyush},
  title = {falsesafe-bench: Measuring False-Safe Errors in Hamilton-Jacobi Reachability Under Learned Dynamics},
  year = {2026},
  url = {https://github.com/nisaral/falsesafe-bench}
}
```

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">

**⚠️ This is a research tool, not a safety certification system. A collision-free simulation replay is not a safety guarantee.**

</div>


## Framework + 3D viewer (exploratory)

Build your own dynamics/hazard/policy/certifier and evaluate it against HJ ground truth with FSV/FUV/gap-FSV: see [docs/FRAMEWORK.md](docs/FRAMEWORK.md). View results in 3D: open `demos/viewer3d/index.html` (offline). Framework numbers are exploratory, not preregistered results.
