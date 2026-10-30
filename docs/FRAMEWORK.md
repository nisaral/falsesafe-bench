# lcert.framework — bring your own dynamics, hazards, policies and certifiers

`src/lcert/framework/` lets you plug in an environment (true dynamics **with a hidden-physics gap**
plus the nominal model a designer/learner believes), a hazard, task policies and a certifier, and
evaluates the certifier against a **Hamilton-Jacobi ground truth** (hj_reachability) using the
repo's own `lcert.metrics.compute_certification_metrics` (FSV, FUV, gap-FSV, IoU, …). Results are
exported as JSON that the 3D viewer (`demos/viewer3d/index.html`) renders.

> Framework runs are **exploratory**. They are not the preregistered P1/P2 pipeline and their numbers
> must not be quoted as confirmatory results (see `PREREGISTRATION_v2.md`, `claims_ledger.md`).

## Quickstart

```bash
source /path/to/venv/bin/activate
export PYTHONPATH=src
python -m lcert.framework list                                   # registered plug-ins
python -m lcert.framework run configs/framework/di_nominal_sweep.yaml   # -> demos/viewer3d/data/
python -m lcert.framework run configs/framework/my_rover.yaml --plugin examples/framework/my_rover.py
xdg-open demos/viewer3d/index.html     # works from file:// (data/bundle.js), no network needed
```

## Plug-in interfaces (`lcert/framework/core.py`)

| Piece | What you provide |
|---|---|
| `Environment` | `true_dynamics(x,u)`, `nominal_dynamics(x,u)` (jax.numpy, single state), `controls` (finite candidate set, include box vertices), `lo/hi/grid_size/horizon/periodic_dims`, optional `ground_truth_solver` override |
| `Hazard` | `fn(states[..., n])`; failure set is `fn < 0` |
| `GapRegion` | box where physics differ / no training data → gap-FSV and training exclusion |
| `Policy` | `act(x, t) -> u` (`FunctionPolicy(name, fn)` wraps a function) |
| `Certifier` | `certify(env, grid) -> CertifierOutput(value=V_hat)`; must never read the true value |

Built-ins: envs `di_hidden_physics` (ground truth reuses `lcert.hj_di.solve_di_hidden_physics_brt`),
`dubins_slippery` (3D, periodic heading); certifiers `nominal_inflated` (robust HJ on the nominal model
with disturbance `k*sigma`), `ensemble` (adapter: `lcert.train.train_dynamics_model` + the JAX
evaluator from `lcert.hj_learned`, disturbance `k * max_u std`), `oracle` (sanity check),
`FunctionCertifier`; policies `constant`, `di_go_to`, `dubins_goal`.

Generic HJ: `solver.CallableDynamics` builds `H(x,p) = max_u p·f(x,u) − Σ|p_i| d_i(x)` from any
`f`. On the double integrator it matches `lcert.hj_di` to < 0.05 in value (tested).

## Your own sim in ~25 lines

See `examples/framework/my_rover.py` (rover, hidden downhill slope, cliff hazard, cruise policy) and
`configs/framework/my_rover.yaml`. Register with `@register_env("name")` / `@register_policy` /
`@register_certifier`; factories take keyword params so any of them can be swept:

```yaml
sweep:
  env.slope: [0.0, 0.4, 0.8]   # changes true physics -> ground truth re-solved per value
  certifier.k: [0, 1, 2, 3]
```

Or in Python: `run_experiment(ExperimentSpec(name=..., env=my_factory, certifier=MyCert, ...), out_dir=...)`.

## What gets exported (`schema: falsesafe.framework/v1`)

Per sweep case: params, full metrics dict, `v_true`/`v_hat`/`hazard_l` (downsampled to
`display_n` per axis; per heading slice for 3D envs), gap box, and trajectories (each policy × start ×
{`unfiltered`, `filtered_by_certifier`}) simulated with RK4 under the **true** dynamics, with V_true
and V_hat interpolated along them. Provenance includes git commit and timestamp. The viewer reads
`data/bundle.js` (rebuilt on every export from all `data/*.json`) or JSON files dropped onto it.

## Known limitations

* Grids are dense: ≤3D practical (Dubins 41³ ≈ seconds; 101³ is slow).
* Controls are a finite candidate set; exact only if it contains the optimal (vertex) controls.
* The safety filter is one-step lookahead on V_hat with the nominal model; finite-horizon BRTs are
  not control-invariant, so short horizons can let filtered rollouts crash (that is real output).
* The ensemble adapter trains on i.i.d. samples, not the preregistered trajectory generator; the
  conformal / distance-aware certifiers are not yet adapted (they use NumPy bounds, not JAX).
