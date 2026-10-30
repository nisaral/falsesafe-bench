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
`FunctionCertifier`, and the learned-bound certifiers `conformal` (variants C0, C1b, C2) and
`distance_aware` (L) described below; policies `constant`, `di_go_to`, `dubins_goal`.

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
* The ensemble / conformal / distance-aware adapters train on i.i.d. samples, not the preregistered
  trajectory generator, and support scalar controls + 2-hidden-layer ensembles only (`lcert.hj_learned`
  JAX evaluator). C1a (epistemic-std normalised conformal) is not adapted.
* Exported data (`demos/viewer3d/data/`) is ~19 MB in total (JSONs + `bundle.js`, which duplicates them).

## Conformal (C0 / C1b / C2) and distance-aware (L) certifiers

`lcert/framework/bound_certifiers.py` adapts `lcert.certifiers.conformal` and
`lcert.certifiers.distance_aware`. All variants use the **same learned model** as the `ensemble`
adapter (ensemble mean `f_hat`; shared in-process cache keyed on env + training hyper-parameters +
seed) and a disturbance bound `d(x,u)` on the derivative dims `dims` (DI configs: `dims: [1]`, i.e. v'):

| Certifier | Bound | Knob |
|---|---|---|
| `conformal` `variant: C0` | `q_hat` = split-conformal (1-alpha) quantile (`compute_conformal_quantile`, (n+1) correction) of `max_i∈dims |y_i - f_hat_i|` on `n_calib` i.i.d. calibration samples outside the gap (independent seed) | `alpha` |
| `conformal` `variant: C1b` | `q_hat * max(dist((x,u), D_train), 1e-4)`, `q_hat` from scores `|y - f_hat| / max(dist, 1e-4)` | `alpha` |
| `conformal` `variant: C2` | `q_hat` calibrated only on `calib_region` (held-out state box, must be disjoint from the gap; checked) | `alpha` |
| `distance_aware` (L) | `eps0 + lam * L_hat * dist((x,u), D_train)`; `eps0` = max training residual, `L_hat` = max pairwise slope on a seeded subsample of training pairs (floor 0.1), as in `lcert.certifiers.distance_aware` | `lam` |

`dist` is Euclidean in joint (x, u) space with each axis divided by its range (state box, control
range), via `scipy.spatial.cKDTree`.

**JAX conversion (how and what is approximated).** A k-d tree cannot run inside the JAX HJ solver, so
the bound is evaluated **offline on every HJ grid node** for every candidate control in `env.controls`
and reduced with `max_u` → `D[x]` (a state-only bound, conservative w.r.t. the control the Hamiltonian
chooses). Inside the dynamics `D` is read by multilinear interpolation
(`grid_lookup` → `jax.scipy.ndimage.map_coordinates(order=1, mode="nearest")`). The solver only
evaluates the Hamiltonian at grid nodes, so the lookup is exact there (tested); the approximations are
(i) max over the finite control set rather than the continuous box and (ii) linear interpolation if the
dynamics are queried off-grid (e.g. by other code). The robust HJ is then
`H = max_u p·f_hat(x,u) − Σ_{i∈dims} |p_i| D(x)`.

**Differences from the repo's S2 scripts** (`scripts/run_s2_benchmark_all6.py`): those use
trajectory-generated data and pass a *scalar* (q_hat, or the grid-mean of d) into
`solve_learned_di_brt`; here C1b and L keep the state-dependent bound, data are i.i.d., and the
ensemble is trained with framework settings. Numbers are therefore **exploratory** and not comparable
1:1 with `results/`.

Configs: `configs/framework/di_conformal.yaml` (variant × alpha sweep), `di_distance_aware.yaml`
(lam sweep), `di_certifier_comparison.yaml` (one setting each of nominal/E/C0/C1b/C2/L on the same
model, using `certifier_presets` + the special sweep key `certifier`, which the viewer shows as a
slider). `scripts/framework_aupc.py` computes S2-style AUPC (same knob grids, nominal = E at k=0,
condition-wide FUV penalty) on the framework pipeline; it prints to stdout and writes only where
`--out` points (never `results/`).

Exploratory sanity check (grid 101, seeds 0/1/2, mean AUPC, lower is better): E 0.00245, L 0.00369,
C1b 0.00370, C0 0.00544, C2 0.00546. The repo's S2 ordering (`results/certifier_comparison/
s2_full6_grid101_n10.json`: L < C1b < E < C0 = C2) is reproduced only **partially**: L and C1b beat
C0/C2 and C0 ≈ C2, but in the framework E is best and L ≈ C1b. C0 and C2 are not identical here
because C2 is calibrated on its own region. Likely causes: i.i.d. data (better coverage around the gap)
and a finer E knob grid vs. different training data in the S2 pipeline. Not a confirmatory result.
