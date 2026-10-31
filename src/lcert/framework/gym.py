"""Interactive "gym" export: scenarios (env + hidden-physics gap + hazard + certifier choices)
packaged for the in-browser simulator in ``demos/gym/``.

What is exported per scenario (one ``data/<name>.js`` file, no JSON duplicate):

* env metadata (bounds, grid shape, periodic dims, candidate controls, gap box, hazard text),
* ``V_true`` (HJ ground truth under the TRUE dynamics) on the FULL solver grid,
* for each certifier choice, ``V_hat`` on the same grid plus FSV / FUV / gap-FSV computed by
  ``lcert.metrics.compute_certification_metrics`` on that grid (same as ``run_experiment``),
* the hazard function l(x) on the grid (used for crash detection when no JS mirror exists),
* for environments without a hand-written JS mirror (user plug-ins), tabulated f_true / f_nom
  at every grid node for every candidate control (the browser interpolates them),
* provenance (git commit, timestamp, quantisation step).

Arrays are quantised to int16 with one scale per array (``q = round(v / scale)``,
``scale = max|v| / 32767``), little-endian, base64.  The max quantisation error is
``scale / 2`` and is stored next to each array.
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from lcert.metrics import compute_certification_metrics

from . import registry
from .runner import _git_commit, _jsonable
from .solver import ground_truth

GYM_SCHEMA = "falsesafe.gym/v1"
SCENARIOS: dict[str, "Scenario"] = {}

# envs whose dynamics/hazard are mirrored by hand in demos/gym/gym_core.js
JS_MIRRORED = {"di_hidden_physics", "dubins_slippery", "drone_vertical", "pendulum_payload", "boat_current",
               "rover_cliff3d"}


@dataclass
class Scenario:
    """One gym scenario.

    certifiers: list of {"label", "certifier", "params"}; the first is the default.
    render: viewer scene kind ("cart", "car", "drone", "pendulum", "boat", "rover", or "point").
    manual: maps input axes to control dims, e.g. {"x": 0} (left/right -> u[0]) or {"x": 0, "y": 1}.
    goal: goal state for the go-to-goal policy (``None`` entries = don't care).
    start: default start state; start_box: box for random starts in batch mode.
    """

    name: str
    title: str
    env: str
    env_params: dict = field(default_factory=dict)
    certifiers: list = field(default_factory=list)
    render: str = "point"
    manual: dict = field(default_factory=lambda: {"x": 0})
    goal: list | None = None
    goal_weights: list | None = None
    start: list | None = None
    start_box: tuple | None = None
    dt: float = 0.02
    max_steps: int = 750
    filter_margin: float = 0.05
    border_cells: int = 3
    blurb: str = ""
    reduction: str = ""
    user_defined: bool = False


def register_scenario(s: Scenario) -> Scenario:
    SCENARIOS[s.name] = s
    return s


def _q16(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    m = float(np.max(np.abs(a))) or 1.0
    scale = m / 32767.0
    q = np.clip(np.round(a / scale), -32767, 32767).astype("<i2")
    return {"shape": list(a.shape), "scale": scale, "max_abs_err": scale / 2,
            "b64": base64.b64encode(q.tobytes(order="C")).decode("ascii")}


def dequantize(d: dict) -> np.ndarray:
    q = np.frombuffer(base64.b64decode(d["b64"]), dtype="<i2").astype(np.float64)
    return (q * d["scale"]).reshape(d["shape"])


def _tabulate(env, grid, f) -> dict:
    X = jnp.asarray(grid.states).reshape(-1, env.dim)
    fj = jax.jit(jax.vmap(lambda x, u: jnp.asarray(f(x, u)), in_axes=(0, None)))
    T = np.stack([np.asarray(fj(X, jnp.asarray(u))) for u in env.controls])  # (m, N, dim)
    return _q16(T.reshape((len(env.controls),) + tuple(env.shape) + (env.dim,)))


def build_scenario(s: Scenario, verbose=True) -> dict:
    env = registry.build(registry.ENVS, "env", s.env, s.env_params)
    grid, v_true = ground_truth(env)
    v_true = np.asarray(v_true)
    states = np.asarray(grid.states)
    cases = []
    for c in s.certifiers:
        cert = registry.build(registry.CERTIFIERS, "certifier", c["certifier"], c.get("params", {}))
        out = cert.certify(env, grid)
        m = compute_certification_metrics(v_true, out.value, border_cells=s.border_cells,
                                          periodic_dims=tuple(env.periodic_dims), states=states,
                                          coverage_gap=env.gap.as_coverage_gap() if env.gap else None).to_dict()
        if verbose:
            print(f"[gym:{s.name}] {c['label']}: FSV={m['fsv']:.4f} FUV={m['fuv']:.4f} gap-FSV={m['gap_fsv']}", flush=True)
        cases.append({"label": c["label"], "certifier": cert.name, "params": _jsonable(c.get("params", {})),
                      "info": _jsonable({k: v for k, v in out.info.items() if k != "training"}),
                      "metrics": _jsonable(m), "v_hat": _q16(out.value)})
    mirrored = env.name in JS_MIRRORED and not s.user_defined
    doc = {
        "schema": GYM_SCHEMA, "name": s.name, "title": s.title, "blurb": s.blurb, "reduction": s.reduction,
        "render": s.render, "manual": s.manual, "goal": s.goal, "goal_weights": s.goal_weights,
        "start": s.start, "start_box": s.start_box, "dt": s.dt, "max_steps": s.max_steps,
        "filter_margin": s.filter_margin, "user_defined": bool(s.user_defined),
        "env": {"name": env.name, "description": env.description, "state_names": list(env.state_names),
                "lo": list(map(float, env.lo)), "hi": list(map(float, env.hi)), "shape": list(env.shape),
                "periodic_dims": list(env.periodic_dims), "horizon": env.horizon,
                "axes": [[float(env.lo[i]), (float(env.hi[i]) - float(env.lo[i])) /
                          (env.shape[i] if i in env.periodic_dims else env.shape[i] - 1)]
                         for i, c in enumerate(grid.coordinate_vectors)],
                "controls": env.controls.tolist(), "hazard": env.hazard.description,
                "gap": ({"lo": list(map(float, env.gap.lo)), "hi": list(map(float, env.gap.hi)),
                         "description": env.gap.description} if env.gap else None),
                "params": _jsonable(env.params), "js_dynamics": env.name if mirrored else None},
        "v_true": _q16(v_true), "hazard_l": _q16(np.asarray(env.hazard(grid.states))),
        "cases": cases,
        "provenance": {"source": "lcert.framework.gym.build_scenario: hj_reachability ground truth vs certifier "
                                 "on the full solver grid; metrics from lcert.metrics.compute_certification_metrics",
                       "created": _dt.datetime.now(_dt.timezone.utc).isoformat(), "git_commit": _git_commit(),
                       "quantization": "int16 per-array scale; see max_abs_err",
                       "border_cells": s.border_cells},
    }
    if not mirrored:
        doc["f_true_table"] = _tabulate(env, grid, env.true_dynamics)
        doc["f_nom_table"] = _tabulate(env, grid, env.nominal_dynamics)
    return doc


def export_scenario(doc: dict, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{doc['name']}.js"
    p.write_text("// generated by lcert.framework.gym; do not edit\n(window.FSB_GYM=window.FSB_GYM||[]).push("
                 + json.dumps(doc, separators=(",", ":")) + ");\n")
    rebuild_index(out)
    return p


def rebuild_index(out_dir) -> Path:
    """data/index.js lists every scenario file so index.html can load them from file://."""
    out = Path(out_dir)
    names = sorted(p.stem for p in out.glob("*.js") if p.name != "index.js")
    builtin = [n for n in BUILTIN_ORDER if n in names]
    ordered = builtin + [n for n in names if n not in builtin]
    p = out / "index.js"
    p.write_text("// generated by lcert.framework.gym.rebuild_index; do not edit\nwindow.FSB_GYM_FILES = "
                 + json.dumps(ordered) + ";\n")
    return p


def load_scenario_file(path) -> dict:
    t = Path(path).read_text()
    return json.loads(t[t.index("push(") + 5: t.rindex(");")])


# ---------------------------------------------------------------- built-in scenarios
_NAIVE = {"label": "naive nominal model (k=0)", "certifier": "nominal_inflated", "params": {"k": 0.0}}


def _infl(k, sigma, dims):
    return {"label": f"nominal + robust margin k*sigma={k * sigma:g} on dims {list(dims)}",
            "certifier": "nominal_inflated", "params": {"k": k, "sigma": sigma, "dims": list(dims)}}


_ENS = {"label": "learned ensemble (k=2 std), trained outside gap", "certifier": "ensemble",
        "params": {"k": 2.0, "epochs": 60, "n_samples": 2000}}
_ORACLE = {"label": "oracle: TRUE dynamics (sanity check, not a real certifier)", "certifier": "oracle", "params": {}}
_C0 = {"label": "conformal C0 (alpha=0.05) on ensemble", "certifier": "conformal",
       "params": {"variant": "C0", "alpha": 0.05, "dims": [1]}}

register_scenario(Scenario(
    "cart_wall", "Cart vs wall (double integrator)", "di_hidden_physics", {"horizon": 2.5},
    [_NAIVE, _infl(2.0, 0.1, [1]), _infl(6.0, 0.1, [1]), _ENS, _C0, _ORACLE], render="cart", manual={"x": 0},
    goal=[0.15, 0.0], goal_weights=[1.0, 0.3], start=[2.0, -0.5], start_box=([0.2, -1.5], [2.8, 1.0]),
    blurb="Park the cart as close to the wall (p=0) as you dare. A hidden band of weak actuators + drift "
          "toward the wall sits at p in [0.4, 1.0].",
    reduction="Exact: the system is a 2D double integrator (cart on a frictionless track)."))
register_scenario(Scenario(
    "drone_hover", "Drone altitude (quadrotor vertical channel)", "drone_vertical", {"horizon": 2.5},
    [_NAIVE, _infl(3.0, 0.1, [1]), _infl(8.0, 0.1, [1]), _ENS, _ORACLE], render="drone", manual={"y": 0},
    goal=[1.3, 0.0], goal_weights=[1.0, 0.3], start=[2.5, 0.0], start_box=([0.3, -1.5], [2.7, 1.5]),
    blurb="Hold altitude inside the hidden rotor-wash band without hitting the floor or the ceiling.",
    reduction="2D reduction of a quadrotor: only altitude z and climb rate; attitude/lateral motion ignored."))
register_scenario(Scenario(
    "pendulum_balance", "Balance an inverted pendulum / arm", "pendulum_payload", {"horizon": 5.0},
    [_NAIVE, _infl(3.0, 0.1, [1]), _infl(6.0, 0.1, [1]), _ENS, _ORACLE], render="pendulum", manual={"x": 0},
    goal=[0.5, 0.0], goal_weights=[1.0, 0.2], start=[0.0, 0.0], start_box=([-0.6, -1.0], [0.6, 1.0]),
    blurb="Keep the pole within its joint limits. Leaning right (theta >= 0.25) a hidden payload shift "
          "weakens the motor.",
    reduction="2D: one-link pendulum/arm (theta, omega). A cart-pole would be 4D and exceeds the 3D HJ grid limit."))
register_scenario(Scenario(
    "boat_rocks", "Boat in a cross-current", "boat_current", {"horizon": 3.0},
    [_NAIVE, _infl(4.0, 0.15, [0, 1]), _infl(7.0, 0.15, [0, 1]), _ORACLE], render="boat", manual={"x": 0, "y": 1}, filter_margin=0.1,
    goal=[1.8, -1.2], goal_weights=[1.0, 1.0], start=[-2.4, 1.6], start_box=([-2.8, -1.8], [2.8, 2.8]),
    blurb="Steer around the rock and along the shore. A hidden current (y in [-2.4, 0.2], 1 m/s, stronger than your 0.7 m/s engine) pushes toward the rocks and shore.",
    reduction="2D holonomic kinematic reduction (x, y); hull heading/inertia ignored; 2D control (velocity)."))
register_scenario(Scenario(
    "car_slippery", "Dubins car with a slippery patch", "dubins_slippery", {"horizon": 4.0},
    [_NAIVE, _infl(3.0, 0.1, [2]), _infl(6.0, 0.1, [2]), _ORACLE], render="car", manual={"x": [0, -1]}, filter_margin=0.1,
    goal=[2.5, 0.0, None], goal_weights=[1.0, 1.0, 0.0], start=[-2.5, 0.3, 0.0],
    start_box=([-2.8, -2.8, -3.14], [2.8, 2.8, 3.14]), max_steps=600,
    blurb="Constant-speed car, you steer. In the hidden patch (x in [0.8, 2], |y| <= 1.5) turning authority halves.",
    reduction="Exact 3D Dubins kinematics (x, y, heading) at 41^3 grid."))
register_scenario(Scenario(
    "rover_cliff", "Rover near a cliff edge", "rover_cliff3d", {"horizon": 4.0},
    [_NAIVE, _infl(3.0, 0.1, [1]), _infl(6.0, 0.1, [1]), _ORACLE], render="rover", manual={"x": [0, -1]}, filter_margin=0.1,
    goal=[1.5, 1.7, None], goal_weights=[1.0, 1.0, 0.0], start=[-2.4, -2.0, 0.6],
    start_box=([-2.8, -2.8, -3.14], [2.8, 1.8, 3.14]), max_steps=600,
    blurb="Survey near the cliff (y > 2 is a drop) and avoid the pit. Hidden scree (y >= 0.8) slides you toward the edge.",
    reduction="3D unicycle (x, y, heading) with constant forward speed; you steer."))

BUILTIN_ORDER = list(SCENARIOS)


def main(argv=None):
    import argparse
    from .__main__ import _load_plugin
    ap = argparse.ArgumentParser(prog="python -m lcert.framework.gym")
    ap.add_argument("names", nargs="*", help="scenario names (default: all built-ins)")
    ap.add_argument("--out", default="demos/gym/data")
    ap.add_argument("--plugin", action="append", default=[])
    a = ap.parse_args(argv)
    import lcert.framework.gym as g  # not __main__: plug-ins register into this module's table
    for p in a.plugin:
        _load_plugin(p)
    for n in a.names or g.BUILTIN_ORDER:
        if n not in g.SCENARIOS:
            raise SystemExit(f"unknown scenario {n!r}; registered: {sorted(g.SCENARIOS)}")
        print("exported", g.export_scenario(g.build_scenario(g.SCENARIOS[n]), a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
