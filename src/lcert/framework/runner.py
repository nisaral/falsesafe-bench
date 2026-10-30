"""Experiment runner: ground-truth HJ vs certifier -> FSV/FUV/gap-FSV + viewer JSON."""
from __future__ import annotations

import datetime as _dt
import itertools
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from lcert.metrics import compute_certification_metrics

from . import registry
from .core import Environment
from .policies import GridValue, simulate
from .solver import ground_truth, make_grid

SCHEMA = "falsesafe.framework/v1"


@dataclass
class ExperimentSpec:
    """Everything needed for one experiment (or a sweep).

    ``env``/``certifier``: either a registered name or a factory callable.
    ``sweep``: {"env.<param>": [...], "certifier.<param>": [...]}; the cartesian
    product is run and exported, which is what the 3D viewer's sliders scrub.
    """

    name: str
    env: str | Callable = "di_hidden_physics"
    env_params: dict = field(default_factory=dict)
    certifier: str | Callable = "nominal_inflated"
    certifier_params: dict = field(default_factory=dict)
    policies: list = field(default_factory=list)   # [{"name":..., "params":{...}}] or Policy objects
    initial_states: list = field(default_factory=list)
    sweep: dict = field(default_factory=dict)
    dt: float = 0.02
    steps: int = 150
    filter_margin: float = 0.05
    border_cells: int = 3
    display_n: int = 61
    slice_values: list | None = None   # for 3D envs: values of the last dim to export
    illustrative: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentSpec":
        return cls(**d)

    @classmethod
    def from_yaml(cls, path) -> "ExperimentSpec":
        import yaml
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


def _build(table, kind, what, params):
    return what(**params) if callable(what) else registry.build(table, kind, what, params)


def _policy(p):
    if hasattr(p, "act"):
        return p
    return registry.build(registry.POLICIES, "policy", p["name"], p.get("params", {}))


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _display(env: Environment, grid, V, spec, slice_idx):
    """Downsample V for display; returns list (per slice) of 2D lists."""
    n0, n1 = V.shape[0], V.shape[1]
    s0, s1 = max(1, int(np.ceil(n0 / spec.display_n))), max(1, int(np.ceil(n1 / spec.display_n)))
    out = []
    for k in slice_idx:
        A = V if V.ndim == 2 else V[..., k]
        out.append(np.round(A[::s0, ::s1], 4).tolist())
    axes = [np.asarray(grid.coordinate_vectors[0])[::s0], np.asarray(grid.coordinate_vectors[1])[::s1]]
    return out, axes


def run_case(env: Environment, certifier, spec: ExperimentSpec, gt_cache: dict | None = None,
             policies=None) -> dict:
    gt_cache = {} if gt_cache is None else gt_cache
    key = (env.name, json.dumps(env.params, sort_keys=True, default=str))
    t0 = time.monotonic()
    if key not in gt_cache:
        gt_cache[key] = ground_truth(env)
    grid, v_true = gt_cache[key]
    t_gt = time.monotonic() - t0
    t0 = time.monotonic()
    out = certifier.certify(env, grid)
    t_cert = time.monotonic() - t0
    v_hat = out.value
    m = compute_certification_metrics(
        v_true, v_hat, border_cells=spec.border_cells, periodic_dims=tuple(env.periodic_dims),
        states=np.asarray(grid.states), coverage_gap=env.gap.as_coverage_gap() if env.gap else None)
    true_fn, hat_fn = GridValue(env, grid, v_true), GridValue(env, grid, v_hat)
    trajs = []
    for pol in (policies or []):
        for x0 in spec.initial_states:
            for mode, fv in (("unfiltered", None), ("filtered_by_certifier", hat_fn)):
                r = simulate(env, pol, x0=x0, dt=spec.dt, steps=spec.steps, filter_value=fv, margin=spec.filter_margin)
                S = r["states"]
                trajs.append({
                    "policy": pol.name, "mode": mode, "x0": list(map(float, x0)),
                    "states": np.round(S, 4).tolist(), "v_true": np.round(true_fn(S), 4).tolist(),
                    "v_hat": np.round(hat_fn(S), 4).tolist(), "filter_active": r["filter_active"],
                    "hit_failure": r["hit_failure"], "hit_failure_step": r["hit_failure_step"], "dt": r["dt"]})
    return {"metrics": m.to_dict(), "info": _jsonable(out.info), "grid": grid, "v_true": v_true, "v_hat": v_hat,
            "trajectories": trajs, "timing_s": {"ground_truth": t_gt, "certifier": t_cert}}


def _jsonable(o):
    return json.loads(json.dumps(o, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


def run_experiment(spec: ExperimentSpec, out_dir: str | Path | None = None, verbose=True) -> dict:
    keys = list(spec.sweep)
    combos = list(itertools.product(*[spec.sweep[k] for k in keys])) or [()]
    policies = [_policy(p) for p in spec.policies]
    gt_cache, cases, env_meta, axes, slice_idx = {}, [], None, None, None
    for combo in combos:
        ep, cp = dict(spec.env_params), dict(spec.certifier_params)
        params = dict(zip(keys, combo))
        for k, v in params.items():
            scope, name = k.split(".", 1)
            {"env": ep, "certifier": cp}[scope][name] = v
        env = _build(registry.ENVS, "env", spec.env, ep)
        cert = _build(registry.CERTIFIERS, "certifier", spec.certifier, cp)
        res = run_case(env, cert, spec, gt_cache, policies)
        grid = res["grid"]
        if slice_idx is None:
            if env.dim == 2:
                slice_idx, slices = [0], [None]
            else:
                cv = np.asarray(grid.coordinate_vectors[2])
                vals = spec.slice_values if spec.slice_values is not None else [-np.pi / 2, 0.0, np.pi / 2, np.pi]
                slice_idx = [int(np.argmin(np.abs(((cv - v + np.pi) % (2 * np.pi)) - np.pi))) for v in vals]
                slices = [float(cv[i]) for i in slice_idx]
        vt, axes = _display(env, grid, res["v_true"], spec, slice_idx)
        vh, _ = _display(env, grid, res["v_hat"], spec, slice_idx)
        lh, _ = _display(env, grid, np.asarray(env.hazard(grid.states)), spec, slice_idx)
        if env_meta is None:
            env_meta = {"name": env.name, "description": env.description, "state_names": list(env.state_names),
                        "lo": list(map(float, env.lo)), "hi": list(map(float, env.hi)), "grid_shape": list(env.shape),
                        "horizon": env.horizon, "periodic_dims": list(env.periodic_dims), "hazard": env.hazard.description,
                        "slice_dim": 2 if env.dim == 3 else None, "slice_values": slices}
        cases.append({"params": _jsonable(params), "env_params": _jsonable(env.params),
                      "certifier": cert.name, "certifier_params": _jsonable(cp), "certifier_info": res["info"],
                      "gap": ({"lo": list(map(float, env.gap.lo)), "hi": list(map(float, env.gap.hi)),
                               "description": env.gap.description} if env.gap else None),
                      "metrics": res["metrics"], "v_true": vt, "v_hat": vh, "hazard_l": lh,
                      "trajectories": res["trajectories"], "timing_s": res["timing_s"]})
        if verbose:
            m = res["metrics"]
            print(f"[{spec.name}] {params or '(single case)'}: FSV={m['fsv']:.4f} FUV={m['fuv']:.4f} "
                  f"gap-FSV={m['gap_fsv'] if m['gap_fsv'] is None else round(m['gap_fsv'], 4)}", flush=True)
    doc = {
        "schema": SCHEMA, "name": spec.name, "illustrative": bool(spec.illustrative),
        "provenance": {"source": "lcert.framework.run_experiment: hj_reachability ground truth vs certifier; "
                                 "metrics from lcert.metrics.compute_certification_metrics on the full solver grid",
                       "created": _dt.datetime.now(_dt.timezone.utc).isoformat(), "git_commit": _git_commit(),
                       "display_note": f"surfaces downsampled to <= {spec.display_n} pts/axis for display; metrics use full grid",
                       "border_cells": spec.border_cells},
        "spec": _jsonable({k: v for k, v in spec.__dict__.items() if k not in ("policies", "env", "certifier")}
                          | {"env": spec.env if isinstance(spec.env, str) else getattr(spec.env, "__name__", "custom"),
                             "certifier": spec.certifier if isinstance(spec.certifier, str) else getattr(spec.certifier, "__name__", "custom"),
                             "policies": [p.name for p in policies]}),
        "sweep_keys": keys, "sweep_values": _jsonable(spec.sweep),
        "env": env_meta, "axes": [a.round(5).tolist() for a in axes], "cases": cases,
    }
    if out_dir is not None:
        export(doc, out_dir)
    return doc


def export(doc: dict, out_dir) -> Path:
    """Write <name>.json and register it in the viewer bundle (data/bundle.js, works from file://)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{doc['name']}.json"
    p.write_text(json.dumps(doc, separators=(",", ":")))
    rebuild_bundle(out)
    return p


def rebuild_bundle(out_dir) -> Path:
    out = Path(out_dir)
    docs = [json.loads(f.read_text()) for f in sorted(out.glob("*.json"))]
    docs = [d for d in docs if d.get("schema") == SCHEMA]
    b = out / "bundle.js"
    b.write_text("// generated by lcert.framework.runner.rebuild_bundle; do not edit\nwindow.FSB_DATASETS = "
                 + json.dumps(docs, separators=(",", ":")) + ";\n")
    return b
