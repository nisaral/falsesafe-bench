"""EXPLORATORY: S2-style AUPC for framework certifiers (E, L, C0, C1b, C2) on di_hidden_physics.

Mirrors the knob grids and AUPC definition of scripts/run_s2_benchmark_all6.py
(lcert.certifiers.pareto.compute_aupc_and_fuv_at_ratios, nominal = E at k=0 of the same seed,
condition-wide penalty = max FUV over all certifiers) but uses the framework pipeline
(i.i.d. training data outside the gap, state-dependent bounds for C1b/L).  NOT confirmatory;
does not write to results/.

    PYTHONPATH=src python scripts/framework_aupc.py --seeds 0 1 2 --grid 101 [--out FILE.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from lcert.certifiers.pareto import compute_aupc_and_fuv_at_ratios
from lcert.framework import (ENVS, ConformalBoundCertifier, DistanceAwareBoundCertifier, EnsembleCertifier,
                             ExperimentSpec, run_case)

K_GRID = [round(float(k), 1) for k in np.arange(0.0, 3.1, 0.2)]
LAM_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
ALPHA_GRID = [0.5, 0.2, 0.1, 0.05, 0.01]
C2_REGION = {"lo": [1.8, -2.0], "hi": [2.4, 2.0]}


def run_seed(seed, grid):
    env = ENVS["di_hidden_physics"](grid_size=grid)
    spec, cache = ExperimentSpec(name="aupc"), {}
    common = dict(members=3, hidden_sizes=(32, 32), epochs=60, n_samples=2000, seed=seed)
    curves = {}

    def go(name, certs):
        ms = [run_case(env, c, spec, cache)["metrics"] for c in certs]
        curves[name] = {"fsv": [m["fsv"] for m in ms], "fuv": [m["fuv"] for m in ms],
                        "gap_fsv": [m["gap_fsv"] for m in ms]}

    go("E", [EnsembleCertifier(k=k, **common) for k in K_GRID])
    go("L", [DistanceAwareBoundCertifier(lam=lam, dims=(1,), **common) for lam in LAM_GRID])
    for v in ("C0", "C1b", "C2"):
        go(v, [ConformalBoundCertifier(variant=v, alpha=a, dims=(1,), calib_region=C2_REGION, **common)
               for a in ALPHA_GRID])
    nom = curves["E"]["fsv"][0]
    pen = max(max(c["fuv"]) for c in curves.values()) or 1.0
    out = {}
    for n, c in curves.items():
        r = compute_aupc_and_fuv_at_ratios(c["fsv"], c["fuv"], fsv_nominal=nom, penalty_fuv=pen)
        out[n] = {"aupc": r["aupc"], **c}
    return {"seed": seed, "fsv_nominal": nom, "penalty_fuv": pen, "results": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--grid", type=int, default=101)
    ap.add_argument("--out")
    a = ap.parse_args()
    per = []
    for s in a.seeds:
        r = run_seed(s, a.grid)
        per.append(r)
        print(f"seed {s}: FSV_nominal={r['fsv_nominal']:.4f} " +
              " ".join(f"{k}={v['aupc']:.5f}" for k, v in r["results"].items()), flush=True)
    summ = {k: {"mean_aupc": float(np.mean([p["results"][k]["aupc"] for p in per])),
                "std_aupc": float(np.std([p["results"][k]["aupc"] for p in per]))} for k in per[0]["results"]}
    print("EXPLORATORY mean AUPC (lower = better):",
          ", ".join(f"{k}={v['mean_aupc']:.5f}±{v['std_aupc']:.5f}" for k, v in sorted(summ.items(), key=lambda kv: kv[1]["mean_aupc"])))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"exploratory": True, "grid": a.grid, "seeds": a.seeds, "summary": summ, "per_seed": per}, f, indent=1)


if __name__ == "__main__":
    main()
