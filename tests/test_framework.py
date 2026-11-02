import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lcert.framework import (ENVS, ExperimentSpec, FunctionCertifier, GridValue, NominalInflatedCertifier,
                             OracleCertifier, ground_truth, make_grid, rebuild_bundle, run_case, run_experiment,
                             simulate, solve_brt, POLICIES)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def di_env():
    return ENVS["di_hidden_physics"](grid_size=41)


def test_generic_solver_matches_hj_di(di_env):
    g, v_ref = ground_truth(di_env)  # lcert.hj_di.solve_di_hidden_physics_brt
    _, v_gen = solve_brt(di_env, di_env.true_dynamics)
    assert v_ref.shape == v_gen.shape == (41, 41)
    assert np.max(np.abs(v_ref - v_gen)) < 0.05
    assert np.mean((v_ref >= 0) == (v_gen >= 0)) > 0.99


def test_oracle_has_no_false_safe(di_env):
    spec = ExperimentSpec(name="t")
    res = run_case(di_env, OracleCertifier(), spec)
    assert res["metrics"]["fsv"] < 1e-3


def test_inflation_reduces_fsv_and_raises_fuv(di_env):
    spec = ExperimentSpec(name="t")
    cache = {}
    m0 = run_case(di_env, NominalInflatedCertifier(k=0, sigma=0.15, dims=(1,)), spec, cache)["metrics"]
    m3 = run_case(di_env, NominalInflatedCertifier(k=3, sigma=0.15, dims=(1,)), spec, cache)["metrics"]
    assert m0["fsv"] > 0 and m3["fsv"] <= m0["fsv"] and m3["fuv"] >= m0["fuv"]
    assert m0["gap_fsv"] is not None


def test_function_certifier_all_safe_is_maximally_false_safe(di_env):
    spec = ExperimentSpec(name="t")
    res = run_case(di_env, FunctionCertifier("all_safe", lambda env, grid: np.ones(env.shape)), spec)
    assert res["metrics"]["fuv"] == 0.0
    assert res["metrics"]["fsv"] == pytest.approx(1 - res["metrics"]["safe_true_volume"])


def test_simulate_and_filter():
    # finite-horizon BRTs are only invariant-ish for long horizons; use T=2 for the filter check
    di_env = ENVS["di_hidden_physics"](grid_size=41, horizon=2.0)
    g, v = ground_truth(di_env)
    pol = POLICIES["di_go_to"](target=-0.5, kp=3.0, kd=0.2)
    raw = simulate(di_env, pol, x0=[1.0, -0.5], steps=150)
    assert raw["hit_failure"]
    filt = simulate(di_env, pol, x0=[1.0, -0.5], steps=150, filter_value=GridValue(di_env, g, v))
    assert any(filt["filter_active"])
    assert not filt["hit_failure"]


def test_track_race_is_on_track_and_ice_changes_dynamics():
    """CarRacing analog: on-track is safe, off-track is failure, ice slows and understeers."""
    env = ENVS["track_race"](grid_size=15)
    mid = 0.5 * (env.params["r_inner"] + env.params["r_outer"])
    on = np.array([0.0, -mid, 0.0])
    off = np.array([0.0, 0.0, 0.0])  # origin is the infield
    assert float(env.hazard(on)) > 0
    assert float(env.hazard(off)) < 0
    ice = np.array([1.8, 0.0, 0.0])  # right-hand corner, on the ring
    assert float(env.hazard(ice)) > 0
    u = np.array([env.params["turn_rate"]])
    nom = np.asarray(env.nominal_dynamics(ice, u))
    tru = np.asarray(env.true_dynamics(ice, u))
    dry = np.asarray(env.true_dynamics(on, u))
    assert tru[2] < nom[2]  # less turn on ice
    assert np.hypot(tru[0], tru[1]) < np.hypot(nom[0], nom[1])  # slower on ice
    assert np.allclose(dry, np.asarray(env.nominal_dynamics(on, u)))


def test_track_race_oracle_fsv_near_zero():
    env = ENVS["track_race"](grid_size=15, horizon=0.6, accuracy="low")
    res = run_case(env, OracleCertifier(), ExperimentSpec(name="t"))
    assert res["metrics"]["fsv"] < 1e-3
    g = make_grid(env)
    _, v_true = ground_truth(env)
    _, v_nom = solve_brt(env, env.nominal_dynamics, None, g)
    # Coarse 15^3 + short T can give FSV=0 after the 3-cell border; the ice still
    # has to change the value field, which the gym 41^3 export measures as FSV.
    assert float(np.max(np.abs(np.asarray(v_true) - np.asarray(v_nom)))) > 0.01


def test_gridvalue_periodic():
    env = ENVS["dubins_slippery"](grid_size=15)
    g = make_grid(env)
    V = np.asarray(g.states[..., 2])  # value == heading coordinate
    gv = GridValue(env, g, np.cos(V))
    assert gv([0.0, 0.0, np.pi + 0.1])[0] == pytest.approx(gv([0.0, 0.0, -np.pi + 0.1])[0], abs=1e-6)


def test_export_schema_and_bundle(tmp_path):
    spec = ExperimentSpec(name="tiny", env_params={"grid_size": 31}, certifier_params={"sigma": 0.1},
                          policies=[{"name": "di_go_to"}], initial_states=[[1.5, -1.0]],
                          sweep={"certifier.k": [0, 2]}, steps=20, display_n=16)
    doc = run_experiment(spec, out_dir=tmp_path, verbose=False)
    d = json.loads((tmp_path / "tiny.json").read_text(encoding="utf-8"))
    assert d["schema"] == "falsesafe.framework/v1" and d["illustrative"] is False
    assert len(d["cases"]) == 2 and d["sweep_keys"] == ["certifier.k"]
    c = d["cases"][0]
    assert len(c["v_true"][0]) == len(d["axes"][0]) and len(c["v_true"][0][0]) == len(d["axes"][1])
    assert {t["mode"] for t in c["trajectories"]} == {"unfiltered", "filtered_by_certifier"}
    assert c["metrics"]["fsv"] == doc["cases"][0]["metrics"]["fsv"]
    assert "window.FSB_DATASETS" in rebuild_bundle(tmp_path).read_text(encoding="utf-8")


def test_cli_with_user_plugin(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: rover_t\nenv: rover_cliff\nenv_params: {grid_size: 31}\n"
                   "certifier: nominal_inflated\ncertifier_params: {k: 1, sigma: 0.1}\n"
                   "policies: [{name: rover_cruise}]\ninitial_states: [[0.0, 0.5]]\nsteps: 20\n")
    r = subprocess.run([sys.executable, "-m", "lcert.framework", "run", str(cfg), "--out", str(tmp_path),
                        "--plugin", str(ROOT / "examples/framework/my_rover.py")],
                       capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})  # full env: Windows needs SYSTEMROOT etc.
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads((tmp_path / "rover_t.json").read_text(encoding="utf-8"))["env"]["name"] == "rover_cliff"
