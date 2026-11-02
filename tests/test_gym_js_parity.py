"""Cross-check the in-browser gym (demos/gym/gym_core.js) against the Python framework.

* JS dynamics mirrors + RK4 + safety filter vs lcert.framework.policies.simulate / SafetyFilter
  (open-loop control sequences, shield off and on) for every JS-mirrored built-in env.
* Grid-tabulated dynamics (user-defined env) vs the Python env.
* Browser interpolation of the exported (int16) grids vs GridValue on the dequantised grids.
* Export round-trip / quantisation error bound.
Skipped if node is not installed.
"""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lcert.framework import GridValue, make_grid, simulate
from lcert.framework import gym as G
from lcert.framework import registry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "demos" / "gym" / "data"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

# load the user plug-in so the my_rover scenario/env is registered
_spec = importlib.util.spec_from_file_location("my_rover_plugin", ROOT / "examples" / "framework" / "my_rover.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)


def _js(runs, files):
    job = {"files": [str(f) for f in files], "runs": runs}
    p = subprocess.run([NODE, str(ROOT / "tests" / "gym_rollout.js")], input=json.dumps(job),
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


class OpenLoop:
    def __init__(self, us, dt):
        self.us, self.dt = us, dt

    def act(self, x, t):
        return np.asarray(self.us[min(int(round(t / self.dt)), len(self.us) - 1)], float)


def _doc(name):
    return G.load_scenario_file(DATA / f"{name}.js")


def _env(name):
    s = G.SCENARIOS[name]
    return s, registry.build(registry.ENVS, "env", s.env, s.env_params)


def _controls(env, rng, n, hold=10):
    us = []
    while len(us) < n:
        us += [env.controls[rng.integers(len(env.controls))].tolist()] * hold
    return us[:n]


MIRRORED = ["cart_wall", "drone_hover", "pendulum_balance", "boat_rocks", "car_slippery", "rover_cliff"]


@pytest.mark.parametrize("name", MIRRORED)
def test_js_rollouts_match_python(name):
    s, env = _env(name)
    doc = _doc(name)
    assert doc["env"]["js_dynamics"] == env.name
    rng = np.random.default_rng(0)
    grid = make_grid(env)
    vh = GridValue(env, grid, G.dequantize(doc["cases"][0]["v_hat"]))
    runs, py = [], []
    for k in range(3):
        x0 = list(map(float, s.start)) if k == 0 else [float(rng.uniform(a, b)) for a, b in zip(*s.start_box)]
        us = _controls(env, rng, 120)
        for shield in (False, True):
            runs.append({"scenario": name, "case": 0, "x0": x0, "us": us, "shield": shield})
            py.append(simulate(env, OpenLoop(us, s.dt), x0=x0, dt=s.dt, steps=len(us),
                               filter_value=vh if shield else None, margin=s.filter_margin))
    js = _js(runs, [DATA / f"{name}.js"])
    checked = []
    for r, p, j in zip(runs, py, js):
        assert j["dyn"] == "js-mirror"
        a, b = np.asarray(j["states"]), p["states"]
        n = min(len(a), len(b))
        # shield decisions identical (up to float32 ties right at the margin)
        agree = np.mean(np.asarray(j["active"][:n - 1]) == np.asarray(p["filter_active"][:n - 1])) if n > 1 else 1
        assert agree >= 0.97, (name, r["shield"], agree)
        assert agree == 1 or r["shield"], (name, agree)
        if agree == 1:
            err = np.abs(a[:n] - b[:n])
            for d in env.periodic_dims:
                err[:, d] = np.minimum(err[:, d], (env.hi[d] - env.lo[d]) - err[:, d])
            assert err.max() < 5e-3, (name, r["shield"], err.max())
            checked.append(r["shield"])
            assert (j["hit"] is None) == (p["hit_failure_step"] is None)
    assert checked.count(False) == 3 and checked.count(True) >= 2, checked


def test_tabulated_user_env_matches_python():
    name = "my_rover"
    s, env = _env(name)
    doc = _doc(name)
    assert doc["env"]["js_dynamics"] is None and "f_true_table" in doc
    rng = np.random.default_rng(1)
    pts = [[float(rng.uniform(lo, hi)) for lo, hi in zip(env.lo, env.hi)] for _ in range(200)]
    u = env.controls[-1].tolist()
    j = _js([{"scenario": name, "case": 0, "kind": "f", "points": pts, "u": u}], [DATA / f"{name}.js"])[0]
    ft = np.array([np.asarray(env.true_dynamics(np.asarray(p), np.asarray(u))) for p in pts])
    # piecewise dynamics are interpolated on the grid: exact away from the gap boundary
    e = np.abs(np.asarray(j["ft"]) - ft).max(axis=1)
    assert np.median(e) < 1e-3 and np.mean(e < 0.05) > 0.9
    # and a closed rollout stays close to the Python one
    us = _controls(env, rng, 100)
    x0 = list(map(float, s.start))
    p = simulate(env, OpenLoop(us, s.dt), x0=x0, dt=s.dt, steps=len(us))
    r = _js([{"scenario": name, "case": 0, "x0": x0, "us": us, "shield": False}], [DATA / f"{name}.js"])[0]
    n = min(len(p["states"]), len(r["states"]))
    assert r["dyn"] == "tabulated"
    assert np.abs(np.asarray(r["states"][:n]) - p["states"][:n]).max() < 0.05


@pytest.mark.parametrize("name", MIRRORED + ["my_rover"])
def test_js_interpolation_matches_gridvalue(name):
    s, env = _env(name)
    doc = _doc(name)
    grid = make_grid(env)
    vt = GridValue(env, grid, G.dequantize(doc["v_true"]))
    vh = GridValue(env, grid, G.dequantize(doc["cases"][-1]["v_hat"]))
    rng = np.random.default_rng(2)
    pad = 0.1 * (np.asarray(env.hi) - np.asarray(env.lo))
    pts = rng.uniform(np.asarray(env.lo) - pad, np.asarray(env.hi) + pad, size=(300, env.dim))
    j = _js([{"scenario": name, "case": len(doc["cases"]) - 1, "kind": "interp", "points": pts.tolist()}],
            [DATA / f"{name}.js"])[0]
    assert np.abs(np.asarray(j["vt"]) - vt(pts)).max() < 1e-4  # hj grid axes are float32
    assert np.abs(np.asarray(j["vh"]) - vh(pts)).max() < 1e-4
    if doc["env"]["js_dynamics"]:  # hazard l(x): JS closed form vs Python
        l = np.asarray(env.hazard(pts))
        assert np.abs(np.asarray(j["l"]) - l).max() < 1e-5


def test_export_roundtrip_and_quantisation(tmp_path):
    s = G.Scenario(name="tiny_di", title="tiny", env="di_hidden_physics", env_params={"grid_size": 31},
                   certifiers=[{"label": "naive", "certifier": "nominal_inflated", "params": {"k": 0}}],
                   start=[1.5, 0.0], start_box=([0.5, -1.0], [2.0, 1.0]))
    doc = G.build_scenario(s, verbose=False)
    p = G.export_scenario(doc, tmp_path)
    G.rebuild_index(tmp_path)
    back = G.load_scenario_file(p)
    assert back["schema"] == "falsesafe.gym/v1" and back["env"]["js_dynamics"] == "di_hidden_physics"
    _, env = None, registry.build(registry.ENVS, "env", "di_hidden_physics", {"grid_size": 31})
    from lcert.framework import ground_truth
    _, v = ground_truth(env)
    q = G.dequantize(back["v_true"])
    assert q.shape == (31, 31) and np.abs(q - np.asarray(v)).max() <= back["v_true"]["max_abs_err"] + 1e-9
    assert "tiny_di" in (tmp_path / "index.js").read_text()
    m = back["cases"][0]["metrics"]
    assert 0 <= m["fsv"] <= 1 and m["fsv"] > 0  # naive certifier is false-safe in the gap


def test_shipped_data_is_complete():
    idx = (DATA / "index.js").read_text()
    for n in MIRRORED + ["my_rover"]:
        assert f'"{n}"' in idx
        d = _doc(n)
        assert d["cases"] and all("fsv" in c["metrics"] for c in d["cases"])
        assert any("oracle" in c["label"].lower() for c in d["cases"])


def test_rebuild_bundle_keeps_datasets_without_json(tmp_path):
    from lcert.framework.runner import SCHEMA, _bundle_docs, rebuild_bundle
    a = {"schema": SCHEMA, "name": "a", "x": 1}
    (tmp_path / "a.json").write_text(json.dumps(a)); rebuild_bundle(tmp_path)
    (tmp_path / "a.json").unlink()
    (tmp_path / "b.json").write_text(json.dumps({"schema": SCHEMA, "name": "b"})); rebuild_bundle(tmp_path)
    assert [d["name"] for d in _bundle_docs(tmp_path / "bundle.js")] == ["a", "b"]
