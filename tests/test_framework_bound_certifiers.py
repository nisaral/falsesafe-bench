"""Framework adapters of the conformal (C0/C1b/C2) and distance-aware (L) certifiers."""
import json

import numpy as np
import pytest

from lcert.certifiers.conformal import compute_conformal_quantile
from lcert.framework import (ENVS, ConformalBoundCertifier, DistanceAwareBoundCertifier, ExperimentSpec,
                             grid_lookup, make_grid, run_case, run_experiment)

FAST = dict(members=2, hidden_sizes=(16, 16), epochs=15, n_samples=600, seed=0, n_calib=200)
C2_REGION = {"lo": [1.8, -2.0], "hi": [2.4, 2.0]}


@pytest.fixture(scope="module")
def env():
    return ENVS["di_hidden_physics"](grid_size=31)


@pytest.fixture(scope="module")
def cache():
    return {}


def test_grid_lookup_exact_on_nodes_and_linear_between(env):
    g = make_grid(env)
    S = np.asarray(g.states)
    vals = 2.0 * S[..., 0] - 0.5 * S[..., 1] + 1.0  # linear -> multilinear interp is exact everywhere
    f = grid_lookup(env, g, vals)
    assert float(f(S[3, 7])) == pytest.approx(vals[3, 7], abs=1e-5)
    x = np.array([0.123, -0.777])
    assert float(f(x)) == pytest.approx(2 * 0.123 + 0.5 * 0.777 + 1.0, abs=1e-4)


@pytest.mark.parametrize("variant", ["C0", "C1b", "C2"])
def test_conformal_variants_run_and_bounds(env, cache, variant):
    c = ConformalBoundCertifier(variant=variant, alpha=0.1, dims=(1,), calib_region=C2_REGION, **FAST)
    res = run_case(env, c, ExperimentSpec(name="t"), cache)
    m, info = res["metrics"], res["info"]
    assert res["v_hat"].shape == env.shape and np.isfinite(res["v_hat"]).all()
    assert 0 <= m["fsv"] <= 1 and m["gap_fsv"] is not None
    assert info["q_hat"] > 0 and info["dims"] == [1]
    st = info["bound_grid_stats"]
    if variant == "C1b":
        assert st["max"] > st["min"]                        # state-dependent
        assert info["bound_in_gap_mean"] > st["min"]        # larger away from data
    else:
        assert st["max"] == pytest.approx(st["min"]) == pytest.approx(info["q_hat"])


def test_conformal_quantile_reproduced(env, cache):
    c = ConformalBoundCertifier(variant="C0", alpha=0.2, dims=(1,), **FAST)
    c.certify(env, make_grid(env))
    # recompute from scratch with the same calibration draw
    from lcert.framework.bound_certifiers import _sample
    rng = np.random.default_rng(FAST["seed"] + c.calib_seed_offset)
    X, U = _sample(env, FAST["n_calib"], rng)
    Y = c._truth(env, X, U, rng)
    assert c.q_hat == pytest.approx(compute_conformal_quantile(c._resid(X, U, Y), 0.2))


def test_alpha_monotone_and_inflation_reduces_fsv(env, cache):
    g = make_grid(env)
    q = [ConformalBoundCertifier(variant="C0", alpha=a, dims=(1,), **FAST) for a in (0.5, 0.01)]
    for c in q:
        c.certify(env, g)
    assert q[1].q_hat >= q[0].q_hat
    spec = ExperimentSpec(name="t")
    m0 = run_case(env, DistanceAwareBoundCertifier(lam=0.0, dims=(1,), **FAST), spec, cache)["metrics"]
    m3 = run_case(env, DistanceAwareBoundCertifier(lam=3.0, dims=(1,), **FAST), spec, cache)["metrics"]
    assert m3["fsv"] <= m0["fsv"] and m3["fuv"] >= m0["fuv"]


def test_distance_aware_matches_repo_formula(env):
    c = DistanceAwareBoundCertifier(lam=1.5, dims=(1,), **FAST)
    c.certify(env, make_grid(env))
    assert c.eps0 > 0 and c.l_hat >= 0.1
    x, u = np.array([[0.7, 0.0]]), np.array([[1.0]])
    d = c._dist(x, u)
    assert c._bound(env, x, u)[0] == pytest.approx(c.eps0 + 1.5 * c.l_hat * d[0])
    # bound grid is the max over candidate controls
    D = c._bound_grid
    S = np.asarray(make_grid(env).states).reshape(-1, 2)
    i = 123
    expect = max(c._bound(env, S[i:i + 1], np.array([[uu]]))[0] for uu in env.controls[:, 0])
    assert D.reshape(-1)[i] == pytest.approx(expect)


def test_c2_rejects_region_overlapping_gap(env):
    c = ConformalBoundCertifier(variant="C2", dims=(1,), calib_region={"lo": [0.5, -2], "hi": [0.9, 2]}, **FAST)
    with pytest.raises(ValueError):
        c.certify(env, make_grid(env))
    with pytest.raises(ValueError):
        ConformalBoundCertifier(variant="C1a")


def test_preset_comparison_export(tmp_path):
    spec = ExperimentSpec(
        name="cmp", env_params={"grid_size": 25}, certifier="ensemble",
        certifier_params={k: v for k, v in FAST.items() if k != "n_calib"},
        certifier_presets={"C0": {"certifier": "conformal", "params": {"variant": "C0", "dims": [1]}},
                           "L": {"certifier": "distance_aware", "params": {"lam": 1.0, "dims": [1]}},
                           "E": {"certifier": "ensemble", "params": {"k": 2}}},
        sweep={"certifier": ["E", "C0", "L"]}, display_n=13)
    run_experiment(spec, out_dir=tmp_path, verbose=False)
    d = json.loads((tmp_path / "cmp.json").read_text())
    assert [c["certifier"] for c in d["cases"]] == ["ensemble", "conformal_C0", "distance_aware"]
    assert d["sweep_keys"] == ["certifier"] and "eps0" in d["cases"][2]["certifier_info"]
    assert "window.FSB_DATASETS" in (tmp_path / "bundle.js").read_text()
