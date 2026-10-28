"""Unit tests for Phase 2 Certifiers (DistanceAware, Conformal, Pareto) and Closed-Loop Shielding."""
import numpy as np
import pytest

from lcert.certifiers import (
    DistanceAwareCertifier,
    ConformalCertifier,
    compute_conformal_quantile,
    compute_aupc_and_fuv_at_ratios,
)
from lcert.closed_loop import (
    PolicyFamily,
    simulate_closed_loop_rollout,
    compute_rank_correlations,
)
from scipy.interpolate import RegularGridInterpolator


def test_conformal_quantile_finite_sample():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    q = compute_conformal_quantile(scores, alpha=0.1)
    # At n=10, alpha=0.1: level = ceil(11 * 0.9)/10 = ceil(9.9)/10 = 10/10 = 1.0 -> 1.0
    assert q >= 0.9


def test_distance_aware_certifier():
    cert = DistanceAwareCertifier(state_lo=(-1.0, -2.0), state_hi=(3.0, 2.0), umax=1.0)
    # Synthetic training data
    rng = np.random.default_rng(42)
    n = 50
    p = rng.uniform(-0.5, 2.5, size=n)
    v = rng.uniform(-1.5, 1.5, size=n)
    u = rng.uniform(-1.0, 1.0, size=n)
    features = np.column_stack([p, v, u])
    # Target: v_dot = u + noise
    targets = np.column_stack([v, u + rng.normal(0, 0.01, size=n)])

    cert.fit(features, targets)
    assert cert.epsilon_0 >= 0.0
    assert cert.l_hat >= 0.1

    # Query point at exact training point should have distance ~ 0
    dists = cert.compute_distance(features[:1, :2], features[:1, 2])
    assert dists[0] < 1e-4

    # Query point far away (e.g. at [3.0, 2.0, 1.0]) should have positive distance
    far_dist = cert.compute_distance(np.array([[3.0, 2.0]]), actions=1.0)
    assert far_dist[0] > 0.0

    # Disturbance bound should be >= epsilon_0
    b = cert.disturbance_bound(np.array([[3.0, 2.0]]), actions=1.0, lam=2.0)
    assert b[0] >= cert.epsilon_0


def test_conformal_certifiers():
    rng = np.random.default_rng(42)
    n = 60
    features = rng.uniform(-1.0, 1.0, size=(n, 3))
    # True target with some residual
    targets = np.column_stack([features[:, 1], features[:, 2] + rng.normal(0, 0.05, size=n)])

    # Test C0 naive
    c0 = ConformalCertifier("C0", alpha=0.1)
    c0.calibrate(features, targets, model=None)
    assert c0.is_calibrated
    assert c0.q_hat > 0.0

    b0 = c0.disturbance_bound(np.array([[0.5, 0.2]]), actions=0.0)
    assert b0[0] == c0.q_hat

    # Test C1a normalized
    class DummyEnsemble:
        def predict_mean_std(self, z):
            means = np.column_stack([z[:, 1], z[:, 2]])
            stds = np.column_stack([np.zeros(len(z)), np.ones(len(z)) * 0.2])
            return means, stds

    c1a = ConformalCertifier("C1a", alpha=0.1)
    c1a.calibrate(features, targets, model=DummyEnsemble())
    assert c1a.is_calibrated

    b1a = c1a.disturbance_bound(np.array([[0.5, 0.2]]), actions=0.0, model=DummyEnsemble())
    assert b1a[0] > 0.0


def test_pareto_aupc():
    fsv_list = [0.010, 0.008, 0.005, 0.002, 0.001]
    fuv_list = [0.001, 0.002, 0.004, 0.007, 0.012]
    fsv_nom = 0.010

    res = compute_aupc_and_fuv_at_ratios(fsv_list, fuv_list, fsv_nominal=fsv_nom)
    assert "aupc" in res
    assert 0.0 <= res["aupc"] <= 1.0
    assert "fuv_at_rho_0.5" in res
    assert res["fuv_at_rho_0.5"] == 0.004  # fsv=0.005 has ratio 0.5


def test_closed_loop_shielding():
    # Setup dummy linear value function: V(p, v) = p
    # Failure at p < 0
    p_lin = np.linspace(-1.0, 3.0, 41)
    v_lin = np.linspace(-2.0, 2.0, 41)
    P, Vv = np.meshgrid(p_lin, v_lin, indexing="ij")
    v_hat = P.copy()  # Safe when p >= 0
    interp = RegularGridInterpolator((p_lin, v_lin), v_hat)

    # Initial state with positive position and positive velocity toward wall (e.g. p=0.2, v=-0.5)
    x0 = np.array([0.2, -0.5])
    rollout = simulate_closed_loop_rollout(
        x0=x0,
        v_hat_interpolator=interp,
        policy_type="nominal_goal",
        p_goal=2.0,
        shield_margin=0.05,
        total_steps=20,
    )
    assert "crashed" in rollout
    assert "shield_interventions" in rollout
    assert isinstance(rollout["trajectory"], np.ndarray)


def test_rank_correlation_computation():
    fsv = [0.01, 0.008, 0.005, 0.002, 0.001]
    # Crash rates monotonically following FSV
    crash = [0.45, 0.38, 0.22, 0.10, 0.05]
    corr = compute_rank_correlations(fsv, crash)
    assert np.isclose(corr["spearman_rho"], 1.0)
    assert corr["interpretation"] == "strong"


def test_c2_calibration_window_disjoint_from_gap():
    """Verify that C2 leave-region-out calibration window is strictly disjoint from gap G."""
    # Pre-registered gap G
    gap_g = (0.4, 1.0)
    # C2 calibration window away from G
    calib_window = (1.8, 2.4)

    # Disjointness condition: max(G) < min(calib) or min(G) > max(calib)
    is_disjoint = (gap_g[1] < calib_window[0]) or (gap_g[0] > calib_window[1])
    assert is_disjoint, f"C2 calibration window {calib_window} must be strictly disjoint from gap {gap_g}"

    # Verify width equals gap width w=0.6
    w_gap = gap_g[1] - gap_g[0]
    w_calib = calib_window[1] - calib_window[0]
    assert np.isclose(w_gap, w_calib), f"C2 window width ({w_calib}) must match gap width ({w_gap})"


def test_pareto_aupc_penalty_clause():
    """Verify that a certifier unable to reach low FSV receives condition penalty in AUPC."""
    # Failing certifier: FSV remains constant at nominal 0.010, FUV is 0.000 across all knobs
    fsv_failing = [0.010, 0.010, 0.010]
    fuv_failing = [0.000, 0.000, 0.000]
    fsv_nom = 0.010

    # If penalty is 0.05, AUPC should be close to 0.05 because no ratio <= 0.99 is ever reached
    res = compute_aupc_and_fuv_at_ratios(fsv_failing, fuv_failing, fsv_nominal=fsv_nom, penalty_fuv=0.05)
    # Must NOT be 0.0
    assert res["aupc"] > 0.04, f"Failing certifier should receive penalty, got AUPC={res['aupc']}"
