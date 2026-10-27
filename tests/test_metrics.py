"""Comprehensive tests for safety certification metrics (FSV, FUV, IoU, Gap-FSV, CI)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from lcert.metrics import (
    CertificationMetrics,
    ConfidenceInterval,
    PairedBootstrapResult,
    aggregate_metric_runs,
    compute_asymmetric_safety_cost,
    compute_certification_metrics,
    compute_confidence_interval,
    compute_paired_bootstrap_reduction,
    coverage_gap_mask,
    interior_mask,
)
from lcert.systems import di_safe_analytic

hj = pytest.importorskip("hj_reachability")
from lcert.hj_di import solve_di_brt  # noqa: E402


def test_interior_mask_shapes_and_counts():
    # 2D (101, 101) with border_cells=3
    mask2d = interior_mask((101, 101), border_cells=3)
    assert mask2d.shape == (101, 101)
    assert mask2d.sum() == 95 * 95
    assert not mask2d[0, 0]
    assert not mask2d[2, 50]
    assert mask2d[3, 3]
    assert mask2d[97, 97]
    assert not mask2d[98, 97]

    # 3D with periodic dim 2 (e.g. Dubins heading)
    mask3d = interior_mask((21, 21, 25), border_cells=3, periodic_dims=(2,))
    assert mask3d.shape == (21, 21, 25)
    # dims 0 and 1 trimmed by 6 cells each (15 cells remaining), dim 2 unchanged (25 cells)
    assert mask3d.sum() == 15 * 15 * 25
    assert mask3d[3, 3, 0]
    assert mask3d[3, 3, 24]
    assert not mask3d[2, 3, 0]


def test_interior_mask_validation():
    with pytest.raises(ValueError, match="too small"):
        interior_mask((6, 20), border_cells=3)

    with pytest.raises(ValueError, match="nonnegative"):
        interior_mask((20, 20), border_cells=-1)


def test_coverage_gap_mask():
    states = np.zeros((10, 10, 2))
    p = np.linspace(-2.0, 2.0, 10)
    v = np.linspace(-1.0, 1.0, 10)
    P, V = np.meshgrid(p, v, indexing="ij")
    states[..., 0] = P
    states[..., 1] = V

    gap_lo = np.array([-0.5, -0.5])
    gap_hi = np.array([0.5, 0.5])
    mask = coverage_gap_mask(states, gap_lo, gap_hi)

    expected = (P >= -0.5) & (P <= 0.5) & (V >= -0.5) & (V <= 0.5)
    np.testing.assert_array_equal(mask, expected)

    # Invalid bounds
    with pytest.raises(ValueError, match="strictly less than"):
        coverage_gap_mask(states, [0.5, 0.0], [-0.5, 0.5])


def test_certification_metrics_perfect_match():
    # Synthetic 2D value function
    v_true = np.ones((20, 20))
    v_true[:10, :] = -1.0  # top half unsafe, bottom half safe
    v_hat = v_true.copy()

    metrics = compute_certification_metrics(v_true, v_hat, border_cells=2)
    assert metrics.fsv == 0.0
    assert metrics.fuv == 0.0
    assert metrics.iou == 1.0
    assert metrics.max_abs_error == 0.0
    assert metrics.rmse == 0.0
    # Interior size is (20-4) * (20-4) = 256. Half are safe -> 128 cells.
    assert metrics.num_eval_cells == 256
    assert metrics.safe_hat_volume == 0.5
    assert metrics.safe_true_volume == 0.5


def test_certification_metrics_fsv_and_fuv():
    # Grid of size (10, 10), border=1 -> interior is 8x8 = 64 cells
    v_true = np.zeros((10, 10))
    v_hat = np.zeros((10, 10))

    # All interior cells are unsafe in ground truth (v_true = -1)
    v_true[1:9, 1:9] = -1.0
    # Model says 16 of them are safe (v_hat = +1)
    v_hat[1:5, 1:5] = 1.0
    v_hat[5:9, 1:9] = -1.0
    v_hat[1:5, 5:9] = -1.0

    metrics = compute_certification_metrics(v_true, v_hat, border_cells=1)
    # 16 false-safe cells out of 64 interior cells -> FSV = 16 / 64 = 0.25
    assert metrics.fsv == pytest.approx(0.25)
    # 0 false-unsafe cells -> FUV = 0.0
    assert metrics.fuv == 0.0
    # Ground truth has 0 safe cells, hat has 16 -> intersection = 0, IoU = 0.0
    assert metrics.iou == 0.0

    # Now test FUV (conservatism error)
    # Flip: Ground truth is safe everywhere (v_true = +1), model says 16 are unsafe (v_hat = -1)
    v_true[1:9, 1:9] = 1.0
    v_hat[1:9, 1:9] = 1.0
    v_hat[1:5, 1:5] = -1.0  # 16 cells falsely unsafe

    metrics_fuv = compute_certification_metrics(v_true, v_hat, border_cells=1)
    assert metrics_fuv.fsv == 0.0
    assert metrics_fuv.fuv == pytest.approx(0.25)
    assert metrics_fuv.iou == pytest.approx((64 - 16) / 64)


def test_gap_fsv_computation():
    grid_n = 20
    states = np.zeros((grid_n, grid_n, 2))
    coords = np.linspace(-2.0, 2.0, grid_n)
    P, V = np.meshgrid(coords, coords, indexing="ij")
    states[..., 0] = P
    states[..., 1] = V

    # Ground truth: safe when P >= 0
    v_true = P.copy()
    # Learned model: falsely believes safe everywhere (v_hat = 1.0)
    v_hat = np.ones_like(v_true)

    # Coverage gap is on negative P: [-1.5, -0.5] x [-1.5, 1.5]
    gap = {
        "enabled": True,
        "lo": [-1.5, -1.5],
        "hi": [-0.5, 1.5],
    }

    metrics = compute_certification_metrics(
        v_true, v_hat, border_cells=2, states=states, coverage_gap=gap
    )
    assert metrics.fsv > 0.0
    # Inside the coverage gap, P is always in [-1.5, -0.5], so V_true < 0 everywhere in gap.
    # Model says safe everywhere (V_hat = 1.0 >= 0).
    # Therefore, Gap-FSV must be 1.0 (100% of the gap cells in eval domain are false-safe)!
    assert metrics.gap_fsv == pytest.approx(1.0)
    assert metrics.gap_fuv == 0.0
    assert metrics.num_gap_cells is not None and metrics.num_gap_cells > 0


def test_certification_metrics_on_analytic_double_integrator():
    """Verify that compute_certification_metrics matches the established di_hj_analytic benchmark."""
    T = 0.5
    grid, V_num = solve_di_brt(umax=1.0, T=T, n=101)
    P = np.asarray(grid.states[..., 0])
    Vv = np.asarray(grid.states[..., 1])
    V_exact = di_safe_analytic(P, Vv, 1.0, T)

    metrics = compute_certification_metrics(V_exact, V_num, border_cells=3)
    dx = float(grid.spacings[0])

    assert metrics.max_abs_error < 5 * dx
    assert metrics.iou > 0.98
    assert metrics.fsv < 0.005
    assert metrics.fuv < 0.005


def test_confidence_interval_computation():
    # 5 runs with known values
    samples = [0.10, 0.12, 0.11, 0.09, 0.13]
    ci = compute_confidence_interval(samples, confidence=0.95, min_samples=5)

    assert ci.n == 5
    assert ci.mean == pytest.approx(0.11)
    assert ci.ci_lower < ci.mean < ci.ci_upper
    assert ci.half_width > 0.0
    assert math.isclose(ci.mean - ci.half_width, ci.ci_lower)

    # Identical values: zero variance
    ci_const = compute_confidence_interval([0.05, 0.05, 0.05, 0.05, 0.05])
    assert ci_const.half_width == 0.0
    assert ci_const.ci_lower == 0.05
    assert ci_const.ci_upper == 0.05

    # Enforce minimum samples check
    with pytest.raises(ValueError, match="reported results require n >= 5"):
        compute_confidence_interval([0.1, 0.2, 0.3], min_samples=5, enforce_min_samples=True)


def test_aggregate_metric_runs():
    m1 = CertificationMetrics(
        fsv=0.10, fuv=0.02, iou=0.88, safe_hat_volume=0.45, safe_true_volume=0.40,
        max_abs_error=0.15, mean_abs_error=0.03, rmse=0.05, num_eval_cells=1000
    )
    m2 = CertificationMetrics(
        fsv=0.12, fuv=0.03, iou=0.85, safe_hat_volume=0.47, safe_true_volume=0.40,
        max_abs_error=0.18, mean_abs_error=0.04, rmse=0.06, num_eval_cells=1000
    )
    m3 = CertificationMetrics(
        fsv=0.11, fuv=0.01, iou=0.87, safe_hat_volume=0.46, safe_true_volume=0.40,
        max_abs_error=0.16, mean_abs_error=0.03, rmse=0.05, num_eval_cells=1000
    )

    agg = aggregate_metric_runs([m1, m2, m3], confidence=0.95, enforce_min_samples=False)
    assert "fsv" in agg
    assert "fuv" in agg
    assert "iou" in agg
    assert agg["fsv"].mean == pytest.approx(0.11)
    assert agg["fsv"].n == 3
    # Volume metrics should have non-negative lower bounds
    assert agg["fsv"].ci_lower >= 0.0


def test_confidence_interval_clipping():
    # Mean near 0 with variance: unbounded CI lower bound would be negative
    samples = [0.0001, 0.0002, 0.0000, 0.0008, 0.0001]
    ci_unclipped = compute_confidence_interval(samples, clip_bounds=None)
    assert ci_unclipped.ci_lower < 0.0

    ci_clipped = compute_confidence_interval(samples, clip_bounds=(0.0, 1.0))
    assert ci_clipped.ci_lower == 0.0
    assert ci_clipped.ci_upper <= 1.0


def test_paired_bootstrap_reduction():
    # Nominal vs k=2.0 on 5 matched seeds
    nominal = [0.0035, 0.0028, 0.0041, 0.0019, 0.0027]
    inflated = [0.0009, 0.0007, 0.0011, 0.0005, 0.0008]

    res = compute_paired_bootstrap_reduction(nominal, inflated, target_reduction_pct=50.0, seed=42)
    assert res.mean_reduction_pct > 70.0
    assert res.ci_reduction_pct[0] > 60.0  # Lower bound of reduction is well above 50%
    assert res.p_value_meets_target >= 0.99
    assert res.n == 5


def test_asymmetric_safety_cost():
    # Nominal: high FSV, low FUV
    cost_nominal = compute_asymmetric_safety_cost(fsv=0.0030, fuv=0.0005, cost_false_safe=10.0, cost_false_unsafe=1.0)
    # Inflated: low FSV, higher FUV
    cost_inflated = compute_asymmetric_safety_cost(fsv=0.0008, fuv=0.0042, cost_false_safe=10.0, cost_false_unsafe=1.0)

    # Cost of nominal = 10*0.0030 + 1*0.0005 = 0.0305
    # Cost of inflated = 10*0.0008 + 1*0.0042 = 0.0122
    assert cost_nominal == pytest.approx(0.0305)
    assert cost_inflated == pytest.approx(0.0122)
    assert cost_inflated < cost_nominal * 0.5  # More than 50% net cost reduction!

