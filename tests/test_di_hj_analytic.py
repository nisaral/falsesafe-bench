"""Sanity: hj_reachability BRT for the double integrator vs the analytic solution.
Thresholds are fixed here and must not be loosened without human sign-off (SCOPE.md)."""
import numpy as np
import pytest
from lcert.systems import di_safe_analytic

hj = pytest.importorskip("hj_reachability")
from lcert.hj_di import solve_di_brt  # noqa: E402


@pytest.mark.parametrize("T", [0.5, 1.0])
def test_di_value_and_set_match_analytic(T):
    grid, V = solve_di_brt(umax=1.0, T=T, n=101)
    P, Vv = np.asarray(grid.states[..., 0]), np.asarray(grid.states[..., 1])
    Va = di_safe_analytic(P, Vv, 1.0, T)
    # interior mask: drop cells within 3 cells of the grid edge (boundary-condition artefacts)
    m = np.zeros_like(V, bool); m[3:-3, 3:-3] = True
    err = np.abs(V - Va)[m]
    safe_hj, safe_a = V[m] >= 0, Va[m] >= 0
    iou = (safe_hj & safe_a).sum() / (safe_hj | safe_a).sum()
    false_safe = (safe_hj & ~safe_a).mean()
    dx = float(grid.spacings[0])
    print(f"T={T} max|V-Va|={err.max():.4g} mean={err.mean():.4g} IoU={iou:.5f} "
          f"false_safe_frac={false_safe:.5f} cell={dx:.4f}")
    assert err.max() < 5 * dx
    assert iou > 0.98
    assert false_safe < 0.005


def test_negative_control_wrong_dynamics_is_detected():
    """Solver run with umax=0.5 but compared against the umax=1.0 truth must NOT match.
    Guards against a comparison that passes trivially."""
    grid, V = solve_di_brt(umax=0.5, T=1.0, n=101)
    P, Vv = np.asarray(grid.states[..., 0]), np.asarray(grid.states[..., 1])
    Va = di_safe_analytic(P, Vv, 1.0, 1.0)
    m = np.zeros_like(V, bool); m[3:-3, 3:-3] = True
    err = np.abs(V - Va)[m].max()
    print(f"negative control max|V-Va|={err:.4g}")
    assert err > 0.1
