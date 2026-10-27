from __future__ import annotations

import numpy as np
import pytest

from lcert.artifacts import check_confirmatory_guard
from lcert.systems import (
    double_integrator_f,
    double_integrator_hidden_physics_f,
    double_integrator_benign_drag_f,
    di_safe_analytic,
)


def test_hidden_physics_gap_dynamics():
    # Outside gap (p=0.2, v=-1.0): normal DI dynamics v'=u
    x_outside = np.array([[0.2, -1.0]])
    u = np.array([1.0])
    dx_out = double_integrator_hidden_physics_f(x_outside, u, p_c=0.7, w=0.6, eta=0.5, b=0.3, s=-1.0)
    assert dx_out[0, 0] == -1.0
    assert dx_out[0, 1] == 1.0  # Full umax authority

    # Inside gap (p=0.7, v=-1.0): authority cut by 50% + push of -0.3
    # v' = (1 - 0.5)*1.0 - 0.3 = 0.2
    x_inside = np.array([[0.7, -1.0]])
    dx_in = double_integrator_hidden_physics_f(x_inside, u, p_c=0.7, w=0.6, eta=0.5, b=0.3, s=-1.0)
    assert dx_in[0, 0] == -1.0
    assert dx_in[0, 1] == pytest.approx(0.2)  # Much worse deceleration -> causes crash!


def test_benign_drag_control_dynamics():
    # Inside gap with v=-1.0, u=1.0: drag opposes motion (pushes positive v)
    # v' = 1.0 - 0.5 * (-1.0) * 1.0 = 1.5 (drag aids braking)
    x_inside = np.array([[0.7, -1.0]])
    u = np.array([1.0])
    dx_drag = double_integrator_benign_drag_f(x_inside, u, p_c=0.7, w=0.6, drag_coeff=0.5)
    assert dx_drag[0, 1] == pytest.approx(1.5)  # Reality is safer than model


def test_confirmatory_guard_dev_seeds_pass():
    # Development seeds 0-9 should pass without error
    check_confirmatory_guard([0, 1, 2, 3, 4])
