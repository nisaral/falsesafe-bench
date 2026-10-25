import numpy as np
from lcert.systems import double_integrator_f, rk4_step, double_integrator_exact


def test_rk4_matches_closed_form():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(50, 2))
    for u in (-1.0, 0.0, 0.7):
        x = x0.copy()
        for _ in range(100):
            x = rk4_step(double_integrator_f, x, u, 0.01)
        np.testing.assert_allclose(x, double_integrator_exact(x0, u, 1.0), atol=1e-10)
