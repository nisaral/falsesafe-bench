from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import hj_reachability as hj


class DubinsCar(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, speed: float = 1.0, turn_rate: float = 1.6,
                 disturbance_bounds: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        bounds = np.asarray(disturbance_bounds, float)
        if not np.isfinite(speed) or not np.isfinite(turn_rate) or speed <= 0 or turn_rate <= 0:
            raise ValueError('Dubins speed and control authority must be positive and finite')
        if bounds.shape != (3,) or not np.isfinite(bounds).all() or np.any(bounds < 0):
            raise ValueError('Disturbance bounds must be three nonnegative finite values')
        super().__init__(control_mode='max', disturbance_mode='min',
                         control_space=hj.sets.Box(jnp.array([-turn_rate]), jnp.array([turn_rate])),
                         disturbance_space=hj.sets.Box(jnp.asarray(-bounds), jnp.asarray(bounds)))
        self.speed = speed

    def open_loop_dynamics(self, state, time):
        return jnp.array([self.speed * jnp.cos(state[2]), self.speed * jnp.sin(state[2]), 0.0])

    def control_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [1.0]])

    def disturbance_jacobian(self, state, time):
        return jnp.eye(3)


def solve_dubins_brt(speed: float = 1.0, turn_rate: float = 1.6, horizon: float = 1.0,
                     n: int = 101, lo=(-4.0, -3.0, -np.pi), hi=(4.0, 3.0, np.pi),
                     obstacles=((0.0, 0.0, 0.94),), accuracy: str = 'very_high',
                     disturbance_bounds=(0.0, 0.0, 0.0)):
    if isinstance(n, bool) or not isinstance(n, int) or n < 9:
        raise ValueError('Dubins grid n must be an integer >= 9; do not shrink a configured grid silently')
    lower, upper = np.asarray(lo, float), np.asarray(hi, float)
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all() or not np.isfinite(upper).all() or not np.all(lower < upper):
        raise ValueError('Dubins grid requires finite, ordered 3D bounds')
    if not np.isclose(upper[2] - lower[2], 2 * np.pi, rtol=0, atol=1e-12) or not np.isfinite(horizon) or horizon <= 0:
        raise ValueError('Heading must span one full period and horizon must be positive')
    circles = np.asarray(obstacles, float)
    if circles.ndim != 2 or circles.shape[1] != 3 or circles.shape[0] == 0 or not np.isfinite(circles).all() or np.any(circles[:, 2] <= 0):
        raise ValueError('At least one finite positive-radius failure circle is required')
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(lower, upper), (n, n, n), periodic_dims=(2,))
    target = jnp.min(jnp.stack([
        jnp.sqrt((grid.states[..., 0] - center_x)**2 + (grid.states[..., 1] - center_y)**2) - radius
        for center_x, center_y, radius in circles]), axis=0)
    settings = hj.SolverSettings.with_accuracy(
        accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    value = hj.step(settings, DubinsCar(speed, turn_rate, disturbance_bounds), grid,
                    0.0, target, -horizon)
    return grid, np.asarray(value)
