"""Generic HJ reachability for any :class:`Environment` (wraps hj_reachability)."""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj

from .core import Environment


class CallableDynamics(hj.Dynamics):
    """hj.Dynamics from a plain ``f(x, u)`` + a state-dependent disturbance bound.

    H(x, p) = max_{u in U_finite} p . f(x, u) - sum_i |p_i| d_i(x)
    (control maximises safety, disturbance d in [-d(x), d(x)] opposes it).
    """

    def __init__(self, f: Callable, controls: np.ndarray, dbound: Callable | None = None):
        controls = np.atleast_2d(np.asarray(controls, dtype=float))
        super().__init__(
            control_mode="max",
            disturbance_mode="min",
            control_space=hj.sets.Box(jnp.asarray(controls.min(0)), jnp.asarray(controls.max(0))),
            disturbance_space=hj.sets.Box(jnp.asarray([-1.0]), jnp.asarray([1.0])),
        )
        self.f = f
        self.controls = jnp.asarray(controls)
        self.dbound = dbound

    def _d(self, state):
        if self.dbound is None:
            return jnp.zeros_like(state)
        return jnp.broadcast_to(jnp.asarray(self.dbound(state)), state.shape)

    def _cand(self, state):
        return jax.vmap(lambda u: jnp.asarray(self.f(state, u)))(self.controls)

    def __call__(self, state, control, disturbance, time):
        return jnp.asarray(self.f(state, control)) + disturbance

    def optimal_control_and_disturbance(self, state, time, grad_value):
        idx = jnp.argmax(self._cand(state) @ grad_value)
        return self.controls[idx], -self._d(state) * jnp.sign(grad_value)

    def hamiltonian(self, state, time, value, grad_value):
        return jnp.max(self._cand(state) @ grad_value) - jnp.sum(jnp.abs(grad_value) * self._d(state))

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        return jnp.max(jnp.abs(self._cand(state)), axis=0) + self._d(state)


def make_grid(env: Environment) -> hj.Grid:
    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(np.asarray(env.lo, float), np.asarray(env.hi, float)), env.shape,
        periodic_dims=tuple(env.periodic_dims) or None)


def solve_brt(env: Environment, f: Callable, dbound: Callable | None = None, grid=None):
    """Solve the BRT of env.hazard under dynamics f (+ optional disturbance bound)."""
    grid = grid if grid is not None else make_grid(env)
    target = jnp.asarray(env.hazard(grid.states))
    settings = hj.SolverSettings.with_accuracy(
        env.accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    V = hj.step(settings, CallableDynamics(f, env.controls, dbound), grid, 0.0, target, -env.horizon)
    V = np.asarray(V)
    if not np.isfinite(V).all():
        raise FloatingPointError(f"HJ solve for {env.name} produced non-finite values")
    return grid, V


def ground_truth(env: Environment):
    """Ground-truth BRT under env.true_dynamics (or the env's own solver override)."""
    if env.ground_truth_solver is not None:
        return env.ground_truth_solver(env)
    return solve_brt(env, env.true_dynamics)
