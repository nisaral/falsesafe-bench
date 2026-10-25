"""Double integrator BRT with hj_reachability (JAX)."""
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj


class DoubleIntegrator(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, umax=1.0, dmax=0.0):
        super().__init__(
            control_mode="max",  # control tries to stay safe (maximize value)
            disturbance_mode="min",
            control_space=hj.sets.Box(jnp.array([-umax]), jnp.array([umax])),
            disturbance_space=hj.sets.Box(jnp.array([-dmax]), jnp.array([dmax])),
        )

    def open_loop_dynamics(self, state, time):
        return jnp.array([state[1], 0.0])

    def control_jacobian(self, state, time):
        return jnp.array([[0.0], [1.0]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0], [1.0]])


def solve_di_brt(umax=1.0, T=1.0, n=101, lo=(-1.0, -2.0), hi=(3.0, 2.0), accuracy="very_high"):
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(np.array(lo), np.array(hi)), (n, n))
    l = grid.states[..., 0]  # failure set: p < 0
    settings = hj.SolverSettings.with_accuracy(
        accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    V = hj.step(settings, DoubleIntegrator(umax), grid, 0.0, l, -T)
    return grid, np.asarray(V)
