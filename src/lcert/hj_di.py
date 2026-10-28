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


class DoubleIntegratorHiddenPhysics(hj.Dynamics):
    """True dynamics with pre-registered hidden physics:
    dot_p = v
    dot_v = (1 - eta * 1_G(p)) * u + b * s * 1_G(p)
    where G = [p_c - w/2, p_c + w/2].
    """
    def __init__(self, umax=1.0, pc=0.7, w=0.6, eta=0.5, b=0.3, s=-1.0):
        super().__init__(
            control_mode="max",
            disturbance_mode="min",
            control_space=hj.sets.Box(jnp.array([-umax]), jnp.array([umax])),
            disturbance_space=hj.sets.Box(jnp.array([0.0]), jnp.array([0.0])),
        )
        self.umax = umax
        self.pc = pc
        self.w = w
        self.eta = eta
        self.b = b
        self.s = s

    def __call__(self, state, control, disturbance, time):
        p, v = state[0], state[1]
        u = control[0]
        in_gap = (p >= (self.pc - 0.5 * self.w)) & (p <= (self.pc + 0.5 * self.w))
        v_dot = jnp.where(in_gap, (1.0 - self.eta) * u + self.b * self.s, u)
        return jnp.array([v, v_dot])

    def optimal_control_and_disturbance(self, state, time, grad_value):
        p, v = state[0], state[1]
        p_v = grad_value[1]
        u_opt = jnp.where(p_v >= 0.0, self.umax, -self.umax)
        return jnp.array([u_opt]), jnp.array([0.0])

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        del time, value, grad_value_box
        return jnp.array([jnp.abs(state[1]) + 0.5, 2.5])


def solve_di_hidden_physics_brt(
    umax=1.0, T=0.5, n=101, lo=(-1.0, -2.0), hi=(3.0, 2.0),
    pc=0.7, w=0.6, eta=0.5, b=0.3, s=-1.0, accuracy="very_high"
):
    """Solve exact ground-truth BRT under the true hidden-physics dynamics."""
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(np.array(lo), np.array(hi)), (n, n))
    l = grid.states[..., 0]
    settings = hj.SolverSettings.with_accuracy(
        accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    dynamics = DoubleIntegratorHiddenPhysics(umax=umax, pc=pc, w=w, eta=eta, b=b, s=s)
    V = hj.step(settings, dynamics, grid, 0.0, l, -T)
    return grid, np.asarray(V)


class DoubleIntegratorBenignDrag(hj.Dynamics):
    """Benign drag control dynamics:
    dot_p = v
    dot_v = u - gamma * v * |v| * 1_G(p)
    where G = [p_c - w/2, p_c + w/2].
    Drag opposes velocity and aids braking.
    """
    def __init__(self, umax=1.0, pc=0.7, w=0.6, drag_coeff=0.5):
        super().__init__(
            control_mode="max",
            disturbance_mode="min",
            control_space=hj.sets.Box(jnp.array([-umax]), jnp.array([umax])),
            disturbance_space=hj.sets.Box(jnp.array([0.0]), jnp.array([0.0])),
        )
        self.umax = umax
        self.pc = pc
        self.w = w
        self.drag_coeff = drag_coeff

    def __call__(self, state, control, disturbance, time):
        p, v = state[0], state[1]
        u = control[0]
        in_gap = (p >= (self.pc - 0.5 * self.w)) & (p <= (self.pc + 0.5 * self.w))
        drag = jnp.where(in_gap, -self.drag_coeff * v * jnp.abs(v), 0.0)
        return jnp.array([v, u + drag])

    def optimal_control_and_disturbance(self, state, time, grad_value):
        p, v = state[0], state[1]
        p_v = grad_value[1]
        u_opt = jnp.where(p_v >= 0.0, self.umax, -self.umax)
        return jnp.array([u_opt]), jnp.array([0.0])

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        del time, value, grad_value_box
        return jnp.array([jnp.abs(state[1]) + 0.5, 3.5])


def solve_di_benign_drag_brt(
    umax=1.0, T=0.5, n=101, lo=(-1.0, -2.0), hi=(3.0, 2.0),
    pc=0.7, w=0.6, drag_coeff=0.5, accuracy="very_high"
):
    """Solve ground-truth BRT under benign drag (C4 control)."""
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(np.array(lo), np.array(hi)), (n, n))
    l = grid.states[..., 0]
    settings = hj.SolverSettings.with_accuracy(
        accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    dynamics = DoubleIntegratorBenignDrag(umax=umax, pc=pc, w=w, drag_coeff=drag_coeff)
    V = hj.step(settings, dynamics, grid, 0.0, l, -T)
    return grid, np.asarray(V)


