"""Ground-truth systems and integrators. State x=(p, v) for the double integrator."""
import numpy as np


def double_integrator_f(x, u):
    """dx/dt for p'' = u. x: (..., 2), u: (...,)."""
    x = np.asarray(x, dtype=float)
    return np.stack([x[..., 1], np.asarray(u, dtype=float) * np.ones_like(x[..., 1])], axis=-1)


def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def double_integrator_exact(x0, u, t):
    """Closed form under constant u."""
    p, v = x0[..., 0], x0[..., 1]
    return np.stack([p + v * t + 0.5 * u * t**2, v + u * t], axis=-1)


def di_safe_analytic(p, v, umax, T):
    """Analytic safe set for failure set {p < 0}, control |u|<=umax, horizon T.

    Best action is u=+umax. p(t)=p+v t+umax t^2/2, minimized on [0,T] at
    t*=clip(-v/umax, 0, T). Safe iff min_t p(t) >= 0. Returns min_t p(t),
    which is also the exact HJ value function for l(x)=p under this system.
    """
    p = np.asarray(p, float); v = np.asarray(v, float)
    ts = np.clip(-v / umax, 0.0, T)
    return p + v * ts + 0.5 * umax * ts**2


def double_integrator_hidden_physics_f(
    x,
    u,
    p_c: float = 0.7,
    w: float = 0.6,
    eta: float = 0.5,
    b: float = 0.3,
    s: float = -1.0,
):
    """True dynamics with pre-registered hidden physics inside gap G = [p_c - w/2, p_c + w/2].

    p' = v
    v' = (1 - eta * 1_G(p)) * u + b * s * 1_G(p)
    """
    x = np.asarray(x, dtype=float)
    u_arr = np.asarray(u, dtype=float) * np.ones_like(x[..., 1])
    p = x[..., 0]
    v = x[..., 1]

    in_gap = (p >= (p_c - 0.5 * w)) & (p <= (p_c + 0.5 * w))
    effective_u = np.where(in_gap, (1.0 - eta) * u_arr + b * s, u_arr)
    return np.stack([v, effective_u], axis=-1)


def double_integrator_benign_drag_f(
    x,
    u,
    p_c: float = 0.7,
    w: float = 0.6,
    drag_coeff: float = 0.5,
):
    """Benign-drag control (C4): drag opposes motion and assists braking.

    p' = v
    v' = u - drag_coeff * v * |v| * 1_G(p)
    """
    x = np.asarray(x, dtype=float)
    u_arr = np.asarray(u, dtype=float) * np.ones_like(x[..., 1])
    p = x[..., 0]
    v = x[..., 1]

    in_gap = (p >= (p_c - 0.5 * w)) & (p <= (p_c + 0.5 * w))
    drag = np.where(in_gap, -drag_coeff * v * np.abs(v), 0.0)
    return np.stack([v, u_arr + drag], axis=-1)

