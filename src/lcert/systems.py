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
