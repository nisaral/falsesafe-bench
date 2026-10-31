"""Built-in policies, a value-function safety filter and a true-dynamics simulator."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .core import Environment, FunctionPolicy
from .registry import register_policy


@register_policy("constant")
def constant(u=(0.0,), name="constant"):
    u = np.atleast_1d(np.asarray(u, float))
    return FunctionPolicy(name, lambda x, t: u)


@register_policy("di_go_to")
def di_go_to(target=0.2, kp=2.0, kd=1.5, umax=1.0, name="di_go_to"):
    """PD controller driving a double integrator to position ``target`` (aggressive near the wall)."""
    return FunctionPolicy(name, lambda x, t: np.clip([kp * (target - x[0]) - kd * x[1]], -umax, umax))


@register_policy("dubins_goal")
def dubins_goal(goal=(2.5, 0.0), turn_rate=1.6, gain=3.0, name="dubins_goal"):
    """Turn toward a goal point, ignoring obstacles."""
    def fn(x, t):
        err = np.arctan2(goal[1] - x[1], goal[0] - x[0]) - x[2]
        err = (err + np.pi) % (2 * np.pi) - np.pi
        return np.clip([gain * err], -turn_rate, turn_rate)
    return FunctionPolicy(name, fn)


class GridValue:
    """Linear interpolation of a value function on an hj grid (periodic dims wrapped)."""

    def __init__(self, env: Environment, grid, V):
        self.env = env
        axes = [np.asarray(c) for c in grid.coordinate_vectors]
        self.periodic = tuple(env.periodic_dims)
        self.lo, self.hi = np.asarray(env.lo, float), np.asarray(env.hi, float)
        Vp, axp = np.asarray(V), list(axes)
        for d in self.periodic:  # append wrap-around slice
            Vp = np.concatenate([Vp, np.take(Vp, [0], axis=d)], axis=d)
            axp[d] = np.append(axes[d], axes[d][0] + (self.hi[d] - self.lo[d]))
        self.interp = RegularGridInterpolator(axp, Vp, bounds_error=False, fill_value=None)

    def __call__(self, x):
        x = np.array(x, float, ndmin=2)
        for d in self.periodic:
            x[:, d] = self.lo[d] + np.mod(x[:, d] - self.lo[d], self.hi[d] - self.lo[d])
        return self.interp(x)


@dataclass
class SafetyFilter:
    """Least-restrictive filter: use the task policy while V_hat(x) >= margin,
    otherwise pick the candidate control maximising V_hat one step ahead under
    the *nominal* model.  This is what a deployed learned certifier would do."""

    policy: object
    value: GridValue
    margin: float = 0.05
    dt: float = 0.02

    def act(self, env, f_nom, x, t):
        u_task = self.policy.act(x, t)
        if float(self.value(x)[0]) >= self.margin:
            return u_task, False
        nxt = np.stack([x + self.dt * np.asarray(f_nom(jnp.asarray(x), jnp.asarray(u))) for u in env.controls])
        return env.controls[int(np.argmax(self.value(nxt)))], True


def simulate(env: Environment, policy, *, x0, dt=0.02, steps=150, filter_value=None, margin=0.05):
    """Roll out under env.true_dynamics (RK4).  Returns dict with states, controls,
    whether the safety filter intervened, and whether the TRUE failure set was hit."""
    f_true = jax.jit(lambda x, u: jnp.asarray(env.true_dynamics(x, u)))
    f_nom = jax.jit(lambda x, u: jnp.asarray(env.nominal_dynamics(x, u)))
    filt = SafetyFilter(policy, filter_value, margin, dt) if filter_value is not None else None
    x = np.asarray(x0, float)
    xs, us, flags, hit = [x.copy()], [], [], None
    for i in range(steps):
        t = i * dt
        if filt is not None:
            u, flag = filt.act(env, f_nom, x, t)
        else:
            u, flag = policy.act(x, t), False
        u = np.asarray(u, float)
        k1 = np.asarray(f_true(x, u)); k2 = np.asarray(f_true(x + dt / 2 * k1, u))
        k3 = np.asarray(f_true(x + dt / 2 * k2, u)); k4 = np.asarray(f_true(x + dt * k3, u))
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        for d in env.periodic_dims:
            x[d] = env.lo[d] + np.mod(x[d] - env.lo[d], env.hi[d] - env.lo[d])
        xs.append(x.copy()); us.append(u.tolist()); flags.append(bool(flag))
        if hit is None and float(env.hazard(jnp.asarray(x))) < 0:
            hit = i + 1
            break
    return {"states": np.asarray(xs), "controls": us, "filter_active": flags,
            "hit_failure_step": hit, "hit_failure": hit is not None, "dt": dt}
