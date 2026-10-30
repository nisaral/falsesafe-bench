"""Built-in environments.  Each factory takes keyword params so sweeps can vary them."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .core import Environment, GapRegion, Hazard
from .registry import register_env


@register_env("di_hidden_physics")
def double_integrator_hidden_physics(umax=1.0, pc=0.7, w=0.6, eta=0.5, b=0.3, s=-1.0,
                                     horizon=0.5, grid_size=101, accuracy="medium",
                                     lo=(-1.0, -2.0), hi=(3.0, 2.0), control_steps=5):
    """The repo's preregistered double integrator with hidden physics.

    True: p' = v, v' = (1 - eta 1_G(p)) u + b s 1_G(p), G = [pc - w/2, pc + w/2].
    Nominal (what a learner/designer believes): plain double integrator.
    Failure: p < 0 (wall).  Ground truth reuses ``lcert.hj_di.solve_di_hidden_physics_brt``.
    """
    lo_g, hi_g = pc - 0.5 * w, pc + 0.5 * w

    def f_true(x, u):
        in_gap = (x[0] >= lo_g) & (x[0] <= hi_g)
        return jnp.array([x[1], jnp.where(in_gap, (1.0 - eta) * u[0] + b * s, u[0])])

    def f_nom(x, u):
        return jnp.array([x[1], u[0]])

    def gt(env):
        from lcert.hj_di import solve_di_hidden_physics_brt
        return solve_di_hidden_physics_brt(umax=umax, T=horizon, n=int(grid_size), lo=tuple(lo), hi=tuple(hi),
                                           pc=pc, w=w, eta=eta, b=b, s=s, accuracy=accuracy)

    return Environment(
        name="di_hidden_physics", state_names=("position p", "velocity v"), lo=lo, hi=hi,
        true_dynamics=f_true, nominal_dynamics=f_nom,
        controls=np.linspace(-umax, umax, control_steps)[:, None],
        hazard=Hazard(lambda s_: s_[..., 0], "wall: p < 0"),
        horizon=horizon, grid_size=int(grid_size), accuracy=accuracy,
        gap=GapRegion((lo_g, lo[1]), (hi_g, hi[1]), f"hidden physics: p in [{lo_g:.2f},{hi_g:.2f}], eta={eta}, b*s={b*s}"),
        ground_truth_solver=gt,
        params=dict(umax=umax, pc=pc, w=w, eta=eta, b=b, s=s, horizon=horizon, grid_size=grid_size),
        description="Double integrator, hidden actuator loss + drift band (lcert.hj_di).")


@register_env("dubins_slippery")
def dubins_slippery(speed=1.0, turn_rate=1.6, obstacle=(0.0, 0.0, 0.8), patch=(0.8, 2.0, -1.5, 1.5),
                    slip=0.5, horizon=1.0, grid_size=41, accuracy="low"):
    """Dubins car (x, y, theta) avoiding a disc; inside a 'slippery' x-y patch the
    true turn rate is scaled by (1 - slip).  Nominal model = lcert.hj_dubins.DubinsCar
    dynamics without the slip.  Gap box covers the patch for all headings."""
    cx, cy, r = obstacle
    x0, x1, y0, y1 = patch

    def f_nom(x, u):
        return jnp.array([speed * jnp.cos(x[2]), speed * jnp.sin(x[2]), u[0]])

    def f_true(x, u):
        inside = (x[0] >= x0) & (x[0] <= x1) & (x[1] >= y0) & (x[1] <= y1)
        return jnp.array([speed * jnp.cos(x[2]), speed * jnp.sin(x[2]), jnp.where(inside, (1 - slip) * u[0], u[0])])

    return Environment(
        name="dubins_slippery", state_names=("x", "y", "heading"), lo=(-3.0, -3.0, -np.pi), hi=(3.0, 3.0, np.pi),
        true_dynamics=f_true, nominal_dynamics=f_nom, controls=np.array([[-turn_rate], [0.0], [turn_rate]]),
        hazard=Hazard(lambda s_: jnp.sqrt((s_[..., 0] - cx) ** 2 + (s_[..., 1] - cy) ** 2) - r, f"disc ({cx},{cy}) r={r}"),
        horizon=horizon, grid_size=int(grid_size), periodic_dims=(2,), accuracy=accuracy,
        gap=GapRegion((x0, y0, -np.pi), (x1, y1, np.pi), f"slippery patch x in [{x0},{x1}], y in [{y0},{y1}], slip={slip}"),
        params=dict(speed=speed, turn_rate=turn_rate, slip=slip, horizon=horizon, grid_size=grid_size),
        description="Dubins car with a hidden low-traction patch (generic solver).")
