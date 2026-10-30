"""Example user plug-in: a rover approaching a cliff edge, with a hidden downhill slope.

Run:  PYTHONPATH=src python -m lcert.framework run configs/framework/my_rover.yaml \
          --plugin examples/framework/my_rover.py
"""
import jax.numpy as jnp
import numpy as np

from lcert.framework import Environment, FunctionPolicy, GapRegion, Hazard, register_env, register_policy


@register_env("rover_cliff")
def rover_cliff(slope=0.4, slope_lo=1.0, slope_hi=1.8, umax=1.0, grid_size=81, horizon=0.8):
    def f_true(x, u):  # hidden slope pushes the rover toward the cliff
        on_slope = (x[0] >= slope_lo) & (x[0] <= slope_hi)
        return jnp.array([x[1], u[0] + jnp.where(on_slope, slope, 0.0)])

    return Environment(
        name="rover_cliff", state_names=("position (m)", "velocity (m/s)"), lo=(-1.0, -2.0), hi=(3.0, 2.0),
        true_dynamics=f_true, nominal_dynamics=lambda x, u: jnp.array([x[1], u[0]]),
        controls=np.linspace(-umax, umax, 5)[:, None],
        hazard=Hazard(lambda s: 2.5 - s[..., 0], "cliff edge: p > 2.5"),
        gap=GapRegion((slope_lo, -2.0), (slope_hi, 2.0), f"hidden slope {slope} m/s^2 on [{slope_lo},{slope_hi}]"),
        horizon=horizon, grid_size=grid_size, params=dict(slope=slope, umax=umax, grid_size=grid_size, horizon=horizon))


@register_policy("rover_cruise")
def rover_cruise(speed=1.0, gain=2.0, name="rover_cruise"):
    return FunctionPolicy(name, lambda x, t: np.clip([gain * (speed - x[1])], -1.0, 1.0))
