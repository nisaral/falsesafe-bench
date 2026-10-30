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


# Optional: make this plug-in appear in the interactive gym (demos/gym).  Export with
#   PYTHONPATH=src python -m lcert.framework.gym my_rover --plugin examples/framework/my_rover.py
# The browser has no hand-written JS for this env, so the exporter tabulates f_true / f_nom
# on the grid and the gym interpolates them (marked "user-defined" in the UI).
try:
    from lcert.framework.gym import Scenario, register_scenario

    register_scenario(Scenario(
        "my_rover", "User plug-in: rover approaching a cliff (1D track)", "rover_cliff", {"grid_size": 81, "horizon": 2.5},
        [{"label": "naive nominal model (k=0)", "certifier": "nominal_inflated", "params": {"k": 0.0}},
         {"label": "nominal + margin k*sigma=0.3 on v'", "certifier": "nominal_inflated",
          "params": {"k": 3.0, "sigma": 0.1, "dims": [1]}},
         {"label": "oracle: TRUE dynamics (sanity check)", "certifier": "oracle", "params": {}}],
        render="point", manual={"x": 0}, goal=[2.2, 0.0], goal_weights=[1.0, 0.3], start=[0.0, 0.0],
        start_box=([-0.8, -1.0], [2.3, 1.5]), user_defined=True,
        blurb="Example user-defined env from examples/framework/my_rover.py (dynamics tabulated on the grid).",
        reduction="2D (position, velocity)."))
except ImportError:  # pragma: no cover
    pass
