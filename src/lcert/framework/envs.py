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
        params=dict(speed=speed, turn_rate=turn_rate, slip=slip, horizon=horizon, grid_size=grid_size,
                    obstacle=list(obstacle), patch=list(patch)),
        description="Dubins car with a hidden low-traction patch (generic solver).")


# ---------------------------------------------------------------------------
# Interactive-gym environments (also usable in YAML experiments).  All are low-
# dimensional (<= 3D) so the dense HJ grid stays tractable; each docstring says
# what physical system it is a reduction of.  JS mirrors of these dynamics live
# in demos/gym/gym_core.js and are cross-checked by tests/test_gym_js_parity.py.
# ---------------------------------------------------------------------------


@register_env("drone_vertical")
def drone_vertical(umax=1.0, floor=0.0, ceiling=3.0, wash_lo=0.9, wash_hi=1.7, wash=0.45, loss=0.35,
                   horizon=0.8, grid_size=81, accuracy="medium", lo=(-0.3, -2.5), hi=(3.3, 2.5), control_steps=5):
    """Quadrotor *vertical channel* (2D reduction: altitude z, climb rate vz).

    u is net vertical acceleration relative to hover (thrust - g), |u| <= umax.
    True: inside the hidden rotor-wash band z in [wash_lo, wash_hi] thrust efficiency drops by
    ``loss`` and a downdraft adds ``-wash``.  Nominal: z'' = u.  Failure: z < floor or z > ceiling.
    """
    def f_true(x, u):
        band = (x[0] >= wash_lo) & (x[0] <= wash_hi)
        return jnp.array([x[1], jnp.where(band, (1.0 - loss) * u[0] - wash, u[0])])

    return Environment(
        name="drone_vertical", state_names=("altitude z (m)", "climb rate vz (m/s)"), lo=lo, hi=hi,
        true_dynamics=f_true, nominal_dynamics=lambda x, u: jnp.array([x[1], u[0]]),
        controls=np.linspace(-umax, umax, control_steps)[:, None],
        hazard=Hazard(lambda s: jnp.minimum(s[..., 0] - floor, ceiling - s[..., 0]),
                      f"ground z < {floor} or ceiling z > {ceiling}"),
        horizon=horizon, grid_size=int(grid_size), accuracy=accuracy,
        gap=GapRegion((wash_lo, lo[1]), (wash_hi, hi[1]), f"hidden rotor wash z in [{wash_lo},{wash_hi}]: "
                      f"thrust x{1 - loss:.2f}, downdraft -{wash} m/s^2"),
        params=dict(umax=umax, floor=floor, ceiling=ceiling, wash_lo=wash_lo, wash_hi=wash_hi, wash=wash, loss=loss,
                    horizon=horizon, grid_size=grid_size),
        description="Quadrotor altitude channel (2D reduction) with a hidden rotor-wash band.")


@register_env("pendulum_payload")
def pendulum_payload(umax=1.2, g_over_l=1.0, damping=0.1, theta_max=0.8, heavy_lo=0.25, gain_loss=0.45,
                     horizon=0.8, grid_size=81, accuracy="medium", lo=(-1.1, -3.0), hi=(1.1, 3.0), control_steps=5):
    """Inverted pendulum / single-link arm balancing upright (2D: angle theta, rate omega).

    theta'' = g/l sin(theta) - damping omega + b(theta) u.  Nominal b = 1.  True: for
    theta >= heavy_lo a hidden payload shift reduces control authority by ``gain_loss``.
    Failure: |theta| > theta_max (falls over / leaves the joint limit)."""
    def f_true(x, u):
        heavy = x[0] >= heavy_lo
        return jnp.array([x[1], g_over_l * jnp.sin(x[0]) - damping * x[1] + jnp.where(heavy, 1.0 - gain_loss, 1.0) * u[0]])

    def f_nom(x, u):
        return jnp.array([x[1], g_over_l * jnp.sin(x[0]) - damping * x[1] + u[0]])

    return Environment(
        name="pendulum_payload", state_names=("angle theta (rad)", "rate omega (rad/s)"), lo=lo, hi=hi,
        true_dynamics=f_true, nominal_dynamics=f_nom, controls=np.linspace(-umax, umax, control_steps)[:, None],
        hazard=Hazard(lambda s: theta_max - jnp.abs(s[..., 0]), f"|theta| > {theta_max} rad"),
        horizon=horizon, grid_size=int(grid_size), accuracy=accuracy,
        gap=GapRegion((heavy_lo, lo[1]), (hi[0], hi[1]), f"hidden payload shift for theta >= {heavy_lo}: "
                      f"torque gain x{1 - gain_loss:.2f}"),
        params=dict(umax=umax, g_over_l=g_over_l, damping=damping, theta_max=theta_max, heavy_lo=heavy_lo,
                    gain_loss=gain_loss, horizon=horizon, grid_size=grid_size),
        description="Inverted pendulum / one-link arm (2D) with a hidden payload shift.")


@register_env("boat_current")
def boat_current(speed=0.7, rock=(0.6, -0.4, 0.55), shore=-2.4, band=(-2.4, 0.2), current=(0.2, -1.0),
                 horizon=1.5, grid_size=81, accuracy="medium", lo=(-3.0, -3.0), hi=(3.0, 3.0), n_dirs=8):
    """Boat / ship as a holonomic planar point (2D kinematic reduction: x, y; heading and
    hull dynamics ignored).  u = commanded water-relative velocity, |u| <= speed (finite set
    of ``n_dirs`` headings + stop).  True: inside a hidden current band y in ``band`` the water
    moves at ``current`` (toward the rocky shore).  Nominal: still water.
    Failure: hit the rock disc or the shoreline y < shore."""
    cx, cy, r = rock
    b0, b1 = band
    ang = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    ctrl = np.vstack([np.zeros((1, 2)), speed * np.stack([np.cos(ang), np.sin(ang)], 1)])

    def f_true(x, u):
        inb = (x[1] >= b0) & (x[1] <= b1)
        return jnp.array([u[0] + jnp.where(inb, current[0], 0.0), u[1] + jnp.where(inb, current[1], 0.0)])

    return Environment(
        name="boat_current", state_names=("x (m)", "y (m)"), lo=lo, hi=hi,
        true_dynamics=f_true, nominal_dynamics=lambda x, u: jnp.array([u[0], u[1]]), controls=ctrl,
        hazard=Hazard(lambda s: jnp.minimum(jnp.sqrt((s[..., 0] - cx) ** 2 + (s[..., 1] - cy) ** 2) - r,
                                            s[..., 1] - shore), f"rock ({cx},{cy}) r={r} or shore y < {shore}"),
        horizon=horizon, grid_size=int(grid_size), accuracy=accuracy,
        gap=GapRegion((lo[0], b0), (hi[0], b1), f"hidden current {tuple(current)} m/s for y in [{b0},{b1}]"),
        params=dict(speed=speed, rock=list(rock), shore=shore, band=list(band), current=list(current),
                    horizon=horizon, grid_size=grid_size),
        description="Boat (2D holonomic kinematic reduction) with a hidden cross-current toward rocks.")


@register_env("rover_cliff3d")
def rover_cliff3d(speed=0.6, turn_rate=1.5, cliff=2.0, slope_lo=0.8, drift=0.35, pit=(-1.2, -0.8, 0.5),
                  horizon=1.2, grid_size=41, accuracy="low", lo=(-3.0, -3.0, -np.pi), hi=(3.0, 3.0, np.pi)):
    """Mobile robot (unicycle, constant forward speed; 3D: x, y, heading) on a plateau whose
    edge y > cliff is a drop, plus a pit (disc).  True: on the hidden scree slope
    y in [slope_lo, cliff] the robot slides +``drift`` m/s toward the edge.  Nominal: no slide."""
    px, py, pr = pit

    def f_nom(x, u):
        return jnp.array([speed * jnp.cos(x[2]), speed * jnp.sin(x[2]), u[0]])

    def f_true(x, u):
        on = x[1] >= slope_lo
        return jnp.array([speed * jnp.cos(x[2]), speed * jnp.sin(x[2]) + jnp.where(on, drift, 0.0), u[0]])

    return Environment(
        name="rover_cliff3d", state_names=("x (m)", "y (m)", "heading (rad)"), lo=lo, hi=hi,
        true_dynamics=f_true, nominal_dynamics=f_nom, controls=np.array([[-turn_rate], [0.0], [turn_rate]]),
        hazard=Hazard(lambda s: jnp.minimum(cliff - s[..., 1], jnp.sqrt((s[..., 0] - px) ** 2 + (s[..., 1] - py) ** 2) - pr),
                      f"cliff y > {cliff} or pit ({px},{py}) r={pr}"),
        horizon=horizon, grid_size=int(grid_size), periodic_dims=(2,), accuracy=accuracy,
        gap=GapRegion((lo[0], slope_lo, lo[2]), (hi[0], hi[1], hi[2]), f"hidden scree slope y >= {slope_lo}: drift +{drift} m/s in y"),
        params=dict(speed=speed, turn_rate=turn_rate, cliff=cliff, slope_lo=slope_lo, drift=drift, pit=list(pit),
                    horizon=horizon, grid_size=grid_size),
        description="Unicycle rover near a cliff (3D) with a hidden scree slope.")
