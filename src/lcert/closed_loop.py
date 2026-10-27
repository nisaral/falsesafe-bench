"""Closed-Loop Policy Family & Shielded Crash Validation (Section 4.2 S6).

Pre-registered S6 Protocol:
- Goal behind the gap (p_goal = 2.5).
- Policy family:
  1. random: random exploratory actions u in [-umax, umax].
  2. nominal_goal: proportional goal-seeking policy targeting p_goal.
  3. shortcut_seeking: maximum speed path driving through the gap.
- Initial states: sampled uniformly from the true safe set S* (crashes avoidable in principle).
- Safety shield: monitors V_hat(p, v). If V_hat <= delta_margin, overrides with u* = +umax.
- Crash: true hidden-physics trajectory enters failure set {p < 0}.
- Rank correlation: Spearman rho_s and Kendall tau of FSV and FSV_pi against crash rate.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import spearmanr, kendalltau

from .systems import double_integrator_hidden_physics_f, rk4_step


class PolicyFamily:
    """Closed-loop control policies targeting goal behind the coverage gap."""

    @staticmethod
    def get_action(policy_type: str, state: np.ndarray, umax: float = 1.0, p_goal: float = 2.5, rng=None) -> float:
        p, v = float(state[0]), float(state[1])
        if policy_type == "random":
            if rng is None:
                rng = np.random.default_rng()
            return float(rng.uniform(-umax, umax))
        elif policy_type == "nominal_goal":
            # PD control towards p_goal
            kp = 1.5
            kd = 1.0
            u_cmd = kp * (p_goal - p) - kd * v
            return float(np.clip(u_cmd, -umax, umax))
        elif policy_type == "shortcut_seeking":
            # Drive at maximum speed forward through the gap towards goal
            if p < p_goal:
                return float(umax) if v < 1.5 else 0.0
            else:
                return float(-umax)
        else:
            return 0.0


def simulate_closed_loop_rollout(
    x0: np.ndarray,
    v_hat_interpolator: RegularGridInterpolator,
    policy_type: str = "nominal_goal",
    umax: float = 1.0,
    p_goal: float = 2.5,
    shield_margin: float = 0.02,
    dt: float = 0.05,
    total_steps: int = 60,
    pc: float = 0.7,
    w: float = 0.6,
    eta: float = 0.5,
    b: float = 0.3,
    s: float = -1.0,
    rng=None,
) -> dict:
    """Simulate a single closed-loop trajectory under true hidden physics with safety shield.
    
    Returns:
        dict containing 'crashed': bool, 'shield_interventions': int, 'trajectory': np.ndarray.
    """
    state = np.asarray(x0, dtype=float).copy()
    trajectory = [state.copy()]
    crashed = False
    interventions = 0

    for step in range(total_steps):
        # 1. Check shield value function
        # Clip state within interpolator bounds
        p_c = float(np.clip(state[0], -0.99, 2.99))
        v_c = float(np.clip(state[1], -1.99, 1.99))
        v_val = float(v_hat_interpolator((p_c, v_c)))

        # 2. Action selection
        if v_val <= shield_margin:
            # Shield intervenes: maximum braking acceleration
            u_applied = umax
            interventions += 1
        else:
            u_applied = PolicyFamily.get_action(policy_type, state, umax=umax, p_goal=p_goal, rng=rng)

        # 3. Step forward under true hidden physics
        state = rk4_step(
            lambda x, u: double_integrator_hidden_physics_f(x, u, p_c=pc, w=w, eta=eta, b=b, s=s),
            state,
            u_applied,
            dt,
        )
        trajectory.append(state.copy())

        # 4. Check failure boundary p < 0
        if state[0] < 0.0:
            crashed = True
            break

    return {
        "crashed": bool(crashed),
        "shield_interventions": int(interventions),
        "trajectory": np.asarray(trajectory),
    }


def evaluate_closed_loop_crash_rates(
    v_true_grid: np.ndarray,
    v_hat_grid: np.ndarray,
    p_lin: np.ndarray,
    v_lin: np.ndarray,
    num_initial_states: int = 100,
    policy_types: tuple[str, ...] = ("random", "nominal_goal", "shortcut_seeking"),
    umax: float = 1.0,
    shield_margin: float = 0.02,
    seed: int = 20261027,
    pc: float = 0.7,
    w: float = 0.6,
    eta: float = 0.5,
    b: float = 0.3,
    s: float = -1.0,
) -> dict[str, float]:
    """Sample initial states from true safe set S* and evaluate empirical crash rates."""
    rng = np.random.default_rng(seed)

    # Find valid initial states in true safe set S* (V_true >= 0.05)
    # Also restrict to initial states that can potentially reach the gap (p in [0.8, 2.2], v in [-1.5, 0.5])
    safe_mask = (v_true_grid >= 0.05)
    P, Vv = np.meshgrid(p_lin, v_lin, indexing="ij")
    cand_p = P[safe_mask]
    cand_v = Vv[safe_mask]

    if len(cand_p) == 0:
        return {p_type: 0.0 for p_type in policy_types}

    # Sample initial states
    indices = rng.choice(len(cand_p), size=min(num_initial_states, len(cand_p)), replace=False)
    init_states = np.column_stack([cand_p[indices], cand_v[indices]])

    # Build interpolator for learned value function
    interpolator = RegularGridInterpolator((p_lin, v_lin), v_hat_grid, bounds_error=False, fill_value=None)

    crash_rates = {}
    for p_type in policy_types:
        crashes = 0
        for x0 in init_states:
            res = simulate_closed_loop_rollout(
                x0=x0,
                v_hat_interpolator=interpolator,
                policy_type=p_type,
                umax=umax,
                shield_margin=shield_margin,
                pc=pc,
                w=w,
                eta=eta,
                b=b,
                s=s,
                rng=rng,
            )
            if res["crashed"]:
                crashes += 1
        crash_rates[p_type] = float(crashes / len(init_states))

    crash_rates["mean_crash_rate"] = float(np.mean(list(crash_rates.values())))
    return crash_rates


def compute_rank_correlations(fsv_values: list[float], crash_rates: list[float]) -> dict:
    """Compute Spearman rho_s and Kendall tau rank correlations against crash rates."""
    fsv_arr = np.asarray(fsv_values, dtype=float)
    cr_arr = np.asarray(crash_rates, dtype=float)

    if len(fsv_arr) < 3 or np.all(cr_arr == cr_arr[0]):
        return {
            "spearman_rho": 0.0,
            "spearman_pvalue": 1.0,
            "kendall_tau": 0.0,
            "kendall_pvalue": 1.0,
            "interpretation": "insufficient_variance",
        }

    rho, rho_p = spearmanr(fsv_arr, cr_arr)
    tau, tau_p = kendalltau(fsv_arr, cr_arr)

    # Pre-registered labels: >= 0.7 strong, 0.4-0.7 moderate, < 0.4 weak
    if rho >= 0.7:
        interp = "strong"
    elif rho >= 0.4:
        interp = "moderate"
    else:
        interp = "weak"

    return {
        "spearman_rho": float(rho),
        "spearman_pvalue": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_pvalue": float(tau_p),
        "interpretation": interp,
    }
