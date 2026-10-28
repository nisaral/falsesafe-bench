"""Pareto Trade-Off & Area Under the Pareto Curve (AUPC).

Pre-registered S2 definition:
- rho = FSV / FSV_nominal (where nominal = E at k=0 for the same seed).
- FUV*(rho) = min { FUV(theta) | FSV_ratio(theta) <= rho }.
- AUPC = integral_0^1 FUV*(rho) d rho (trapezoidal integration).
- Where no knob setting reaches rho, FUV*(rho) is set to penalty_fuv (largest FUV observed).
- Also reports FUV*(rho) at rho in {0.5, 0.25, 0.1}.
"""
from __future__ import annotations
import numpy as np


def compute_aupc_and_fuv_at_ratios(
    fsv_list: list[float] | np.ndarray,
    fuv_list: list[float] | np.ndarray,
    fsv_nominal: float,
    penalty_fuv: float | None = None,
    evaluation_ratios: tuple[float, ...] = (0.5, 0.25, 0.1),
    num_grid_points: int = 101,
) -> dict:
    """Compute pre-registered AUPC and FUV at fixed FSV ratios."""
    fsv_arr = np.asarray(fsv_list, dtype=float)
    fuv_arr = np.asarray(fuv_list, dtype=float)

    if fsv_nominal <= 1e-7:
        # If nominal FSV is zero or negligible
        ratios = np.zeros_like(fsv_arr)
    else:
        ratios = fsv_arr / fsv_nominal

    if penalty_fuv is None or penalty_fuv <= 0.0:
        max_obs = float(np.max(fuv_arr)) if len(fuv_arr) > 0 else 0.0
        penalty_fuv = max_obs if max_obs > 0.0 else 1.0

    # Grid of rho from 0.0 to 1.0
    rho_grid = np.linspace(0.0, 1.0, num_grid_points)
    fuv_star_grid = np.zeros_like(rho_grid)

    for i, rho in enumerate(rho_grid):
        # Settings that achieve FSV ratio <= rho
        valid_mask = ratios <= rho
        if np.any(valid_mask):
            fuv_star_grid[i] = float(np.min(fuv_arr[valid_mask]))
        else:
            fuv_star_grid[i] = float(penalty_fuv)

    # Trapezoidal integration of FUV*(rho) over [0, 1]
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aupc = float(trapz_func(fuv_star_grid, rho_grid))

    # Evaluate FUV at fixed ratios
    fuv_at_rho = {}
    for r in evaluation_ratios:
        valid_mask = ratios <= r
        if np.any(valid_mask):
            fuv_at_rho[f"fuv_at_rho_{r}"] = float(np.min(fuv_arr[valid_mask]))
        else:
            fuv_at_rho[f"fuv_at_rho_{r}"] = float(penalty_fuv)

    return {
        "aupc": aupc,
        "penalty_fuv": float(penalty_fuv),
        "fuv_star_curve": fuv_star_grid.tolist(),
        "rho_grid": rho_grid.tolist(),
        **fuv_at_rho,
    }
