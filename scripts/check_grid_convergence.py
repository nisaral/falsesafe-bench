"""C5 Grid Convergence Check across 51, 101, 201 grids on seed 1000."""
from __future__ import annotations
import numpy as np

from lcert.hj_di import solve_di_hidden_physics_brt
from lcert.hj_learned import solve_learned_di_brt
from lcert.models import EnsembleMLP
from lcert.data import generate_trajectory_splits
from lcert.metrics import compute_certification_metrics

def check_grid_convergence():
    print("=" * 60)
    print("=== C5 GRID CONVERGENCE CHECK (Grid 51 vs 101 vs 201) ===")
    print("=" * 60)
    lo = (-1.0, -2.0)
    hi = (3.0, 2.0)
    umax = 1.0
    horizon = 0.5
    gap_cfg = {"enabled": True, "lo": [0.4, -2.0], "hi": [1.0, 2.0]}

    # Load data and fit model for seed 1000
    data_cfg = {
        "system": "double_integrator",
        "state_lo": list(lo),
        "state_hi": list(hi),
        "umax": umax,
        "dt": 0.05,
        "trajectory_steps": 10,
        "trajectory_counts": {"train": 100, "validation": 10, "test": 10},
        "derivative_observation_noise_std": 0.01,
        "split_seed": 42,
        "seed": 1000,
        "max_sampling_attempts": 20000,
        "coverage_gap": gap_cfg,
    }
    splits, _ = generate_trajectory_splits(data_cfg)
    model = EnsembleMLP(members=3, hidden_sizes=[32, 32])
    model.fit(splits["train"]["features"], splits["train"]["targets"], epochs=40, seed=1000)

    fsv_by_grid = {}
    for n in (51, 101, 201):
        p_lin = np.linspace(lo[0], hi[0], n)
        v_lin = np.linspace(lo[1], hi[1], n)
        P, Vv = np.meshgrid(p_lin, v_lin, indexing="ij")
        states = np.stack([P, Vv], axis=-1)

        _, v_true = solve_di_hidden_physics_brt(umax=umax, T=horizon, n=n, lo=lo, hi=hi, accuracy="medium")
        _, v_hat = solve_learned_di_brt(model=model, umax=umax, horizon=horizon, grid_size=n, lo=lo, hi=hi, k_epistemic=0.0, accuracy="medium")

        cert = compute_certification_metrics(v_true, v_hat, states=states, coverage_gap=gap_cfg)
        fsv_by_grid[n] = cert.fsv
        print(f"Grid {n:3d}x{n:3d}: FSV = {cert.fsv:.6f}, Gap-FSV = {cert.gap_fsv:.6f}")

    diff_pct = abs(fsv_by_grid[101] - fsv_by_grid[201]) / fsv_by_grid[201] * 100.0
    print(f"\nRelative difference |FSV_101 - FSV_201| / FSV_201: {diff_pct:.2f}%")
    passed = diff_pct <= 25.0
    print(f"Pre-registered C5 Check (<= 25% tolerance): {'PASSED' if passed else 'WARNING'}")
    return passed

if __name__ == "__main__":
    check_grid_convergence()
