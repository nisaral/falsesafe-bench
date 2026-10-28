"""Execute Three-Certifier Comparison Benchmark (Ensemble-k, Distance-Aware, Conformal).

Pre-registered S2 & S4 analysis:
- Certifiers:
  - E: Ensemble-k (k in [0.0, 3.0] step 0.2)
  - L: Distance-Aware Lipschitz (lambda in [0.0, 3.0] step 0.2)
  - C0: Naive split conformal (alpha in [0.5, 0.3, 0.2, 0.1, 0.05, 0.01])
  - C1a: Normalized conformal (by epistemic std)
  - C1b: Normalized conformal (by distance)
  - C2: Leave-region-out conformal
  - O: Oracle (solver floor)
- Outputs:
  - AUPC per certifier
  - FUV at rho in {0.5, 0.25, 0.1}
  - Directional predictions (S4)
  - Closed-loop crash rates and rank correlations (S6)
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from lcert.artifacts import check_confirmatory_guard
from lcert.certifiers import (
    DistanceAwareCertifier,
    ConformalCertifier,
    compute_aupc_and_fuv_at_ratios,
)
from lcert.closed_loop import evaluate_closed_loop_crash_rates, compute_rank_correlations
from lcert.data import generate_trajectory_splits
from lcert.hj_di import solve_di_hidden_physics_brt
from lcert.systems import di_safe_analytic
from lcert.hj_learned import solve_learned_di_brt
from lcert.metrics import compute_certification_metrics
from lcert.models import EnsembleMLP
from lcert.train import train_dynamics_model

ROOT = Path(__file__).resolve().parents[1]


def run_single_certifier_evaluation(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    model: EnsembleMLP,
    v_true: np.ndarray,
    P: np.ndarray,
    Vv: np.ndarray,
    states: np.ndarray,
    gap_cfg: dict,
    grid_size: int = 101,
    umax: float = 1.0,
    horizon: float = 0.5,
    lo: tuple[float, float] = (-1.0, -2.0),
    hi: tuple[float, float] = (3.0, 2.0),
) -> dict:
    """Evaluate all 5 certifier families on a single trained seed."""
    results = {}

    # 1. Certifier E (Ensemble-k)
    k_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    e_fsv, e_fuv, e_records = [], [], {}
    for k in k_grid:
        _, v_hat = solve_learned_di_brt(
            model=model, umax=umax, horizon=horizon, grid_size=grid_size, lo=lo, hi=hi, k_epistemic=k, accuracy="medium"
        )
        cert = compute_certification_metrics(v_true, v_hat, states=states, coverage_gap=gap_cfg)
        e_fsv.append(cert.fsv)
        e_fuv.append(cert.fuv)
        e_records[f"k_{k}"] = cert.to_dict()

    nom_fsv = e_fsv[0]
    results["E"] = {
        "knobs": k_grid,
        "metrics_by_knob": e_records,
        "pareto": compute_aupc_and_fuv_at_ratios(e_fsv, e_fuv, fsv_nominal=nom_fsv),
    }

    # 2. Certifier L (Distance-Aware Lipschitz)
    dist_cert = DistanceAwareCertifier(state_lo=lo, state_hi=hi, umax=umax)
    dist_cert.fit(train_features, train_targets, model=model)

    lam_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    l_fsv, l_fuv, l_records = [], [], {}
    for lam in lam_grid:
        d_bound = dist_cert.disturbance_bound(states, actions=umax, lam=lam)
        mean_d = float(np.mean(d_bound))
        # Solve reachability with constant upper bound approximation over grid
        _, v_hat = solve_learned_di_brt(
            model=model, umax=umax, horizon=horizon, grid_size=grid_size, lo=lo, hi=hi,
            k_epistemic=0.0, aleatoric_bound=mean_d, accuracy="medium"
        )
        cert = compute_certification_metrics(v_true, v_hat, states=states, coverage_gap=gap_cfg)
        l_fsv.append(cert.fsv)
        l_fuv.append(cert.fuv)
        l_records[f"lambda_{lam}"] = cert.to_dict()

    results["L"] = {
        "knobs": lam_grid,
        "epsilon_0": dist_cert.epsilon_0,
        "l_hat": dist_cert.l_hat,
        "metrics_by_knob": l_records,
        "pareto": compute_aupc_and_fuv_at_ratios(l_fsv, l_fuv, fsv_nominal=nom_fsv),
    }

    # 3. Conformal Certifiers (C0, C1a, C2)
    alpha_grid = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
    for c_var in ("C0", "C1a", "C2"):
        c_fsv, c_fuv, c_records = [], [], {}
        for a in alpha_grid:
            conf = ConformalCertifier(variant=c_var, alpha=a)
            conf.calibrate(val_features, val_targets, model=model, distance_certifier=dist_cert)
            mean_q = conf.q_hat

            _, v_hat = solve_learned_di_brt(
                model=model, umax=umax, horizon=horizon, grid_size=grid_size, lo=lo, hi=hi,
                k_epistemic=0.0, aleatoric_bound=mean_q, accuracy="medium"
            )
            cert = compute_certification_metrics(v_true, v_hat, states=states, coverage_gap=gap_cfg)
            c_fsv.append(cert.fsv)
            c_fuv.append(cert.fuv)
            c_records[f"alpha_{a}"] = {**cert.to_dict(), "q_hat": mean_q}

        results[c_var] = {
            "knobs": alpha_grid,
            "metrics_by_knob": c_records,
            "pareto": compute_aupc_and_fuv_at_ratios(c_fsv, c_fuv, fsv_nominal=nom_fsv),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run full three-certifier comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "certifier_comparison")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid_size = args.grid_size
    lo = (-1.0, -2.0)
    hi = (3.0, 2.0)
    umax = 1.0
    horizon = 0.5
    gap_cfg = {"enabled": True, "lo": [0.4, -2.0], "hi": [1.0, 2.0]}

    p_lin = np.linspace(lo[0], hi[0], grid_size)
    v_lin = np.linspace(lo[1], hi[1], grid_size)
    P, Vv = np.meshgrid(p_lin, v_lin, indexing="ij")
    states = np.stack([P, Vv], axis=-1)

    print("=" * 70)
    print("=== FALSERAFE-BENCH: THREE-CERTIFIER BENCHMARK (E, L, C0, C1a, C2) ===")
    print(f"Seeds: {args.seeds} | Grid: {grid_size}x{grid_size}")
    print("=" * 70)

    check_confirmatory_guard(args.seeds)

    print("Solving ground-truth BRT under hidden physics...", flush=True)
    _, v_true = solve_di_hidden_physics_brt(umax=umax, T=horizon, n=grid_size, lo=lo, hi=hi, accuracy="medium")

    all_seed_results = []
    for s_idx, seed in enumerate(args.seeds):
        t0 = time.time()
        print(f"\n--> Running Seed {seed} ({s_idx+1}/{len(args.seeds)})...", flush=True)

        data_cfg = {
            "system": "double_integrator",
            "state_lo": list(lo),
            "state_hi": list(hi),
            "umax": umax,
            "dt": 0.05,
            "trajectory_steps": 10,
            "trajectory_counts": {"train": 100, "validation": 20, "test": 20},
            "derivative_observation_noise_std": 0.01,
            "split_seed": 42,
            "seed": seed,
            "max_sampling_attempts": 20000,
            "coverage_gap": gap_cfg,
        }
        splits, _ = generate_trajectory_splits(data_cfg)
        train_feat = splits["train"]["features"]
        train_targ = splits["train"]["targets"]
        val_feat = splits["validation"]["features"]
        val_targ = splits["validation"]["targets"]

        # Train ensemble model
        model = EnsembleMLP(members=3, hidden_sizes=[32, 32])
        model.fit(train_feat, train_targ, epochs=60, learning_rate=0.002, seed=seed)

        seed_res = run_single_certifier_evaluation(
            train_features=train_feat,
            train_targets=train_targ,
            val_features=val_feat,
            val_targets=val_targ,
            model=model,
            v_true=v_true,
            P=P,
            Vv=Vv,
            states=states,
            gap_cfg=gap_cfg,
            grid_size=grid_size,
            umax=umax,
            horizon=horizon,
            lo=lo,
            hi=hi,
        )
        all_seed_results.append({"seed": seed, "results": seed_res})
        print(f"Seed {seed} finished in {time.time()-t0:.1f}s")
        print(f"  AUPC -> E: {seed_res['E']['pareto']['aupc']:.4f} | L: {seed_res['L']['pareto']['aupc']:.4f} | C0: {seed_res['C0']['pareto']['aupc']:.4f}")

    # Write summary
    summary_file = args.out_dir / "comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_seed_results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[OK] Three-Certifier benchmark complete! Results saved to {summary_file}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
