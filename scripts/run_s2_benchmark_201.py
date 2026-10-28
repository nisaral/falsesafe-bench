"""Execute Confirmatory Certifier Benchmark S2 on Grid 201x201 across Confirmatory Seeds.

Pre-registered S2 Analysis:
- E: Re-uses exact confirmatory sweep (seeds 1000-1029, k in [0.0, 3.0], grid 201x201).
- L: Distance-Aware Lipschitz (lambda in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]).
- C0: Naive Split Conformal (alpha in [0.5, 0.2, 0.1, 0.05, 0.01]).
- C1a: Normalized Conformal by Epistemic Std.
- C2: Leave-Region-Out Conformal.
- Computes Pareto AUPC = integral_0^1 FUV*(rho) d(rho) and FUV at rho in {0.5, 0.25, 0.1}.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

from lcert.certifiers import (
    DistanceAwareCertifier,
    ConformalCertifier,
    compute_aupc_and_fuv_at_ratios,
)
from lcert.data import generate_trajectory_splits
from lcert.hj_di import solve_di_hidden_physics_brt
from lcert.hj_learned import solve_learned_di_brt
from lcert.metrics import compute_certification_metrics
from lcert.models import EnsembleMLP

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "results" / "sweeps" / "20261027_184119"


def run_s2_benchmark(num_seeds: int = 10, grid_size: int = 201):
    print("=" * 75)
    print(f"=== S2 CERTIFIER BENCHMARK (Grid {grid_size}x{grid_size}, Seeds: {num_seeds}) ===")
    print("=" * 75)

    lo = (-1.0, -2.0)
    hi = (3.0, 2.0)
    umax = 1.0
    horizon = 0.5
    gap_cfg = {"enabled": True, "lo": [0.4, -2.0], "hi": [1.0, 2.0]}

    p_lin = np.linspace(lo[0], hi[0], grid_size)
    v_lin = np.linspace(lo[1], hi[1], grid_size)
    P, Vv = np.meshgrid(p_lin, v_lin, indexing="ij")
    states = np.stack([P, Vv], axis=-1)

    print("Computing ground-truth BRT under hidden physics on 201x201...", flush=True)
    _, v_true = solve_di_hidden_physics_brt(umax=umax, T=horizon, n=grid_size, lo=lo, hi=hi, accuracy="medium")

    seed_dirs = sorted(list(SWEEP_DIR.glob("gap_N100_seed*")))[:num_seeds]
    seeds = [int(d.name.split("seed")[-1]) for d in seed_dirs]

    from lcert.artifacts import check_confirmatory_guard
    check_confirmatory_guard(seeds)

    all_seed_results = []
    
    for seed_idx, seed in enumerate(seeds):
        t_seed = time.time()
        print(f"\nEvaluating Seed {seed} ({seed_idx + 1}/{len(seeds)})...", flush=True)
        seed_res = {}

        # 1. Certifier E: Load from existing confirmatory sweep
        runlog_path = SWEEP_DIR / f"gap_N100_seed{seed}" / "runlog.json"
        with open(runlog_path) as f:
            runlog = json.load(f)
        m = runlog["metrics"]
        k_grid = [round(float(k), 1) for k in np.arange(0.0, 3.1, 0.2)]
        e_fsv = [m[f"k_{k:.1f}"]["fsv"] for k in k_grid]
        e_fuv = [m[f"k_{k:.1f}"]["fuv"] for k in k_grid]
        nom_fsv = e_fsv[0]
        seed_res["E"] = {
            "knobs": k_grid,
            "pareto": compute_aupc_and_fuv_at_ratios(e_fsv, e_fuv, fsv_nominal=nom_fsv),
        }

        # Generate data splits & fit model
        data_cfg = {
            "system": "double_integrator", "state_lo": list(lo), "state_hi": list(hi), "umax": umax,
            "dt": 0.05, "trajectory_steps": 10, "trajectory_counts": {"train": 100, "validation": 20, "test": 20},
            "derivative_observation_noise_std": 0.01, "split_seed": 42, "seed": seed,
            "max_sampling_attempts": 20000, "coverage_gap": gap_cfg,
        }
        splits, _ = generate_trajectory_splits(data_cfg)
        train_feat = splits["train"]["features"]
        train_targ = splits["train"]["targets"]
        val_feat = splits["validation"]["features"]
        val_targ = splits["validation"]["targets"]

        model = EnsembleMLP(members=3, hidden_sizes=[32, 32])
        model.fit(train_feat, train_targ, epochs=40, seed=seed)

        # 2. Certifier L (Distance-Aware Lipschitz)
        dist_cert = DistanceAwareCertifier(state_lo=lo, state_hi=hi, umax=umax)
        dist_cert.fit(train_feat, train_targ, model=model)
        lam_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        l_fsv, l_fuv = [], []
        for lam in lam_grid:
            d_bound = dist_cert.disturbance_bound(states, actions=umax, lam=lam)
            mean_d = float(np.mean(d_bound))
            _, v_hat = solve_learned_di_brt(
                model=model, umax=umax, horizon=horizon, grid_size=grid_size, lo=lo, hi=hi,
                k_epistemic=0.0, aleatoric_bound=mean_d, accuracy="medium"
            )
            cert = compute_certification_metrics(v_true, v_hat, states=states, coverage_gap=gap_cfg)
            l_fsv.append(cert.fsv)
            l_fuv.append(cert.fuv)
        seed_res["L"] = {
            "knobs": lam_grid,
            "pareto": compute_aupc_and_fuv_at_ratios(l_fsv, l_fuv, fsv_nominal=nom_fsv),
        }

        # 3. Conformal Variants: C0, C1a, C2
        alpha_grid = [0.5, 0.2, 0.1, 0.05, 0.01]
        for c_var in ("C0", "C1a", "C2"):
            c_fsv, c_fuv = [], []
            for a in alpha_grid:
                conf = ConformalCertifier(variant=c_var, alpha=a)
                conf.calibrate(val_feat, val_targ, model=model, distance_certifier=dist_cert)
                mean_q = conf.q_hat
                _, v_hat = solve_learned_di_brt(
                    model=model, umax=umax, horizon=horizon, grid_size=grid_size, lo=lo, hi=hi,
                    k_epistemic=0.0, aleatoric_bound=mean_q, accuracy="medium"
                )
                cert = compute_certification_metrics(v_true, v_hat, states=states, coverage_gap=gap_cfg)
                c_fsv.append(cert.fsv)
                c_fuv.append(cert.fuv)
            seed_res[c_var] = {
                "knobs": alpha_grid,
                "pareto": compute_aupc_and_fuv_at_ratios(c_fsv, c_fuv, fsv_nominal=nom_fsv),
            }

        print(f"  Seed {seed} finished in {time.time()-t_seed:.1f}s | AUPC -> E: {seed_res['E']['pareto']['aupc']:.5f} | L: {seed_res['L']['pareto']['aupc']:.5f} | C1a: {seed_res['C1a']['pareto']['aupc']:.5f} | C0: {seed_res['C0']['pareto']['aupc']:.5f}")
        all_seed_results.append({"seed": seed, "results": seed_res})

    # Summary
    summary = {}
    for c_name in ("E", "L", "C1a", "C0", "C2"):
        aupcs = [s["results"][c_name]["pareto"]["aupc"] for s in all_seed_results]
        summary[c_name] = {
            "mean_aupc": float(np.mean(aupcs)),
            "std_aupc": float(np.std(aupcs)),
            "aupc_values": aupcs,
        }

    print("\n" + "=" * 75)
    print("=== S2 MULTI-CERTIFIER AUPC SUMMARY (201x201 Grid) ===")
    for c_name, res in summary.items():
        print(f"  {c_name:5s}: Mean AUPC = {res['mean_aupc']:.6f} +/- {res['std_aupc']:.6f}")
    print("=" * 75)

    out_file = ROOT / "results" / "certifier_comparison" / "s2_confirmatory_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_seed": all_seed_results}, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=201)
    args = parser.parse_args()
    run_s2_benchmark(num_seeds=args.num_seeds, grid_size=args.grid_size)
