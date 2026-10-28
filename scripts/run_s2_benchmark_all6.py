"""Execute Full Six-Certifier Benchmark S2 on Grid 201x201 (PREREGISTRATION.md Section 4.2 S2).

Certifiers:
- E: Ensemble-k (k in [0.0, 3.0], re-used from confirmatory sweep)
- L: Distance-Aware Lipschitz (lambda in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
- C0: Naive Split Conformal (alpha in [0.5, 0.2, 0.1, 0.05, 0.01])
- C1a: Normalized Conformal by Epistemic Std
- C1b: Normalized Conformal by Distance to Training Set
- C2: Leave-Region-Out Conformal (calibrated on disjoint spatial window [1.8, 2.4])

Pre-registered AUPC penalty clause:
- rho = FSV / FSV_nominal
- FUV*(rho) = min { FUV | ratio <= rho }
- Where no setting reaches rho, FUV*(rho) = largest FUV observed across ALL certifiers in condition.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

from lcert.artifacts import check_confirmatory_guard
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


def evaluate_s2_all_certifiers(seeds: list[int], grid_size: int = 201, accuracy: str = "medium") -> dict:
    # 1. Enforce confirmatory guard
    check_confirmatory_guard(seeds)

    print("=" * 80)
    print(f"=== S2 BENCHMARK: 6 CERTIFIERS (Grid {grid_size}x{grid_size}, n={len(seeds)}) ===")
    print("=" * 80)

    lo = (-1.0, -2.0)
    hi = (3.0, 2.0)
    umax = 1.0
    horizon = 0.5
    gap_cfg = {"enabled": True, "lo": [0.4, -2.0], "hi": [1.0, 2.0]}
    c2_calib_window = (1.8, 2.4)

    # Disjointness check assertion
    assert gap_cfg["hi"][0] < c2_calib_window[0], "C2 calibration window must be strictly disjoint from gap G"

    p_lin = np.linspace(lo[0], hi[0], grid_size)
    v_lin = np.linspace(lo[1], hi[1], grid_size)
    P, Vv = np.meshgrid(p_lin, v_lin, indexing="ij")
    states = np.stack([P, Vv], axis=-1)

    print("Solving ground-truth BRT under hidden physics...", flush=True)
    _, v_true = solve_di_hidden_physics_brt(umax=umax, T=horizon, n=grid_size, lo=lo, hi=hi, accuracy=accuracy)

    all_seed_results = []

    for seed_idx, seed in enumerate(seeds):
        t0 = time.time()
        print(f"\nProcessing Seed {seed} ({seed_idx + 1}/{len(seeds)})...", flush=True)

        # 1. Certifier E (Ensemble-k): load pre-computed confirmatory sweep
        runlog_file = SWEEP_DIR / f"gap_N100_seed{seed}" / "runlog.json"
        with open(runlog_file) as f:
            rlog = json.load(f)
        m = rlog["metrics"]
        k_grid = [round(float(k), 1) for k in np.arange(0.0, 3.1, 0.2)]
        e_fsv = [m[f"k_{k:.1f}"]["fsv"] for k in k_grid]
        e_fuv = [m[f"k_{k:.1f}"]["fuv"] for k in k_grid]
        nom_fsv = e_fsv[0]

        # 2. Train seed ensemble model
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

        # Distance-Aware Certifier
        dist_cert = DistanceAwareCertifier(state_lo=lo, state_hi=hi, umax=umax)
        dist_cert.fit(train_feat, train_targ, model=model)

        # 3. Certifier L (Distance-Aware Lipschitz)
        lam_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        l_fsv, l_fuv = [], []
        for lam in lam_grid:
            d_bound = dist_cert.disturbance_bound(states, actions=umax, lam=lam)
            mean_d = float(np.mean(d_bound))
            _, v_hat = solve_learned_di_brt(
                model=model, umax=umax, horizon=horizon, grid_size=grid_size, lo=lo, hi=hi,
                k_epistemic=0.0, aleatoric_bound=mean_d, accuracy=accuracy
            )
            cert = compute_certification_metrics(v_true, v_hat, states=states, coverage_gap=gap_cfg)
            l_fsv.append(cert.fsv)
            l_fuv.append(cert.fuv)

        # 4. Conformal Certifiers: C0, C1a, C1b, C2
        alpha_grid = [0.5, 0.2, 0.1, 0.05, 0.01]
        conf_results = {}

        # C2 disjoint calibration set
        in_c2_calib = (val_feat[:, 0] >= c2_calib_window[0]) & (val_feat[:, 0] <= c2_calib_window[1])
        c2_calib_feat = val_feat[in_c2_calib]
        c2_calib_targ = val_targ[in_c2_calib]

        for c_var in ("C0", "C1a", "C1b", "C2"):
            c_fsv, c_fuv = [], []
            for a in alpha_grid:
                conf = ConformalCertifier(variant=c_var, alpha=a)
                if c_var == "C2":
                    conf.calibrate(c2_calib_feat, c2_calib_targ, model=model, distance_certifier=dist_cert)
                else:
                    conf.calibrate(val_feat, val_targ, model=model, distance_certifier=dist_cert)
                mean_q = conf.q_hat
                _, v_hat = solve_learned_di_brt(
                    model=model, umax=umax, horizon=horizon, grid_size=grid_size, lo=lo, hi=hi,
                    k_epistemic=0.0, aleatoric_bound=mean_q, accuracy=accuracy
                )
                cert = compute_certification_metrics(v_true, v_hat, states=states, coverage_gap=gap_cfg)
                c_fsv.append(cert.fsv)
                c_fuv.append(cert.fuv)
            conf_results[c_var] = {"fsv": c_fsv, "fuv": c_fuv}

        # 5. Condition-Wide Maximum FUV for Pre-Registered Penalty Clause
        all_observed_fuv = e_fuv + l_fuv + conf_results["C0"]["fuv"] + conf_results["C1a"]["fuv"] + conf_results["C1b"]["fuv"] + conf_results["C2"]["fuv"]
        penalty_fuv = float(np.max(all_observed_fuv))
        if penalty_fuv <= 0.0:
            penalty_fuv = 1.0

        # 6. Compute AUPC with condition-wide penalty
        seed_res = {
            "E": compute_aupc_and_fuv_at_ratios(e_fsv, e_fuv, fsv_nominal=nom_fsv, penalty_fuv=penalty_fuv),
            "L": compute_aupc_and_fuv_at_ratios(l_fsv, l_fuv, fsv_nominal=nom_fsv, penalty_fuv=penalty_fuv),
            "C0": compute_aupc_and_fuv_at_ratios(conf_results["C0"]["fsv"], conf_results["C0"]["fuv"], fsv_nominal=nom_fsv, penalty_fuv=penalty_fuv),
            "C1a": compute_aupc_and_fuv_at_ratios(conf_results["C1a"]["fsv"], conf_results["C1a"]["fuv"], fsv_nominal=nom_fsv, penalty_fuv=penalty_fuv),
            "C1b": compute_aupc_and_fuv_at_ratios(conf_results["C1b"]["fsv"], conf_results["C1b"]["fuv"], fsv_nominal=nom_fsv, penalty_fuv=penalty_fuv),
            "C2": compute_aupc_and_fuv_at_ratios(conf_results["C2"]["fsv"], conf_results["C2"]["fuv"], fsv_nominal=nom_fsv, penalty_fuv=penalty_fuv),
            "penalty_fuv": penalty_fuv,
        }

        print(f"  Seed {seed} in {time.time()-t0:.1f}s | AUPC -> E: {seed_res['E']['aupc']:.5f} | L: {seed_res['L']['aupc']:.5f} | C1a: {seed_res['C1a']['aupc']:.5f} | C1b: {seed_res['C1b']['aupc']:.5f} | C0: {seed_res['C0']['aupc']:.5f} | C2: {seed_res['C2']['aupc']:.5f}")
        all_seed_results.append({"seed": seed, "results": seed_res})

    # Summary Across Seeds
    summary = {}
    for c_name in ("E", "L", "C1a", "C1b", "C0", "C2"):
        aupcs = [s["results"][c_name]["aupc"] for s in all_seed_results]
        summary[c_name] = {
            "mean_aupc": float(np.mean(aupcs)),
            "std_aupc": float(np.std(aupcs)),
            "aupc_values": aupcs,
        }

    print("\n" + "=" * 80)
    print(f"=== S2 SUMMARY: ALL 6 CERTIFIERS (Grid {grid_size}x{grid_size}, n={len(seeds)}) ===")
    for c_name, res in summary.items():
        print(f"  {c_name:5s}: Mean AUPC = {res['mean_aupc']:.6f} +/- {res['std_aupc']:.6f}")
    print("=" * 80)

    out_file = ROOT / "results" / "certifier_comparison" / f"s2_full6_grid{grid_size}_n{len(seeds)}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_seed": all_seed_results, "grid_size": grid_size, "n": len(seeds)}, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=201)
    parser.add_argument("--accuracy", type=str, default="medium")
    args = parser.parse_args()

    seed_dirs = sorted(list(SWEEP_DIR.glob("gap_N100_seed*")))[:args.num_seeds]
    seeds = [int(d.name.split("seed")[-1]) for d in seed_dirs]

    evaluate_s2_all_certifiers(seeds=seeds, grid_size=args.grid_size, accuracy=args.accuracy)


if __name__ == "__main__":
    main()
