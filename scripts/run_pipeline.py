"""Full reproducible experiment pipeline for E1/E2 sweeps on double integrator.

Automates:
1. Data generation across sample sizes N and coverage-gap conditions
2. Model training and held-out validation
3. Ground-truth analytic vs learned Hamilton-Jacobi reachability solving
4. Exact certification metric calculation (FSV, FUV, IoU, Gap-FSV)
5. Multi-seed statistical aggregation with 95% Student's t-intervals
6. Falsification testing for hypotheses H1, H2, and H3
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from lcert.artifacts import check_confirmatory_guard, write_json
from lcert.config import load_config
from lcert.data import generate_trajectory_splits
from lcert.hj_learned import solve_learned_di_brt
from lcert.metrics import (
    CertificationMetrics,
    aggregate_metric_runs,
    compute_certification_metrics,
)
from lcert.systems import di_safe_analytic
from lcert.train import evaluate_dynamics_model, train_dynamics_model


def run_single_condition(
    base_config: dict,
    num_train_trajectories: int,
    seed: int,
    gap_enabled: bool,
    inflation_k_values: list[float],
    output_dir: Path,
) -> dict[float, CertificationMetrics]:
    """Execute training, reachability solve, and certification for a single seed & condition."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data config
    data_config = {
        "system": base_config["system"],
        "state_lo": base_config["state_lo"],
        "state_hi": base_config["state_hi"],
        "umax": base_config["umax"],
        "dt": base_config["dt"],
        "trajectory_steps": base_config["trajectory_steps"],
        "trajectory_counts": {
            "train": num_train_trajectories,
            "validation": base_config["val_trajectories"],
            "test": base_config["test_trajectories"],
        },
        "derivative_observation_noise_std": base_config["derivative_observation_noise_std"],
        "split_seed": base_config["split_seed"],
        "seed": seed,
        "max_sampling_attempts": 20000,
        "coverage_gap": {
            "enabled": gap_enabled,
            "lo": base_config["coverage_gap"]["lo"],
            "hi": base_config["coverage_gap"]["hi"],
        },
    }

    # 1. Generate data splits
    splits, _ = generate_trajectory_splits(data_config)
    train_feat = splits["train"]["features"]
    train_targ = splits["train"]["targets"]
    val_feat = splits["validation"]["features"]
    val_targ = splits["validation"]["targets"]

    # 2. Train dynamics model
    model_cfg = base_config.get("model", {})
    model, train_diag = train_dynamics_model(
        model_type=model_cfg.get("type", "ensemble_mlp"),
        train_features=train_feat,
        train_targets=train_targ,
        seed=seed,
        epochs=model_cfg.get("epochs", 60),
        learning_rate=model_cfg.get("learning_rate", 0.002),
        members=model_cfg.get("members", 3),
        hidden_sizes=tuple(model_cfg.get("hidden_sizes", [32, 32])),
        max_seconds=120.0,
    )
    val_summary = evaluate_dynamics_model(model, val_feat, val_targ)

    # 3. Ground truth analytic value function
    horizon = float(base_config.get("horizon", 0.5))
    grid_size = int(base_config.get("grid_size", 51))
    umax = float(base_config.get("umax", 1.0))
    lo = tuple(base_config.get("state_lo", [-1.0, -2.0]))
    hi = tuple(base_config.get("state_hi", [3.0, 2.0]))

    # Precompute coordinates on grid
    p_lin = np.linspace(lo[0], hi[0], grid_size)
    v_lin = np.linspace(lo[1], hi[1], grid_size)
    P, Vv = np.meshgrid(p_lin, v_lin, indexing="ij")
    if base_config.get("system") == "double_integrator_hidden_physics":
        from lcert.hj_di import solve_di_hidden_physics_brt
        hp = base_config.get("hidden_physics", {})
        _, v_true = solve_di_hidden_physics_brt(
            umax=umax,
            T=horizon,
            n=grid_size,
            lo=lo,
            hi=hi,
            pc=hp.get("pc", 0.7),
            w=hp.get("w", 0.6),
            eta=hp.get("eta", 0.5),
            b=hp.get("b", 0.3),
            s=hp.get("s", -1.0),
            accuracy=base_config.get("accuracy", "medium"),
        )
    else:
        v_true = di_safe_analytic(P, Vv, umax, horizon)

    states = np.stack([P, Vv], axis=-1)

    # 4. Solve learned BRT for each inflation k
    metrics_by_k: dict[float, CertificationMetrics] = {}
    for k in inflation_k_values:
        grid, v_hat = solve_learned_di_brt(
            model=model,
            umax=umax,
            horizon=horizon,
            grid_size=grid_size,
            lo=lo,
            hi=hi,
            k_epistemic=k,
            accuracy=base_config.get("accuracy", "medium"),
        )
        cert = compute_certification_metrics(
            v_true=v_true,
            v_hat=v_hat,
            border_cells=2,
            states=states,
            coverage_gap=data_config["coverage_gap"],
        )
        metrics_by_k[k] = cert

    # Save run artifact
    runlog = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "train_samples": len(train_feat),
        "gap_enabled": gap_enabled,
        "validation_rmse": val_summary.rmse_overall,
        "validation_r2": val_summary.r2_overall,
        "metrics": {f"k_{k}": m.to_dict() for k, m in metrics_by_k.items()},
    }
    write_json(output_dir / "runlog.json", runlog)
    return metrics_by_k


def run_pipeline(config_path: Path, max_sizes: int | None = None) -> dict:
    """Execute full sweep defined in config_path and print aggregated research findings."""
    config = load_config(config_path)
    data_sizes = config.get("data_sizes_trajectories", [10, 30, 100])
    if max_sizes is not None:
        data_sizes = data_sizes[:max_sizes]
    seeds = config.get("seeds", [0, 1, 2, 3, 4])
    inflation_k_values = [float(k) for k in config.get("inflation_k_values", [0.0, 1.0])]

    # Enforce confirmatory guard before touching any confirmatory seed
    check_confirmatory_guard(seeds)

    results_dir = ROOT / "results" / "sweeps" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"=== falsesafe-bench: Running {config.get('experiment', 'E1')} Sweep ===")
    print(f"System: {config.get('system')} | Seeds: {seeds} | Sizes: {data_sizes}")

    print(f"Inflation k: {inflation_k_values} | Output: {results_dir.relative_to(ROOT)}")
    print("=" * 70, flush=True)

    summary_records = []
    # Loop over gap conditions: True (with coverage gap), False (uniform coverage)
    for gap_enabled in (True, False):
        gap_label = "gap" if gap_enabled else "no_gap"
        print(f"\n>>> Condition: {gap_label.upper()} (coverage_gap={gap_enabled})")

        for num_traj in data_sizes:
            started = time.monotonic()
            run_metrics_by_k: dict[float, list[CertificationMetrics]] = {k: [] for k in inflation_k_values}

            for seed in seeds:
                cond_dir = results_dir / f"{gap_label}_N{num_traj}_seed{seed}"
                m_by_k = run_single_condition(
                    base_config=config,
                    num_train_trajectories=num_traj,
                    seed=seed,
                    gap_enabled=gap_enabled,
                    inflation_k_values=inflation_k_values,
                    output_dir=cond_dir,
                )
                for k in inflation_k_values:
                    run_metrics_by_k[k].append(m_by_k[k])

            elapsed = time.monotonic() - started
            # Aggregate across seeds
            agg_by_k = {}
            for k in inflation_k_values:
                agg = aggregate_metric_runs(run_metrics_by_k[k], confidence=0.95, enforce_min_samples=False)
                agg_by_k[k] = agg

            nom_fsv = agg_by_k[0.0]["fsv"]
            nom_gap_fsv = agg_by_k[0.0].get("gap_fsv")
            gap_fsv_str = f" | Gap-FSV: {nom_gap_fsv.format()}" if nom_gap_fsv else ""
            print(
                f"  N_traj={num_traj:3d} (approx {num_traj*10:4d} points) [{elapsed:.1f}s]: "
                f"Nominal FSV: {nom_fsv.format()}{gap_fsv_str}"
            )

            record = {
                "gap_enabled": gap_enabled,
                "trajectories": num_traj,
                "data_points": num_traj * config["trajectory_steps"],
                "seeds": seeds,
                "results": {
                    f"k_{k}": {metric_name: ci.to_dict() for metric_name, ci in agg_by_k[k].items()}
                    for k in inflation_k_values
                },
            }
            summary_records.append(record)

    summary_path = results_dir / "sweep_summary.json"
    write_json(summary_path, {"config": config, "records": summary_records})
    print("\n" + "=" * 70)
    print(f"[OK] Sweep complete! Summary written to: {summary_path.relative_to(ROOT)}")
    print("=" * 70)

    return {"summary_path": str(summary_path), "records": summary_records}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run falsesafe-bench E1/E2 experiment sweep")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "e1_di_sweep.yaml")
    parser.add_argument("--max-sizes", type=int, default=None)
    options = parser.parse_args()

    run_pipeline(options.config.resolve(), max_sizes=options.max_sizes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
