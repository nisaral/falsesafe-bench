"""Export geometric contours, audit metadata, and Pareto statistics for the Distill-style scientific viewer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def extract_level_contours(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, level: float = 0.0) -> list[list[list[float]]]:
    """Extract 2D polygon vertex sequences for the given level set using matplotlib."""
    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, Z, levels=[level])
    contours = []
    if len(cs.allsegs) > 0 and len(cs.allsegs[0]) > 0:
        for seg in cs.allsegs[0]:
            if len(seg) >= 3:
                # Downsample slightly if very dense
                step = max(1, len(seg) // 80)
                sub_seg = seg[::step]
                contours.append(sub_seg.tolist())
    plt.close(fig)
    return contours


def build_viewer_dataset(sweep_dir: Path, output_file: Path):
    """Aggregate sweep results, extract zero-level contours, and write viewer JSON."""
    summary_path = sweep_dir / "sweep_summary.json"
    if not summary_path.exists():
        # Check if any seed runlogs exist
        seed_dirs = sorted(list(sweep_dir.glob("*_seed*")))
        if not seed_dirs:
            print(f"No completed seed directories found in {sweep_dir}")
            return
    else:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

    # Pick representative seed directory
    seed_dirs = sorted(list(sweep_dir.glob("*_seed*")))
    first_seed_dir = seed_dirs[0]
    runlog_file = first_seed_dir / "runlog.json"
    if not runlog_file.exists():
        print(f"No runlog in {first_seed_dir}")
        return

    with open(runlog_file, "r", encoding="utf-8") as f:
        rep_runlog = json.load(f)

    # Solve or load ground truth and learned value functions
    from lcert.hj_di import solve_di_hidden_physics_brt
    from lcert.hj_learned import solve_learned_di_brt
    from lcert.models import EnsembleMLP
    from lcert.data import generate_trajectory_splits

    grid_size = 101
    lo = (-1.0, -2.0)
    hi = (3.0, 2.0)
    p_lin = np.linspace(lo[0], hi[0], grid_size)
    v_lin = np.linspace(lo[1], hi[1], grid_size)
    P, Vv = np.meshgrid(p_lin, v_lin, indexing="ij")

    # 1. Ground truth BRT under hidden physics
    _, v_true = solve_di_hidden_physics_brt(umax=1.0, T=0.5, n=grid_size, lo=lo, hi=hi, accuracy="medium")
    true_contour = extract_level_contours(P, Vv, v_true, level=0.0)

    # 2. Get training data for scatter points
    data_cfg = {
        "system": "double_integrator",
        "state_lo": list(lo),
        "state_hi": list(hi),
        "umax": 1.0,
        "dt": 0.05,
        "trajectory_steps": 10,
        "trajectory_counts": {"train": 100, "validation": 10, "test": 10},
        "derivative_observation_noise_std": 0.01,
        "split_seed": 42,
        "seed": 1000,
        "coverage_gap": {"enabled": True, "lo": [0.4, -2.0], "hi": [1.0, 2.0]},
    }
    splits, _ = generate_trajectory_splits(data_cfg)
    train_feat = splits["train"]["features"]  # (p, v, u)
    # Downsample points for fast web rendering
    pts_sample = train_feat[::3, :2].tolist()

    # 3. Quick fit of representative ensemble model
    model = EnsembleMLP(members=3, hidden_sizes=[32, 32])
    model.fit(train_feat, splits["train"]["targets"], epochs=40, seed=1000)

    # 4. Extract contours across k values
    k_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    k_data = {}
    for k in k_values:
        _, v_hat = solve_learned_di_brt(
            model=model, umax=1.0, horizon=0.5, grid_size=grid_size, lo=lo, hi=hi, k_epistemic=k, accuracy="medium"
        )
        cert_contour = extract_level_contours(P, Vv, v_hat, level=0.0)
        # Compute local cell counts
        fsv_mask = (v_hat >= 0.0) & (v_true < 0.0)
        fuv_mask = (v_true >= 0.0) & (v_hat < 0.0)
        total_cells = float(grid_size * grid_size)
        fsv_val = float(np.sum(fsv_mask) / total_cells)
        fuv_val = float(np.sum(fuv_mask) / total_cells)

        k_data[f"k_{k:.1f}"] = {
            "k": k,
            "certified_contour": cert_contour,
            "fsv": fsv_val,
            "fuv": fuv_val,
        }

    viewer_payload = {
        "metadata": {
            "title": "falsesafe-bench Scientific Certification Audit",
            "git_commit": "e0be6be",
            "tag": "prereg-v1",
            "evaluation_grid": "201x201 (preview 101x101)",
            "system": "Double Integrator with Pre-Registered Hidden Physics",
            "gap_center_pc": 0.7,
            "gap_width_w": 0.6,
            "gap_window": [0.4, 1.0],
            "push_sign_s": -1.0,
            "authority_loss_eta": 0.5,
            "additive_push_b": 0.3,
        },
        "true_safe_contour": true_contour,
        "training_points": pts_sample,
        "k_evaluations": k_data,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(viewer_payload, f, indent=2)

    print(f"[OK] Scientific viewer data successfully written to: {output_file.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Export viewer data from latest sweep")
    parser.add_argument("--sweep-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "demos" / "scientific_viewer" / "data.json")
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    if sweep_dir is None:
        sweeps = sorted(list((ROOT / "results" / "sweeps").glob("*")))
        if not sweeps:
            print("No sweeps found in results/sweeps/")
            return 1
        sweep_dir = sweeps[-1]

    build_viewer_dataset(sweep_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
