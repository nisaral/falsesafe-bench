"""Integration tests for learned 3D Dubins reachability and separated vs blended uncertainty bounds (H4)."""
import numpy as np
import pytest

from lcert.data import generate_trajectory_splits
from lcert.metrics import compute_certification_metrics
from lcert.train import train_dynamics_model

hj = pytest.importorskip("hj_reachability")
from lcert.hj_dubins import solve_dubins_brt  # noqa: E402
from lcert.hj_dubins_learned import solve_learned_dubins_brt  # noqa: E402


@pytest.fixture
def dubins_data():
    config = {
        "system": "dubins",
        "state_lo": [-3.0, -3.0, -np.pi],
        "state_hi": [3.0, 3.0, np.pi],
        "speed": 1.0,
        "turn_rate": 1.6,
        "dt": 0.05,
        "trajectory_steps": 10,
        "trajectory_counts": {"train": 30, "validation": 8, "test": 8},
        "derivative_observation_noise_std": 0.01,
        "split_seed": 100,
        "seed": 0,
        "max_sampling_attempts": 20000,
        "coverage_gap": {"enabled": True, "lo": [-0.5, -0.5, -np.pi], "hi": [0.5, 0.5, np.pi]},
    }
    splits, _ = generate_trajectory_splits(config)
    return splits, config


def test_learned_dubins_reachability_and_h4(dubins_data):
    splits, config = dubins_data
    train_feat = splits["train"]["features"]
    train_targ = splits["train"]["targets"]

    # Train 3D Dubins ensemble model
    model, _ = train_dynamics_model(
        model_type="ensemble_mlp",
        train_features=train_feat,
        train_targets=train_targ,
        seed=42,
        epochs=30,
        members=3,
        hidden_sizes=(32, 32),
        max_seconds=60.0,
    )

    grid_n = 21  # Fast 3D grid: 21x21x21 = 9261 states
    horizon = 0.4
    obstacles = ((0.0, 0.0, 0.8),)

    # 1. Ground truth Dubins BRT
    grid_true, v_true = solve_dubins_brt(
        speed=1.0,
        turn_rate=1.6,
        horizon=horizon,
        grid_size=grid_n,
        lo=(-3.0, -3.0, -np.pi),
        hi=(3.0, 3.0, np.pi),
        obstacles=obstacles,
        accuracy="low",
    )
    assert v_true.shape == (grid_n, grid_n, grid_n)
    assert np.isfinite(v_true).all()

    # 2. Nominal learned Dubins BRT (k=0)
    _, v_nom = solve_learned_dubins_brt(
        model=model,
        speed=1.0,
        turn_rate=1.6,
        horizon=horizon,
        grid_size=grid_n,
        lo=(-3.0, -3.0, -np.pi),
        hi=(3.0, 3.0, np.pi),
        obstacles=obstacles,
        k_epistemic=0.0,
        accuracy="low",
    )
    assert v_nom.shape == (grid_n, grid_n, grid_n)
    assert np.isfinite(v_nom).all()

    # 3. Inflated Separated Uncertainty (E3 / H4)
    _, v_sep = solve_learned_dubins_brt(
        model=model,
        speed=1.0,
        turn_rate=1.6,
        horizon=horizon,
        grid_size=grid_n,
        lo=(-3.0, -3.0, -np.pi),
        hi=(3.0, 3.0, np.pi),
        obstacles=obstacles,
        k_epistemic=1.0,
        aleatoric_bound=(0.02, 0.02, 0.02),
        uncertainty_mode="separated",
        accuracy="low",
    )

    # 4. Inflated Blended Uncertainty (H4 comparator)
    _, v_blend = solve_learned_dubins_brt(
        model=model,
        speed=1.0,
        turn_rate=1.6,
        horizon=horizon,
        grid_size=grid_n,
        lo=(-3.0, -3.0, -np.pi),
        hi=(3.0, 3.0, np.pi),
        obstacles=obstacles,
        k_epistemic=1.0,
        aleatoric_bound=(0.02, 0.02, 0.02),
        uncertainty_mode="blended",
        accuracy="low",
    )

    # Epistemic inflation adds disturbance against the safety objective, so V_inflated <= V_nom
    safe_nom = np.sum(v_nom >= 0.0)
    safe_sep = np.sum(v_sep >= 0.0)
    assert safe_sep <= safe_nom

    # Verify metrics computation on 3D grid with periodic dim 2
    metrics_nom = compute_certification_metrics(
        v_true, v_nom, border_cells=2, periodic_dims=(2,)
    )
    assert 0.0 <= metrics_nom.fsv <= 1.0
    assert 0.0 <= metrics_nom.fuv <= 1.0
    assert 0.0 <= metrics_nom.iou <= 1.0
