"""Integration tests for model training and learned-dynamics HJ reachability solving."""
import tempfile
from pathlib import Path
import numpy as np
import pytest

from lcert.data import generate_trajectory_splits
from lcert.metrics import compute_certification_metrics
from lcert.models.ensemble import EnsembleMLP
from lcert.systems import di_safe_analytic
from lcert.train import evaluate_dynamics_model, train_dynamics_model

hj = pytest.importorskip("hj_reachability")
from lcert.hj_learned import solve_learned_di_brt  # noqa: E402


@pytest.fixture
def di_data_splits():
    config = {
        "system": "double_integrator",
        "state_lo": [-1.0, -2.0],
        "state_hi": [3.0, 2.0],
        "umax": 1.0,
        "dt": 0.05,
        "trajectory_steps": 10,
        "trajectory_counts": {"train": 24, "validation": 8, "test": 8},
        "derivative_observation_noise_std": 0.005,
        "split_seed": 42,
        "seed": 0,
        "max_sampling_attempts": 10000,
        "coverage_gap": {"enabled": True, "lo": [0.5, -0.5], "hi": [1.5, 0.5]},
    }
    splits, _ = generate_trajectory_splits(config)
    return splits, config


def test_ensemble_training_save_load_and_evaluation(di_data_splits):
    splits, _ = di_data_splits
    train_feat, train_targ = splits["train"]["features"], splits["train"]["targets"]
    val_feat, val_targ = splits["validation"]["features"], splits["validation"]["targets"]

    model, diag = train_dynamics_model(
        model_type="ensemble_mlp",
        train_features=train_feat,
        train_targets=train_targ,
        seed=10,
        epochs=40,
        members=3,
        hidden_sizes=(32, 32),
        max_seconds=60.0,
    )

    eval_summary = evaluate_dynamics_model(model, val_feat, val_targ)
    assert eval_summary.rmse_overall > 0.0
    assert eval_summary.sample_count == len(val_feat)
    assert eval_summary.epistemic_std_mean >= 0.0

    # Test save and load roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ensemble.pt"
        model.save(ckpt_path)

        loaded_model = EnsembleMLP.load(ckpt_path)
        pred_orig = model.predict(val_feat[:5])
        pred_loaded = loaded_model.predict(val_feat[:5])

        np.testing.assert_allclose(pred_orig.mean, pred_loaded.mean, atol=1e-6)
        np.testing.assert_allclose(pred_orig.epistemic_std, pred_loaded.epistemic_std, atol=1e-6)


def test_learned_di_hj_solve_and_epistemic_inflation(di_data_splits):
    splits, config = di_data_splits
    train_feat, train_targ = splits["train"]["features"], splits["train"]["targets"]

    # Train a quick ensemble model
    model, _ = train_dynamics_model(
        model_type="ensemble_mlp",
        train_features=train_feat,
        train_targets=train_targ,
        seed=1,
        epochs=30,
        members=3,
        hidden_sizes=(32, 32),
        max_seconds=60.0,
    )

    grid_n = 41
    horizon = 0.5
    # 1. Nominal solve (E1: k=0)
    grid, v_nominal = solve_learned_di_brt(
        model,
        umax=1.0,
        horizon=horizon,
        grid_size=grid_n,
        k_epistemic=0.0,
        accuracy="medium",
    )
    assert v_nominal.shape == (grid_n, grid_n)
    assert np.isfinite(v_nominal).all()

    # 2. Epistemic-inflated solve (E2: k=1.0)
    _, v_inflated = solve_learned_di_brt(
        model,
        umax=1.0,
        horizon=horizon,
        grid_size=grid_n,
        k_epistemic=1.0,
        accuracy="medium",
    )
    assert v_inflated.shape == (grid_n, grid_n)
    assert np.isfinite(v_inflated).all()

    # Epistemic inflation adds disturbance against the safety objective, so V_inflated <= V_nominal
    # Safe set {V >= 0} shrinks or stays equal
    safe_nominal_count = np.sum(v_nominal >= 0.0)
    safe_inflated_count = np.sum(v_inflated >= 0.0)
    assert safe_inflated_count <= safe_nominal_count

    # 3. Certification metrics against true analytic solution
    P = np.asarray(grid.states[..., 0])
    Vv = np.asarray(grid.states[..., 1])
    v_exact = di_safe_analytic(P, Vv, 1.0, horizon)

    metrics_nom = compute_certification_metrics(
        v_exact,
        v_nominal,
        border_cells=2,
        states=grid.states,
        coverage_gap=config["coverage_gap"],
    )
    assert 0.0 <= metrics_nom.fsv <= 1.0
    assert 0.0 <= metrics_nom.fuv <= 1.0
    assert 0.0 <= metrics_nom.iou <= 1.0
    assert metrics_nom.gap_fsv is not None
