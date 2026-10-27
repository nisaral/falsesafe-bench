"""Dynamics model training, evaluation, and held-out validation.

Supports:
- 5-member MLP ensemble (PyTorch) with bootstrap sampling
- Independent exact GP (GPyTorch) with RBF/Scale kernel
- Held-out one-step prediction evaluation (RMSE, MAE, R^2, epistemic uncertainty)
- Shuffled-target negative control to confirm model-recovery significance
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models.ensemble import DynamicsPrediction, EnsembleMLP
from .models.gp import IndependentExactGP


@dataclass(frozen=True)
class EvaluationSummary:
    """Held-out prediction quality summary."""
    rmse_per_dim: list[float]
    rmse_overall: float
    mae_per_dim: list[float]
    mae_overall: float
    r2_per_dim: list[float]
    r2_overall: float
    epistemic_std_mean: float
    epistemic_std_max: float
    aleatoric_std_mean: float | None = None
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_dynamics_model(
    model: EnsembleMLP | IndependentExactGP,
    features: np.ndarray,
    targets: np.ndarray,
) -> EvaluationSummary:
    """Evaluate a trained dynamics model on held-out features and targets."""
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(features) == 0 or len(features) != len(targets):
        raise ValueError("Non-empty matching features and targets required for evaluation")

    pred: DynamicsPrediction = model.predict(features)
    errors = pred.mean - targets

    rmse_dim = [float(np.sqrt(np.mean(errors[:, i] ** 2))) for i in range(targets.shape[1])]
    rmse_overall = float(np.sqrt(np.mean(errors ** 2)))

    mae_dim = [float(np.mean(np.abs(errors[:, i]))) for i in range(targets.shape[1])]
    mae_overall = float(np.mean(np.abs(errors)))

    # R^2 calculation per dimension
    r2_dim = []
    for i in range(targets.shape[1]):
        var_tot = float(np.var(targets[:, i]))
        if var_tot > 1e-12:
            r2_dim.append(float(1.0 - np.mean(errors[:, i] ** 2) / var_tot))
        else:
            r2_dim.append(1.0 if rmse_dim[i] < 1e-6 else 0.0)
    var_all = float(np.var(targets))
    r2_overall = float(1.0 - np.mean(errors ** 2) / var_all) if var_all > 1e-12 else 0.0

    epistemic_mean = float(np.mean(pred.epistemic_std))
    epistemic_max = float(np.max(pred.epistemic_std))
    aleatoric_mean = float(np.mean(pred.aleatoric_std)) if pred.aleatoric_std is not None else None

    return EvaluationSummary(
        rmse_per_dim=rmse_dim,
        rmse_overall=rmse_overall,
        mae_per_dim=mae_dim,
        mae_overall=mae_overall,
        r2_per_dim=r2_dim,
        r2_overall=r2_overall,
        epistemic_std_mean=epistemic_mean,
        epistemic_std_max=epistemic_max,
        aleatoric_std_mean=aleatoric_mean,
        sample_count=len(features),
    )


def train_dynamics_model(
    model_type: str,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    *,
    seed: int = 0,
    epochs: int = 200,
    learning_rate: float = 0.001,
    members: int = 5,
    hidden_sizes: tuple[int, ...] = (64, 64),
    max_seconds: float = 600.0,
) -> tuple[EnsembleMLP | IndependentExactGP, dict[str, Any]]:
    """Train either an MLP ensemble or Exact GP on trajectory data.

    Returns:
        (fitted_model, training_diagnostics_dict)
    """
    started = time.monotonic()
    if model_type in ("ensemble_mlp", "ensemble"):
        model = EnsembleMLP(members=members, hidden_sizes=hidden_sizes)
        model.fit(
            train_features,
            train_targets,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            max_seconds=max_seconds,
        )
    elif model_type in ("gp", "exact_gp"):
        model = IndependentExactGP(max_exact_samples=max(2000, len(train_features)))
        model.fit(
            train_features,
            train_targets,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            max_seconds=max_seconds,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'; choose 'ensemble_mlp' or 'gp'")

    elapsed = time.monotonic() - started
    diagnostics = {
        "model_type": model_type,
        "train_samples": len(train_features),
        "feature_dim": train_features.shape[1],
        "target_dim": train_targets.shape[1],
        "training_seconds": elapsed,
        "seed": seed,
    }
    return model, diagnostics


def run_shuffle_control(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    *,
    seed: int = 42,
    model_type: str = "ensemble_mlp",
    epochs: int = 100,
) -> EvaluationSummary:
    """Fit a control model with randomly permuted targets to verify data signal."""
    rng = np.random.default_rng(seed)
    shuffled_targets = rng.permutation(train_targets, axis=0)
    model, _ = train_dynamics_model(
        model_type=model_type,
        train_features=train_features,
        train_targets=shuffled_targets,
        seed=seed,
        epochs=epochs,
        members=3,
        hidden_sizes=(32, 32),
        max_seconds=60.0,
    )
    return evaluate_dynamics_model(model, val_features, val_targets)
