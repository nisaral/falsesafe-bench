"""Conformal Prediction Certifiers (C0, C1a, C1b, C2).

Implements pre-registered conformal uncertainty bounds:
- C0: Naive split conformal (exchangeability failure demonstration in OOD gaps).
- C1a: Normalized split conformal scaled by ensemble epistemic std sigma_ep(x).
- C1b: Normalized split conformal scaled by distance dist((x,u), D_train).
- C2: Leave-region-out split conformal (calibrated on a separate withheld spatial window).
"""
from __future__ import annotations
import numpy as np


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Compute standard split-conformal quantile with finite-sample (n+1)/n correction."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        return 0.0
    # Desired quantile level: ceil((n + 1) * (1 - alpha)) / n
    level = np.ceil((n + 1) * (1.0 - alpha)) / float(n)
    level = min(1.0, max(0.0, level))
    idx = int(np.clip(int(np.ceil(level * n)) - 1, 0, n - 1))
    return float(scores[idx])


class ConformalCertifier:
    """Unified split conformal certifier supporting C0, C1a, C1b, and C2."""

    def __init__(self, variant: str = "C0", alpha: float = 0.1):
        if variant not in ("C0", "C1a", "C1b", "C2"):
            raise ValueError(f"Unknown conformal variant: {variant}. Expected C0, C1a, C1b, or C2.")
        self.variant = variant
        self.alpha = float(alpha)
        self.q_hat: float = 0.0
        self.is_calibrated: bool = False

    def calibrate(
        self,
        calib_features: np.ndarray,
        calib_targets: np.ndarray,
        model,
        distance_certifier=None,
    ) -> ConformalCertifier:
        """Calibrate non-conformity scores on calibration data split."""
        z = np.asarray(calib_features, dtype=float)
        y = np.asarray(calib_targets, dtype=float)
        acc_targets = y[:, 1]

        # Model predictions
        if hasattr(model, "predict_mean_std"):
            pred_means, pred_stds = model.predict_mean_std(z)
            pred_acc = pred_means[:, 1]
            sigma_ep = pred_stds[:, 1]
        elif callable(model):
            pred = model(z)
            pred_acc = pred[:, 1] if pred.ndim > 1 else pred
            sigma_ep = np.ones_like(pred_acc) * 0.1
        else:
            pred_acc = z[:, 2]
            sigma_ep = np.ones_like(pred_acc) * 0.1

        residuals = np.abs(acc_targets - pred_acc)

        if self.variant == "C0":
            # Unnormalized absolute residuals
            scores = residuals
        elif self.variant == "C1a":
            # Normalized by epistemic standard deviation (with floor to avoid division by 0)
            scores = residuals / np.maximum(sigma_ep, 1e-4)
        elif self.variant == "C1b":
            # Normalized by distance to training set
            if distance_certifier is not None:
                dists = distance_certifier.compute_distance(z[:, :2], z[:, 2])
            else:
                dists = np.ones_like(residuals)
            scores = residuals / np.maximum(dists, 1e-4)
        elif self.variant == "C2":
            # Leave-region-out: uses residuals from withheld region
            scores = residuals
        else:
            scores = residuals

        self.q_hat = compute_conformal_quantile(scores, self.alpha)
        self.is_calibrated = True
        return self

    def disturbance_bound(
        self,
        states: np.ndarray,
        actions: np.ndarray | float | None = None,
        model=None,
        distance_certifier=None,
    ) -> np.ndarray:
        """Evaluate conformal disturbance bound across states."""
        if not self.is_calibrated:
            raise RuntimeError("ConformalCertifier must be calibrated before evaluation.")

        states = np.asarray(states, dtype=float)
        orig_shape = states.shape[:-1]
        flat_states = states.reshape(-1, 2)
        n = flat_states.shape[0]

        if self.variant in ("C0", "C2"):
            # Constant bound everywhere
            return np.full(orig_shape, self.q_hat, dtype=float)

        if actions is None:
            flat_actions = np.ones(n, dtype=float)
        elif np.isscalar(actions):
            flat_actions = np.full(n, float(actions), dtype=float)
        else:
            flat_actions = np.asarray(actions, dtype=float).reshape(-1)

        z_query = np.column_stack([flat_states, flat_actions])

        if self.variant == "C1a":
            if model is not None and hasattr(model, "predict_mean_std"):
                _, pred_stds = model.predict_mean_std(z_query)
                sigma_ep = pred_stds[:, 1]
            else:
                sigma_ep = np.ones(n, dtype=float) * 0.1
            bounds = self.q_hat * np.maximum(sigma_ep, 1e-4)
            return bounds.reshape(orig_shape)

        elif self.variant == "C1b":
            if distance_certifier is not None:
                dists = distance_certifier.compute_distance(flat_states, flat_actions)
            else:
                dists = np.ones(n, dtype=float)
            bounds = self.q_hat * np.maximum(dists, 1e-4)
            return bounds.reshape(orig_shape)

        return np.full(orig_shape, self.q_hat, dtype=float)
