"""Distance-Aware Lipschitz Certifier (Certifier L).

Pre-registered bound:
d(x, u) = epsilon_0 + lambda * L_hat * dist((x, u), D_train)

where:
- dist is Euclidean in joint (p, v, u) normalized space (each axis divided by range).
- L_hat = max_{i != j} ||y_i - y_j|| / ||z_i - z_j||_norm from training pairs.
- epsilon_0 = max_i ||y_i - f_hat(z_i)|| (maximum training residual).
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


class DistanceAwareCertifier:
    def __init__(
        self,
        state_lo: tuple[float, float] = (-1.0, -2.0),
        state_hi: tuple[float, float] = (3.0, 2.0),
        umax: float = 1.0,
        subsample_pairs_for_lipschitz: int = 500,
    ):
        self.state_lo = np.asarray(state_lo, dtype=float)
        self.state_hi = np.asarray(state_hi, dtype=float)
        self.umax = float(umax)
        self.subsample_pairs = subsample_pairs_for_lipschitz

        # Normalization scale vector across (p, v, u)
        range_p = float(self.state_hi[0] - self.state_lo[0])
        range_v = float(self.state_hi[1] - self.state_lo[1])
        range_u = 2.0 * self.umax
        self.scale = np.array([range_p, range_v, range_u], dtype=float)

        self.kdtree: cKDTree | None = None
        self.train_z_norm: np.ndarray | None = None
        self.epsilon_0: float = 0.0
        self.l_hat: float = 1.0

    def fit(self, train_features: np.ndarray, train_targets: np.ndarray, model=None) -> DistanceAwareCertifier:
        """Fit epsilon_0 and empirical Lipschitz constant L_hat from training data."""
        z = np.asarray(train_features, dtype=float)  # (N, 3): (p, v, u)
        y = np.asarray(train_targets, dtype=float)   # (N, 2): (v, v_dot)
        acc_target = y[:, 1]  # acceleration channel where dynamics uncertainty lives

        # 1. Normalize features to unit hypercube
        self.train_z_norm = z / self.scale
        self.kdtree = cKDTree(self.train_z_norm)

        # 2. Compute epsilon_0 (max training residual)
        if model is not None:
            if hasattr(model, "predict_mean_std"):
                mean_pred, _ = model.predict_mean_std(z)
                pred_acc = mean_pred[:, 1]
            elif callable(model):
                pred = model(z)
                pred_acc = pred[:, 1] if pred.ndim > 1 else pred
            else:
                pred_acc = z[:, 2]  # nominal double integrator: v_dot = u
            residuals = np.abs(acc_target - pred_acc)
            self.epsilon_0 = float(np.max(residuals))
        else:
            # Nominal double integrator residual: v_dot - u
            residuals = np.abs(acc_target - z[:, 2])
            self.epsilon_0 = float(np.max(residuals))

        # 3. Compute empirical Lipschitz constant L_hat
        n_samples = len(z)
        if n_samples < 2:
            self.l_hat = 1.0
            return self

        # If dataset is large, subsample pairs for pairwise slope calculation
        if n_samples <= self.subsample_pairs:
            idx_i, idx_j = np.triu_indices(n_samples, k=1)
        else:
            rng = np.random.default_rng(20261027)
            idx_i = rng.integers(0, n_samples, size=self.subsample_pairs * 10)
            idx_j = rng.integers(0, n_samples, size=self.subsample_pairs * 10)
            valid = idx_i != idx_j
            idx_i = idx_i[valid]
            idx_j = idx_j[valid]

        diff_z = np.linalg.norm(self.train_z_norm[idx_i] - self.train_z_norm[idx_j], axis=-1)
        diff_y = np.abs(acc_target[idx_i] - acc_target[idx_j])

        # Filter out identical or extremely close points to avoid numerical division by zero
        non_zero = diff_z > 1e-5
        if np.any(non_zero):
            slopes = diff_y[non_zero] / diff_z[non_zero]
            self.l_hat = float(np.max(slopes))
        else:
            self.l_hat = 1.0

        # Safety floor for L_hat
        self.l_hat = max(self.l_hat, 0.1)
        return self

    def compute_distance(self, states: np.ndarray, actions: np.ndarray | float | None = None) -> np.ndarray:
        """Compute Euclidean distance in normalized (p, v, u) space to training set.
        
        Args:
            states: (..., 2) array of (p, v) states.
            actions: (...,) array of control actions u, or float (e.g. +umax). If None, defaults to +umax.
        """
        if self.kdtree is None:
            raise RuntimeError("DistanceAwareCertifier must be fit before querying distances.")
        states = np.asarray(states, dtype=float)
        orig_shape = states.shape[:-1]
        flat_states = states.reshape(-1, 2)
        n = flat_states.shape[0]

        if actions is None:
            flat_actions = np.full(n, self.umax, dtype=float)
        elif np.isscalar(actions):
            flat_actions = np.full(n, float(actions), dtype=float)
        else:
            flat_actions = np.asarray(actions, dtype=float).reshape(-1)

        z_query = np.column_stack([flat_states, flat_actions])
        z_query_norm = z_query / self.scale
        dists, _ = self.kdtree.query(z_query_norm, k=1)
        return dists.reshape(orig_shape)

    def disturbance_bound(
        self,
        states: np.ndarray,
        actions: np.ndarray | float | None = None,
        lam: float = 1.0,
    ) -> np.ndarray:
        """Pre-registered bound d(x, u) = epsilon_0 + lambda * L_hat * dist((x,u), D_train)."""
        dists = self.compute_distance(states, actions)
        return self.epsilon_0 + float(lam) * self.l_hat * dists
