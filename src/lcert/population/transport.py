from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot


@dataclass(frozen=True)
class TransportSolution:
    coupling: np.ndarray
    cost: np.ndarray
    marginal_residual: float


def entropic_transport(source_positions, target_positions, regularization: float,
                       source_weights=None, target_weights=None,
                       tolerance: float = 1e-7, max_iterations: int = 10000) -> TransportSolution:
    source = np.asarray(source_positions, float)
    target = np.asarray(target_positions, float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1] or not len(source) or not len(target):
        raise ValueError('Population positions must be nonempty, dimension-matched matrices')
    if not np.isfinite(source).all() or not np.isfinite(target).all() or not np.isfinite(regularization) or regularization <= 0:
        raise ValueError('Positions must be finite and entropic regularization positive')
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError('Invalid transport solver tolerance or iteration budget')
    source_mass = np.full(len(source), 1 / len(source)) if source_weights is None else np.asarray(source_weights, float)
    target_mass = np.full(len(target), 1 / len(target)) if target_weights is None else np.asarray(target_weights, float)
    for weights, size in ((source_mass, len(source)), (target_mass, len(target))):
        if weights.shape != (size,) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError('Transport weights must be strictly positive and normalised; no silent pruning')
    cost = np.sum((source[:, None, :] - target[None, :, :])**2, axis=-1)
    coupling = ot.sinkhorn(source_mass, target_mass, cost, regularization,
                          numItermax=max_iterations, stopThr=tolerance / 10, method='sinkhorn_log')
    residual = max(float(np.abs(coupling.sum(axis=1) - source_mass).sum()),
                   float(np.abs(coupling.sum(axis=0) - target_mass).sum()))
    if not np.isfinite(coupling).all() or residual > tolerance:
        raise RuntimeError(f'Entropic OT did not satisfy marginals: residual={residual}')
    return TransportSolution(coupling, cost, residual)
