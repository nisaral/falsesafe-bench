from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _logit_response(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = values / temperature
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


@dataclass(frozen=True)
class QRESolution:
    first_policy: np.ndarray
    second_policy: np.ndarray
    best_response_residual: float


def solve_logit_qre(first_payoffs, second_payoffs, first_temperature: float,
                    second_temperature: float, damping: float = 0.5,
                    tolerance: float = 1e-8, max_iterations: int = 10000) -> QRESolution:
    first = np.asarray(first_payoffs, float)
    second = np.asarray(second_payoffs, float)
    if first.ndim != 2 or not first.size or second.shape != first.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Two finite, shape-matched payoff matrices are required')
    if not all(np.isfinite(value) and value > 0 for value in (first_temperature, second_temperature, tolerance)):
        raise ValueError('Temperatures and solver tolerance must be positive and finite')
    if not 0 < damping <= 1 or max_iterations < 1:
        raise ValueError('Invalid QRE damping or iteration budget')
    first_policy = np.full(first.shape[0], 1 / first.shape[0])
    second_policy = np.full(first.shape[1], 1 / first.shape[1])
    for _iteration in range(max_iterations):
        first_response = _logit_response(first @ second_policy, first_temperature)
        second_response = _logit_response(first_policy @ second, second_temperature)
        residual = max(float(np.max(np.abs(first_response - first_policy))),
                       float(np.max(np.abs(second_response - second_policy))))
        if residual <= tolerance:
            return QRESolution(first_policy, second_policy, residual)
        first_policy = (1 - damping) * first_policy + damping * first_response
        second_policy = (1 - damping) * second_policy + damping * second_response
    raise RuntimeError('QRE fixed point did not converge; no equilibrium or inverse estimate is claimed')


def sample_play(solution: QRESolution, sample_count: int, seed: int) -> np.ndarray:
    if sample_count < 1:
        raise ValueError('sample_count must be positive')
    random = np.random.default_rng(seed)
    return np.column_stack((random.choice(len(solution.first_policy), sample_count, p=solution.first_policy),
                            random.choice(len(solution.second_policy), sample_count, p=solution.second_policy)))
