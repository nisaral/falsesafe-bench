from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bridge import probability_vector, stochastic_kernels


@dataclass(frozen=True)
class MeanFieldSolution:
    population: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    consistency_residual: float


def solve_congestion_game(initial, action_kernels, base_cost, horizon: int,
                          congestion_weight: float = 1.0, temperature: float = 0.5,
                          damping: float = 0.5, tolerance: float = 1e-7,
                          max_iterations: int = 1000) -> MeanFieldSolution:
    """Entropy-regularised finite-state MFG fixed point, not a safety HJ solver."""
    kernels = stochastic_kernels(action_kernels)
    state_count = kernels.shape[1]
    initial = probability_vector(initial, state_count)
    costs = np.asarray(base_cost, float)
    if costs.shape != (state_count,) or not np.isfinite(costs).all() or horizon < 1:
        raise ValueError('Invalid mean-field costs or horizon')
    if not np.isfinite(congestion_weight) or congestion_weight < 0 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Congestion weight must be nonnegative and temperature positive')
    if not 0 < damping <= 1 or not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError('Invalid fixed-point damping, tolerance, or budget')
    population = np.tile(initial, (horizon + 1, 1))
    for _iteration in range(max_iterations):
        value = np.zeros((horizon + 1, state_count))
        policy = np.empty((horizon, state_count, len(kernels)))
        for time_index in range(horizon - 1, -1, -1):
            action_costs = costs[None, :] + congestion_weight * population[time_index][None, :] + kernels @ value[time_index + 1]
            logits = -action_costs.T / temperature
            maximum = logits.max(axis=1, keepdims=True)
            exponentials = np.exp(logits - maximum)
            policy[time_index] = exponentials / exponentials.sum(axis=1, keepdims=True)
            value[time_index] = -temperature * (maximum[:, 0] + np.log(exponentials.sum(axis=1)))
        response_population = [initial]
        for time_index in range(horizon):
            transition = np.einsum('sa,asj->sj', policy[time_index], kernels)
            response_population.append(response_population[-1] @ transition)
        response_population = np.asarray(response_population)
        residual = float(np.max(np.abs(response_population - population)))
        if residual <= tolerance:
            if not np.allclose(response_population.sum(axis=1), 1.0, rtol=0, atol=tolerance):
                raise RuntimeError('Mean-field population did not conserve mass')
            return MeanFieldSolution(response_population, policy, value, residual)
        population = (1 - damping) * population + damping * response_population
    raise RuntimeError('Mean-field fixed point did not converge; no equilibrium is claimed')
