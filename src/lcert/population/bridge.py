from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def probability_vector(values, size: int | None = None) -> np.ndarray:
    probabilities = np.asarray(values, float)
    if probabilities.ndim != 1 or not probabilities.size or (size is not None and len(probabilities) != size):
        raise ValueError('Probability vector has invalid shape')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError('Probabilities must be finite, nonnegative, and normalised')
    return probabilities


def stochastic_kernels(values) -> np.ndarray:
    kernels = np.asarray(values, float)
    if kernels.ndim != 3 or not kernels.shape[0] or kernels.shape[1] != kernels.shape[2] or not kernels.shape[1]:
        raise ValueError('Reference kernels must have shape [steps, states, states]')
    if not np.isfinite(kernels).all() or np.any(kernels < 0) or not np.allclose(kernels.sum(axis=2), 1.0, rtol=0, atol=1e-12):
        raise ValueError('Reference kernels must be finite row-stochastic matrices')
    return kernels


def _supported_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    if np.any((numerator > 0) & (denominator <= 0)):
        raise ValueError('Requested marginal is unreachable under the declared reference process')
    result = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


@dataclass(frozen=True)
class DiscreteBridge:
    endpoint_coupling: np.ndarray
    transition_kernels: np.ndarray
    population: np.ndarray
    marginal_residual: float


def solve_discrete_bridge(initial, terminal, reference_kernels,
                          tolerance: float = 1e-8, max_iterations: int = 10000) -> DiscreteBridge:
    """Finite-state path-space bridge relative to explicit Markov reference kernels."""
    reference = stochastic_kernels(reference_kernels)
    state_count = reference.shape[1]
    initial = probability_vector(initial, state_count)
    terminal = probability_vector(terminal, state_count)
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError('Invalid bridge tolerance or iteration budget')
    endpoint_reference = np.eye(state_count)
    for kernel in reference:
        endpoint_reference = endpoint_reference @ kernel
    right = np.ones(state_count)
    coupling = None
    for _iteration in range(max_iterations):
        left = _supported_divide(initial, endpoint_reference @ right)
        right = _supported_divide(terminal, endpoint_reference.T @ left)
        coupling = left[:, None] * endpoint_reference * right[None, :]
        residual = max(float(np.abs(coupling.sum(axis=1) - initial).sum()),
                       float(np.abs(coupling.sum(axis=0) - terminal).sum()))
        if not np.isfinite(coupling).all():
            raise FloatingPointError('Bridge scaling potentials overflowed')
        if residual <= tolerance:
            break
        normalizer = right.max()
        if normalizer <= 0:
            raise ValueError('Reference process cannot reach the terminal marginal')
        right = right / normalizer
    else:
        raise RuntimeError('Bridge scaling did not converge; no alternate reference process was substituted')
    potentials = [None] * (len(reference) + 1)
    potentials[-1] = right
    for index in range(len(reference) - 1, -1, -1):
        potentials[index] = reference[index] @ potentials[index + 1]
    driven = []
    population = [initial]
    for index, kernel in enumerate(reference):
        numerator = kernel * potentials[index + 1][None, :]
        transition = np.zeros_like(kernel)
        np.divide(numerator, potentials[index][:, None], out=transition,
                  where=potentials[index][:, None] > 0)
        if np.any((population[-1] > tolerance) & (potentials[index] <= 0)):
            raise ValueError('Bridge has mass on an unsupported reference state')
        driven.append(transition)
        population.append(population[-1] @ transition)
    population = np.asarray(population)
    if not np.allclose(population.sum(axis=1), 1.0, rtol=0, atol=tolerance) or not np.allclose(population[-1], terminal, rtol=0, atol=tolerance):
        raise RuntimeError('Driven bridge did not conserve mass or satisfy the terminal marginal')
    return DiscreteBridge(coupling, np.asarray(driven), population, residual)
