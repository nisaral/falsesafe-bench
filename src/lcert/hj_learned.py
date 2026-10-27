"""Hamilton-Jacobi reachability solver using learned dynamics and uncertainty bounds.

Implements the core research formulations (E1, E2, E3):
- E1: Nominal learned model (k=0, no disturbance)
- E2: Epistemic-inflated HJ: disturbance space scaled by k * ensemble_std(x)
- E3: Separated epistemic + aleatoric disturbance bounds
Evaluated via JAX-native compilation inside hj_reachability.
"""
from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj

from .models.ensemble import EnsembleMLP


class LearnedDoubleIntegratorDynamics(hj.Dynamics):
    """JAX-native Double Integrator dynamics evaluated from a trained neural ensemble."""

    def __init__(
        self,
        ensemble_weights: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
        ensemble_biases: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
        input_mean: np.ndarray,
        input_scale: np.ndarray,
        umax: float = 1.0,
        k_epistemic: float = 0.0,
        aleatoric_bound: float = 0.0,
        control_steps: int = 5,
    ):
        controls = tuple(float(u) for u in np.linspace(-umax, umax, control_steps))
        super().__init__(
            control_mode="max",
            disturbance_mode="min",
            control_space=hj.sets.Box(jnp.array([-umax]), jnp.array([umax])),
            disturbance_space=hj.sets.Box(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])),
        )
        # Store weights as JAX arrays for each member
        self.weights = tuple(
            tuple(jnp.asarray(w, dtype=jnp.float32) for w in m_weights)
            for m_weights in ensemble_weights
        )
        self.biases = tuple(
            tuple(jnp.asarray(b, dtype=jnp.float32) for b in m_biases)
            for m_biases in ensemble_biases
        )
        self.input_mean = jnp.asarray(input_mean, dtype=jnp.float32)
        self.input_scale = jnp.asarray(input_scale, dtype=jnp.float32)
        self.controls = controls
        self.k_epistemic = float(k_epistemic)
        self.aleatoric_bound = float(aleatoric_bound)

    def _eval_member(self, member_idx: int, state: jnp.ndarray, u: float) -> jnp.ndarray:
        w = self.weights[member_idx]
        b = self.biases[member_idx]
        feature = (jnp.concatenate([state, jnp.asarray([u], dtype=jnp.float32)]) - self.input_mean) / self.input_scale
        h1 = jnp.tanh(w[0] @ feature + b[0])
        h2 = jnp.tanh(w[1] @ h1 + b[1])
        out = w[2] @ h2 + b[2]
        return out

    def _eval_ensemble(self, state: jnp.ndarray, u: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        preds = jnp.stack([self._eval_member(m, state, u) for m in range(len(self.weights))])
        mean = jnp.mean(preds, axis=0)
        std = jnp.std(preds, axis=0)
        return mean, std

    def __call__(self, state, control, disturbance, time):
        del time
        u = control[0]
        mean, _ = self._eval_ensemble(state, u)
        return mean + disturbance

    def optimal_control_and_disturbance(self, state, time, grad_value):
        del time
        # Pick control from candidate set that maximizes grad_value @ mean_prediction
        cand_means = jnp.stack([self._eval_ensemble(state, u)[0] for u in self.controls])
        scores = cand_means @ grad_value
        best_idx = jnp.argmax(scores)
        best_u = jnp.array([jnp.array(self.controls)[best_idx]])

        # Disturbance bound: epistemic (k * std) + aleatoric
        _, std = self._eval_ensemble(state, self.controls[best_idx])
        d_bound = self.k_epistemic * std + self.aleatoric_bound
        # Worst-case disturbance opposes gradient direction: -d_bound * sign(grad_value)
        worst_d = -d_bound * jnp.sign(grad_value)
        return best_u, worst_d

    def hamiltonian(self, state, time, value, grad_value):
        del time, value
        # H = max_u [grad_value @ f_mean(x, u)] - sum_i (|grad_value_i| * d_bound_i)
        cand_means_stds = [self._eval_ensemble(state, u) for u in self.controls]
        cand_means = jnp.stack([m for m, _ in cand_means_stds])
        cand_stds = jnp.stack([s for _, s in cand_means_stds])

        scores = cand_means @ grad_value
        best_idx = jnp.argmax(scores)
        h_ctrl = scores[best_idx]

        d_bound = self.k_epistemic * cand_stds[best_idx] + self.aleatoric_bound
        h_dist = -jnp.sum(jnp.abs(grad_value) * d_bound)
        return h_ctrl + h_dist

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        del time, value, grad_value_box
        # Conservative velocity and acceleration upper bounds for CFL condition
        return jnp.array([jnp.abs(state[1]) + 0.5, 2.5])


def extract_ensemble_jax_params(model: EnsembleMLP) -> tuple[tuple, tuple, np.ndarray, np.ndarray]:
    """Extract PyTorch weights and normalization parameters for JAX execution."""
    weights = []
    biases = []
    for torch_model in model.models:
        m_weights = []
        m_biases = []
        for layer in torch_model:
            if hasattr(layer, "weight"):
                m_weights.append(layer.weight.detach().cpu().numpy())
                m_biases.append(layer.bias.detach().cpu().numpy())
        weights.append(tuple(m_weights))
        biases.append(tuple(m_biases))
    return tuple(weights), tuple(biases), np.asarray(model.input_mean), np.asarray(model.input_scale)


def solve_learned_di_brt(
    model: EnsembleMLP,
    umax: float = 1.0,
    horizon: float = 1.0,
    grid_size: int = 101,
    lo: tuple[float, float] = (-1.0, -2.0),
    hi: tuple[float, float] = (3.0, 2.0),
    k_epistemic: float = 0.0,
    aleatoric_bound: float = 0.0,
    accuracy: str = "very_high",
) -> tuple[hj.Grid, np.ndarray]:
    """Solve the Backwards Reachable Tube for Double Integrator under learned dynamics.

    Args:
        model: Trained EnsembleMLP model.
        umax: Maximum control authority |u| <= umax.
        horizon: Backward time horizon T.
        grid_size: Number of points along each dimension.
        lo: Lower bounds of grid domain.
        hi: Upper bounds of grid domain.
        k_epistemic: Uncertainty multiplier (0.0 for nominal E1, >0 for inflated E2/E3).
        aleatoric_bound: Constant aleatoric disturbance bound (for E3 separated).
        accuracy: hj_reachability numerical accuracy setting.

    Returns:
        (grid, value_array) where value_array is shape (grid_size, grid_size).
    """
    weights, biases, mean, scale = extract_ensemble_jax_params(model)
    dynamics = LearnedDoubleIntegratorDynamics(
        ensemble_weights=weights,
        ensemble_biases=biases,
        input_mean=mean,
        input_scale=scale,
        umax=umax,
        k_epistemic=k_epistemic,
        aleatoric_bound=aleatoric_bound,
    )
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(np.array(lo), np.array(hi)), (grid_size, grid_size)
    )
    # Failure set: p < 0
    target = grid.states[..., 0]
    settings = hj.SolverSettings.with_accuracy(
        accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
    )
    value = hj.step(settings, dynamics, grid, 0.0, target, -horizon)
    return grid, np.asarray(value)
