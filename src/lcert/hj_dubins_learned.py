"""Learned Hamilton-Jacobi reachability solver for 3D Dubins Car (E1, E2, E3, E4).

Implements 3D state-space (x, y, theta) reachability with periodic heading wrapping,
obstacle collision target function, and separated vs blended uncertainty bounds (H4).
"""
from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj

from .models.ensemble import EnsembleMLP


class LearnedDubinsDynamics(hj.Dynamics):
    """JAX-native 3D Dubins Car dynamics evaluated from a trained neural ensemble."""

    def __init__(
        self,
        ensemble_weights: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
        ensemble_biases: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
        input_mean: np.ndarray,
        input_scale: np.ndarray,
        turn_rate: float = 1.6,
        speed: float = 1.0,
        k_epistemic: float = 0.0,
        aleatoric_bound: tuple[float, float, float] = (0.0, 0.0, 0.0),
        uncertainty_mode: str = "separated",  # "separated" or "blended"
        control_steps: int = 5,
    ):
        controls = tuple(float(u) for u in np.linspace(-turn_rate, turn_rate, control_steps))
        super().__init__(
            control_mode="max",
            disturbance_mode="min",
            control_space=hj.sets.Box(jnp.array([-turn_rate]), jnp.array([turn_rate])),
            disturbance_space=hj.sets.Box(jnp.array([-1.0, -1.0, -1.0]), jnp.array([1.0, 1.0, 1.0])),
        )
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
        self.speed = float(speed)
        self.k_epistemic = float(k_epistemic)
        self.aleatoric_bound = jnp.asarray(aleatoric_bound, dtype=jnp.float32)
        self.uncertainty_mode = uncertainty_mode

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
        cand_means = jnp.stack([self._eval_ensemble(state, u)[0] for u in self.controls])
        scores = cand_means @ grad_value
        best_idx = jnp.argmax(scores)
        best_u = jnp.array([jnp.array(self.controls)[best_idx]])

        _, std = self._eval_ensemble(state, self.controls[best_idx])
        if self.uncertainty_mode == "blended":
            total_sigma = jnp.sqrt((self.k_epistemic * std) ** 2 + self.aleatoric_bound ** 2)
            d_bound = total_sigma
        else:
            d_bound = self.k_epistemic * std + self.aleatoric_bound

        worst_d = -d_bound * jnp.sign(grad_value)
        return best_u, worst_d

    def hamiltonian(self, state, time, value, grad_value):
        del time, value
        cand_means_stds = [self._eval_ensemble(state, u) for u in self.controls]
        cand_means = jnp.stack([m for m, _ in cand_means_stds])
        cand_stds = jnp.stack([s for _, s in cand_means_stds])

        scores = cand_means @ grad_value
        best_idx = jnp.argmax(scores)
        h_ctrl = scores[best_idx]

        if self.uncertainty_mode == "blended":
            d_bound = jnp.sqrt((self.k_epistemic * cand_stds[best_idx]) ** 2 + self.aleatoric_bound ** 2)
        else:
            d_bound = self.k_epistemic * cand_stds[best_idx] + self.aleatoric_bound

        h_dist = -jnp.sum(jnp.abs(grad_value) * d_bound)
        return h_ctrl + h_dist

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        del time, value, grad_value_box
        # Max |dx/dt| = speed + max_disturbance, |dy/dt| = speed + max_dist, |dtheta/dt| = turn_rate + max_dist
        return jnp.array([self.speed + 0.5, self.speed + 0.5, 2.5])


def extract_dubins_ensemble_params(model: EnsembleMLP):
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


def solve_learned_dubins_brt(
    model: EnsembleMLP,
    speed: float = 1.0,
    turn_rate: float = 1.6,
    horizon: float = 1.0,
    grid_size: int = 41,
    lo: tuple[float, float, float] = (-3.0, -3.0, -np.pi),
    hi: tuple[float, float, float] = (3.0, 3.0, np.pi),
    obstacles: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.8),),
    k_epistemic: float = 0.0,
    aleatoric_bound: tuple[float, float, float] = (0.0, 0.0, 0.0),
    uncertainty_mode: str = "separated",
    accuracy: str = "medium",
) -> tuple[hj.Grid, np.ndarray]:
    """Solve 3D Backwards Reachable Tube for Dubins Car under learned dynamics.

    Args:
        model: Trained EnsembleMLP on 4D features (x, y, theta, u) -> 3D targets (dx, dy, dtheta).
        speed: Forward speed v.
        turn_rate: Maximum steering rate |omega| <= turn_rate.
        horizon: Backward time horizon T.
        grid_size: Grid resolution along each of the 3 axes.
        lo: Lower bounds [x_min, y_min, -pi].
        hi: Upper bounds [x_max, y_max, pi].
        obstacles: Obstacles as ((cx, cy, radius), ...).
        k_epistemic: Epistemic inflation factor (0 for nominal).
        aleatoric_bound: Bound on observation/process noise per dimension.
        uncertainty_mode: 'separated' (k*std + aleatoric) or 'blended' (sqrt((k*std)^2 + aleatoric^2)).
        accuracy: 'low', 'medium', or 'very_high'.

    Returns:
        (grid, value_3d_array) with shape (grid_size, grid_size, grid_size).
    """
    weights, biases, mean, scale = extract_dubins_ensemble_params(model)
    dynamics = LearnedDubinsDynamics(
        ensemble_weights=weights,
        ensemble_biases=biases,
        input_mean=mean,
        input_scale=scale,
        turn_rate=turn_rate,
        speed=speed,
        k_epistemic=k_epistemic,
        aleatoric_bound=aleatoric_bound,
        uncertainty_mode=uncertainty_mode,
    )

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(np.asarray(lo, float), np.asarray(hi, float)),
        (grid_size, grid_size, grid_size),
        periodic_dims=(2,),
    )

    circles = np.asarray(obstacles, float)
    target = jnp.min(
        jnp.stack([
            jnp.sqrt((grid.states[..., 0] - cx)**2 + (grid.states[..., 1] - cy)**2) - cr
            for cx, cy, cr in circles
        ]),
        axis=0,
    )

    settings = hj.SolverSettings.with_accuracy(
        accuracy, hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
    )
    value = hj.step(settings, dynamics, grid, 0.0, target, -horizon)
    return grid, np.asarray(value)
