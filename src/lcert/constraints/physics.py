from __future__ import annotations

import math

import torch


def dubins_kinematic_prior_loss(predicted_derivatives: torch.Tensor,
                                observed_states: torch.Tensor, known_speed: float) -> torch.Tensor:
    """A public fixed-speed kinematic prior, not a complete PINN or unknown-dynamics oracle."""
    if predicted_derivatives.shape != observed_states.shape or observed_states.shape[-1] != 3 or not observed_states.numel():
        raise ValueError('Expected nonempty matching [..., 3] state and derivative tensors')
    if not math.isfinite(known_speed) or known_speed <= 0 or not torch.isfinite(predicted_derivatives).all() or not torch.isfinite(observed_states).all():
        raise ValueError('Prior inputs and public speed must be finite, speed positive')
    headings = observed_states[..., 2]
    nominal = torch.stack((known_speed * headings.cos(), known_speed * headings.sin()), dim=-1)
    return (predicted_derivatives[..., :2] - nominal).square().mean()
