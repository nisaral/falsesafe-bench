from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical


class CategoricalActorCritic(nn.Module):
    """A bounded-action robot policy network, not a trained checkpoint."""

    def __init__(self, observation_size: int = 6, action_count: int = 3,
                 hidden_sizes: tuple[int, ...] = (64, 64)):
        super().__init__()
        if observation_size < 1 or action_count < 2 or not hidden_sizes or min(hidden_sizes) < 1:
            raise ValueError('Invalid policy network dimensions')
        layers = []
        previous_size = observation_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(previous_size, hidden_size), nn.Tanh()))
            previous_size = hidden_size
        self.encoder = nn.Sequential(*layers)
        self.actor = nn.Linear(previous_size, action_count)
        self.critic = nn.Linear(previous_size, 1)
        self.observation_size = observation_size
        self.action_count = action_count

    def forward(self, observations: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        if observations.shape[-1] != self.observation_size or not torch.isfinite(observations).all():
            raise ValueError('Invalid policy observations')
        features = self.encoder(observations)
        return Categorical(logits=self.actor(features)), self.critic(features).squeeze(-1)

    def action_log_probabilities(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        distribution, _values = self(observations)
        return distribution.log_prob(actions)


def discrete_turn_rates(turn_rate: float) -> tuple[float, float, float]:
    if isinstance(turn_rate, bool) or not math.isfinite(turn_rate) or turn_rate <= 0:
        raise ValueError('Turn-rate authority must be positive and finite')
    return (-turn_rate, 0.0, turn_rate)
