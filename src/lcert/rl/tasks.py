from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationReward:
    progress_weight: float = 1.0
    goal_bonus: float = 5.0
    failure_penalty: float = -5.0
    time_cost: float = 0.01

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.progress_weight, self.goal_bonus, self.failure_penalty, self.time_cost)):
            raise ValueError('Reward parameters must be finite')
        if self.progress_weight < 0 or self.goal_bonus < 0 or self.failure_penalty > 0 or self.time_cost < 0:
            raise ValueError('Reward signs must be explicitly consistent with their definitions')

    def transition_rewards(self, previous: dict, current: dict) -> list[float]:
        if len(previous['agents']) != len(current['agents']):
            raise ValueError('Reward frames must describe the same agents')
        rewards = []
        for before, after in zip(previous['agents'], current['agents'], strict=True):
            if before['status'] != 'active':
                rewards.append(0.0)
                continue
            before_distance = math.hypot(before['observation'][-2], before['observation'][-1])
            after_distance = math.hypot(after['observation'][-2], after['observation'][-1])
            reward = self.progress_weight * (before_distance - after_distance) - self.time_cost
            if after['status'] == 'goal':
                reward += self.goal_bonus
            elif after['status'] in ('collision', 'domain_exit'):
                reward += self.failure_penalty
            rewards.append(reward)
        return rewards
