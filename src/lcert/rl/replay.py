from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Transition:
    observation: tuple[float, ...]
    commanded_action: float
    executed_action: float
    reward: float
    next_observation: tuple[float, ...]
    terminated: bool
    truncated: bool

    def __post_init__(self):
        if not self.observation or len(self.observation) != len(self.next_observation):
            raise ValueError('Transition observations must be nonempty and shape-matched')
        if not all(math.isfinite(value) for value in (*self.observation, *self.next_observation,
                                                     self.commanded_action, self.executed_action, self.reward)):
            raise ValueError('Transition values must be finite')
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError('Termination and time-limit truncation must be explicit booleans')


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError('Replay capacity must be a positive integer')
        self.transitions = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def append(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise TypeError('Only validated transitions can be appended')
        self.transitions.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 0 < batch_size <= len(self.transitions):
            raise ValueError('Requested batch is unavailable; no smaller batch is substituted')
        return self.random.sample(list(self.transitions), batch_size)

    def __len__(self) -> int:
        return len(self.transitions)
