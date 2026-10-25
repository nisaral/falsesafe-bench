from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import ConfigurationError, require_integer, require_positive


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % math.tau - math.pi


@dataclass(frozen=True)
class Pose:
    position_x: float
    position_y: float
    heading: float

    def __post_init__(self):
        if not all(math.isfinite(value) for value in self.to_list()):
            raise ConfigurationError('Pose must be finite')

    def to_list(self) -> list[float]:
        return [self.position_x, self.position_y, self.heading]

    @classmethod
    def from_list(cls, values) -> Pose:
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ConfigurationError('Pose must contain position_x, position_y, and heading')
        return cls(float(values[0]), float(values[1]), wrap_angle(float(values[2])))


@dataclass(frozen=True)
class Circle:
    center_x: float
    center_y: float
    radius: float

    def __post_init__(self):
        if not all(math.isfinite(value) for value in self.to_list()) or self.radius <= 0:
            raise ConfigurationError('Circle coordinates must be finite and radius positive')

    def to_list(self) -> list[float]:
        return [self.center_x, self.center_y, self.radius]


@dataclass(frozen=True)
class ArenaSettings:
    bounds: tuple[float, float, float, float]
    dt: float
    max_steps: int
    speed: float
    turn_rate: float
    robot_radius: float
    goal_radius: float
    collision_substeps: int
    process_turn_noise_std: float
    obstacles: tuple[Circle, ...]

    @classmethod
    def from_dict(cls, config: dict) -> ArenaSettings:
        bounds = config.get('bounds')
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise ConfigurationError('arena.bounds must be [min_x, max_x, min_y, max_y]')
        bounds = tuple(float(value) for value in bounds)
        if not all(math.isfinite(value) for value in bounds):
            raise ConfigurationError('Bounds must be finite')
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ConfigurationError('Arena bounds are reversed or empty')
        noise = float(config.get('process_turn_noise_std', 0.0))
        if not math.isfinite(noise) or noise < 0:
            raise ConfigurationError('Process turn noise must be finite and nonnegative')
        obstacles = []
        for values in config.get('obstacles', []):
            if not isinstance(values, list) or len(values) != 3:
                raise ConfigurationError('Obstacle must be [center_x, center_y, radius]')
            obstacles.append(Circle(*(float(value) for value in values)))
        return cls(bounds, require_positive(config, 'dt'), require_integer(config, 'max_steps', 1),
                   require_positive(config, 'speed'), require_positive(config, 'turn_rate'),
                   require_positive(config, 'robot_radius'), require_positive(config, 'goal_radius'),
                   require_integer(config, 'collision_substeps', 1), noise, tuple(obstacles))

    def inside(self, pose: Pose) -> bool:
        return (self.bounds[0] + self.robot_radius <= pose.position_x <= self.bounds[1] - self.robot_radius
                and self.bounds[2] + self.robot_radius <= pose.position_y <= self.bounds[3] - self.robot_radius)


def advance_dubins(pose: Pose, turn_rate: float, speed: float, dt: float) -> Pose:
    if not all(math.isfinite(value) for value in (turn_rate, speed, dt)) or dt <= 0 or speed <= 0:
        raise ValueError('Dubins speed and dt must be positive; inputs must be finite')
    half_turn = 0.5 * turn_rate * dt
    distance = speed * dt * (math.sin(half_turn) / half_turn if half_turn != 0 else 1.0)
    middle_heading = pose.heading + half_turn
    return Pose(pose.position_x + distance * math.cos(middle_heading),
                pose.position_y + distance * math.sin(middle_heading),
                wrap_angle(pose.heading + 2.0 * half_turn))


def segment_distance(start: tuple[float, float], end: tuple[float, float],
                     point: tuple[float, float]) -> float:
    delta_x, delta_y = end[0] - start[0], end[1] - start[1]
    squared_length = delta_x**2 + delta_y**2
    fraction = ((point[0] - start[0]) * delta_x + (point[1] - start[1]) * delta_y) / squared_length if squared_length else 0.0
    fraction = min(1.0, max(0.0, fraction))
    return math.hypot(start[0] + fraction * delta_x - point[0],
                      start[1] + fraction * delta_y - point[1])
