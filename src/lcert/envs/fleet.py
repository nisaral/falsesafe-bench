from __future__ import annotations

import math
import random

from ..config import ConfigurationError
from .dubins import ArenaSettings, Pose, advance_dubins, segment_distance, wrap_angle


class RobotArena:
    """Substepped simulation, not a continuous-time safety certificate."""

    def __init__(self, settings: ArenaSettings, agents: list[dict], seed: int):
        if not isinstance(agents, list) or not agents:
            raise ConfigurationError('At least one agent must be configured')
        self.settings = settings
        self.poses = [Pose.from_list(agent['start']) for agent in agents]
        self.goals = []
        for agent in agents:
            goal = agent.get('goal')
            if not isinstance(goal, list) or len(goal) != 2 or not all(math.isfinite(float(value)) for value in goal):
                raise ConfigurationError('Each agent needs a finite 2D goal')
            self.goals.append((float(goal[0]), float(goal[1])))
        self.statuses = ['active'] * len(agents)
        self.random = random.Random(seed)
        self.steps = 0
        self.time = 0.0
        for index, pose in enumerate(self.poses):
            if not settings.inside(pose) or not settings.inside(Pose(*self.goals[index], 0.0)):
                raise ConfigurationError('Starts and goals must be inside the robot-radius inset')
            for obstacle in settings.obstacles:
                if math.hypot(pose.position_x - obstacle.center_x, pose.position_y - obstacle.center_y) <= obstacle.radius + settings.robot_radius:
                    raise ConfigurationError('An initial robot overlaps an obstacle')
            for other in self.poses[:index]:
                if math.hypot(pose.position_x - other.position_x, pose.position_y - other.position_y) <= 2 * settings.robot_radius:
                    raise ConfigurationError('Initial robots overlap')

    @property
    def finished(self) -> bool:
        return all(status != 'active' for status in self.statuses)

    def observation(self, index: int) -> list[float]:
        pose = self.poses[index]
        goal = self.goals[index]
        return [pose.position_x, pose.position_y, math.cos(pose.heading), math.sin(pose.heading),
                goal[0] - pose.position_x, goal[1] - pose.position_y]

    def frame(self, actions: list[float] | None = None, events: list[dict] | None = None) -> dict:
        return {'time': self.time, 'agents': [
            {'id': index, 'pose': pose.to_list(), 'goal': list(self.goals[index]),
             'status': self.statuses[index], 'observation': self.observation(index),
             'commanded_action': None if actions is None else actions[index],
             'executed_action': None if actions is None else actions[index], 'shield': 'none'}
            for index, pose in enumerate(self.poses)], 'events': events or []}

    def scripted_actions(self) -> list[float]:
        actions = []
        for index, pose in enumerate(self.poses):
            if self.statuses[index] != 'active':
                actions.append(0.0)
                continue
            goal = self.goals[index]
            distance = max(math.hypot(goal[0] - pose.position_x, goal[1] - pose.position_y), 1e-9)
            force_x, force_y = (goal[0] - pose.position_x) / distance, (goal[1] - pose.position_y) / distance
            obstacles = [(obstacle.center_x, obstacle.center_y, obstacle.radius + self.settings.robot_radius)
                         for obstacle in self.settings.obstacles]
            obstacles += [(other.position_x, other.position_y, 2 * self.settings.robot_radius)
                          for other_index, other in enumerate(self.poses)
                          if other_index != index and self.statuses[other_index] == 'active']
            for center_x, center_y, radius in obstacles:
                offset_x, offset_y = pose.position_x - center_x, pose.position_y - center_y
                separation = max(math.hypot(offset_x, offset_y), 1e-9)
                clearance = separation - radius
                if clearance < 1.0:
                    repulsion = 0.7 * (1.0 - clearance) / max(clearance, 0.05)
                    force_x += repulsion * offset_x / separation
                    force_y += repulsion * offset_y / separation
            angle_error = wrap_angle(math.atan2(force_y, force_x) - pose.heading)
            actions.append(max(-self.settings.turn_rate, min(self.settings.turn_rate, 2.5 * angle_error)))
        return actions

    def step(self, actions: list[float]) -> dict:
        if self.finished:
            raise RuntimeError('Cannot step a terminated arena')
        if len(actions) != len(self.poses):
            raise ValueError('Exactly one action is required per agent')
        if not all(math.isfinite(action) and abs(action) <= self.settings.turn_rate + 1e-12 for action in actions):
            raise ValueError('Actions must be finite and within the configured turn-rate bounds')
        actual_turns = [action + self.random.gauss(0.0, self.settings.process_turn_noise_std)
                        for action in actions]
        events = []
        sub_dt = self.settings.dt / self.settings.collision_substeps
        for substep in range(self.settings.collision_substeps):
            previous = self.poses.copy()
            active = [status == 'active' for status in self.statuses]
            self.poses = [advance_dubins(pose, actual_turns[index], self.settings.speed, sub_dt)
                          if active[index] else pose for index, pose in enumerate(previous)]
            event_time = self.time + (substep + 1) * sub_dt
            def event(kind: str, identifiers: list[int]) -> None:
                events.append({'kind': kind, 'agent_ids': identifiers,
                               'time': event_time, 'substep': substep + 1})
            for index, pose in enumerate(self.poses):
                if not active[index]:
                    continue
                start = (previous[index].position_x, previous[index].position_y)
                end = (pose.position_x, pose.position_y)
                if not self.settings.inside(pose):
                    self.statuses[index] = 'domain_exit'
                    event('domain_exit', [index])
                for obstacle in self.settings.obstacles:
                    if segment_distance(start, end, (obstacle.center_x, obstacle.center_y)) <= obstacle.radius + self.settings.robot_radius:
                        self.statuses[index] = 'collision'
                        event('obstacle_collision', [index])
            for first in range(len(self.poses)):
                for second in range(first + 1, len(self.poses)):
                    if not (active[first] and active[second]):
                        continue
                    relative_start = (previous[first].position_x - previous[second].position_x,
                                      previous[first].position_y - previous[second].position_y)
                    relative_end = (self.poses[first].position_x - self.poses[second].position_x,
                                    self.poses[first].position_y - self.poses[second].position_y)
                    if segment_distance(relative_start, relative_end, (0.0, 0.0)) <= 2 * self.settings.robot_radius:
                        self.statuses[first] = self.statuses[second] = 'collision'
                        event('agent_collision', [first, second])
            for index, pose in enumerate(self.poses):
                if self.statuses[index] == 'active' and math.hypot(pose.position_x - self.goals[index][0], pose.position_y - self.goals[index][1]) <= self.settings.goal_radius:
                    self.statuses[index] = 'goal'
                    event('goal_reached', [index])
        self.steps += 1
        self.time = self.steps * self.settings.dt
        if self.steps == self.settings.max_steps:
            for index, status in enumerate(self.statuses):
                if status == 'active':
                    self.statuses[index] = 'timeout'
                    events.append({'kind': 'horizon_expired', 'agent_ids': [index], 'time': self.time})
        result = self.frame(actions, events)
        for index, agent in enumerate(result['agents']):
            agent['realized_turn_rate'] = actual_turns[index]
        return result
