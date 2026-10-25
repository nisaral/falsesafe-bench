from __future__ import annotations

import random

from .config import ConfigurationError
from .envs import ArenaSettings, RobotArena


def simulate(config: dict) -> dict:
    settings = ArenaSettings.from_dict(config['arena'])
    arena = RobotArena(settings, config['agents'], config['seed'])
    controller = config.get('controller')
    if controller not in ('scripted_goal', 'random'):
        raise ConfigurationError('Starter simulation supports explicitly scripted_goal or random controllers')
    action_random = random.Random(config['seed'] + 1)
    frames = [arena.frame()]
    while not arena.finished:
        actions = arena.scripted_actions() if controller == 'scripted_goal' else [
            action_random.uniform(-settings.turn_rate, settings.turn_rate) for _ in arena.poses]
        frames.append(arena.step(actions))
    return {'schema_version': 1, 'metadata': {
        'purpose': 'simulation_diagnostic', 'controller': controller, 'learning': False,
        'safety_certificate': False, 'seed': config['seed'], 'split_seed': config['split_seed'],
        'split_role': config.get('split_role', 'validation'), 'dt': settings.dt,
        'collision_substeps': settings.collision_substeps,
        'collision_check': 'substepped segment checks; not a continuous-time guarantee',
        'terminal_agents': 'removed from active dynamics after a terminal event',
        'bounds': list(settings.bounds), 'robot_radius': settings.robot_radius,
        'goal_radius': settings.goal_radius, 'process_turn_noise_std': settings.process_turn_noise_std,
        'obstacles': [obstacle.to_list() for obstacle in settings.obstacles]}, 'frames': frames}
