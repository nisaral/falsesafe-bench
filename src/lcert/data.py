from __future__ import annotations

import math

import numpy as np

from .config import ConfigurationError, require_integer, require_positive
from .envs.dubins import Pose, advance_dubins
from .systems import double_integrator_exact, double_integrator_f


def _in_box(state: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    return bool(np.all(state >= lower) and np.all(state <= upper))


def generate_trajectory_splits(config: dict) -> tuple[dict, dict]:
    """Synthetic noisy derivative observations along fixed, disjoint trajectories."""
    system = config.get('system')
    if system not in ('double_integrator', 'dubins'):
        raise ConfigurationError('Trajectory generator supports double_integrator and dubins')
    dimensions = 2 if system == 'double_integrator' else 3
    lower = np.asarray(config.get('state_lo'), dtype=float)
    upper = np.asarray(config.get('state_hi'), dtype=float)
    if lower.shape != (dimensions,) or upper.shape != lower.shape:
        raise ConfigurationError('State bounds must match the system dimension')
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or not np.all(lower < upper):
        raise ConfigurationError('State bounds must be finite and ordered')
    dt = require_positive(config, 'dt')
    steps = require_integer(config, 'trajectory_steps', 1)
    attempts_limit = require_integer(config, 'max_sampling_attempts', 1)
    action_limit = require_positive(config, 'umax' if dimensions == 2 else 'turn_rate')
    speed = require_positive(config, 'speed') if dimensions == 3 else None
    noise = float(config.get('derivative_observation_noise_std', 0.0))
    if not math.isfinite(noise) or noise < 0:
        raise ConfigurationError('Derivative-observation noise must be finite and nonnegative')
    counts = config.get('trajectory_counts')
    if not isinstance(counts, dict):
        raise ConfigurationError('trajectory_counts must explicitly define all three splits')
    for role in ('train', 'validation', 'test'):
        require_integer(counts, role, 1)
    gap = config.get('coverage_gap', {'enabled': False})
    gap_lower = gap_upper = None
    if gap.get('enabled'):
        gap_lower, gap_upper = np.asarray(gap.get('lo'), float), np.asarray(gap.get('hi'), float)
        if gap_lower.shape != lower.shape or gap_upper.shape != lower.shape:
            raise ConfigurationError('Coverage gap needs explicit, dimension-matched lo/hi bounds')
        if not np.isfinite(gap_lower).all() or not np.isfinite(gap_upper).all() or not np.all(gap_lower < gap_upper):
            raise ConfigurationError('Coverage gap must be finite and ordered')
        if not np.all(gap_lower >= lower) or not np.all(gap_upper <= upper):
            raise ConfigurationError('Coverage gap must be contained in the declared state domain')
    splits, sampling = {}, {}
    for role_index, role in enumerate(('train', 'validation', 'test')):
        state_random = np.random.default_rng(np.random.SeedSequence([config['split_seed'], role_index]))
        noise_random = np.random.default_rng(np.random.SeedSequence([config['seed'], config['split_seed'], role_index, 1]))
        trajectories, action_sequences, features, targets = [], [], [], []
        attempts = 0
        while len(trajectories) < counts[role] and attempts < attempts_limit:
            attempts += 1
            state = state_random.uniform(lower, upper)
            states = [state.copy()]
            actions = state_random.uniform(-action_limit, action_limit, size=steps)
            derivatives = []
            valid = True
            for action in actions:
                if not _in_box(state, lower, upper) or (role != 'test' and gap_lower is not None and _in_box(state, gap_lower, gap_upper)):
                    valid = False
                    break
                if dimensions == 2:
                    derivatives.append(double_integrator_f(state, action))
                    state = double_integrator_exact(state, action, dt)
                else:
                    derivatives.append(np.array([speed * np.cos(state[2]), speed * np.sin(state[2]), action]))
                    state = np.asarray(advance_dubins(Pose.from_list(state.tolist()), float(action), speed, dt).to_list())
                states.append(state.copy())
            if not valid or not _in_box(state, lower, upper) or (role != 'test' and gap_lower is not None and _in_box(state, gap_lower, gap_upper)):
                continue
            states = np.asarray(states)
            observations = np.asarray(derivatives) + noise_random.normal(0.0, noise, size=(steps, dimensions))
            trajectories.append(states)
            action_sequences.append(actions)
            features.append(np.column_stack((states[:-1], actions)))
            targets.append(observations)
        if len(trajectories) != counts[role]:
            raise RuntimeError(f'Could not generate {counts[role]} {role} trajectories within {attempts_limit} attempts; no data-size fallback was applied')
        identifiers = np.array([f'{role}-{config["split_seed"]}-{index}' for index in range(counts[role])])
        splits[role] = {'states': np.asarray(trajectories), 'actions': np.asarray(action_sequences),
                        'features': np.concatenate(features), 'targets': np.concatenate(targets),
                        'trajectory_ids': identifiers}
        sampling[role] = {'trajectories': len(trajectories), 'sampling_attempts': attempts}
    manifest = {'observation_kind': 'synthetic derivative observations along simulated trajectories',
                'physical_process_noise': 'none in this starter generator',
                'derivative_observation_noise_std': noise, 'split_seed': config['split_seed'],
                'observation_noise_seed': config['seed'], 'sampling': sampling,
                'test_artifact_written': True, 'test_split_evaluated': False,
                'coverage_gap_applies_to': ['train', 'validation']}
    return splits, manifest
