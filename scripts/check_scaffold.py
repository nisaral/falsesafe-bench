from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from lcert.cli import environment_report
from lcert.config import ConfigurationError, load_config, validate_common
from lcert.constraints.geometry import rotate_vectors, world_to_body
from lcert.constraints.physics import dubins_kinematic_prior_loss
from lcert.data import generate_trajectory_splits
from lcert.envs import ArenaSettings, RobotArena
from lcert.envs.dubins import Pose, advance_dubins, segment_distance
from lcert.experiments import EXPERIMENTS, ExperimentUnavailable, require_runnable
from lcert.games.qre import sample_play, solve_logit_qre
from lcert.models.ensemble import EnsembleMLP
from lcert.models.gp import IndependentExactGP
from lcert.population.bridge import probability_vector, solve_discrete_bridge, stochastic_kernels
from lcert.population.mean_field import solve_congestion_game
from lcert.population.transport import entropic_transport
from lcert.rl import objectives
from lcert.rl.catalog import ALGORITHMS
from lcert.rl.policies import CategoricalActorCritic, discrete_turn_rates
from lcert.rl.replay import ReplayBuffer, Transition
from lcert.rl.tasks import NavigationReward
from lcert.simulation import simulate


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def expect_error(operation, exception_type) -> None:
    try:
        operation()
    except exception_type:
        return
    raise RuntimeError(f'Expected {exception_type.__name__} was not raised')


def check_environment_and_contracts() -> None:
    require(environment_report()['ready'], 'Use the approved pinned Python environment')
    # AGENTS.md / CLAUDE.md are local agent configs and are not tracked in git;
    # only compare the scope copies that actually exist (e.g. not in CI).
    scope_copies = [ROOT / name for name in ('SCOPE.md', 'AGENTS.md', 'CLAUDE.md')
                    if (ROOT / name).exists()]
    require(len({path.read_bytes() for path in scope_copies}) <= 1, 'Scope copies differ')
    for config_path in sorted((ROOT / 'configs' / 'starters').glob('*.json')):
        load_config(config_path)
    expect_error(lambda: validate_common({'experiment': 'simulation', 'seed': True, 'split_seed': 42}),
                 ConfigurationError)
    for specification in EXPERIMENTS:
        if specification.identifier != 'E0':
            expect_error(lambda: require_runnable(specification.identifier), ExperimentUnavailable)
    require({'ppo', 'grpo_robot', 'dpo_robot', 'sac', 'td3', 'iql', 'cql', 'mappo', 'independent_ppo'}
            <= {item.identifier for item in ALGORITHMS}, 'Algorithm catalogue is incomplete')


def check_robot_kinematics() -> None:
    initial = Pose(0.0, 0.0, 0.0)
    require(np.allclose(advance_dubins(initial, 0.0, 1.0, 0.5).to_list(), [0.5, 0.0, 0.0],
                        rtol=0, atol=1e-12), 'Straight-line motion differs from its closed form')
    turned = advance_dubins(initial, 1.0, 1.0, math.pi / 2)
    require(np.allclose(turned.to_list(), [1.0, 1.0, math.pi / 2], rtol=0, atol=1e-12),
            'Constant-turn motion differs from its closed form')
    require(np.allclose(advance_dubins(initial, 1e-12, 1.0, 0.5).to_list(), [0.5, 0.0, 0.0],
                        rtol=0, atol=1e-12), 'Near-zero turn is discontinuous')
    require(segment_distance((-1.0, 0.0), (1.0, 0.0), (0.0, 0.5)) == 0.5,
            'Segment clearance is incorrect')


def arena_settings(**overrides) -> ArenaSettings:
    settings = {'bounds': [-3.0, 3.0, -3.0, 3.0], 'dt': 1.0, 'max_steps': 3,
                'speed': 1.0, 'turn_rate': 1.6, 'robot_radius': 0.1, 'goal_radius': 0.2,
                'collision_substeps': 8, 'process_turn_noise_std': 0.0, 'obstacles': []}
    return ArenaSettings.from_dict(settings | overrides)


def check_robot_events() -> None:
    collision = RobotArena(arena_settings(), [
        {'start': [-0.6, 0.0, 0.0], 'goal': [2.5, 0.0]},
        {'start': [0.6, 0.0, math.pi], 'goal': [-2.5, 0.0]}], 0)
    frame = collision.step([0.0, 0.0])
    require(any(event['kind'] == 'agent_collision' for event in frame['events']) and collision.finished,
            'Crossing agents did not produce a terminal collision')
    expect_error(lambda: collision.step([0.0, 0.0]), RuntimeError)
    cases = [
        (arena_settings(obstacles=[[0.0, 0.0, 0.2]]), [-1.1, 0.0, 0.0], [2.0, 0.0], 'obstacle_collision'),
        (arena_settings(), [2.7, 0.0, 0.0], [-2.0, 0.0], 'domain_exit'),
        (arena_settings(), [-0.6, 0.0, 0.0], [0.0, 0.0], 'goal_reached'),
        (arena_settings(dt=0.05, max_steps=1), [-1.0, 0.0, 0.0], [2.0, 0.0], 'horizon_expired')]
    for settings, start, goal, kind in cases:
        arena = RobotArena(settings, [{'start': start, 'goal': goal}], 0)
        before = arena.frame()
        current = arena.step([0.0])
        require(any(event['kind'] == kind for event in current['events']) and arena.finished,
                f'Missing terminal event: {kind}')
        require(math.isfinite(NavigationReward().transition_rewards(before, current)[0]),
                'Reward primitive returned a nonfinite value')
    invalid = RobotArena(arena_settings(), [{'start': [-1.0, 0.0, 0.0], 'goal': [2.0, 0.0]}], 0)
    expect_error(lambda: invalid.step([2.0]), ValueError)


def check_replay_determinism() -> None:
    for filename in ('single_robot.json', 'fleet.json'):
        config = load_config(ROOT / 'configs' / 'starters' / filename)
        first, second = simulate(config), simulate(config)
        require(first == second, f'Same-seed simulation differs: {filename}')
        require(first['metadata']['learning'] is False and first['metadata']['safety_certificate'] is False,
                'Scripted simulation is labelled learned or certified')
        require(all(agent['status'] != 'active' for agent in first['frames'][-1]['agents']),
                'Replay ended before all terminal states were recorded')
        require(all(before['time'] < after['time'] for before, after
                    in zip(first['frames'], first['frames'][1:])), 'Replay time did not advance')


def check_data_generation() -> None:
    config = copy.deepcopy(load_config(ROOT / 'configs' / 'starters' / 'trajectory_data.json'))
    config.update(seed=19, split_seed=23, trajectory_steps=2,
                  trajectory_counts={'train': 2, 'validation': 1, 'test': 1})
    first, manifest = generate_trajectory_splits(config)
    second, repeated_manifest = generate_trajectory_splits(config)
    require(manifest == repeated_manifest and not manifest['test_split_evaluated'],
            'Data generator changed provenance or evaluated the test split')
    for role in ('train', 'validation'):
        for name in first[role]:
            require(np.array_equal(first[role][name], second[role][name]),
                    f'Seeded training/validation data differ: {role}/{name}')
        states = first[role]['states']
        lower, upper = np.asarray(config['coverage_gap']['lo']), np.asarray(config['coverage_gap']['hi'])
        require(not np.any(np.all((states >= lower) & (states <= upper), axis=-1)),
                'Training/validation trajectory entered the declared coverage gap')


def check_policy_objectives() -> None:
    for loss_function in (
        lambda current: objectives.reinforce_loss(current, torch.tensor([1.0, -1.0])),
        lambda current: objectives.a2c_loss(current, torch.tensor([1.0, -1.0]), torch.ones(2)),
        lambda current: objectives.ppo_loss(current, torch.tensor([-1.0, -1.0]), torch.tensor([1.0, -1.0]))):
        current = torch.tensor([-0.9, -1.1], requires_grad=True)
        loss = loss_function(current)
        loss.backward()
        require(torch.isfinite(loss).item() and torch.isfinite(current.grad).all().item(),
                'Policy objective or gradient is nonfinite')
    chosen = torch.tensor([-1.0], requires_grad=True)
    rejected = torch.tensor([-2.0], requires_grad=True)
    reference_chosen = chosen.detach().clone().requires_grad_()
    reference_rejected = rejected.detach().clone().requires_grad_()
    loss = objectives.dpo_loss(chosen, rejected, reference_chosen, reference_rejected)
    require(torch.allclose(loss, torch.tensor(math.log(2))), 'DPO reference-ratio baseline is incorrect')
    loss.backward()
    require(chosen.grad.item() < 0 < rejected.grad.item() and reference_chosen.grad is None
            and reference_rejected.grad is None, 'DPO gradients or frozen reference are incorrect')
    predictions = torch.zeros(2, requires_grad=True)
    objectives.expectile_loss(predictions, torch.tensor([-1.0, 2.0])).backward()
    require(torch.isfinite(predictions.grad).all().item(), 'Expectile gradient is nonfinite')
    logits = torch.zeros((2, 3), requires_grad=True)
    objectives.behavior_cloning_loss(logits, torch.tensor([0, 2])).backward()
    require(torch.isfinite(logits.grad).all().item(), 'Behaviour-cloning gradient is nonfinite')


def check_grouped_objective() -> None:
    rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    advantages = objectives.group_relative_advantages(rewards)
    require(torch.equal(advantages[0], torch.zeros(3)) and torch.isfinite(advantages).all().item(),
            'Equal-reward group produced a nonzero or nonfinite advantage')
    require(torch.allclose(advantages.mean(dim=1), torch.zeros(2), atol=1e-7),
            'Group-relative advantages are not centred by context')
    current = torch.full((2, 3, 4), -0.5, requires_grad=True)
    old = current.detach().clone().requires_grad_()
    reference = current.detach().clone().requires_grad_()
    mask = torch.ones_like(current, dtype=torch.bool)
    mask[..., -1] = False
    loss = objectives.grpo_loss(current, old, reference, rewards, mask)
    loss.backward()
    require(torch.isfinite(current.grad).all().item() and torch.equal(current.grad[~mask], torch.zeros(6)),
            'GRPO gradient is nonfinite or updates padded actions')
    require(old.grad is None and reference.grad is None, 'GRPO old/reference policy was not frozen')
    expect_error(lambda: objectives.grpo_loss(current, old, reference, rewards, torch.zeros_like(mask)), ValueError)


def check_policy_and_replay_buffer() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        policy = CategoricalActorCritic()
        distribution, values = policy(torch.zeros((4, 6)))
    require(values.shape == (4,) and torch.allclose(distribution.probs.sum(dim=-1), torch.ones(4)),
            'Actor/critic shape or probability contract failed')
    require(discrete_turn_rates(1.6) == (-1.6, 0.0, 1.6), 'Discrete action authority changed')
    expect_error(lambda: discrete_turn_rates(math.inf), ValueError)
    first, second = ReplayBuffer(3, 42), ReplayBuffer(3, 42)
    for index in range(4):
        transition = Transition((float(index), 0.0), 0.5, 0.25, 1.0,
                                (float(index + 1), 0.0), False, False)
        first.append(transition)
        second.append(transition)
    require(len(first) == 3 and first.sample(2) == second.sample(2), 'Replay capacity or seeded sampling failed')
    expect_error(lambda: first.sample(4), ValueError)


def check_unfitted_models() -> None:
    for model in (EnsembleMLP(), IndependentExactGP()):
        expect_error(lambda: model.predict(np.zeros((1, 3))), RuntimeError)
    expect_error(lambda: EnsembleMLP().save(ROOT / '.cache' / 'untrained.pt'), RuntimeError)


def check_transport_and_bridge() -> None:
    result = entropic_transport([[0.0, 0.0], [1.0, 0.0]], [[0.25, 0.0], [0.75, 0.0]],
                               0.5, [0.7, 0.3], [0.4, 0.6], tolerance=1e-8)
    require(result.marginal_residual <= 1e-8 and np.all(result.coupling >= 0), 'OT marginal check failed')
    reference = np.tile(np.array([[0.8, 0.2], [0.3, 0.7]]), (4, 1, 1))
    result = solve_discrete_bridge([0.75, 0.25], [0.3, 0.7], reference)
    require(np.allclose(result.population.sum(axis=1), 1.0, rtol=0, atol=1e-8)
            and np.allclose(result.population[-1], [0.3, 0.7], rtol=0, atol=1e-8),
            'Bridge mass or endpoint constraint failed')
    expect_error(lambda: solve_discrete_bridge([1.0, 0.0], [0.0, 1.0], np.eye(2)[None, ...]), ValueError)
    expect_error(lambda: probability_vector([0.5, 0.500001]), ValueError)
    expect_error(lambda: stochastic_kernels(np.array([[[0.5, 0.500001], [0.5, 0.5]]])), ValueError)


def check_mean_field_and_games() -> None:
    kernels = np.array([[[0.9, 0.1], [0.2, 0.8]], [[0.2, 0.8], [0.8, 0.2]]])
    result = solve_congestion_game([0.65, 0.35], kernels, [0.1, 0.4], 4,
                                   congestion_weight=0.25, temperature=1.0)
    require(result.consistency_residual <= 1e-7
            and np.allclose(result.population.sum(axis=1), 1.0, rtol=0, atol=1e-8)
            and np.allclose(result.policy.sum(axis=2), 1.0, rtol=0, atol=1e-8),
            'Mean-field probability or best-response consistency failed')
    first = np.array([[0.4, -0.2], [0.1, 0.3]])
    second = np.array([[0.2, 0.4], [-0.1, 0.3]])
    original = solve_logit_qre(first, second, 0.9, 1.2)
    scaled = solve_logit_qre(3 * first, 3 * second, 2.7, 3.6)
    require(np.allclose(original.first_policy, scaled.first_policy, rtol=0, atol=1e-10)
            and np.allclose(original.second_policy, scaled.second_policy, rtol=0, atol=1e-10),
            'QRE simultaneous payoff/temperature scale invariance failed')
    require(np.array_equal(sample_play(original, 20, 42), sample_play(original, 20, 42)),
            'QRE play sampling is not seeded')


def check_geometry_and_physics() -> None:
    vectors = np.array([[1.0, 2.0], [-0.5, 1.5]])
    headings = np.array([0.3, -0.2])
    original = world_to_body(vectors, headings)
    transformed = world_to_body(rotate_vectors(vectors, 0.7), headings + 0.7)
    require(np.allclose(original, transformed, rtol=0, atol=1e-12), 'Body-frame rotation invariance failed')
    states = torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, -0.2]])
    predicted = torch.stack((states[:, 2].cos(), states[:, 2].sin(), torch.zeros(2)), dim=-1)
    require(dubins_kinematic_prior_loss(predicted, states, 1.0).item() == 0,
            'Correct public kinematic prior is inconsistent')
    require(dubins_kinematic_prior_loss(predicted, states, 2.0).item() > 0,
            'Incorrect speed prior was not detected')


def main() -> int:
    parser = argparse.ArgumentParser(description='Synthetic starter checks; no training or benchmark evaluation')
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    if options.output and not options.output.resolve().is_relative_to(ROOT):
        parser.error('Check output must remain inside this project')
    torch.set_num_threads(1)
    checks = (check_environment_and_contracts, check_robot_kinematics, check_robot_events,
              check_replay_determinism, check_data_generation, check_policy_objectives,
              check_grouped_objective, check_policy_and_replay_buffer, check_unfitted_models,
              check_transport_and_bridge, check_mean_field_and_games, check_geometry_and_physics)
    passed = []
    for operation in checks:
        operation()
        passed.append(operation.__name__)
        print(f'PASS {operation.__name__}')
    report = {'purpose': 'synthetic_scaffold_checks', 'training_performed': False,
              'benchmark_evaluation': False, 'passed': passed}
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Starter checks passed; no trained-policy, uncertainty-bound, or safety claim.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
