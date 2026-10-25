from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AlgorithmSpec:
    identifier: str
    family: str
    readiness: str
    input_contract: str
    reference: str

    def to_dict(self) -> dict:
        return asdict(self)


ALGORITHMS = (
    AlgorithmSpec('reinforce', 'on-policy', 'objective_ready',
                  'Action log-probabilities and trajectory returns.', 'Policy-gradient baseline'),
    AlgorithmSpec('a2c', 'on-policy actor-critic', 'objective_ready',
                  'Action log-probabilities, advantages, and entropy; critic targets separate.', 'Actor-critic baseline'),
    AlgorithmSpec('ppo', 'on-policy actor-critic', 'objective_ready',
                  'Frozen old-policy log-probabilities, advantages, and clipped likelihood ratios.', '1707.06347'),
    AlgorithmSpec('grpo_robot', 'grouped on-policy adaptation', 'objective_ready',
                  'Grouped same-context robot rollouts, rewards, action masks, old/reference log-probabilities.', '2402.03300'),
    AlgorithmSpec('dpo_robot', 'offline trajectory-preference adaptation', 'objective_ready',
                  'Chosen/rejected robot trajectories from the same context and a frozen reference policy.', '2305.18290'),
    AlgorithmSpec('behavior_cloning', 'offline supervised baseline', 'objective_ready',
                  'Observed actions with explicit demonstrator provenance.', 'Supervised baseline'),
    AlgorithmSpec('iql', 'offline actor-critic', 'expectile_objective_ready',
                  'Fixed replay dataset; expectile value loss is provided, full trainer is not.', '2110.06169'),
    AlgorithmSpec('sac', 'off-policy maximum-entropy control', 'planned',
                  'Replay buffer, twin critics, stochastic bounded actor, and entropy-temperature updates.', '1812.05905'),
    AlgorithmSpec('td3', 'off-policy continuous control', 'planned',
                  'Replay buffer, twin critics, delayed actor updates, and target-action smoothing.', '1802.09477'),
    AlgorithmSpec('cql', 'offline conservative control', 'planned',
                  'Fixed replay support, conservative Q regularisation, and OOD-action checks.', '2006.04779'),
    AlgorithmSpec('mappo', 'cooperative multi-agent actor-critic', 'planned',
                  'Per-agent masks and observations with a reviewed centralised-critic contract.', '2103.01955'),
    AlgorithmSpec('independent_ppo', 'multi-agent baseline', 'planned',
                  'Separate per-agent policies, seeds, trajectories, and intervention logs.', '1707.06347'),
)


def get_algorithm(identifier: str) -> AlgorithmSpec:
    for specification in ALGORITHMS:
        if specification.identifier == identifier:
            return specification
    raise ValueError(f'Unknown algorithm: {identifier}')
