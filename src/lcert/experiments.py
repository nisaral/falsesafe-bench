from __future__ import annotations

from dataclasses import asdict, dataclass


class ExperimentUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentSpec:
    identifier: str
    title: str
    readiness: str
    blockers: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


EXPERIMENTS = (
    ExperimentSpec('E0', 'True-dynamics ground-truth snapshots', 'diagnostic_only',
                   ('Full metric evaluation awaits protected-metric review.',)),
    ExperimentSpec('E1', 'Nominal FSV vs data size and coverage', 'gated',
                   ('Reviewed metrics, Dubins convergence, and model-recovery checks required.',)),
    ExperimentSpec('E2', 'Uncertainty inflation vs conservatism', 'gated',
                   ('E1 verification and tuned constant-disturbance baseline required.',)),
    ExperimentSpec('E3', 'Separated vs blended uncertainty', 'gated',
                   ('E2 verification and reviewed noise/metric specification required.',)),
    ExperimentSpec('E4', 'Pairwise fleet safety', 'gated',
                   ('MVR review and relative-state/constraint-conflict checks required.',)),
    ExperimentSpec('E5', 'QRE incentive identification', 'gated',
                   ('Identifiability normalisation and payoff-set metric review required.',)),
    ExperimentSpec('E6', 'Single-robot learning algorithms', 'gated',
                   ('MVR sign-off, training budget, and rollout-metric review required.',)),
    ExperimentSpec('E7', 'Cooperative and competitive fleet learning', 'gated',
                   ('E6 verification and joint-control conflict checks required.',)),
    ExperimentSpec('E8', 'Population optimal transport', 'gated',
                   ('Transport baseline and held-out population-metric review required.',)),
    ExperimentSpec('E9', 'Finite-state Schrodinger bridges', 'gated',
                   ('Reference-process and path-space diagnostic review required.',)),
    ExperimentSpec('E10', 'Toy mean-field congestion game', 'gated',
                   ('Best-response/population consistency and metric review required.',)),
    ExperimentSpec('E11', 'Physics and geometric learning', 'gated',
                   ('Prior assumptions and constraint-mismatch control review required.',)),
    ExperimentSpec('E12', 'Emergent swarm exploration', 'gated',
                   ('Exploration/claim boundary and coordination diagnostics review required.',)),
)


def get_experiment(identifier: str) -> ExperimentSpec:
    for specification in EXPERIMENTS:
        if specification.identifier == identifier:
            return specification
    raise ExperimentUnavailable(f'Unknown experiment: {identifier}')


def require_runnable(identifier: str) -> ExperimentSpec:
    specification = get_experiment(identifier)
    if specification.readiness != 'diagnostic_only':
        raise ExperimentUnavailable(
            f'{identifier} is registered but not a runnable reported study. '
            + ' '.join(specification.blockers)
        )
    return specification
