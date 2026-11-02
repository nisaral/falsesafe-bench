"""Plug-in interfaces for the falsesafe framework.

Every plug-in is a small Python object.  States are arrays whose *last* axis is
the state dimension; dynamics and hazards must be written with ``jax.numpy``
so the same function drives the HJ solver and the trajectory simulator.

Sign convention (same as the rest of lcert): value >= 0 means *safe*, the
hazard/failure function ``l(x) < 0`` marks the failure set, and the HJ solver
computes the backward reachable tube (BRT) of that failure set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class Hazard:
    """Failure set {x : fn(x) < 0}.  ``fn`` maps states[..., n] -> [...]."""

    fn: Callable[[Any], Any]
    description: str = ""

    def __call__(self, states):
        return self.fn(states)


@dataclass
class GapRegion:
    """Axis-aligned box where the learner has no data / physics differ.

    Used for the gap-FSV metric (``lcert.metrics.coverage_gap_mask``) and for
    excluding training samples in learned certifiers.
    """

    lo: Sequence[float]
    hi: Sequence[float]
    description: str = ""

    def contains(self, states: np.ndarray) -> np.ndarray:
        s = np.asarray(states)
        return np.all((s >= np.asarray(self.lo)) & (s <= np.asarray(self.hi)), axis=-1)

    def as_coverage_gap(self) -> dict:
        return {"enabled": True, "lo": list(map(float, self.lo)), "hi": list(map(float, self.hi))}


@dataclass
class Environment:
    """A control-affine-or-not system with a (possibly) hidden-physics gap.

    Attributes:
        true_dynamics:    f_true(x, u) -> xdot  (jnp, single state).  Ground truth.
        nominal_dynamics: f_nom(x, u)  -> xdot  what a model-based certifier
                          believes (typically f_true without the hidden physics).
        controls:         finite candidate control set, shape (m, udim).  The HJ
                          Hamiltonian maximises over this set (exact for
                          control-affine systems with box inputs if the vertices
                          are included).
        hazard:           failure set {l(x) < 0}.
        gap:              optional GapRegion (hidden-physics / coverage gap).
        ground_truth_solver: optional override ``(env) -> (grid, V)``; e.g. the
                          built-in double integrator reuses ``lcert.hj_di``.
    """

    name: str
    state_names: Sequence[str]
    lo: Sequence[float]
    hi: Sequence[float]
    true_dynamics: Callable
    nominal_dynamics: Callable
    controls: np.ndarray
    hazard: Hazard
    horizon: float = 0.5
    grid_size: int | Sequence[int] = 101
    periodic_dims: Sequence[int] = ()
    gap: GapRegion | None = None
    accuracy: str = "medium"
    ground_truth_solver: Callable | None = None
    params: dict = field(default_factory=dict)
    description: str = ""

    @property
    def dim(self) -> int:
        return len(self.lo)

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.grid_size, int):
            return (self.grid_size,) * self.dim
        return tuple(int(n) for n in self.grid_size)

    def __post_init__(self):
        self.controls = np.atleast_2d(np.asarray(self.controls, dtype=float))
        if self.controls.shape[0] == 1 and self.controls.shape[1] > 1:
            self.controls = self.controls.T
        if len(self.lo) != len(self.hi) or len(self.state_names) != len(self.lo):
            raise ValueError("lo, hi and state_names must have the same length")
        if not np.all(np.asarray(self.lo) < np.asarray(self.hi)):
            raise ValueError("lo must be strictly below hi")


@runtime_checkable
class Policy(Protocol):
    """A (possibly unsafe) task policy: ``act(x, t) -> u`` (array of udim)."""

    name: str

    def act(self, state: np.ndarray, t: float) -> np.ndarray: ...


@dataclass
class CertifierOutput:
    value: np.ndarray            # learned/certified value on the env grid
    info: dict = field(default_factory=dict)


@runtime_checkable
class Certifier(Protocol):
    """Produces a certified value function V_hat on the environment grid.

    ``certify(env, grid) -> CertifierOutput``; ``V_hat >= 0`` is the set the
    certifier *claims* is safe.  It must never look at the true value function.
    """

    name: str

    def certify(self, env: Environment, grid) -> CertifierOutput: ...


@dataclass
class FunctionPolicy:
    """Wrap a plain function ``fn(x, t) -> u`` as a Policy."""

    name: str
    fn: Callable

    def act(self, state, t):
        return np.atleast_1d(np.asarray(self.fn(np.asarray(state), t), dtype=float))
