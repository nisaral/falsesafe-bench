"""Built-in certifiers (and adapters around the repo's existing learned models)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .core import CertifierOutput, Environment
from .registry import register_certifier
from .solver import solve_brt

_MODEL_CACHE: dict = {}


@register_certifier("nominal_inflated")
@dataclass
class NominalInflatedCertifier:
    """Robust HJ on env.nominal_dynamics with a constant disturbance bound k*sigma
    on every state derivative listed in ``dims`` (default: all).  k=0 is the
    naive nominal certifier.  No learning involved."""

    k: float = 0.0
    sigma: float = 0.1
    dims: tuple | None = None
    name: str = "nominal_inflated"

    def certify(self, env: Environment, grid):
        mask = np.zeros(env.dim) if self.dims is not None else np.ones(env.dim)
        if self.dims is not None:
            mask[list(self.dims)] = 1.0
        d = jnp.asarray(mask * self.k * self.sigma)
        _, V = solve_brt(env, env.nominal_dynamics, (lambda x: d) if self.k > 0 else None, grid)
        return CertifierOutput(V, {"k": self.k, "sigma": self.sigma, "learned": False})


@register_certifier("ensemble")
@dataclass
class EnsembleCertifier:
    """Adapter around ``lcert.train.train_dynamics_model('ensemble_mlp')`` and the
    JAX ensemble evaluator in ``lcert.hj_learned``.

    Data: ``n_samples`` i.i.d. (x, u) samples drawn uniformly in the state box,
    *excluding* env.gap (the coverage gap), labelled with f_true(x, u) + Gaussian
    noise.  NOTE: this is NOT the preregistered trajectory generator, so numbers
    from this adapter are not P1 numbers.
    HJ: f = ensemble mean, disturbance bound = k * max_u ensemble_std(x, u).
    """

    k: float = 2.0
    members: int = 3
    hidden_sizes: tuple = (32, 32)
    epochs: int = 60
    learning_rate: float = 0.002
    n_samples: int = 2000
    noise_std: float = 0.01
    seed: int = 0
    name: str = "ensemble"

    def _train(self, env: Environment):
        key = (env.name, tuple(sorted((k, str(v)) for k, v in env.params.items())), self.members,
               tuple(self.hidden_sizes), self.epochs, self.n_samples, self.noise_std, self.seed)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        from lcert.train import train_dynamics_model
        rng = np.random.default_rng(self.seed)
        lo, hi = np.asarray(env.lo), np.asarray(env.hi)
        xs = []
        while sum(len(a) for a in xs) < self.n_samples:
            cand = rng.uniform(lo, hi, size=(self.n_samples, env.dim))
            if env.gap is not None:
                cand = cand[~env.gap.contains(cand)]
            xs.append(cand)
        X = np.concatenate(xs)[: self.n_samples]
        U = rng.uniform(env.controls.min(0), env.controls.max(0), size=(len(X), env.controls.shape[1]))
        f = jax.jit(jax.vmap(lambda x, u: jnp.asarray(env.true_dynamics(x, u))))
        Y = np.asarray(f(X, U)) + rng.normal(0, self.noise_std, size=(len(X), env.dim))
        t0 = time.monotonic()
        model, diag = train_dynamics_model("ensemble_mlp", np.column_stack([X, U]).astype(np.float32),
                                           Y.astype(np.float32), seed=self.seed, epochs=self.epochs,
                                           learning_rate=self.learning_rate, members=self.members,
                                           hidden_sizes=tuple(self.hidden_sizes))
        diag["wall_seconds"] = time.monotonic() - t0
        _MODEL_CACHE[key] = (model, diag)
        return model, diag

    def certify(self, env: Environment, grid):
        if len(self.hidden_sizes) != 2:
            raise ValueError("lcert.hj_learned's JAX evaluator supports exactly 2 hidden layers")
        from lcert.hj_learned import LearnedDoubleIntegratorDynamics, extract_ensemble_jax_params
        model, diag = self._train(env)
        w, b, mean, scale = extract_ensemble_jax_params(model)
        ev = LearnedDoubleIntegratorDynamics(w, b, mean, scale, umax=1.0)
        controls = jnp.asarray(env.controls)

        def f_mean(x, u):
            return ev._eval_ensemble(x, u[0])[0]

        def dbound(x):
            stds = jnp.stack([ev._eval_ensemble(x, controls[i, 0])[1] for i in range(controls.shape[0])])
            return self.k * jnp.max(stds, axis=0)

        _, V = solve_brt(env, f_mean, dbound if self.k > 0 else None, grid)
        return CertifierOutput(V, {"k": self.k, "learned": True, "training": diag,
                                   "data": "iid uniform samples outside gap (framework adapter)"})


@register_certifier("oracle")
@dataclass
class OracleCertifier:
    """Sanity check only: certifies with the TRUE dynamics (should give FSV ~ 0)."""

    name: str = "oracle"

    def certify(self, env, grid):
        _, V = solve_brt(env, env.true_dynamics, None, grid)
        return CertifierOutput(V, {"oracle": True})


@dataclass
class FunctionCertifier:
    """Wrap ``fn(env, grid) -> V_hat`` (numpy array on grid) as a Certifier."""

    name: str
    fn: Callable
    info: dict = field(default_factory=dict)

    def certify(self, env, grid):
        return CertifierOutput(np.asarray(self.fn(env, grid), dtype=float), dict(self.info))
