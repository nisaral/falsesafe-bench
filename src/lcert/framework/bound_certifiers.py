"""Conformal (C0, C1b, C2) and distance-aware Lipschitz (L) certifiers for the framework.

These adapt ``lcert.certifiers.conformal`` / ``lcert.certifiers.distance_aware`` (NumPy,
scipy cKDTree) to the framework ``Certifier`` interface.

Pipeline (all variants)
-----------------------
1. Learned model f_hat = ensemble mean, trained exactly like the framework ``ensemble``
   adapter (i.i.d. uniform (x, u) samples outside ``env.gap``; shared cache, so E, C*, L
   compare against the *same* model for the same seed).
2. A per-(x, u) disturbance bound d(x, u) >= 0 on the state-derivative dims ``dims``:

   * C0  : d = q_hat, split-conformal (1-alpha) quantile of |y - f_hat| (inf-norm over
           ``dims``) on a calibration set drawn i.i.d. from the training distribution
           (outside the gap, independent seed).
   * C1b : d(x,u) = q_hat * max(dist((x,u), D_train), 1e-4) with q_hat the conformal
           quantile of |y - f_hat| / max(dist, 1e-4) (distance-normalised scores).
   * C2  : d = q_hat with calibration restricted to a held-out state box
           ``calib_region`` (leave-region-out; must be disjoint from the gap).
   * L   : d(x,u) = eps0 + lam * L_hat * dist((x,u), D_train), eps0 = max training
           residual, L_hat = max pairwise slope |y_i - y_j| / ||z_i - z_j|| on a seeded
           subsample of training pairs (same recipe as lcert.certifiers.distance_aware).

   dist is Euclidean in the joint (x, u) space with every axis divided by its range
   (state box ranges, control range), nearest neighbour via scipy cKDTree.
3. JAX conversion (the k-d tree cannot run inside the JAX HJ solver): d is evaluated
   *offline* on every HJ grid node for every candidate control in ``env.controls`` and
   reduced with max over u -> D[x] (conservative: the disturbance cannot depend on the
   control the Hamiltonian picks).  Inside the dynamics D is read with multilinear
   interpolation (``jax.scipy.ndimage.map_coordinates``, order=1, mode='nearest').
   The solver only queries grid nodes, so the lookup is exact there; the max over the
   finite control set (not the continuous box) is the only approximation.
4. Robust HJ with f_hat and disturbance D[x] on ``dims``: H = max_u p.f_hat - sum |p_i| D.

Differences from the repo's S2 scripts (``scripts/run_s2_benchmark_all6.py``): those use
trajectory-generated data and feed a *scalar* (q_hat, or mean of d over the grid) into
``solve_learned_di_brt``; here C1b and L use the state-dependent bound, and data are i.i.d.
Framework numbers are therefore exploratory and not comparable 1:1.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from lcert.certifiers.conformal import compute_conformal_quantile

from .certifiers import EnsembleCertifier
from .core import CertifierOutput, Environment
from .registry import register_certifier
from .solver import solve_brt


def grid_lookup(env: Environment, grid, values: np.ndarray):
    """Return jnp fn x -> multilinear interpolation of ``values`` (on grid nodes) at x."""
    V = jnp.asarray(values)
    lo = jnp.asarray([float(np.asarray(c)[0]) for c in grid.coordinate_vectors])
    sp = jnp.asarray([float(np.asarray(c)[1] - np.asarray(c)[0]) for c in grid.coordinate_vectors])

    def fn(x):
        idx = (jnp.asarray(x) - lo) / sp
        return jax.scipy.ndimage.map_coordinates(V, [idx[i] for i in range(V.ndim)], order=1, mode="nearest")
    return fn


def _sample(env, n, rng, box=None, exclude_gap=True):
    lo, hi = (np.asarray(env.lo, float), np.asarray(env.hi, float)) if box is None else \
        (np.asarray(box["lo"], float), np.asarray(box["hi"], float))
    xs, tries = [], 0
    while sum(len(a) for a in xs) < n:
        tries += 1
        if tries > 1000:
            raise RuntimeError("could not sample calibration states (region empty after gap exclusion?)")
        c = rng.uniform(lo, hi, size=(max(n, 64), env.dim))
        if exclude_gap and env.gap is not None:
            c = c[~env.gap.contains(c)]
        xs.append(c)
    X = np.concatenate(xs)[:n]
    U = rng.uniform(env.controls.min(0), env.controls.max(0), size=(n, env.controls.shape[1]))
    return X, U


@dataclass
class _BoundCertifierBase(EnsembleCertifier):
    """Shared model/data/lookup machinery; subclasses define ``_bound``."""

    dims: tuple | None = None          # derivative dims that get the disturbance (default all)
    n_calib: int = 500
    calib_seed_offset: int = 1000
    name: str = "bound_base"

    # ---- helpers -------------------------------------------------------------------
    def _mask(self, env):
        m = np.zeros(env.dim) if self.dims is not None else np.ones(env.dim)
        if self.dims is not None:
            m[list(self.dims)] = 1.0
        return m

    def _scale(self, env):
        cr = env.controls.max(0) - env.controls.min(0)
        return np.concatenate([np.asarray(env.hi, float) - np.asarray(env.lo, float), np.where(cr > 0, cr, 1.0)])

    def _setup(self, env):
        from lcert.hj_learned import LearnedDoubleIntegratorDynamics, extract_ensemble_jax_params
        if len(self.hidden_sizes) != 2:
            raise ValueError("lcert.hj_learned's JAX evaluator supports exactly 2 hidden layers")
        if env.controls.shape[1] != 1:
            raise ValueError("learned certifiers currently support scalar controls (lcert.hj_learned evaluator)")
        model, diag, (X, U, Y) = self._train_with_data(env)
        ev = LearnedDoubleIntegratorDynamics(*extract_ensemble_jax_params(model), umax=1.0)
        f_mean = lambda x, u: ev._eval_ensemble(x, u[0])[0]  # noqa: E731
        pred = jax.jit(jax.vmap(lambda x, u: f_mean(x, u)))
        mask = self._mask(env).astype(bool)
        scale = self._scale(env)
        tree = cKDTree(np.column_stack([X, U]) / scale)
        self._ctx = dict(model=model, diag=diag, X=X, U=U, Y=Y, f_mean=f_mean, pred=pred, mask=mask,
                         scale=scale, tree=tree)
        return self._ctx

    def _resid(self, X, U, Y):
        c = self._ctx
        P = np.asarray(c["pred"](jnp.asarray(X, jnp.float32), jnp.asarray(U, jnp.float32)))
        return np.max(np.abs(Y - P)[:, c["mask"]], axis=1)

    def _dist(self, X, U):
        return self._ctx["tree"].query(np.column_stack([X, U]) / self._ctx["scale"], k=1)[0]

    def _truth(self, env, X, U, noise_rng):
        f = jax.jit(jax.vmap(lambda x, u: jnp.asarray(env.true_dynamics(x, u))))
        return np.asarray(f(X, U)) + noise_rng.normal(0, self.noise_std, size=(len(X), env.dim))

    def _bound(self, env, X, U) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- Certifier -----------------------------------------------------------------
    def certify(self, env: Environment, grid):
        c = self._setup(env)
        info = self._fit(env)
        S = np.asarray(grid.states).reshape(-1, env.dim)
        D = np.zeros(len(S))
        for u in env.controls:                      # max over candidate controls
            D = np.maximum(D, self._bound(env, S, np.broadcast_to(u, (len(S), env.controls.shape[1]))))
        D = D.reshape(env.shape)
        look = grid_lookup(env, grid, D)
        m = jnp.asarray(self._mask(env))
        _, V = solve_brt(env, c["f_mean"], lambda x: m * look(x), grid)
        info.update({"learned": True, "dims": list(np.flatnonzero(c["mask"]).tolist()),
                     "bound_grid_stats": {"min": float(D.min()), "mean": float(D.mean()), "max": float(D.max())},
                     "bound_in_gap_mean": (float(D.reshape(-1)[env.gap.contains(S)].mean()) if env.gap is not None
                                           and env.gap.contains(S).any() else None),
                     "jax_conversion": "bound precomputed on HJ grid (max over env.controls), multilinear lookup",
                     "training": c["diag"], "data": "iid uniform samples outside gap (framework adapter)"})
        self._bound_grid = D
        return CertifierOutput(V, info)


@register_certifier("conformal")
@dataclass
class ConformalBoundCertifier(_BoundCertifierBase):
    """Split-conformal disturbance bound: variant C0 | C1b | C2 (see module docstring)."""

    variant: str = "C0"
    alpha: float = 0.1
    calib_region: dict | None = None   # C2 only: {"lo": [...], "hi": [...]} state box disjoint from gap
    name: str = "conformal"

    def __post_init__(self):
        if self.variant not in ("C0", "C1b", "C2"):
            raise ValueError("framework conformal variants: C0, C1b, C2 (C1a needs ensemble-std scores; "
                             "use the 'ensemble' certifier for epistemic-std bounds)")
        self.name = f"conformal_{self.variant}"

    def _fit(self, env):
        rng = np.random.default_rng(self.seed + self.calib_seed_offset)
        if self.variant == "C2":
            if self.calib_region is None:
                raise ValueError("C2 needs calib_region={'lo':..., 'hi':...} (held-out state box)")
            if env.gap is not None:
                glo, ghi = np.asarray(env.gap.lo), np.asarray(env.gap.hi)
                clo, chi = np.asarray(self.calib_region["lo"]), np.asarray(self.calib_region["hi"])
                if np.all((clo <= ghi) & (chi >= glo)):
                    raise ValueError("C2 calib_region must be disjoint from env.gap")
            X, U = _sample(env, self.n_calib, rng, box=self.calib_region)
        else:
            X, U = _sample(env, self.n_calib, rng)
        Y = self._truth(env, X, U, rng)
        r = self._resid(X, U, Y)
        scores = r / np.maximum(self._dist(X, U), 1e-4) if self.variant == "C1b" else r
        self.q_hat = compute_conformal_quantile(scores, self.alpha)
        return {"variant": self.variant, "alpha": self.alpha, "q_hat": self.q_hat, "n_calib": len(X),
                "calib_region": self.calib_region if self.variant == "C2" else "iid outside gap",
                "calib_residual_median": float(np.median(r))}

    def _bound(self, env, X, U):
        if self.variant == "C1b":
            return self.q_hat * np.maximum(self._dist(X, U), 1e-4)
        return np.full(len(X), self.q_hat)


@register_certifier("distance_aware")
@dataclass
class DistanceAwareBoundCertifier(_BoundCertifierBase):
    """Certifier L: d(x,u) = eps0 + lam * L_hat * dist((x,u), D_train)."""

    lam: float = 1.0
    subsample_pairs: int = 500
    name: str = "distance_aware"

    def _fit(self, env):
        c = self._ctx
        X, U, Y = c["X"], c["U"], c["Y"]
        self.eps0 = float(np.max(self._resid(X, U, Y)))
        Z = np.column_stack([X, U]) / c["scale"]
        n = len(Z)
        if n <= self.subsample_pairs:
            i, j = np.triu_indices(n, k=1)
        else:
            r = np.random.default_rng(20261027)  # same seed as lcert.certifiers.distance_aware
            i, j = r.integers(0, n, self.subsample_pairs * 10), r.integers(0, n, self.subsample_pairs * 10)
            i, j = i[i != j], j[i != j]
        dz = np.linalg.norm(Z[i] - Z[j], axis=-1)
        dy = np.max(np.abs(Y[i] - Y[j])[:, c["mask"]], axis=1)
        ok = dz > 1e-5
        self.l_hat = max(float(np.max(dy[ok] / dz[ok])) if ok.any() else 1.0, 0.1)
        return {"lam": self.lam, "eps0": self.eps0, "L_hat": self.l_hat}

    def _bound(self, env, X, U):
        return self.eps0 + self.lam * self.l_hat * self._dist(X, U)
