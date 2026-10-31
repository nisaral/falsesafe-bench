"""falsesafe framework: plug-in dynamics / hazards / policies / certifiers evaluated
against Hamilton-Jacobi ground truth with the repo's FSV/FUV/gap-FSV metrics.

Quickstart::

    from lcert.framework import ExperimentSpec, run_experiment
    run_experiment(ExperimentSpec(name="di_demo", sweep={"certifier.k": [0, 1, 2]}),
                   out_dir="demos/viewer3d/data")

See docs/FRAMEWORK.md.
"""
from .core import CertifierOutput, Environment, FunctionPolicy, GapRegion, Hazard, Policy, Certifier
from .registry import register_certifier, register_env, register_policy, ENVS, POLICIES, CERTIFIERS
from .solver import CallableDynamics, ground_truth, make_grid, solve_brt
from . import envs, certifiers, bound_certifiers, policies  # noqa: F401  (populate registries)
from .bound_certifiers import ConformalBoundCertifier, DistanceAwareBoundCertifier, grid_lookup
from .certifiers import EnsembleCertifier, FunctionCertifier, NominalInflatedCertifier, OracleCertifier
from .policies import GridValue, SafetyFilter, simulate
from .runner import ExperimentSpec, export, rebuild_bundle, run_case, run_experiment

__all__ = [n for n in dir() if not n.startswith("_")]
