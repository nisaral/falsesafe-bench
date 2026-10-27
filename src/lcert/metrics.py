"""Safety certification metrics and statistical aggregation.

Implements the fixed benchmark metric definitions from SCOPE.md Section 3:
- Evaluation domain G: interior cells (border cells excluded)
- Safe_true = {V_true >= 0}, Safe_hat = {V_hat >= 0}
- FSV (false-safe volume) = |Safe_hat AND NOT Safe_true| / |G|
- FUV (false-unsafe volume) = |Safe_true AND NOT Safe_hat| / |G|
- IoU = |Safe_hat AND Safe_true| / |Safe_hat OR Safe_true|
- Gap-FSV = FSV restricted to coverage-gap region, divided by gap volume
- 95% Student's t-interval over seeds (n >= 5 for reported claims)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np
import scipy.stats


@dataclass(frozen=True)
class ConfidenceInterval:
    """Statistical summary with Student's t confidence interval."""
    mean: float
    ci_lower: float
    ci_upper: float
    half_width: float
    std: float
    sem: float
    n: int
    confidence: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self, precision: int = 4) -> str:
        return f"{self.mean:.{precision}f} [{self.ci_lower:.{precision}f}, {self.ci_upper:.{precision}f}] (n={self.n})"


@dataclass(frozen=True)
class CertificationMetrics:
    """Exact certification metrics on a discrete grid domain."""
    fsv: float
    fuv: float
    iou: float
    safe_hat_volume: float
    safe_true_volume: float
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    num_eval_cells: int
    gap_fsv: float | None = None
    gap_fuv: float | None = None
    non_gap_fsv: float | None = None
    num_gap_cells: int | None = None
    num_non_gap_cells: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def interior_mask(
    shape: tuple[int, ...],
    border_cells: int | Sequence[int] = 3,
    periodic_dims: Sequence[int] = (),
) -> np.ndarray:
    """Compute boolean mask of interior grid cells excluding boundary cells.

    Args:
        shape: Grid dimensions (e.g. (101, 101) or (101, 101, 101)).
        border_cells: Number of cells to exclude from each boundary.
            Can be an int (same for all dims) or sequence of ints per dim.
        periodic_dims: Dimensions where boundaries wrap periodically
            (boundary exclusion is skipped for these dimensions).

    Returns:
        Boolean ndarray with True for interior evaluation cells.
    """
    dims = len(shape)
    if dims == 0:
        raise ValueError("Grid shape must have at least 1 dimension")

    if isinstance(border_cells, int):
        borders = [border_cells] * dims
    else:
        borders = list(border_cells)
        if len(borders) != dims:
            raise ValueError(f"border_cells length {len(borders)} must match dimensions {dims}")

    for ax, b in enumerate(borders):
        if b < 0:
            raise ValueError(f"border_cells must be nonnegative, got {b} on axis {ax}")
        if ax in periodic_dims:
            borders[ax] = 0

    for ax, (size, b) in enumerate(zip(shape, borders, strict=True)):
        if b > 0 and size <= 2 * b:
            raise ValueError(
                f"Grid size {size} along axis {ax} is too small for border exclusion of {b} cells "
                f"(requires size > {2 * b})"
            )

    mask = np.zeros(shape, dtype=bool)
    slices = tuple(
        slice(b, -b if b > 0 else None)
        for b in borders
    )
    mask[slices] = True
    return mask


def coverage_gap_mask(
    states: np.ndarray,
    gap_lo: np.ndarray | Sequence[float],
    gap_hi: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Compute boolean mask for states inside a hyper-rectangular coverage gap [gap_lo, gap_hi].

    Args:
        states: Grid coordinates with shape (*grid_shape, state_dim).
        gap_lo: Lower bounds of coverage gap, shape (state_dim,).
        gap_hi: Upper bounds of coverage gap, shape (state_dim,).

    Returns:
        Boolean ndarray of shape grid_shape.
    """
    states = np.asarray(states, dtype=float)
    lo = np.asarray(gap_lo, dtype=float)
    hi = np.asarray(gap_hi, dtype=float)

    state_dim = states.shape[-1]
    if lo.shape != (state_dim,) or hi.shape != (state_dim,):
        raise ValueError(
            f"Coverage gap bounds shape ({lo.shape}, {hi.shape}) must match state dimension ({state_dim},)"
        )
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("Coverage gap bounds must be finite")
    if not np.all(lo < hi):
        raise ValueError(f"Coverage gap lower bounds must be strictly less than upper bounds: lo={lo}, hi={hi}")

    in_box = np.all((states >= lo) & (states <= hi), axis=-1)
    return in_box


def compute_certification_metrics(
    v_true: np.ndarray,
    v_hat: np.ndarray,
    *,
    border_cells: int | Sequence[int] = 3,
    periodic_dims: Sequence[int] = (),
    eval_mask: np.ndarray | None = None,
    states: np.ndarray | None = None,
    coverage_gap: dict[str, Any] | None = None,
) -> CertificationMetrics:
    """Compute formal safety certification metrics between ground-truth and learned BRT values.

    Args:
        v_true: Ground-truth value function on grid.
        v_hat: Learned-model value function on identical grid.
        border_cells: Margin to exclude from boundary (default 3 cells per SCOPE.md).
        periodic_dims: Periodic grid dimensions where boundary exclusion is skipped.
        eval_mask: Optional precomputed boolean evaluation mask. If None, interior_mask() is used.
        states: Optional grid state coordinates, shape (*grid_shape, state_dim), required for gap metrics.
        coverage_gap: Optional dict with keys 'enabled' (bool), 'lo' (array-like), 'hi' (array-like).

    Returns:
        CertificationMetrics with FSV, FUV, IoU, Gap-FSV, errors, and cell counts.
    """
    v_true = np.asarray(v_true, dtype=float)
    v_hat = np.asarray(v_hat, dtype=float)

    if v_true.shape != v_hat.shape:
        raise ValueError(f"Shape mismatch: v_true {v_true.shape} != v_hat {v_hat.shape}")
    if not (np.isfinite(v_true).all() and np.isfinite(v_hat).all()):
        raise FloatingPointError("Value functions contain non-finite (NaN or Inf) values")

    if eval_mask is not None:
        g_mask = np.asarray(eval_mask, dtype=bool)
        if g_mask.shape != v_true.shape:
            raise ValueError(f"eval_mask shape {g_mask.shape} does not match value shape {v_true.shape}")
    else:
        g_mask = interior_mask(v_true.shape, border_cells=border_cells, periodic_dims=periodic_dims)

    num_eval = int(np.sum(g_mask))
    if num_eval == 0:
        raise ValueError("Evaluation domain G is empty; check grid size and border exclusion")

    # Safe sets on evaluation domain G
    safe_true = (v_true >= 0.0) & g_mask
    safe_hat = (v_hat >= 0.0) & g_mask

    # False-safe: model says safe (v_hat >= 0) but ground truth is unsafe (v_true < 0)
    false_safe_cells = safe_hat & (~safe_true) & g_mask
    fsv = float(np.sum(false_safe_cells) / num_eval)

    # False-unsafe: model says unsafe (v_hat < 0) but ground truth is safe (v_true >= 0)
    false_unsafe_cells = safe_true & (~safe_hat) & g_mask
    fuv = float(np.sum(false_unsafe_cells) / num_eval)

    # Intersection over Union
    intersection = int(np.sum(safe_hat & safe_true))
    union = int(np.sum(safe_hat | safe_true))
    if union == 0:
        # Both sets empty in evaluation domain
        iou = 1.0
    else:
        iou = float(intersection / union)

    safe_hat_vol = float(np.sum(safe_hat) / num_eval)
    safe_true_vol = float(np.sum(safe_true) / num_eval)

    # Error statistics on G
    v_diff = np.abs(v_hat[g_mask] - v_true[g_mask])
    max_abs_err = float(np.max(v_diff))
    mean_abs_err = float(np.mean(v_diff))
    rmse = float(np.sqrt(np.mean(v_diff ** 2)))

    # Coverage gap metrics if enabled and provided
    gap_fsv = None
    gap_fuv = None
    non_gap_fsv = None
    num_gap_cells = None
    num_non_gap_cells = None

    if coverage_gap and coverage_gap.get("enabled", False):
        if states is None:
            raise ValueError("states array is required to compute coverage gap metrics")
        gap_lo = np.asarray(coverage_gap["lo"], dtype=float)
        gap_hi = np.asarray(coverage_gap["hi"], dtype=float)
        raw_gap = coverage_gap_mask(states, gap_lo, gap_hi)

        gap_eval_mask = raw_gap & g_mask
        non_gap_eval_mask = (~raw_gap) & g_mask

        num_gap_cells = int(np.sum(gap_eval_mask))
        num_non_gap_cells = int(np.sum(non_gap_eval_mask))

        if num_gap_cells > 0:
            gap_fsv = float(np.sum(false_safe_cells & gap_eval_mask) / num_gap_cells)
            gap_fuv = float(np.sum(false_unsafe_cells & gap_eval_mask) / num_gap_cells)
        else:
            gap_fsv = 0.0
            gap_fuv = 0.0

        if num_non_gap_cells > 0:
            non_gap_fsv = float(np.sum(false_safe_cells & non_gap_eval_mask) / num_non_gap_cells)
        else:
            non_gap_fsv = 0.0

    return CertificationMetrics(
        fsv=fsv,
        fuv=fuv,
        iou=iou,
        safe_hat_volume=safe_hat_vol,
        safe_true_volume=safe_true_vol,
        max_abs_error=max_abs_err,
        mean_abs_error=mean_abs_err,
        rmse=rmse,
        num_eval_cells=num_eval,
        gap_fsv=gap_fsv,
        gap_fuv=gap_fuv,
        non_gap_fsv=non_gap_fsv,
        num_gap_cells=num_gap_cells,
        num_non_gap_cells=num_non_gap_cells,
    )


def compute_confidence_interval(
    samples: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
    min_samples: int = 5,
    enforce_min_samples: bool = False,
    clip_bounds: tuple[float | None, float | None] | None = None,
) -> ConfidenceInterval:
    """Compute Student's t confidence interval over independent seed runs.

    Per SCOPE.md Section 3: "CI: 95% t-interval over seeds, n >= 5 seeds."
    Optionally clips to physical domain bounds (e.g. [0, 1] for volume fractions).

    Args:
        samples: Array of numerical metric values across runs.
        confidence: Confidence level (default 0.95 for 95% CI).
        min_samples: Minimum required sample size (default 5).
        enforce_min_samples: If True, raises ValueError when len(samples) < min_samples.
        clip_bounds: Optional (min_val, max_val) to clip confidence interval bounds.

    Returns:
        ConfidenceInterval with mean, lower, upper bounds, half_width, std, and n.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"samples must be 1D, got ndim={arr.ndim}")
    if not np.isfinite(arr).all():
        raise ValueError("samples contains NaN or Inf values")

    n = len(arr)
    if n < 2:
        raise ValueError(f"At least 2 samples required to compute confidence interval, got {n}")

    if enforce_min_samples and n < min_samples:
        raise ValueError(
            f"Per SCOPE.md, reported results require n >= {min_samples} seeds; got n={n}"
        )

    mean = float(np.mean(arr))
    sample_std = float(np.std(arr, ddof=1))

    if math.isclose(sample_std, 0.0, abs_tol=1e-15):
        ci_lo, ci_hi = mean, mean
        half_width = 0.0
        sem = 0.0
    else:
        sem = sample_std / math.sqrt(n)
        t_crit = float(scipy.stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
        half_width = t_crit * sem
        ci_lo = mean - half_width
        ci_hi = mean + half_width

    if clip_bounds is not None:
        if clip_bounds[0] is not None:
            ci_lo = max(clip_bounds[0], ci_lo)
        if clip_bounds[1] is not None:
            ci_hi = min(clip_bounds[1], ci_hi)

    return ConfidenceInterval(
        mean=mean,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        half_width=half_width,
        std=sample_std,
        sem=sem,
        n=n,
        confidence=confidence,
    )


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Non-parametric paired bootstrap summary for comparing matched seed runs."""
    mean_baseline: float
    mean_treatment: float
    mean_reduction_abs: float
    ci_reduction_abs: tuple[float, float]
    mean_reduction_pct: float
    ci_reduction_pct: tuple[float, float]
    target_reduction_pct: float
    p_value_meets_target: float
    n: int
    num_resamples: int
    confidence: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self, precision: int = 1) -> str:
        return (
            f"Reduction: {self.mean_reduction_pct:.{precision}f}% "
            f"[{self.ci_reduction_pct[0]:.{precision}f}%, {self.ci_reduction_pct[1]:.{precision}f}%] "
            f"(target >={self.target_reduction_pct:.0f}%, p={self.p_value_meets_target:.4f}, n={self.n})"
        )


def compute_paired_bootstrap_reduction(
    baseline_samples: Sequence[float] | np.ndarray,
    treatment_samples: Sequence[float] | np.ndarray,
    *,
    target_reduction_pct: float = 50.0,
    confidence: float = 0.95,
    num_resamples: int = 10000,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Compute BCa bootstrap confidence interval for ratio-of-means metric reduction.

    Per PREREGISTRATION.md Section 4.1:
        Estimand: R = (mean(FSV_base) - mean(FSV_treat)) / mean(FSV_base) * 100%
    Resampled over matched seeds using BCa (bias-corrected and accelerated) bootstrap.

    Args:
        baseline_samples: Metric values for nominal/baseline model per seed.
        treatment_samples: Metric values for treatment (e.g. k=2.0) on the identical seeds.
        target_reduction_pct: Hypothesis benchmark threshold (e.g. 50.0% for P1/H2).
        confidence: Confidence level (default 0.95).
        num_resamples: Number of bootstrap iterations (default 10,000).
        seed: Random seed for bootstrap reproducibility.

    Returns:
        PairedBootstrapResult with BCa confidence intervals on ratio of means.
    """
    base = np.asarray(baseline_samples, dtype=float)
    treat = np.asarray(treatment_samples, dtype=float)

    if base.shape != treat.shape or base.ndim != 1:
        raise ValueError(f"baseline and treatment must be 1D arrays of identical length, got {base.shape} and {treat.shape}")
    if not (np.isfinite(base).all() and np.isfinite(treat).all()):
        raise ValueError("Samples contain NaN or Inf values")

    n = len(base)
    if n < 2:
        raise ValueError(f"At least 2 matched pairs required, got n={n}")

    mean_base = float(np.mean(base))
    mean_treat = float(np.mean(treat))
    mean_diff_abs = mean_base - mean_treat

    # Point estimate of ratio of means (percentage reduction)
    if mean_base > 0:
        theta_hat = float((mean_diff_abs / mean_base) * 100.0)
    else:
        theta_hat = 0.0

    # Bootstrap Resampling over matched seeds
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(num_resamples, n))

    boot_base = np.mean(base[boot_indices], axis=1)
    boot_treat = np.mean(treat[boot_indices], axis=1)
    boot_diff_abs = boot_base - boot_treat

    with np.errstate(divide="ignore", invalid="ignore"):
        boot_theta = np.where(boot_base > 0, (boot_diff_abs / boot_base) * 100.0, 0.0)

    # 1. Bias correction z0
    prop_less = float(np.mean(boot_theta < theta_hat))
    prop_less = max(1e-6, min(1.0 - 1e-6, prop_less))
    z0 = float(scipy.stats.norm.ppf(prop_less))

    # 2. Acceleration a via Jackknife
    jack_thetas = np.zeros(n)
    for i in range(n):
        jk_base = np.delete(base, i)
        jk_treat = np.delete(treat, i)
        m_b = np.mean(jk_base)
        m_t = np.mean(jk_treat)
        jack_thetas[i] = ((m_b - m_t) / m_b) * 100.0 if m_b > 0 else 0.0

    jack_mean = np.mean(jack_thetas)
    u_i = jack_mean - jack_thetas
    sum_cubes = np.sum(u_i ** 3)
    sum_squares = np.sum(u_i ** 2)

    if sum_squares > 1e-12:
        a = float(sum_cubes / (6.0 * (sum_squares ** 1.5)))
    else:
        a = 0.0

    # 3. BCa percentile endpoints
    alpha = 1.0 - confidence
    z_lo = scipy.stats.norm.ppf(alpha / 2.0)
    z_hi = scipy.stats.norm.ppf(1.0 - alpha / 2.0)

    denom_lo = 1.0 - a * (z0 + z_lo)
    denom_hi = 1.0 - a * (z0 + z_hi)

    a1 = scipy.stats.norm.cdf(z0 + (z0 + z_lo) / (denom_lo if abs(denom_lo) > 1e-8 else 1.0))
    a2 = scipy.stats.norm.cdf(z0 + (z0 + z_hi) / (denom_hi if abs(denom_hi) > 1e-8 else 1.0))

    a1 = max(0.0001, min(0.9999, float(a1)))
    a2 = max(0.0001, min(0.9999, float(a2)))

    ci_rel_lo = float(np.percentile(boot_theta, 100.0 * min(a1, a2)))
    ci_rel_hi = float(np.percentile(boot_theta, 100.0 * max(a1, a2)))

    # Absolute difference intervals
    ci_abs_lo = float(np.percentile(boot_diff_abs, 100.0 * (alpha / 2.0)))
    ci_abs_hi = float(np.percentile(boot_diff_abs, 100.0 * (1.0 - alpha / 2.0)))

    # Empirical p-value: fraction of bootstrap replicates below target reduction threshold
    p_fail = float(np.mean(boot_theta < target_reduction_pct))

    return PairedBootstrapResult(
        mean_baseline=mean_base,
        mean_treatment=mean_treat,
        mean_reduction_abs=mean_diff_abs,
        ci_reduction_abs=(ci_abs_lo, ci_abs_hi),
        mean_reduction_pct=theta_hat,
        ci_reduction_pct=(ci_rel_lo, ci_rel_hi),
        target_reduction_pct=target_reduction_pct,
        p_value_meets_target=1.0 - p_fail,
        n=n,
        num_resamples=num_resamples,
        confidence=confidence,
    )


def compute_asymmetric_safety_cost(
    fsv: float | Sequence[float] | np.ndarray,
    fuv: float | Sequence[float] | np.ndarray,
    cost_false_safe: float = 10.0,
    cost_false_unsafe: float = 1.0,
) -> float | np.ndarray:
    """Compute formal asymmetric safety cost C = c_FS * FSV + c_FU * FUV.

    In safety-critical robotics, a False-Safe (catastrophic physical collision)
    is orders of magnitude more severe than a False-Unsafe (unnecessary braking/conservatism).

    Args:
        fsv: False-safe volume (catastrophic crash risk).
        fuv: False-unsafe volume (conservatism / efficiency loss).
        cost_false_safe: Penalty weight per unit FSV (default 10.0).
        cost_false_unsafe: Penalty weight per unit FUV (default 1.0).

    Returns:
        Scalar or array of composite safety cost values.
    """
    fsv_arr = np.asarray(fsv, dtype=float)
    fuv_arr = np.asarray(fuv, dtype=float)
    return float(cost_false_safe * fsv_arr + cost_false_unsafe * fuv_arr)


def aggregate_metric_runs(
    metrics_list: Sequence[CertificationMetrics],
    confidence: float = 0.95,
    min_samples: int = 5,
    enforce_min_samples: bool = False,
) -> dict[str, ConfidenceInterval]:
    """Aggregate a list of CertificationMetrics across seeds into mean and 95% CIs.

    Args:
        metrics_list: List of CertificationMetrics from different random seeds.
        confidence: Confidence level (default 0.95).
        min_samples: Minimum seeds required.
        enforce_min_samples: Whether to strictly enforce min_samples.

    Returns:
        Dict mapping metric name to ConfidenceInterval.
    """
    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")

    volume_keys = {"fsv", "fuv", "iou", "safe_hat_volume", "safe_true_volume", "gap_fsv", "gap_fuv", "non_gap_fsv"}
    keys = ["fsv", "fuv", "iou", "safe_hat_volume", "safe_true_volume", "rmse", "max_abs_error"]
    if all(m.gap_fsv is not None for m in metrics_list):
        keys.extend(["gap_fsv", "gap_fuv", "non_gap_fsv"])

    result: dict[str, ConfidenceInterval] = {}
    for key in keys:
        values = [getattr(m, key) for m in metrics_list]
        clip = (0.0, 1.0) if key in volume_keys else None
        result[key] = compute_confidence_interval(
            values,
            confidence=confidence,
            min_samples=min_samples,
            enforce_min_samples=enforce_min_samples,
            clip_bounds=clip,
        )
    return result

