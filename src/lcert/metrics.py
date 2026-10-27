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
) -> ConfidenceInterval:
    """Compute Student's t confidence interval over independent seed runs.

    Per SCOPE.md Section 3: "CI: 95% t-interval over seeds, n >= 5 seeds."

    Args:
        samples: Array of numerical metric values across runs.
        confidence: Confidence level (default 0.95 for 95% CI).
        min_samples: Minimum required sample size (default 5).
        enforce_min_samples: If True, raises ValueError when len(samples) < min_samples.

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
        return ConfidenceInterval(
            mean=mean,
            ci_lower=mean,
            ci_upper=mean,
            half_width=0.0,
            std=0.0,
            sem=0.0,
            n=n,
            confidence=confidence,
        )

    sem = sample_std / math.sqrt(n)
    t_crit = float(scipy.stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    half_width = t_crit * sem

    return ConfidenceInterval(
        mean=mean,
        ci_lower=mean - half_width,
        ci_upper=mean + half_width,
        half_width=half_width,
        std=sample_std,
        sem=sem,
        n=n,
        confidence=confidence,
    )


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

    keys = ["fsv", "fuv", "iou", "safe_hat_volume", "safe_true_volume", "rmse", "max_abs_error"]
    # Check if optional gap metrics exist across runs
    if all(m.gap_fsv is not None for m in metrics_list):
        keys.extend(["gap_fsv", "gap_fuv", "non_gap_fsv"])

    result: dict[str, ConfidenceInterval] = {}
    for key in keys:
        values = [getattr(m, key) for m in metrics_list]
        result[key] = compute_confidence_interval(
            values,
            confidence=confidence,
            min_samples=min_samples,
            enforce_min_samples=enforce_min_samples,
        )
    return result
