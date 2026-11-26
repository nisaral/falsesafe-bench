"""Document the current conformal-quantile behavior against the textbook rule.

Behavior is intentionally NOT changed here; these tests pin the existing
implementation so future edits cannot silently shift it.
"""
from __future__ import annotations

import numpy as np

from lcert.certifiers.conformal import compute_conformal_quantile


def _textbook_order_statistic(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float("inf") if rank > n else float(s[rank - 1])


def test_matches_textbook_when_rank_within_n():
    # n=10, alpha=0.1 -> ceil(11*0.9)=10 <= n, so both agree on the 10th (max) score.
    scores = np.arange(1, 11, dtype=float)
    assert compute_conformal_quantile(scores, 0.1) == _textbook_order_statistic(scores, 0.1) == 10.0

    # n=9, alpha=0.1 -> ceil(10*0.9)=9 <= n.
    scores = np.arange(1, 10, dtype=float)
    assert compute_conformal_quantile(scores, 0.1) == _textbook_order_statistic(scores, 0.1) == 9.0


def test_differs_from_textbook_when_rank_exceeds_n():
    # n=3, alpha=0.1 -> ceil(4*0.9)=4 > n: textbook says inf, implementation clamps to max.
    scores = np.array([0.1, 0.2, 0.3])
    assert _textbook_order_statistic(scores, 0.1) == float("inf")
    assert compute_conformal_quantile(scores, 0.1) == 0.3


def test_empty_scores_returns_zero():
    assert compute_conformal_quantile(np.array([]), 0.1) == 0.0
