"""Tests for the extracted P1 aggregation helper in scripts/run_pipeline.py."""
from __future__ import annotations

import json

from scripts.run_pipeline import aggregate_p1


def test_aggregate_p1_returns_expected_fields():
    res = aggregate_p1(
        [0.010, 0.012, 0.011],
        [0.006, 0.007, 0.0065],
        [0.0010, 0.0012, 0.0011],
        [0.0020, 0.0021, 0.0022],
    )
    assert set(res) == {"reduction", "cost", "decision"}
    assert "mean_reduction_pct" in res["reduction"]
    assert "ci_reduction_pct" in res["reduction"]
    assert "delta_fuv_mean" in res["cost"]
    assert "break_even_ratio_r_star" in res["cost"]
    assert res["decision"] in {"SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED"}


def test_aggregate_p1_reduction_positive_when_k2_lower():
    res = aggregate_p1(
        [0.010, 0.012, 0.011],
        [0.006, 0.007, 0.0065],
        [0.0010, 0.0012, 0.0011],
        [0.0020, 0.0021, 0.0022],
    )
    assert res["reduction"]["mean_reduction_pct"] > 0.0
    assert res["cost"]["delta_fsv_mean"] > 0.0


def test_aggregate_p1_result_is_json_serializable():
    res = aggregate_p1(
        [0.010, 0.012, 0.011],
        [0.006, 0.007, 0.0065],
        [0.0010, 0.0012, 0.0011],
        [0.0020, 0.0021, 0.0022],
    )
    assert isinstance(json.dumps(res), str)
