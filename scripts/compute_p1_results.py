"""Compute Primary Confirmatory P1 Estimands on n=30 Confirmatory Seeds (seeds 1000-1029).

Pre-registered Decision Rule:
- Estimand: R = (mean(FSV_k0) - mean(FSV_k2)) / mean(FSV_k0) * 100%
- Interval: 95% BCa Bootstrap, 10,000 resamples (paired), seed 20261027.
- Supported: R >= 50% and lower_bound > 0.
- Partially Supported: lower_bound > 0 but R < 50%.
- Not Supported: lower_bound <= 0.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from lcert.metrics import (
    compute_paired_bootstrap_reduction,
    compute_asymmetric_safety_cost,
    compute_confidence_interval,
)

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "results" / "sweeps" / "20261027_184119"


def evaluate_confirmatory_p1():
    seed_dirs = sorted(list(SWEEP_DIR.glob("gap_N100_seed*")))
    assert len(seed_dirs) == 30, f"Expected 30 seed directories, found {len(seed_dirs)}"

    k_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    metrics_by_k: dict[float, dict[str, list[float]]] = {
        k: {"fsv": [], "fuv": [], "gap_fsv": [], "gap_fuv": []} for k in k_values
    }

    seeds = []
    for s_dir in seed_dirs:
        runlog_path = s_dir / "runlog.json"
        with open(runlog_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        seeds.append(log["seed"])
        for k in k_values:
            k_key = f"k_{k}"
            m = log["metrics"][k_key]
            metrics_by_k[k]["fsv"].append(m["fsv"])
            metrics_by_k[k]["fuv"].append(m["fuv"])
            metrics_by_k[k]["gap_fsv"].append(m["gap_fsv"])
            metrics_by_k[k]["gap_fuv"].append(m["gap_fuv"])

    print("=" * 75)
    print("=== FALSERAFE-BENCH: PRIMARY P1 CONFIRMATORY LEDGER AUDIT (n=30) ===")
    print(f"Directory: {SWEEP_DIR.name} | Confirmatory Seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)})")
    print("=" * 75)

    # 1. Evaluate P1 Primary Estimand at k=2.0 vs k=0.0
    fsv_k0 = metrics_by_k[0.0]["fsv"]
    fsv_k2 = metrics_by_k[2.0]["fsv"]
    fuv_k0 = metrics_by_k[0.0]["fuv"]
    fuv_k2 = metrics_by_k[2.0]["fuv"]

    red = compute_paired_bootstrap_reduction(
        fsv_k0, fsv_k2, confidence=0.95, num_resamples=10000, seed=20261027
    )
    delta_fsv = float(np.mean(fsv_k0) - np.mean(fsv_k2))
    delta_fuv = float(np.mean(fuv_k2) - np.mean(fuv_k0))
    r_star = float(delta_fuv / delta_fsv) if delta_fsv > 1e-7 else None

    cost_curve = {}
    for r in range(1, 51):
        cost_curve[str(r)] = float(r * np.mean(fsv_k2) + np.mean(fuv_k2))

    cost = {
        "delta_fsv_mean": delta_fsv,
        "delta_fuv_mean": delta_fuv,
        "break_even_ratio_r_star": round(r_star, 4) if r_star is not None else None,
        "cost_curve": cost_curve,
    }

    r_est = red.mean_reduction_pct
    r_lo = red.ci_reduction_pct[0]
    r_hi = red.ci_reduction_pct[1]

    # Decision rule from PREREGISTRATION.md Section 4.1
    if r_est >= 50.0 and r_lo > 0:
        decision = "SUPPORTED"
    elif r_lo > 0:
        decision = "PARTIALLY SUPPORTED"
    else:
        decision = "NOT SUPPORTED"

    mean_fsv_k0 = np.mean(fsv_k0)
    ci_fsv_k0 = compute_confidence_interval(fsv_k0, 0.95)
    mean_fsv_k2 = np.mean(fsv_k2)
    ci_fsv_k2 = compute_confidence_interval(fsv_k2, 0.95)

    mean_fuv_k0 = np.mean(fuv_k0)
    ci_fuv_k0 = compute_confidence_interval(fuv_k0, 0.95)
    mean_fuv_k2 = np.mean(fuv_k2)
    ci_fuv_k2 = compute_confidence_interval(fuv_k2, 0.95)

    mean_gap_fsv_k0 = np.mean(metrics_by_k[0.0]["gap_fsv"])
    mean_gap_fsv_k2 = np.mean(metrics_by_k[2.0]["gap_fsv"])

    print(f"\n[P1 PRIMARY FINDINGS]")
    print(f"Nominal FSV (k=0.0): mean = {mean_fsv_k0:.4f} [95% CI: {ci_fsv_k0.ci_lower:.4f}, {ci_fsv_k0.ci_upper:.4f}]")
    print(f"Inflated FSV (k=2.0): mean = {mean_fsv_k2:.4f} [95% CI: {ci_fsv_k2.ci_lower:.4f}, {ci_fsv_k2.ci_upper:.4f}]")
    print(f"Nominal Gap-FSV:     mean = {mean_gap_fsv_k0:.4f} -> Inflated Gap-FSV: {mean_gap_fsv_k2:.4f}")
    print(f"Delta FUV (mean):    {cost['delta_fuv_mean']:+.4f}")
    print(f"Ratio-of-Means Reduction R: {r_est:.2f}% (95% BCa CI: [{r_lo:.2f}%, {r_hi:.2f}%])")
    print(f"Pre-Registered Target Bar:  >= 50.0% reduction with lower CI > 0")
    print(f"Official Pre-Registered Decision Verdict: >>> {decision} <<<")

    print(f"\n[ASYMMETRIC SAFETY COST & BREAK-EVEN]")
    print(f"Break-Even Safety Weight Ratio r*: {cost['break_even_ratio_r_star']}")
    print(f"Interpretation: For any safety criticality r > {cost['break_even_ratio_r_star']}, epistemic inflation strictly reduces net safety cost C(r).")
    print(f"Cost Curve C(r = 1):  {cost['cost_curve']['1']:.4f}")
    print(f"Cost Curve C(r = 10): {cost['cost_curve']['10']:.4f}")
    print(f"Cost Curve C(r = 50): {cost['cost_curve']['50']:.4f}")

    # 2. S1 Full k-curve analysis: test monotonic FUV growth across seeds
    fuv_spearmans = []
    for s_idx in range(len(seeds)):
        seed_fuvs = [metrics_by_k[k]["fuv"][s_idx] for k in k_values]
        rho, _ = spearmanr(k_values, seed_fuvs)
        fuv_spearmans.append(rho)

    median_rho = float(np.median(fuv_spearmans))
    print(f"\n[S1 FULL K-CURVE ANALYSIS]")
    print(f"Spearman rank correlation rho(k, FUV) across 30 seeds: median = {median_rho:.4f}")

    print("\n[FULL K-CURVE SUMMARY TABLE]")
    print(f"{'k':>4} | {'FSV Mean':>10} | {'95% CI':>18} | {'FUV Mean':>10} | {'Gap-FSV':>10}")
    print("-" * 62)
    k_summary_table = {}
    for k in k_values:
        m_fsv = float(np.mean(metrics_by_k[k]["fsv"]))
        ci_f = compute_confidence_interval(metrics_by_k[k]["fsv"], 0.95)
        m_fuv = float(np.mean(metrics_by_k[k]["fuv"]))
        m_gap_fsv = float(np.mean(metrics_by_k[k]["gap_fsv"]))
        print(f"{k:4.1f} | {m_fsv:10.4f} | [{ci_f.ci_lower:7.4f}, {ci_f.ci_upper:7.4f}] | {m_fuv:10.4f} | {m_gap_fsv:10.4f}")
        k_summary_table[f"k_{k}"] = {
            "k": k,
            "fsv_mean": m_fsv,
            "fsv_ci": [ci_f.ci_lower, ci_f.ci_upper],
            "fuv_mean": m_fuv,
            "gap_fsv_mean": m_gap_fsv,
        }

    # 3. Save official sweep summary
    summary_payload = {
        "sweep_dir": SWEEP_DIR.name,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "grid_size": 201,
        "primary_p1": {
            "estimand": "Ratio-of-Means BCa FSV Reduction (k=2.0 vs k=0.0)",
            "reduction_pct": r_est,
            "ci_95_bca": [r_lo, r_hi],
            "fsv_nominal_mean": float(mean_fsv_k0),
            "fsv_nominal_ci": [ci_fsv_k0.ci_lower, ci_fsv_k0.ci_upper],
            "fsv_k2_mean": float(mean_fsv_k2),
            "fsv_k2_ci": [ci_fsv_k2.ci_lower, ci_fsv_k2.ci_upper],
            "fuv_nominal_mean": float(mean_fuv_k0),
            "fuv_k2_mean": float(mean_fuv_k2),
            "delta_fuv": cost["delta_fuv_mean"],
            "break_even_ratio_r_star": cost["break_even_ratio_r_star"],
            "cost_curve": cost["cost_curve"],
            "decision": decision,
        },
        "secondary_s1": {
            "median_spearman_rho_k_fuv": median_rho,
            "k_summary_table": k_summary_table,
        },
    }

    summary_file = SWEEP_DIR / "sweep_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "=" * 75)
    print(f"[OK] Official Confirmatory P1 Summary saved to: {summary_file.relative_to(ROOT)}")
    print("=" * 75 + "\n")
    return summary_payload


if __name__ == "__main__":
    evaluate_confirmatory_p1()
