"""Plot FSV vs inflation k from the stored confirmatory sweep summary.

Reads ONLY committed artifacts (no recomputation, no training). The n=30 P1
run stores per-k ``fsv_mean`` and a 95% CI in
``results/sweeps/20261027_184119/sweep_summary.json`` under
``secondary_s1.k_summary_table``; this script plots exactly those numbers.

Output: docs/figures/fsv_vs_k.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "sweeps" / "20261027_184119" / "sweep_summary.json"
OUT = ROOT / "docs" / "figures" / "fsv_vs_k.png"


def load_curve(summary_path: Path = SUMMARY) -> tuple[list[float], list[float], list[float], list[float]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    table = data["secondary_s1"]["k_summary_table"]
    ks, means, los, his = [], [], [], []
    for key in sorted(table, key=lambda name: float(name.split("_")[1])):
        row = table[key]
        ks.append(float(row["k"]))
        means.append(float(row["fsv_mean"]))
        lo, hi = row["fsv_ci"]
        los.append(float(lo))
        his.append(float(hi))
    if not ks:
        raise ValueError("no k-curve rows found in stored summary")
    return ks, means, los, his


def main() -> int:
    ks, means, los, his = load_curve()
    err_lo = [m - lo for m, lo in zip(means, los, strict=True)]
    err_hi = [hi - m for m, hi in zip(means, his, strict=True)]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.errorbar(
        ks,
        means,
        yerr=[err_lo, err_hi],
        fmt="o-",
        capsize=3,
        linewidth=1.6,
        markersize=4,
        color="#1f77b4",
    )
    ax.set_xlabel("Epistemic inflation multiplier k")
    ax.set_ylabel("False-Safe Volume (FSV)")
    ax.set_title(
        "FSV vs inflation k (n=30 replicates, 95% CI)\n"
        "results/sweeps/20261027_184119",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"wrote {OUT.relative_to(ROOT)} from {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
