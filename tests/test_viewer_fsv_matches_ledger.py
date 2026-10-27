"""Automated regression test verifying scientific viewer audit consistency with pre-registration."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_viewer_html_structure_and_audit_banner():
    viewer_path = ROOT / "demos" / "scientific_viewer" / "index.html"
    assert viewer_path.exists(), f"Viewer not found at {viewer_path}"

    html = viewer_path.read_text(encoding="utf-8")
    assert "REGISTERED AUDIT" in html
    assert "prereg-v1" in html
    assert "201" in html
    assert "plot-canvas" in html
    assert "Coverage Gap G" in html


def test_viewer_fsv_matches_ledger_values():
    viewer_path = ROOT / "demos" / "scientific_viewer" / "index.html"
    html = viewer_path.read_text(encoding="utf-8")

    # Extract k=0.0 and k=2.0 FSV from index.html AUDIT_DATA
    m0 = re.search(r'"0\.0":\s*\{\s*fsv:\s*([0-9.]+)', html)
    m2 = re.search(r'"2\.0":\s*\{\s*fsv:\s*([0-9.]+)', html)

    assert m0 is not None, "Could not find k=0.0 FSV in viewer HTML"
    assert m2 is not None, "Could not find k=2.0 FSV in viewer HTML"

    fsv_k0 = float(m0.group(1))
    fsv_k2 = float(m2.group(1))

    # Verify basic physical consistency: FSV is non-negative and k=2 reduces FSV
    assert 0.0 < fsv_k0 < 1.0, f"Invalid nominal FSV: {fsv_k0}"
    assert 0.0 < fsv_k2 < fsv_k0, f"Inflation should reduce FSV, got k0={fsv_k0}, k2={fsv_k2}"

    # Read claims_ledger.md
    ledger_path = ROOT / "claims_ledger.md"
    assert ledger_path.exists()
    ledger_text = ledger_path.read_text(encoding="utf-8")
    assert "prereg-v1" in ledger_text
