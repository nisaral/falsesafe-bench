"""CLI wiring tests: the sweep subcommand must resolve its runner without ImportError."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sweep_subcommand_help_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "sweep", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout


def test_run_sweep_is_importable_from_cli_path():
    # cli.py does: from scripts.run_pipeline import run_sweep
    from scripts.run_pipeline import run_sweep

    assert callable(run_sweep)
