"""Report which committed runlogs fail the provenance fields in runlog_schema.json.

Read-only: this script never writes to or mutates any file under results/.
It validates each ``runlog.json`` against the ``required`` list (and the
``git_commit`` pattern) declared in ``runlog_schema.json`` and prints a table.

Exit code is 0 even when logs are non-conformant, because the committed results
are a frozen legacy baseline; use the printed table for documentation.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "runlog_schema.json"
RESULTS_DIR = ROOT / "results"

GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def load_required_fields(schema_path: Path = SCHEMA_PATH) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return list(schema.get("required", []))


def check_log(log: dict, required: list[str]) -> list[str]:
    """Return the list of failed requirements for one decoded runlog."""
    problems = []
    for field in required:
        if field not in log:
            problems.append(f"missing:{field}")
    commit = log.get("git_commit")
    if isinstance(commit, str) and commit and not GIT_COMMIT_PATTERN.match(commit):
        problems.append("bad_format:git_commit")
    return problems


def iter_runlogs(results_dir: Path = RESULTS_DIR):
    for path in sorted(results_dir.rglob("runlog.json")):
        yield path


def collect(results_dir: Path = RESULTS_DIR) -> list[tuple[Path, list[str]]]:
    required = load_required_fields()
    rows = []
    for path in iter_runlogs(results_dir):
        try:
            log = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            rows.append((path, [f"unreadable:{exc.__class__.__name__}"]))
            continue
        rows.append((path, check_log(log, required)))
    return rows


def print_report(rows: list[tuple[Path, list[str]]]) -> None:
    total = len(rows)
    conformant = sum(1 for _, problems in rows if not problems)
    print(f"runlogs scanned: {total}")
    print(f"conformant:      {conformant}")
    print(f"non-conformant:  {total - conformant}")
    print()
    header = f"{'status':<12} {'runlog':<62} problems"
    print(header)
    print("-" * len(header))
    for path, problems in rows:
        status = "OK" if not problems else "NONCONFORMANT"
        rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        print(f"{status:<12} {rel:<62} {', '.join(problems) if problems else '-'}")


def print_field_summary(rows: list[tuple[Path, list[str]]], required: list[str]) -> None:
    print()
    print("per-field missing counts (out of %d runlogs):" % len(rows))
    for field in required:
        missing = sum(1 for _, problems in rows if f"missing:{field}" in problems)
        print(f"  {field:<20} missing in {missing}/{len(rows)}")


def main() -> int:
    if not SCHEMA_PATH.exists():
        print(f"schema not found: {SCHEMA_PATH}", file=sys.stderr)
        return 2
    rows = collect()
    print_report(rows)
    print_field_summary(rows, load_required_fields())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
