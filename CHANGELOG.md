# Changelog

All changes below are on top of the upstream fork point
`nisaral/falsesafe-bench@a4fd856`. Grouped by the phase in which they were made.
Each entry is one commit, generated from `git log --oneline` on
`polish/credibility`.

## Phase 1 — Correctness fixes

- `fix(pipeline)`: import the P1 bootstrap and asymmetric-cost helpers that the
  P1 block used without importing.
- `fix(pipeline)`: read the paired-bootstrap result by attribute instead of
  subscripting the dataclass.
- `fix(pipeline)`: compute the asymmetric safety cost separately at k=0 and k=2
  instead of misbinding the cost weights.
- `refactor(pipeline)`: extract the inline P1 block into `aggregate_p1(...)`.
- `test(pipeline)`: cover `aggregate_p1` output fields, positive reduction, and
  JSON serialisability.
- `feat(pipeline)`: add a `run_sweep` entry point for the CLI.
- `test(cli)`: guard that the `sweep` subcommand resolves without ImportError.
- `feat(models)`: add `predict_mean_std` to `EnsembleMLP`.
- `feat(models)`: add `predict_mean_std` to `IndependentExactGP`.
- `fix(certifiers)`: raise `TypeError` on unsupported models in the
  distance-aware certifier instead of silently falling back.
- `fix(certifiers)`: raise `TypeError` on unsupported models in the conformal
  certifier instead of substituting a constant sigma.
- `test(certifiers)`: exercise both certifiers end to end against a real,
  trained ensemble (no stubs).
- `fix(controls)`: derive the C4 status from its pass criterion rather than
  hardcoding `PASSED`.
- `feat(pipeline)`: make `border_cells` a config parameter defaulting to 3 (the
  two legacy configs pin 2 explicitly).
- `test(certifiers)`: document the conformal quantile against the textbook
  order statistic without changing behaviour.
- `ci`: run the test workflow on `polish/credibility`.

## Phase 2 — Honesty and provenance

- `docs(report)`: mark `docs/COMPREHENSIVE_REPORT.md` superseded by the ledger.
- `docs(report)`: correct the test count to the actual suite size.
- `docs(report)`: mark H2 partially supported at 40.34%.
- `docs(report)`: replace the exec-summary 72.7% pilot number with the
  confirmatory P1 result.
- `docs(report)`: add the P1 row and flag the pilot rows superseded.
- `docs(readme)`: clarify that the 30 runs are replicates with a fixed split
  seed.
- `docs(ledger)`: record the fixed-split-seed limitation as a P1 provenance row.
- `feat(pipeline)`: add an opt-in `--vary-split-seed` flag (off by default).
- `docs(ledger)`: add a Corrections section for the C2/C3/C4 control rows.
- `feat(scripts)`: add a read-only `validate_runlogs.py` schema checker.
- `docs(results)`: document the legacy runlog provenance gaps.
- `docs`: add `docs/KNOWN_ISSUES.md` and one entry each for the C0/C2 identity,
  FSV grid granularity, the legacy border, and framework/results
  incomparability.

## Phase 3 — Presentation

- `chore(make)`: add `setup`, `doctor`, `test`, `figures` targets.
- `chore(lint)`: add a minimal ruff config and `requirements-dev.txt`.
- `refactor(pipeline)`: drop an unused `json` import.
- `feat(figures)`: plot the stored FSV-vs-k curve from committed results.
- `docs(figures)`: add `docs/figures/fsv_vs_k.png`.
- `docs(readme)`: problem statement, headline result with figure, "what this
  repo is and is not", five-command quick start, repo map, limitations, and
  origin/attribution sections.
