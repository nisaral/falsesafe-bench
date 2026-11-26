# results/ — frozen legacy baseline

Everything committed under `results/` is a **frozen legacy baseline**. It is kept
for provenance and citation, and is **not** regenerated, edited, or deleted by
the polish work on `polish/credibility`. Where these artifacts disagree with
`claims_ledger.md`, the ledger's Corrections section records the discrepancy.

## Provenance status of committed runlogs

`scripts/validate_runlogs.py` is a read-only checker that validates every
`runlog.json` under `results/` against the `required` fields in
[`runlog_schema.json`](../runlog_schema.json). Output at the time of writing
(`python scripts/validate_runlogs.py`):

```
runlogs scanned: 40
conformant:      0
non-conformant:  40
```

Missing-field counts across those 40 runlogs:

| field | runlogs missing it |
|---|---|
| `git_commit` | 40/40 |
| `config_sha256` | 40/40 |
| `library_versions` | 40/40 |
| `run_id`, `experiment`, `command`, `git_tree_clean`, `config_path` | 40/40 |
| `split_seed`, `start_time_ist`, `end_time_ist`, `python_version` | 40/40 |
| `hardware`, `status`, `metrics_path`, `test_split_used` | 40/40 |
| `seed` | 0/40 (only schema-required field present) |

In short, **none** of the committed runlogs satisfy `runlog_schema.json`; they
record only `seed`, timestamps, train-sample counts, validation scores, and the
metric dictionaries. They therefore cannot prove which commit, config, or
library versions produced them, and must be treated as legacy rather than as a
reproducible provenance record.

The legacy sweeps also used a 2-cell evaluation border
(see `docs/KNOWN_ISSUES.md`), whereas `SCOPE.md` specifies a 3-cell border.

## Contents

- `sweeps/20261027_161358/` — n=5 pilot (superseded; see `claims_ledger.md`).
- `sweeps/20261027_184119/` — n=30 confirmatory P1 run (primary result).
- `certifier_comparison/` — S2 multi-certifier comparisons
  (`comparison_summary.json`, `s2_confirmatory_summary.json`,
  `s2_full6_grid101_n10.json`).
- `controls_c1_c4_summary.json` — C1–C4 control outputs (see ledger Corrections).
- `s2_s4_s6_summary.json` — S4 and S6 secondary results.
- `sanity_test_output_20261025_1654IST.txt` — captured sanity-test output.
