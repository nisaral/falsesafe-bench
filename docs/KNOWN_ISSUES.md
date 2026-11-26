# Known issues

Only verified issues are listed here. Each entry states what was checked and
where the evidence lives.

## 1. C0 and C2 are bit-identical in the stored S2 output

In `results/certifier_comparison/s2_full6_grid101_n10.json`, the C0 and C2
entries have exactly equal summary statistics and exactly equal per-seed
records (checked across all 10 per-seed records, not just the means). Their
`fuv_star_curve` is constant in both, which makes the AUPC reduce to the
penalty value in each case. Because the values are identical to the bit, the
stored output cannot distinguish the two certifiers; treat the C0/C2 comparison
as unresolved rather than as evidence that they agree by design.

## 2. FSV is a grid-cell count, so small disturbances can be invisible

FSV is computed as a fraction of grid cells (`src/lcert/metrics.py`), so a
small change in the learned disturbance bound can leave every cell on one side
of the safety threshold untouched and produce an identical FSV. The stored C0
vs C2 rows are an example: their `fuv_star_curve` is constant across the whole
control-ratio grid. Report FSV differences with this granularity in mind, and do
not read a zero change as proof that the underlying safety set did not move.

## 3. Legacy results used a 2-cell border, SCOPE specifies 3

The committed sweeps under `results/sweeps/` evaluated internal cells with a
2-cell border excluded, while `SCOPE.md` section 3 specifies a 3-cell border
(and `src/lcert/metrics.py` defaults to 3). The stored `num_eval_cells` of 38809
in the n=30 run equals `(201-4)^2`, confirming the 2-cell border was used. The
two configs that reproduce legacy runs now pin `border_cells: 2` explicitly with
a `legacy` comment; all other runs default to 3. Numbers computed under the
2-cell border are therefore not directly comparable to a 3-cell-border re-run.

## 4. Framework numbers and results/ numbers are not comparable

The `src/lcert/framework/` plugin path and the preregistered `results/` pipeline
are separate evaluation stacks. `configs/framework/di_nominal_sweep.yaml` uses
`grid_size: 101`, `horizon: 1.5` and a constant `sigma: 0.15` nominal certifier,
whereas the P1 run uses `grid_size: 201`, `horizon: 0.5`, a learned ensemble and
per-state epistemic inflation. `docs/FRAMEWORK.md` already states that framework
runs are exploratory and must not be quoted as confirmatory results. Never mix
FSV/FUV numbers across the two stacks.

## 5. Legacy runlogs lack provenance fields

`scripts/validate_runlogs.py` checks every committed `runlog.json` under
`results/` against the required fields in `runlog_schema.json`. It reports
**0 conformant of 40 scanned** (40 non-conformant). Across those 40 logs, `seed`
is the only required field present (missing in 0/40); every other required field
(`git_commit`, `config_sha256`, `library_versions`, `split_seed`, `status`, and
the rest) is missing in 40/40. The legacy runlogs therefore cannot prove which
commit, config, or library versions produced them, and are treated as a frozen
legacy baseline rather than a reproducible provenance record.

