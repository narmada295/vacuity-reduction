# Checker coverage for the fixed 25-design reduction set

`run_batch.py` uses `output/fixme_svagen/reduction_25_plan.json` (copied to the
JasperGold host) and the **validated retained assertions** from each design's
reduction. It inserts those assertions into a copy of the matching RTL module.
The source `.v` file and reduced assertions are never changed.

For each design with retained assertions, the runner performs two JasperGold
runs, following the existing Piezo and VGA examples in
`experiments/checker_coverage` on the remote host:

1. Initialize statement and branch coverage for COI and proof core, prove all
   inserted assertions, measure both coverage types, and report COI plus the
   default checker view and assertion results.
2. Initialize proof-core-only coverage, prove the same assertions, and report
   statement and branch coverage.

The runs use every explicit property clock found in the retained assertions,
or the case's default clock when properties have no explicit event. Designs
without a clock use `clock -none`. The reset policy is `reset -none`, matching
the existing examples. A design with no retained assertions is recorded as
`no_retained_assertions`; coverage percentages are undefined for it.

Run on the JasperGold host after sourcing its Cadence environment:

```sh
python3 -u run_batch.py --plan ../reduction_25_plan.json \
  --jg /home/Cadence_tools/jasper_2022.12/bin/jg
```

Results are written to `checker_coverage/<level>/<design>/run_25_final/`:

- `summary.json` and `summary.md`: per-design coverage counts, percentages,
  assertion and precondition statuses, clocks, and input hashes.
- `combined/coverage.log` and `proof/coverage.log`: full JasperGold output.
- `checked.sv`, `source.v`, `retained_assertions.jsonl`, `manifest.json`, and
  the generated Tcl files: exact inputs and commands.

`checker_coverage/coverage_25_summary.json` aggregates item-weighted coverage
counts across completed designs; `coverage_25_status.json` tracks progress.
The reported percentages are **checker coverage** over JasperGold's cover-item
denominators, not a claim that every behavior or requirement in the design is
checked. Proof and COI metrics are kept separate. Incomplete or failed runs
never contribute to aggregate coverage.

## Coverage before reduction

Use the same runner with `--assertion-set unreduced` to measure every row in
`<reduction output>/extracted/selected_assertions.jsonl`. These are the proved
non-vacuous assertions plus proved non-implication assertions marked outside
the two vacuity cases. This set is measured before exact deduplication and
formal pruning; duplicate expressions retain distinct checker labels.

```sh
python3 -u run_batch.py --plan ../reduction_25_plan.json \
  --jg /home/Cadence_tools/jasper_2022.12/bin/jg --assertion-set unreduced
```

Unreduced results go to `checker_coverage/<level>/<design>/run_25_unreduced/`.
The progress and aggregate files are `coverage_25_unreduced_status.json` and
`coverage_25_unreduced_summary.json`. Reduced output remains in `run_25_final`.

Once the remote run completes, from the SANGAM repo root:

```sh
PYTHONPATH=output/fixme_svagen/.python_deps:. \
  python3 experiments/checker_coverage/sync_unreduced.py
python3 experiments/checker_coverage/compare_sets.py
PYTHONPATH=output/fixme_svagen/.python_deps:. \
  python3 experiments/checker_coverage/publish_comparison.py
```

The comparison is in
`output/fixme_svagen/checker_coverage_25_comparison/coverage_25_comparison.md`
and `.json`, plus `comparison.md` and `.json` for each design. It reports all
six checker coverage metrics, numerator and denominator counts, assertion
counts, and reduced-minus-unreduced percentage-point changes. A changed
denominator is flagged because the two percentages then describe different
checker cover-item sets. An empty assertion set has undefined coverage.
The publish command also copies the aggregate comparison to the remote
`checker_coverage/` directory and each design comparison beside its run folders.

After the remote batch finishes, run this from the SANGAM repo root to copy
every design's `summary.json` and `summary.md` plus the aggregate files into
`output/fixme_svagen/checker_coverage_25/`:

```sh
PYTHONPATH=output/fixme_svagen/.python_deps:. \
  python3 experiments/checker_coverage/sync_summaries.py
```
