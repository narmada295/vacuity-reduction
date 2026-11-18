# Fixed 30-design FIXME SVAGen Text result set

The primary result is `output/fixme_svagen/summary_30_designs.md`. It lists
all 30 designs, including two that generated no final assertions. The local
datasets are `*_30_designs.jsonl`. The after-reduction file
`retained_30_designs_after_reduction.jsonl` has all 1,194 retained properties:
973 proved non-vacuous implications and 221 proved non-implications with
`vacuous=null`. The earlier path
`non_vacuous_30_designs_after_reduction.jsonl` contains the same rows for
compatibility. Validated per-design retained assertions
are under `output/fixme_svagen/reduction_30/<level>/<design>`. Reduced,
unreduced, and comparison coverage summaries are under the corresponding
`checker_coverage_30*` directories.

On the JasperGold host, input files are in `experiments/inputs`, reduction
runs in `experiments/results`, and checker coverage runs plus the aggregate
30-design reports in `experiments/checker_coverage`. The three newer designs'
coverage runs are actual directories in this common tree.

Validate the local snapshot with:

```sh
python3 experiments/fixme_30/verify.py
```

Generate the full overall and per-design results report with:

```sh
python3 experiments/fixme_30/report.py
```

This writes `results_30_detailed.md`, `results_30_detailed.json`, and
`results_30_per_design.csv` under `output/fixme_svagen`.

The after-reduction dataset can be regenerated from per-design validated
remote results using `experiments/assertion_extraction/export_non_vacuous_after_reduction.py`
with `--plan output/fixme_svagen/reduction_30_plan.json`,
`--before-reduction output/fixme_svagen/non_vacuous_30_designs_before_reduction.jsonl`,
`--output output/fixme_svagen/non_vacuous_30_designs_after_reduction.jsonl`,
and `--include-outside-scope`.
