# Experiment code

Run modules from the repository root using `python3 -m experiments.<module>`.

| Directory/file | Role |
| --- | --- |
| `assertion_extraction/` | Extraction, exact deduplication, clock handling, and separate reduction tools |
| `training/` | SFT/KTO data preparation, profiling, training, verification, and recovery |
| `evaluation/` | Frozen prompts, generation, source preparation, and JasperGold checking |
| `checker_coverage/`, `fixme_30/`, `fixme_cohort.py` | Separate reduction/coverage exploration |
| `resolve_l3_clock_context.py`, `resolve_piezo_inconclusive.py` | Audits used to resolve saved formal evidence |

The thesis training data is taken **before formal redundancy reduction**. The
presence of reduction/coverage tools does not mean their outputs were used as SFT
or KTO training targets. Existing original raw paths use the `output/` alias.
