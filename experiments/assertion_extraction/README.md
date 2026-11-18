# Assertion extraction and postprocessing

- `extract_assertions.py`: identifies inline and named properties and resolves
  the property expression associated with an assertion.
- `remove_exact_duplicates.py`: compares tokenised expressions while preserving
  representative source references.
- `formal_clocking.py`: validates and reproduces explicit formal clock settings.
- `check_premise_consistency.py`, `formal_redundancy.py`, `validate_reduced_set.py`:
  consistency, redundancy reduction and validation of reduced sets.
- `run_batch.py`, `run_pipeline.py`, `retry_batch_once.py`: orchestration helpers.

The thesis SFT/KTO data uses exact deduplication **before formal redundancy
reduction**. The separate reduction/coverage tools remain as experimental code;
their presence does not mean reduced assertions were used as training targets.

Run from the repository root and use a module's `--help` for its input/output
arguments. Full formal runs require Cadence JasperGold and the associated saved
RTL/context inputs. Source code and tests are included; historical raw cases and
formal logs are supplied separately.
