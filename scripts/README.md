# Code checks

Run the code checks from the repository root:

```bash
bash scripts/run_checks.sh
```

This parses Python sources, checks essential source assets, and runs the pipeline
and mutation test suites. Tests use temporary files and mock external services.
They do not launch paid generation, GPU training, or a licensed formal campaign.
PyTorch-dependent recovery tests and tests requiring omitted raw snapshots skip
when those optional dependencies/artifacts are absent.

The local full-submission packaging/integrity scripts are not needed by the
code-only GitHub checkout and are not included in the public upload.
