#!/usr/bin/env bash
set -euo pipefail

task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_project_dir="$(cd -- "$task_script_dir/.." && pwd)"
cd -- "$task_project_dir"

export PYTHONDONTWRITEBYTECODE=1
python3 scripts/check_code.py
python3 -m unittest discover -s tests -q
python3 -m unittest discover -s mutation_lab/tests -q
