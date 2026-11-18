#!/usr/bin/env python3
"""Check source syntax and required code assets without datasets or credentials."""
from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ('fixme_svagen', 'experiments', 'mutation_lab', 'scripts', 'tests')


def main():
    counts = {}
    for folder in FOLDERS:
        counts[folder] = 0
        for path in (ROOT / folder).rglob('*.py'):
            if any(part in {'__pycache__', 'runs', 'reports'} for part in path.relative_to(ROOT).parts):
                continue
            ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
            counts[folder] += 1
    for name in (
        'fixme_svagen/prompts.py', 'experiments/evaluation/README.md',
        'experiments/training/sft_split_v2.json',
        'mutation_lab/examples/piezo_driver/spec.txt',
        'mutation_lab/examples/piezo_driver/rtl/Piezo_Driver.v',
        'mutation_lab/examples/tiny_counter/config.json',
    ):
        if not (ROOT / name).is_file():
            raise ValueError('Missing code asset: ' + name)
    plan = json.loads((ROOT / 'experiments/training/sft_split_v2.json').read_text())
    if [len(plan[s]) for s in ('train', 'validation', 'test')] != [25, 5, 5]:
        raise ValueError('Original source split sizes differ')
    print(json.dumps({'status': 'PASS', 'python_sources_parsed': counts}, indent=2))


if __name__ == '__main__':
    main()
