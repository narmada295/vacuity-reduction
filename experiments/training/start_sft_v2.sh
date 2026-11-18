#!/usr/bin/env bash
# First launch only. Subsequent recovery uses train_sft.py --resume auto.
set -euo pipefail

task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_project_dir="$(cd -- "$task_script_dir/../.." && pwd)"
cd -- "$task_project_dir"

python - <<'PY'
import json, os, re, subprocess
from pathlib import Path

processes = subprocess.check_output(
    ['ps', '-u', str(os.getuid()), '-o', 'pid=,args='], text=True)
for line in processes.splitlines():
    pid, args = line.strip().split(None, 1)
    if (re.search(r'(?:^|[/ ])train_sft\.py(?: |$)', args)
            and re.search(r'--mode\s+train(?:\s|$)', args)):
        raise SystemExit('Another training process is still running (PID ' + pid + '). Stop it first.')
manifest = json.loads(Path('data/sft_v2/manifest.json').read_text())
required = {'L1': 9, 'L2': 9, 'L3': 7}
if manifest['split_plan'].get('design_quotas', {}).get('train') != required:
    raise SystemExit('Expected the agreed 9/9/7 training split.')
train = manifest['splits']['train']
if train['designs'] != 25 or any(train[level + '_designs'] != n for level, n in required.items()):
    raise SystemExit('Manifest does not contain 25 training designs in the agreed proportions.')
for name in ('logs/token_lengths_v2', 'outputs/qwen3_4b_sft_v2_smoke',
             'outputs/qwen3_4b_sft_v2_train'):
    if Path(name).exists():
        raise SystemExit('Existing v2 artifact: ' + name + '. Preserve it; use stage-specific recovery commands.')
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    raise SystemExit('Paste the CUDA_VISIBLE_DEVICES export from your current allocation first.')
print('Preflight: 25 training designs (L1=9, L2=9, L3=7); old training is stopped.', flush=True)
PY

mkdir -p logs outputs
python -u scripts/sft_v2/profile_sft.py \
  --data-dir data/sft_v2 --out logs/token_lengths_v2 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  2>&1 | tee logs/sft_v2_profile.log

python -u scripts/sft_v2/train_sft.py --mode smoke \
  --data-dir data/sft_v2 --profile-dir logs/token_lengths_v2 \
  --out outputs/qwen3_4b_sft_v2_smoke \
  2>&1 | tee logs/sft_v2_smoke.log

python - <<'PY'
import json
from pathlib import Path
result = json.loads(Path('outputs/qwen3_4b_sft_v2_smoke/result.json').read_text())
if result['status'] != 'PASS' or result['global_step'] != 2 or not result['complete_checkpoints']:
    raise SystemExit('Smoke test did not pass; full training was not started.')
print('Smoke test passed. Starting fresh full SFT on the 25-design dataset.', flush=True)
PY

python -u scripts/sft_v2/train_sft.py --mode train \
  --data-dir data/sft_v2 --profile-dir logs/token_lengths_v2 \
  --out outputs/qwen3_4b_sft_v2_train \
  2>&1 | tee logs/sft_v2_train.log
