# Dataset preparation and model training

The completed experiment fine-tuned Qwen3-4B-Instruct-2507 using SFT followed by
KTO. Both stages use QLoRA: NF4 base weights stay frozen while LoRA adapters train
with BF16 computation. The data, model weights and run outputs are not bundled
with the code-only repository.

## Code

| File | Purpose |
| --- | --- |
| `prepare_sft.py` | Exact deduplication and positive-example export before formal reduction |
| `verify_sft.py` | Checks source references, hashes, labels and formal context |
| `reassess_l3_3.py` | Corrected-clock evidence audit for L3/3 |
| `prepare_kto.py` | Independent desirable/undesirable feedback export and verification |
| `profile_sft.py` | Token lengths and input-integrity checks without truncation |
| `train_sft.py` | Response-only loss masking, QLoRA SFT and complete checkpoints |
| `train_kto.py` | SFT-initialized KTO with a frozen reference and class weights |
| `resume_kto_verified.py` | Checksum-verified recovery of the interrupted KTO run |
| `start_sft_v2.sh` | Original remote first-launch wrapper using `scripts/sft_v2/` |

Run modules from the repository root. Use `--help` to inspect their arguments.
`sft_split_v2.json` contains the 25-design training assignment: nine L1, nine L2,
and seven L3. The older v1 plan remains for source reproducibility. Explicitly
pass the v2 plan and a fresh destination when preparing the completed variant:

```bash
python3 -m experiments.training.prepare_sft \
  --split-plan experiments/training/sft_split_v2.json \
  --out output/fixme_svagen/sft_before_reduction_v2
```

Preparation needs the corresponding raw cases, saved signal information and
corrected formal evidence. Those are separate research artifacts, not source code.
The original five-design source-reference test split differs from the ten-design
reported generation evaluation; do not substitute source-reference assertions for
evaluation prompts.

## Example selection

- SFT: proved nonvacuous implications and proved non-implication properties.
- KTO desirable: the same categories, with stricter complete-cover evidence checks.
- KTO undesirable: proved vacuous implications.
- Falsified assertions and unresolved evidence are excluded from training selection.

Non-implications remain outside the two implication-vacuity checks. Training
inputs contain specification, RTL, target-signal information and timing context;
proof/vacuity labels guide selection rather than appearing in the student prompt.

The completed SFT export contains 4,966 training and 1,569 validation examples.
KTO contains 5,674/1,761 examples, with 4,948 desirable and 726 undesirable training
examples. Its stricter evidence policy excludes 18 SFT training positives and
three validation positives. No test example is used for KTO training.

## Running SFT with a supplied frozen export

Use a separate Python 3.12 GPU environment with `requirements/training.txt`.
Configure the allocated GPU before running the commands. The completed SFT used
a 24 GB MIG slice; actual memory fit should be checked with the smoke run.

Place a complete prepared export in `data/sft/`, including its original manifest,
then use fresh profile/output directories:

```bash
python3 -m experiments.training.profile_sft \
  --data-dir data/sft --out logs/sft_profile

python3 -m experiments.training.train_sft --mode smoke \
  --data-dir data/sft --profile-dir logs/sft_profile \
  --out outputs/sft_smoke

python3 -m experiments.training.train_sft --mode train \
  --data-dir data/sft --profile-dir logs/sft_profile \
  --out outputs/sft_train
```

The original remote wrapper uses different historical directories. For a source
checkout, the module commands above avoid copying files into that remote layout.
The tokenizer/base weights must be available to the training environment. The
SFT objective masks prompt labels to `-100` and learns from reference response
tokens, rather than comparing two fully generated assertion strings.

## KTO

KTO needs a complete selected SFT run directory, including adapter weights and
metadata. The completed experiment selected SFT checkpoint 621 after three epochs;
KTO selected checkpoint 710 after one epoch and used a 48 GB MIG slice.

Place the complete KTO export, including every file listed in its frozen manifest,
in `data/kto/`. The following commands use the historical SFT output location;
change `--sft-dir` to the supplied complete run directory if necessary.

```bash
python3 -m experiments.training.train_kto --mode preflight \
  --data-dir data/kto --sft-dir outputs/qwen3_4b_sft_v2_train

python3 -m experiments.training.train_kto --mode smoke \
  --data-dir data/kto --sft-dir outputs/qwen3_4b_sft_v2_train \
  --out outputs/qwen3_4b_kto_v1_smoke

python3 -m experiments.training.train_kto --mode train \
  --data-dir data/kto --sft-dir outputs/qwen3_4b_sft_v2_train \
  --out outputs/qwen3_4b_kto_v1_train
```

Preflight and a passing four-update smoke test precede full KTO training. The
reference adapter remains frozen; the policy adapter is updated. The original
undesirable weight is 4948/726, approximately 6.8154.

`--resume auto` restores only complete matching checkpoints. Keep data, packages,
configuration, policy and reference identity unchanged when resuming. Do not
overwrite an old output folder with a fresh experiment.
