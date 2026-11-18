# Vacuity Reduction for SystemVerilog Assertion Generation

B.Tech thesis project by **Narmada Pinjala**, under the guidance of
**Prof. Chandan Karfa**.

This project studies whether vacuity information during training helps a small
language model generate fewer vacuous SystemVerilog assertions. The workflow
collects assertions using modified SANGAM, records proof and vacuity evidence
with Cadence JasperGold, and fine-tunes Qwen3-4B-Instruct-2507 using SFT and KTO.

## Code structure

| Folder | Purpose |
| --- | --- |
| [`fixme_svagen/`](fixme_svagen/README.md) | Assertion generation, refinement and formal labelling |
| [`experiments/assertion_extraction/`](experiments/assertion_extraction/README.md) | Extraction, exact deduplication and formal-context tools |
| [`experiments/training/`](experiments/training/README.md) | Dataset preparation, SFT, KTO and checkpoint recovery |
| [`experiments/evaluation/`](experiments/evaluation/README.md) | One-shot generation and JasperGold evaluation |
| `experiments/checker_coverage/`, `experiments/fixme_30/` | Separate reduction and coverage exploration |
| [`mutation_lab/`](mutation_lab/README.md) | Preliminary behavioral RTL mutation work |
| `tests/` | Pipeline, data-preparation and recovery tests |
| [`scripts/`](scripts/README.md) | Source checks and test launcher |
| `requirements/` | Optional GPU-training and source-preparation dependencies |

The GitHub repository contains code, configurations, tests and small examples.
Training datasets, trained adapters, reports, slides and raw run outputs are
separate artifacts and are not included here.

## Setup

Use Python 3.12 and run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/run_checks.sh
```

The tests do not require live API calls, GPU training or licensed formal jobs.
Checks that require optional PyTorch or saved raw snapshots skip when those are
not available. For GPU training, use a separate environment with the recorded
versions in `requirements/training.txt`. Additional upstream input-preparation
packages are listed in `requirements/evaluation-preparation.txt`.

## Workflow

1. **Collect assertions.** Modified SANGAM receives the full specification and
   target-signal name/description. Candidate assertions are refined using critic
   and JasperGold feedback. RTL is used for formal checking, not directly included
   in the teacher assertion-generation prompt.
2. **Record formal evidence.** Proof results are kept separate from implication
   vacuity. Related covers check whether an antecedent can occur and whether a
   consequent can be false. Missing evidence remains unresolved.
3. **Prepare training examples.** Remove exact duplicates within design/formal
   context before the separate formal redundancy-reduction stage. Split by design.
4. **Train.** SFT uses proved nonvacuous implications and proved non-implications.
   KTO continues from SFT using desirable and undesirable assertion responses.
5. **Evaluate.** Compare base, SFT and SFT+KTO using identical tasks and checking
   conditions. Report proof success, conditional vacuity and nonvacuous implication
   yield separately for each design.

Non-implications remain outside the two targeted implication-vacuity checks.
Passing these checks does not establish specification coverage or fault detection.

## Assertion collection

Clone the FIXME benchmark separately and use the revision recorded for the
experiment. Configure your own credentials locally:

```bash
cp .env.example .env
python3 -m fixme_svagen.cli --fixme-root ../FIXME --list-cases
```

Actual collection requires Nebius access and an SSH-accessible JasperGold
installation. Credentials belong only in the ignored `.env` file. The collection
README contains a small-run example.

## Training and evaluation

The completed SFT dataset used 25 training designs (nine L1, nine L2, seven L3)
with 4,966 examples, plus five validation designs with 1,569 examples. KTO used
5,674 training and 1,761 validation examples. Its training feedback contains
4,948 desirable and 726 undesirable examples, after stricter positive-evidence
checks than the original SFT selection.

The reported generation evaluation covers six FIXME designs and four OpenTitan
modules: GPIO, PWM, UART and I2C, with 632 target-signal tasks per model. The
original source-reference test split differs from this final reported population.
The original `L1/test_endian` evaluation is retained in separate raw artifacts but
excluded from the reported comparison after adapted outputs failed elaboration.

Use the training and evaluation READMEs for required input layouts, commands and
saved-context requirements. Model weights, prepared exports and formal return
bundles must be supplied separately; source files alone are not those artifacts.

## Credit

The assertion collection workflow is adapted from
[SANGAM](https://github.com/CoolSunflower/SANGAM) by Adarsh Gupta, Bhabesh Mali and
Chandan Karfa. The original teacher prompt strings are retained in
`fixme_svagen/prompts.py`.

Benchmark inputs come from [FIXME](https://github.com/ChatDesignVerification/FIXME)
and [OpenTitan](https://github.com/lowRISC/opentitan). The student model is
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), adapted
using [QLoRA](https://arxiv.org/abs/2305.14314) and
[KTO](https://proceedings.mlr.press/v235/ethayarajh24a.html).
Upstream code, benchmark inputs and model weights retain their original terms.
