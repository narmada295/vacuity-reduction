# Assertion generation and formal evaluation

Evaluation compares the base, SFT and SFT+KTO models using matching task inputs,
decoding settings and formal conditions. These scripts require separately
provided model adapters, task exports, or saved run bundles for the relevant
stage. Large datasets and run artifacts are not in this code-only repository.

## Main files

| Files | Role |
| --- | --- |
| `prompts.py` | Frozen one-shot specification-grounded instruction template |
| `generate.py` | One response per target signal, saved attempts and restart checks |
| `prepare_inputs.py`, `prepare_test.py` | Original validation/test input preparation |
| `opentitan_sources.py`, `prepare_opentitan.py`, `prepare_mixed.py` | Additional source and task preparation |
| `run_full.py`, `run_test.py`, `run_opentitan.py`, `run_mixed.py` | Cohort generation and identity checks |
| `formal.py`, `formal_test.py`, `formal_kto.py`, `formal_mixed.py` | Formal input preparation and returned-result interpretation |
| `formal_worker.py` | Resumable remote JasperGold jobs and vacuity classification |
| `formal_infrastructure_guard.py` | Licence/infrastructure interruption handling |

## Generation protocol

The reported comparison uses 632 target-signal tasks per model across six FIXME
designs and OpenTitan GPIO, PWM, UART and I2C. Models receive specification, RTL,
signal information and timing context, with no reference response or formal labels.
Each task requests up to 12 assertions in one response, with temperature 0.7,
top-p 0.9, 4,096 new tokens and a matched task-derived seed schedule based on 42.
Generated assertions are evaluated without a repair/refill stage.

With supplied prompts, the pinned model cached locally, and the recorded SFT run
configuration, a new base-model inference run can be launched as follows:

```bash
python3 -m experiments.evaluation.generate \
  --prompts data/evaluation/prompts.jsonl --label base \
  --training-config outputs/qwen3_4b_sft_v2_train/run_config.json \
  --assertion-cap 12 --samples-per-prompt 1 --max-new-tokens 4096 \
  --temperature 0.7 --top-p 0.9 --seed 42 \
  --out outputs/evaluation/base
```

Use `--allow-download` if model downloading is needed. Adapted runs need label
`sft` or `kto` and the corresponding `--adapter` directory. Use fresh output
directories; original returned cohorts have their own frozen prompt-file hashes.

## Formal checking

Each checking job uses original RTL, the saved clock context, unconstrained
reset, and a 180-second invocation limit including startup. Model-generated
assumptions are not added to the environment. Formal preparation needs exact
source-context exports and the appropriate frozen protocol/returned bundle;
an inference prompt file alone is not a complete formal replay bundle.

`formal_worker.assess()` interprets named assertion and related-cover statuses.
A reachable trace is required before implication vacuity classification. An
unreachable precondition supports antecedent vacuity; an unreachable relevant
contrapositive supports consequent vacuity. Both cover families must be present
and fully covered for a targeted nonvacuity label. Missing/time-limited evidence
remains unresolved. Non-implications stay outside these two tests.

Identical complete native checking inputs may share one tool execution, while
every generated assertion occurrence remains in model statistics. Compilation
and extraction errors are retained in the overall assertion denominator. Empty
responses are recorded separately.

## Metrics and cohort scope

- Proof success: P/N, including proved non-implications.
- Conditional vacuity: V/(V+U), excluding non-implications and unresolved vacuity.
- Nonvacuous implication yield: U/N.

Results are design-wise. The original five-design test cohort included
`L1/test_endian`; the ten-design reported comparison excludes it after all adapted
outputs failed elaboration. Its records remain in the original saved comparison.
The additional mixed cohort contains four OpenTitan modules plus FIXME L3/8 and
L3/9. Initial six-OpenTitan source-preparation utilities also remain because the
mixed preparation imports their source/context helpers.
