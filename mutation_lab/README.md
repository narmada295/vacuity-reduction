# Behavioral RTL mutation experiments

This prototype explores generating RTL faults from the specification and golden
RTL. The target faults involve state transitions, signal interactions and behavior
across several cycles. It is preliminary work, not a completed assertion
fault-detection benchmark.

## Main files

| File | Purpose |
| --- | --- |
| `exploration.py`, `exploration_prompts.py` | Full-context fault planning, generation and review |
| `exploration_checks.py` | Checks review consistency and recorded evidence |
| `scenario_validation.py` | Probe validation and simulation checks |
| `planner.py`, `core.py` | Model calls, mutation proposals and application of edits |
| `campaign.py`, `quality_campaign.py`, `trace_guided.py` | Alternative exploration workflows |
| `assertions.py`, `differential_formal.py` | Golden/mutant assertion checking |
| `cli.py` | Command-line entry points |

The examples are self-contained source inputs:

- `examples/piezo_driver/`: specification, golden RTL, assertion inputs and proposals.
- `examples/tiny_counter/`: small configuration, RTL, probes and oracle examples.

## Setup and inspection

Use the root Python environment and inspect the available commands:

```bash
python3 -m mutation_lab --help
python3 -m unittest discover -s mutation_lab/tests -q
```

Live mutation generation requires your configured model API key. Lint and
simulation stages require Verilator and Icarus Verilog; optional synthesis checks
require Yosys. Differential formal checking requires the configured JasperGold
installation. Large run folders and old model responses are not included.

## Example exploration campaign

Run from the repository root with a fresh output directory:

```bash
python3 -m mutation_lab explore-campaign \
  --spec mutation_lab/examples/piezo_driver/spec.txt \
  --rtl mutation_lab/examples/piezo_driver/rtl/Piezo_Driver.v \
  --reset reset --reset-active 1 \
  --out mutation_lab/runs/piezo_example \
  --count 0 --plans-per-batch 12 \
  --max-attempts 100 --max-planning-calls 80 --max-llm-calls 600
```

This is a live API/simulation campaign, not a static demonstration. The configured
call/attempt limits bound the run; `--count 0` collects candidates until another
budget or stopping condition is reached.

The workflow generates proposed faults, applies individual edits, checks interfaces
and lint, and runs probes on golden and mutated RTL. Plan/RTL reviews remain model
judgments. A simulation difference shows an observed behavioral difference, not by
itself an independently validated specification bug. Records retain accepted,
rejected and unresolved outcomes with the corresponding evidence.

## Assertion-based checking

A direct campaign can take an assertion set:

```bash
python3 -m mutation_lab campaign \
  --spec mutation_lab/examples/piezo_driver/spec.txt \
  --rtl mutation_lab/examples/piezo_driver/rtl/Piezo_Driver.v \
  --assertions mutation_lab/examples/piezo_driver/piezo_driver_final_assertions.txt \
  --out mutation_lab/runs/piezo_assertion_example --count 5
```

An assertion detects a candidate fault when it is proved on golden RTL and has a
counterexample on the mutant under the same verification conditions. Errors,
timeouts and missing evidence are unresolved, not successful detections. Full
fault-detection claims additionally need independently qualified mutants; the
preliminary experiments do not establish that every generated edit is such a fault.
