# Assertion collection and formal labels

This package adapts SANGAM to collect SystemVerilog assertions for the FIXME
SVAGen Text designs. The teacher receives the full specification and saved
target-signal name/description. RTL is used for JasperGold checking and feedback,
not included directly in the teacher assertion-generation prompt.

## Main files

| File | Purpose |
| --- | --- |
| `generate.py` | Nebius client, candidate generation, critic feedback and refinement |
| `prompts.py` | Original SANGAM teacher prompt strings |
| `jasper.py` | SSH-based JasperGold checks and proof/cover interpretation |
| `sva.py` | Assertion extraction and insertion into the target RTL module |
| `signal_mapping.py` | Saved signal descriptions and specification-to-signal mapping |
| `cli.py` | Case discovery, collection, saved records and output manifests |
| `resume.py` | Recovery from transient API, SSH or licence failures |

`SangamGenerator.run_signal()` handles one target signal. It generates an initial
answer, refines candidates using critic/formal feedback, combines candidates,
repairs syntax/elaboration errors, and returns selected assertions plus a trace.
Syntax-valid candidates are not automatically proved or nonvacuous positives.

## Setup

Use Python 3.12 and install the root `requirements.txt`. Clone FIXME separately
and use the source revision recorded in the relevant experiment provenance.
Copy `.env.example` to `.env` and supply your own Nebius and JasperGold SSH access.
The configured teacher is `deepseek-ai/DeepSeek-V4-Flash-0731` through Nebius.

```bash
python3 -m fixme_svagen.cli --fixme-root ../FIXME --list-cases
```

A small collection run, requiring live API and licensed formal access:

```bash
python3 -m fixme_svagen.cli \
  --fixme-root ../FIXME --case L1/PIC_8259 \
  --max-signals 1 --iterations 0 --jg-timeout 180
```

Outputs default to `output/fixme_svagen/`. Partial cases are not complete dataset
exports. Existing raw outputs are not part of this code-only GitHub repository.

For proved implications, related covers distinguish unreachable antecedents and
always-true consequents. Missing or inconclusive covers do not prove unreachability.
Non-implications are recorded outside these two checks. The final evaluation
worker additionally requires a reachable formal context before classifying vacuity.
