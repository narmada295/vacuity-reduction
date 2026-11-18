"""Run the frozen v4 full-validation base/SFT comparison sequentially on one GPU."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_inputs(root, protocol):
    for relative, expected in protocol["files"].items():
        if file_hash(root / relative) != expected:
            raise ValueError("Frozen run file changed: " + relative)
    tasks = [json.loads(line) for line in (root / protocol["prompts"]).open() if line.strip()]
    identities = [(t["case_id"], t["target_signal"]) for t in tasks]
    if len(tasks) != protocol["signal_count"] or len(identities) != len(set(identities)):
        raise ValueError("Full run requires exactly one prompt per assigned signal")
    if dict(Counter(t["case_id"] for t in tasks)) != protocol["signals_by_design"]:
        raise ValueError("Full-validation design membership or signal counts changed")
    if any([m["role"] for m in t["messages"]] != ["system", "user"] for t in tasks):
        raise ValueError("Inference inputs contain an unexpected role")
    training = json.loads((root / protocol["training_config"]).read_text())
    for name, expected in protocol["expected_training"].items():
        if training.get(name) != expected:
            raise ValueError("SFT training identity mismatch: " + name)
    return tasks


def validate_outputs(directory, label, tasks, protocol):
    config = json.loads((directory / "generation_config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    expected = {"label": label, "prompt_count": len(tasks),
                "prompt_version": protocol["prompt_version"],
                "prompts_sha256": protocol["files"][protocol["prompts"]],
                "one_shot_system_prompt_sha256": protocol["system_prompt_sha256"],
                "samples_per_prompt": 1, "generation_mode": "one_shot_per_signal"}
    expected.update(protocol["generation"])
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(label + " run configuration mismatch: " + name)
    if summary["status"] != "complete" or summary["saved_attempts"] != len(tasks):
        raise ValueError(label + " generation is incomplete")
    rows = [json.loads(line) for line in (directory / "generations.jsonl").open() if line.strip()]
    if len(rows) != len(tasks) or {r["task_id"] for r in rows} != {t["task_id"] for t in tasks}:
        raise ValueError(label + " output membership differs from assigned tasks")
    assigned = {t["task_id"]: t for t in tasks}
    for row in rows:
        if (row["label"] != label or row["sample"] != 0
                or row["generation_config_fingerprint"] != config["fingerprint"]
                or row["case_id"] != assigned[row["task_id"]]["case_id"]
                or row["target_signal"] != assigned[row["task_id"]]["target_signal"]):
            raise ValueError("Unexpected label, repeated sampling, or mixed generation configuration")
    return rows, config


def run(root, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    tasks = validate_inputs(root, protocol)
    output = root / protocol["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".full_validation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = output / "frozen_protocol.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != protocol:
        raise ValueError("Existing full run uses a different frozen protocol")
    write_json(snapshot, protocol)
    state_path = output / "run_status.json"
    state = {"state": "running", "signals_per_model": len(tasks), "models": {}}
    configurations = {}
    all_rows = {}
    try:
        for label in ("base", "sft"):
            state["current"] = label
            state["models"][label] = {"state": "running"}
            write_json(state_path, state)
            destination = output / label
            command = [sys.executable, "-u", str(root / protocol["generator"]),
                       "--prompts", str(root / protocol["prompts"]), "--label", label,
                       "--training-config", str(root / protocol["training_config"]),
                       "--assertion-cap", str(protocol["generation"]["assertion_cap"]),
                       "--max-new-tokens", str(protocol["generation"]["max_new_tokens"]),
                       "--temperature", str(protocol["generation"]["temperature"]),
                       "--top-p", str(protocol["generation"]["top_p"]),
                       "--seed", str(protocol["generation"]["seed"]),
                       "--out", str(destination)]
            if label == "sft":
                command.extend(["--adapter", str(root / protocol["sft_adapter"])])
            print("START", label, "full validation:", len(tasks), "signals", flush=True)
            logs = root / "logs/eval_one_shot_v4"
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / ("full_" + label + ".log")).open("a", buffering=1) as log:
                process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            if code:
                raise RuntimeError(label + " generation exited " + str(code) + "; saved attempts are preserved")
            rows, config = validate_outputs(destination, label, tasks, protocol)
            all_rows[label] = rows
            configurations[label] = config
            state["models"][label] = {"state": "complete", "saved_attempts": len(rows),
                                      "outcomes": dict(Counter(r["status"] for r in rows))}
            write_json(state_path, state)
            print("DONE", label, state["models"][label], flush=True)
        # Verify actual, not merely requested, equality of settings and effective inputs.
        common = ("model", "model_revision", "chat_template_sha256", "training_config_sha256",
                  "prompts_sha256", "prompt_count", "assertion_cap", "prompt_version",
                  "one_shot_system_prompt_sha256", "samples_per_prompt", "max_new_tokens",
                  "temperature", "top_p", "top_k", "repetition_penalty", "seed",
                  "quantization", "script_sha256", "truncation", "packages")
        for key in common:
            if configurations["base"][key] != configurations["sft"][key]:
                raise ValueError("Base/SFT comparison setting differs: " + key)
        by_task = {label: {r["task_id"]: r for r in rows} for label, rows in all_rows.items()}
        for task_id in by_task["base"]:
            for key in ("seed", "prompt_tokens", "effective_messages_sha256"):
                if by_task["base"][task_id][key] != by_task["sft"][task_id][key]:
                    raise ValueError("Effective base/SFT input differs: " + task_id + "/" + key)
        state.update(state="generation_complete", comparison_input_check="PASS",
                     formal_evaluation="pending JasperGold checks; generation does not establish assertion correctness")
        state.pop("current", None)
        write_json(state_path, state)
        print("Both full-validation generation runs complete; inputs/settings match.", flush=True)
    except BaseException as error:
        state.update(state="failed_or_interrupted", error=type(error).__name__ + ": " + str(error))
        write_json(state_path, state)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--protocol", type=Path, default=Path("data/eval_one_shot_v4/full_validation_protocol.json"))
    args = ap.parse_args()
    root = args.root.resolve()
    path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    run(root, path)


if __name__ == "__main__":
    main()
