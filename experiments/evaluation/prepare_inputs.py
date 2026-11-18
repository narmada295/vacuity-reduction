"""Export unique inference prompts without reference responses or outcome labels."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def prepare(source, split, output, per_signal=False):
    if output.exists():
        raise ValueError("Choose a fresh output directory to preserve existing inputs")
    manifest = json.loads((source / "manifest.json").read_text())
    original = source / (split + ".jsonl")
    if sha(original.read_bytes()) != manifest["files"][original.name]["sha256"]:
        raise ValueError("Source dataset hash mismatch")
    grouped = {}
    reference_rows = 0
    with original.open() as handle:
        for line in handle:
            row = json.loads(line)
            if [m["role"] for m in row["messages"]] != ["system", "user", "assistant"]:
                raise ValueError("Unexpected conversation roles")
            messages = row["messages"][:-1]
            user = messages[1]["content"]
            signal = user.split("\n\nTarget signal: ", 1)[1].split("\n", 1)[0]
            identity = json.dumps([row["case_id"], messages], sort_keys=True).encode()
            task_id = row["case_id"].replace("/", "__") + "__" + sha(identity)[:20]
            task = {"task_id": task_id, "case_id": row["case_id"],
                    "target_signal": signal, "messages": messages}
            if task_id not in grouped:
                grouped[task_id] = {"task": task, "source_example_ids": []}
            elif grouped[task_id]["task"] != task:
                raise ValueError("Task identity collision")
            grouped[task_id]["source_example_ids"].append(row["id"])
            reference_rows += 1
    tasks = sorted((g["task"] for g in grouped.values()), key=lambda t: t["task_id"])
    source_contexts = len(tasks)
    if per_signal:
        by_signal = {}
        for task in tasks:
            by_signal.setdefault((task["case_id"], task["target_signal"]), []).append(task)
        # Choose one complete saved prompt per signal, without synthesizing a new
        # clock environment. Prefer the most represented context; break ties by ID.
        tasks = sorted((min(options, key=lambda t: (
            -len(grouped[t["task_id"]]["source_example_ids"]), t["task_id"]))
            for options in by_signal.values()), key=lambda t: t["task_id"])
    assigned = set(manifest["split_plan"][split])
    observed = {t["case_id"] for t in tasks}
    if observed != assigned:
        raise ValueError("Inference inputs do not contain the complete assigned split")
    # A reproducible pipeline pilot: one context per design, no outcome-based selection.
    pilot = [min((t for t in tasks if t["case_id"] == cid),
                 key=lambda t: sha(("pilot42:" + t["task_id"]).encode()))
             for cid in sorted(observed)]
    output.mkdir(parents=True)
    for name, values in (("prompts.jsonl", tasks), ("pilot_prompts.jsonl", pilot)):
        with (output / name).open("w") as handle:
            for task in values:
                handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    # This map is for offline provenance. It contains no reference response text.
    selected_ids = {t["task_id"] for t in tasks}
    with (output / "reference_map.jsonl").open("w") as handle:
        for group in grouped.values():
            task = group["task"]
            handle.write(json.dumps({"task_id": task["task_id"],
                "selected": task["task_id"] in selected_ids,
                "source_example_ids": grouped[task["task_id"]]["source_example_ids"]}) + "\n")
    report = {"split": split, "source": str(original), "source_sha256": sha(original.read_bytes()),
              "source_manifest_sha256": sha((source / "manifest.json").read_bytes()),
              "reference_rows": reference_rows, "source_unique_input_contexts": source_contexts,
              "unique_input_contexts": len(tasks), "one_prompt_per_signal": per_signal,
              "contexts_by_design": dict(Counter(t["case_id"] for t in tasks)),
              "pilot_contexts": len(pilot), "pilot_tasks": [t["task_id"] for t in pilot],
              "input_policy": "Original system and user messages only; no assistant response, outcome label, or category metadata.",
              "grouping": ("One complete saved prompt per design/signal: most represented context, task-ID tie break. All alternative contexts retained in offline reference map."
                           if per_signal else "Identical complete prompts within the same design; distinct formal contexts remain separate."),
              "files": {p.name: sha(p.read_bytes()) for p in output.glob("*.jsonl")}}
    write_json(output / "manifest.json", report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("output/fixme_svagen/sft_before_reduction_v2_complete"))
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-signal", action="store_true", help="Select exactly one recorded prompt context per design/signal")
    args = ap.parse_args()
    print(json.dumps(prepare(args.source, args.split, args.out, args.per_signal), indent=2))


if __name__ == "__main__":
    main()
