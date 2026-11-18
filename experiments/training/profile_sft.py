"""Standalone remote preflight: verify SFT inputs and measure untruncated Qwen tokens.

Requires the remote training environment's transformers and huggingface_hub.
Downloads the tokenizer only; no GPU allocation or model weights are used.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import tempfile


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_designs(manifest, split, observed):
    """Check full split membership before sampling smoke examples or loading weights."""
    plan = manifest["split_plan"]
    assigned = {}
    quotas = plan.get("design_quotas")
    if quotas is not None and set(quotas) != {"train", "validation", "test"}:
        raise ValueError("Incomplete design quotas")
    for name in ("train", "validation", "test"):
        for case_id in plan[name]:
            if case_id in assigned or case_id in plan.get("excluded", {}):
                raise ValueError("Overlapping or excluded design: " + case_id)
            assigned[case_id] = name
        if quotas is not None:
            required = quotas[name]
            if (set(required) != {"L1", "L2", "L3"}
                    or any(type(n) is not int or n < 0 for n in required.values())):
                raise ValueError("Invalid design quotas: " + name)
            counts = Counter(cid.split("/")[0] for cid in plan[name])
            if any(counts[level] != n for level, n in required.items()):
                raise ValueError("Design quota mismatch: " + name)
    pending = {item["case_id"] for item in manifest.get("pending", [])}
    if not pending.issubset(plan.get("pending_allowed", [])):
        raise ValueError("Unexpected pending design")
    if quotas is not None and any(assigned.get(cid) != "test" for cid in pending):
        raise ValueError("Only test designs may be pending with design quotas")
    expected = set(plan[split]) - pending
    if set(observed) != expected:
        raise ValueError("Dataset design membership differs from split plan: " + split)
    if len(expected) != manifest["splits"][split]["designs"]:
        raise ValueError("Dataset design count differs from manifest: " + split)
    return dict(Counter(cid.split("/")[0] for cid in sorted(expected)))


def statistics(lengths):
    ordered = sorted(lengths)
    if not ordered:
        raise ValueError("No token lengths to summarize")

    def percentile(fraction):
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]

    return {
        "examples": len(ordered), "min": ordered[0],
        "p50": percentile(.50), "p95": percentile(.95), "p99": percentile(.99),
        "max": ordered[-1], "total_tokens": sum(ordered),
        "examples_exceeding": {
            str(limit): sum(length > limit for length in ordered)
            for limit in (4096, 8192, 12288, 16384, 32768)
        },
    }


def measure(tokenizer, record):
    messages = record["messages"]
    if [item["role"] for item in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected one system/user/assistant conversation: " + record["id"])
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        truncation=False, return_dict=False,
    )
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True,
        truncation=False, return_dict=False,
    )
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat-template prompt is not a prefix of the full example: " + record["id"])
    if len(full) <= len(prompt):
        raise ValueError("No assistant tokens: " + record["id"])
    return {"id": record["id"], "case_id": record["case_id"],
            "total_tokens": len(full), "prompt_tokens": len(prompt),
            "assistant_tokens": len(full) - len(prompt)}


def profile(data_dir, output, model, revision=None):
    if output.exists():
        raise ValueError("Report directory already exists; choose a fresh --out")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for split in ("train", "validation"):
        path = data_dir / (split + ".jsonl")
        if file_hash(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError("Dataset checksum mismatch: " + str(path))

    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    # Pin the tokenizer/model revision so later model loading can use the same one.
    resolved_revision = HfApi().model_info(model, revision=revision).sha
    print("Loading tokenizer:", model, resolved_revision, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=resolved_revision)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": model, "model_revision": resolved_revision,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "truncation": False, "model_weights_downloaded": False,
        "splits": {}, "by_design": {}, "longest": {}, "source_files": {},
    }
    with tempfile.TemporaryDirectory(prefix=".token_profile_", dir=output.parent) as temporary:
        temporary = Path(temporary)
        with (temporary / "lengths.jsonl").open("w") as handle:
            for split in ("train", "validation"):
                lengths = []
                completion_lengths = []
                designs = defaultdict(list)
                longest = []
                source = data_dir / (split + ".jsonl")
                report["source_files"][source.name] = file_hash(source)
                allowed = set(manifest["split_plan"][split])
                for line_number, line in enumerate(source.open(), 1):
                    record = json.loads(line)
                    if record["case_id"] not in allowed or record["case_id"] in manifest["split_plan"].get("excluded", {}):
                        raise ValueError("Design excluded or assigned to another split")
                    result = dict(measure(tokenizer, record), split=split, line_number=line_number)
                    lengths.append(result["total_tokens"])
                    completion_lengths.append(result["assistant_tokens"])
                    designs[result["case_id"]].append(result["total_tokens"])
                    handle.write(json.dumps(result) + "\n")
                    longest.append(result)
                    longest = sorted(longest, key=lambda r: (-r["total_tokens"], r["line_number"]))[:5]
                    if line_number % 250 == 0:
                        print(split + ":", line_number, "examples profiled", flush=True)
                if len(lengths) != manifest["splits"][split]["examples"]:
                    raise ValueError("Dataset row count differs from manifest")
                validate_designs(manifest, split, designs)
                report["splits"][split] = statistics(lengths)
                report["splits"][split]["assistant_token_lengths"] = statistics(completion_lengths)
                report["longest"][split] = longest
                report["by_design"][split] = {name: statistics(values) for name, values in designs.items()}
                if file_hash(source) != report["source_files"][source.name]:
                    raise ValueError("Dataset changed while profiling")
        maximum = max(stats["max"] for stats in report["splits"].values())
        report["minimum_sequence_length_to_keep_all_examples"] = maximum
        report["length_rounded_up_to_256"] = math.ceil(maximum / 256) * 256
        (temporary / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.rename(output)
    print("\nTOKEN LENGTH SUMMARY", flush=True)
    print(json.dumps({key: report[key] for key in (
        "model", "model_revision", "splits", "minimum_sequence_length_to_keep_all_examples",
        "length_rounded_up_to_256", "longest",
    )}, indent=2), flush=True)
    print("Reports:", output, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("logs/token_lengths"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    profile(args.data_dir, args.out, args.model, args.revision)


if __name__ == "__main__":
    main()
