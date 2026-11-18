"""Verify an exported SFT snapshot against its unchanged saved source evidence."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from experiments.assertion_extraction.extract_assertions import resolve_property
from experiments.assertion_extraction.remove_exact_duplicates import tokenize
from experiments.assertion_extraction.formal_clocking import clock_commands
from experiments.training.prepare_sft import SPLITS, category, chat_record, checker_clocks, load_split
from experiments.training.reassess_l3_3 import assess, read_states


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audited_evidence(case):
    audit = case.get("formal_sft_audit")
    if not audit:
        if case.get("formal_clocking") is not None:
            raise ValueError("Explicit clock context has no SFT audit")
        return None
    if audit.get("version") != 1 or audit["formal_clocking"] != case["formal_clocking"]:
        raise ValueError("SFT audit clock context mismatch")
    for info in audit["files"].values():
        if file_hash(Path(info["path"])) != info["sha256"]:
            raise ValueError("SFT audit evidence changed: " + info["path"])
    assessment_info = audit["files"]["assessment.json"]
    result = json.loads(Path(assessment_info["path"]).read_text())
    request = json.loads(Path(audit["files"]["request.json"]["path"]).read_text())
    states = read_states(Path(audit["files"]["status.tsv"]["path"]))
    if (result["status"] != "complete" or result["trace_status"] != "covered"
            or states.get("SFT_TRACE_EXISTS") != "covered"
            or result["formal_clocking"] != case["formal_clocking"]
            or request["formal_clocking"] != case["formal_clocking"]):
        raise ValueError("SFT audit is incomplete or uses another clock context")
    originals = json.loads(Path(audit["files"]["input_case.json"]["path"]).read_text())
    requested = {item["label"]: item for item in request["properties"]}
    if set(requested) != set(result["properties"]):
        raise ValueError("SFT audit assessment omits requested properties")
    for label, item in result["properties"].items():
        if any(item[key] != requested[label][key] for key in requested[label]):
            raise ValueError("SFT assessment differs from audited request")
        checked = assess(states, label, item["implication"])
        if any(item[key] != value for key, value in checked.items()):
            raise ValueError("SFT assessment differs from raw property/cover statuses")
    return result, originals, assessment_info["sha256"]


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    plan, assignments = load_split(root / "split_plan.json")
    if plan != manifest["split_plan"]:
        raise ValueError("Split plan differs from manifest")
    if manifest["formal_reduction_applied"] is not False:
        raise ValueError("Expected data before formal reduction")
    for name, info in manifest["files"].items():
        if file_hash(root / name) != info["sha256"]:
            raise ValueError("Export file hash mismatch: " + name)
    for stats in manifest["designs"].values():
        for name, digest in [("source_case", "case_sha256"), ("source_signals", "signals_sha256")]:
            if file_hash(Path(stats[name])) != stats[digest]:
                raise ValueError("Source changed since snapshot: " + stats[name])
    excluded = set(manifest["split_plan"].get("excluded", {}))
    counts, kinds, duplicates = Counter(), Counter(), set()
    ids, source_case, source_id, source_map = set(), None, None, None
    clock_logs_checked = 0
    logged_clock_cache = {}
    audited_source_rows_checked = 0
    current_audit = None
    represented_sources = set()
    with (root / "examples.jsonl").open() as canonical:
        chats = {split: (root / (split + ".jsonl")).open() for split in SPLITS}
        try:
            for line in canonical:
                example = json.loads(line)
                cid, split = example["case_id"], example["split"]
                if cid in excluded or assignments.get(cid) != split or example["id"] in ids:
                    raise ValueError("Excluded design, split leakage, or duplicate ID")
                ids.add(example["id"])
                counts[split] += 1
                kinds[example["category"]] += 1
                if source_id != cid:
                    stats = manifest["designs"][cid]
                    source_case = json.loads(Path(stats["source_case"]).read_text())
                    source_map = {s["name"]: s for s in json.loads(Path(stats["source_signals"]).read_text())}
                    source_id = cid
                    current_audit = audited_evidence(source_case)
                provenance = example["provenance"]
                representative = source_case["rows"][provenance["sources"][0]["row_number"] - 1]
                if example["spec"] != representative["spec"] or example["rtl"] != representative["rtl"]:
                    raise ValueError("Specification/RTL differs from source")
                if example["target_signal"] != representative["signals"][0]:
                    raise ValueError("Target signal differs from source generation task")
                if example["signal_info"] != source_map[example["target_signal"]]:
                    raise ValueError("Signal description differs from saved mapping")
                extracted = dict(representative, assertion=provenance["source_assertion"],
                                 support_code=provenance["selected_support_code"])
                _, expression, _, status, _ = resolve_property(extracted)
                if status != "ok" or tokenize(expression) != tokenize(provenance["property_expression"]):
                    raise ValueError("Reduced support changed or failed to resolve the assertion")
                if example["assertion"] != ((provenance["selected_support_code"] + "\n")
                                            if provenance["selected_support_code"] else "") + representative["assertion"]:
                    raise ValueError("Training target differs from original assertion and selected declarations")
                context = json.dumps(example["formal_context"], sort_keys=True)
                dedup_key = (cid, context, tokenize(expression))
                if dedup_key in duplicates:
                    raise ValueError("Duplicate resolved assertion in exported data")
                duplicates.add(dedup_key)
                for reference in provenance["sources"]:
                    source_row = source_case["rows"][reference["row_number"] - 1]
                    if category(source_row) != example["category"] or source_row["assertion"] != reference["assertion"]:
                        raise ValueError("Source selection/proof mismatch")
                    _, original_expression, _, original_status, _ = resolve_property(source_row)
                    if original_status != "ok" or tokenize(original_expression) != tokenize(expression):
                        raise ValueError("Duplicate provenance combines different property bodies")
                    clocking = source_row.get("formal_clocking", source_case.get("formal_clocking"))
                    if clocking != example["formal_context"].get("formal_clocking"):
                        raise ValueError("Source and example have different formal clock contexts")
                    if checker_clocks(source_row, clocking) != example["formal_context"]["checker_clocks"]:
                        raise ValueError("Duplicate provenance combines different clock environments")
                    if current_audit is not None:
                        result, originals, digest = current_audit
                        recheck = source_row.get("sft_formal_recheck", {})
                        if recheck.get("assessment_sha256") != digest:
                            raise ValueError("Source row has no matching SFT audit")
                        assessment = result["properties"][recheck["label"]]
                        number = reference["row_number"]
                        original = originals["rows"][number - 1]
                        if (number not in assessment["source_rows"]
                                or original["assertion"] != source_row["assertion"]
                                or original.get("support_code") != source_row.get("support_code")
                                or category(assessment) != example["category"]
                                or tokenize(assessment["expression"]) != tokenize(expression)
                                or any(source_row[key] != assessment[key]
                                       for key in ("assertion_status", "vacuous", "reason"))
                                or reference.get("sft_formal_recheck") != recheck):
                            raise ValueError("SFT source row differs from its audited assertion/status")
                        audited_source_rows_checked += 1
                    log_path = Path(manifest["source_cases_root"]).parent / source_row["jasper_log"]
                    if log_path not in logged_clock_cache:
                        logged_clock_cache[log_path] = re.findall(
                            r"^\[<embedded>\]\s*%\s*clock\s+([^\n]+)",
                            log_path.read_text(errors="replace"), re.M,
                        )
                    logged_clocks = logged_clock_cache[log_path]
                    expected_clocks = [line.removeprefix("clock ") for line in
                        clock_commands(checker_clocks(source_row, clocking), clocking).splitlines()]
                    if logged_clocks != expected_clocks:
                        raise ValueError("Recorded clock environment differs from saved JasperGold log")
                    clock_logs_checked += 1
                    represented_sources.add((cid, reference["row_number"]))
                if json.loads(next(chats[split])) != chat_record(example):
                    raise ValueError("Chat-format record differs from canonical example")
            for handle in chats.values():
                if next(handle, None) is not None:
                    raise ValueError("Extra chat examples")
        finally:
            for handle in chats.values():
                handle.close()
    if sum(counts.values()) != manifest["totals"]["unique_examples"]:
        raise ValueError("Incorrect exported example count")
    if len(represented_sources) != manifest["totals"]["selected_before_deduplication"]:
        raise ValueError("Selected source rows missing from duplicate provenance")
    for split in SPLITS:
        if counts[split] != manifest["splits"][split]["examples"]:
            raise ValueError("Incorrect split count")
    return {"status": "PASS", "unique_examples": len(ids), "source_assertions_represented": len(represented_sources),
            "split_examples": dict(counts), "categories": dict(kinds),
            "source_integrity": "unchanged", "source_clock_logs_checked": clock_logs_checked,
            "audited_source_rows_checked": audited_source_rows_checked,
            "formal_jobs_run": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    (args.directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
