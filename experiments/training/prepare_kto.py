"""Export unpaired KTO feedback from the saved SFT design split.

No generation, training, or formal jobs are run. Training uses only the original
25 training designs; a separate validation file uses the five validation designs.
Test examples are never read or exported.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile

from experiments.assertion_extraction.extract_assertions import resolve_property
from experiments.assertion_extraction.remove_exact_duplicates import tokenize
from experiments.assertion_extraction.formal_clocking import clock_commands
from experiments.training.prepare_sft import (
    chat_record, checker_clocks, minimal_support, sha256, source_reference,
)
from experiments.training.verify_sft import audited_evidence, file_hash


REASONS = {"antecedent_never_true", "consequent_always_true"}
EQUIVALENT_PROBES = {
    "one_cycle_delay_to_nonoverlap", "one_cycle_rewrite",
    "equivalent_one_cycle_or_boolean_rewrite_with_related_covers",
    "single_cycle_property_or_to_boolean_or",
}
REPORT_ROW = re.compile(
    r"(?m)^\[\d+\]\s+(\S+)\s+"
    r"(proven|cex|covered|unreachable|undetermined|unknown|error|bounded_proven|bounded_unreachable)\b"
)


def feedback_label(row):
    """Desirability includes proved non-implications, outside vacuity scope."""
    if row.get("assertion_status") != "proven":
        return None
    if row.get("vacuous") is None and row.get("reason") == "outside_scope_non_implication":
        return True
    if row.get("vacuous") is False and row.get("reason") == "neither_targeted_vacuity_reason":
        return True
    reasons = set((row.get("reason") or "").split("; "))
    if row.get("vacuous") is True and reasons and reasons <= REASONS:
        return False
    if row.get("vacuous") is True and row.get("reason") == "related_cover_unreachable_in_audited_context":
        return False
    return None


def related_covers(states, target="sangam_assert"):
    families = {}
    for family in ("precondition", "contrapositive"):
        families[family] = {name: state for name, state in states.items()
                            if re.search(r"(?:^|\.)" + re.escape(target) + ":" + family + r"\d+$", name)}
    return families


def covered_label(families):
    if any(state == "unreachable" for family in families.values() for state in family.values()):
        return False
    if all(family and all(state == "covered" for state in family.values())
           for family in families.values()):
        return True
    return None


def evidence_for(row, case, audit, root, cache):
    """Use named reports or the corrected-clock audit; never infer from absence.

    For this first strict export, reviewed one-cycle equivalent probes may supply
    missing related covers. Other diagnostic/review labels are deferred, rather
    than promoting their free-text explanations to training evidence.
    """
    non_implication = row.get("reason") == "outside_scope_non_implication" and row.get("vacuous") is None
    clocking = row.get("formal_clocking", case.get("formal_clocking"))
    if clocking is not None:
        recheck = row.get("sft_formal_recheck")
        if audit is None or not recheck:
            return None, "explicit_clock_context_without_row_recheck"
        assessment, originals, digest = audit
        if recheck.get("assessment_sha256") != digest:
            raise ValueError("Clock audit assessment differs from row evidence")
        item = assessment["properties"][recheck["label"]]
        if any(item[key] != row[key] for key in ("assertion_status", "vacuous", "reason")):
            raise ValueError("Clock audit label differs from source row")
        if tokenize(item["expression"]) != tokenize(resolve_property(row)[1]):
            raise ValueError("Clock audit expression differs from source row")
        if item["assertion_status"] != "proven":
            return None, "audit_assertion_not_proven"
        evidence = {"method": "corrected_clock_audit", "assessment_sha256": digest,
                    "audit_label": recheck["label"], "related_covers": item["related_covers"]}
        if non_implication:
            if item["implication"]:
                raise ValueError("Non-implication differs from audited scope")
            evidence["vacuity_check"] = "outside_scope_non_implication"
            return evidence, True
        return evidence, covered_label(item["related_covers"])

    def read_report(relative):
        if not relative:
            return None
        path = root / relative
        if path not in cache:
            if not path.is_file():
                cache[path] = None
            else:
                raw = path.read_bytes()
                text = raw.decode(errors="replace")
                cache[path] = {"path": str(path), "sha256": sha256(raw),
                               "states": dict(REPORT_ROW.findall(text)),
                               "clock_declarations": re.findall(r"^\[<embedded>\]\s*%\s*clock\s+([^\n]+)", text, re.M)}
        return cache[path]

    report = read_report(row.get("jasper_log"))
    if not report:
        return None, "missing_original_log"
    main = [value for name, value in report["states"].items()
            if name == "sangam_assert" or name.endswith(".sangam_assert")]
    if main != ["proven"]:
        return None, "original_named_proof_not_confirmed"
    expected_clocks = [line.removeprefix("clock ") for line in
                       clock_commands(checker_clocks(row), None).splitlines()]
    if report["clock_declarations"] != expected_clocks:
        return None, "original_log_clock_context_mismatch"
    if non_implication:
        return {"method": "original_named_proof", "log": report["path"],
                "log_sha256": report["sha256"],
                "vacuity_check": "outside_scope_non_implication"}, True
    families = related_covers(report["states"])
    label = covered_label(families)
    evidence = {"method": "original_related_covers", "log": report["path"],
                "log_sha256": report["sha256"], "related_covers": families}
    if label is not None:
        return evidence, label
    if row.get("followup_probe_method") in EQUIVALENT_PROBES:
        followup = read_report(row.get("followup_jasper_log") or row.get("followup_probe_jasper_log"))
        if followup:
            main = [value for name, value in followup["states"].items()
                    if name == "sangam_assert" or name.endswith(".sangam_assert")]
            families = related_covers(followup["states"])
            label = covered_label(families)
            if main == ["proven"] and label is not None:
                evidence.update(method="saved_equivalent_probe_related_covers",
                                original_related_covers=evidence.pop("related_covers"),
                                probe_method=row["followup_probe_method"],
                                probe_log=followup["path"], probe_log_sha256=followup["sha256"],
                                related_covers=families)
                return evidence, label
    return None, "related_cover_evidence_incomplete_or_deferred_diagnostic"


def kto_record(example):
    chat = chat_record(example)
    # Only these three fields are consumed by KTOTrainer. Provenance lives apart.
    return {"prompt": chat["messages"][:2], "completion": chat["messages"][2:],
            "label": example["label"]}


def collect_case(case, signals, source_path, audit, log_root, log_cache, exclusions=None):
    cid = case["case_id"]
    signal_map = {item["name"]: item for item in signals}
    if len(signal_map) != len(signals):
        raise ValueError("Duplicate saved signal names")
    groups, skipped = {}, Counter()
    for number, row in enumerate(case["rows"], 1):
        def skip(reason):
            skipped[reason] += 1
            if exclusions is not None:
                exclusions.append({"case_id": cid, "row_number": number, "exclusion": reason,
                    "source_case": source_path, "assertion": row["assertion"],
                    "assertion_status": row.get("assertion_status"), "vacuous": row.get("vacuous"),
                    "reason": row.get("reason"), "jasper_log": row.get("jasper_log")})
        label = feedback_label(row)
        if label is None:
            skip("not_proven_eligible_property")
            continue
        _, expression, _, status, error = resolve_property(row)
        if status != "ok":
            raise ValueError(f"{cid}:{number}: {status}: {error}")
        has_implication = any(t in ("|->", "|=>") for t in tokenize(expression))
        non_implication = row.get("reason") == "outside_scope_non_implication"
        if has_implication == non_implication:
            skip("property_scope_conflicts_with_saved_category")
            continue
        evidence, observed = evidence_for(row, case, audit, log_root, log_cache)
        if evidence is None:
            skip(observed)
            continue
        if observed is not label:
            skip("saved_label_conflicts_with_related_covers")
            continue
        if row["case_id"] != cid:
            raise ValueError("Mismatched case identity")
        identity = (sha256(row["spec"]), sha256(row["rtl"]))
        if identity != (row["spec_sha256"], row["rtl_sha256"]):
            raise ValueError("Specification/RTL hash mismatch")
        origins = row.get("signals")
        if not origins or any(name not in signal_map for name in origins):
            raise ValueError("Missing saved origin signal information")
        target = origins[0]
        if not signal_map[target].get("information", "").strip():
            raise ValueError("Empty saved signal description")
        clocking = row.get("formal_clocking", case.get("formal_clocking"))
        context = {"clock": row.get("clock"), "checker_clocks": checker_clocks(row, clocking),
                   "reset_signal": row.get("reset"), "formal_reset_policy": row["formal_reset_policy"],
                   "formal_clocking": clocking}
        key = (identity, json.dumps(context, sort_keys=True), tokenize(expression))
        reference = source_reference(row, number, expression, clocking)
        if clocking is not None:
            assessment, originals, digest = audit
            item = assessment["properties"][row["sft_formal_recheck"]["label"]]
            original = originals["rows"][number - 1]
            if (number not in item["source_rows"] or row["assertion"] != original["assertion"]
                    or row.get("support_code") != original.get("support_code")):
                raise ValueError("Audited row differs from original assertion/support")
        reference["kto_evidence"] = evidence
        # Keep follow-up information as provenance, never as model input.
        reference["followup"] = {k: v for k, v in row.items() if k.startswith("followup_")}
        if key in groups:
            group = groups[key]
            group["provenance"]["sources"].append(reference)
            if group["label"] is not label:
                group["conflicting_labels"] = True
            continue
        support = minimal_support(row)
        groups[key] = {
            "id": cid.replace("/", "__") + "__" + sha256(json.dumps([cid, key], sort_keys=True))[:20],
            "case_id": cid, "target_signal": target, "signal_info": signal_map[target],
            "spec": row["spec"], "rtl": row["rtl"],
            "assertion": (support + "\n" if support else "") + row["assertion"],
            "label": label, "category": ("proven_non_implication" if non_implication else
                "proven_non_vacuous_implication" if label else "proven_vacuous_implication"),
            "assertion_status": "proven", "vacuous": row.get("vacuous"),
            "formal_context": context, "reason": row["reason"],
            "provenance": {"source_case": source_path, "property_expression": expression,
                           "selected_support_code": support, "sources": [reference]},
        }
    conflicts = [example for example in groups.values() if example.get("conflicting_labels")]
    examples = [example for example in groups.values() if not example.get("conflicting_labels")]
    return examples, {"source_rows": len(case["rows"]), "excluded_rows": dict(skipped),
                      "quarantined_conflicting_groups": len(conflicts),
                      "selected_before_deduplication": sum(len(e["provenance"]["sources"]) for e in examples),
                      "unique_examples": len(examples),
                      "categories": dict(Counter(e["category"] for e in examples)),
                      "desirable": sum(e["label"] for e in examples),
                      "undesirable": sum(not e["label"] for e in examples)}, conflicts


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        if file_hash(output / name) != info["sha256"]:
            raise ValueError("KTO dataset file changed: " + name)
    for info in manifest["sources"].values():
        for path_key, hash_key in (("source_case", "case_sha256"), ("source_signals", "signals_sha256")):
            if file_hash(Path(info[path_key])) != info[hash_key]:
                raise ValueError("KTO source file changed: " + info[path_key])
    for path, digest in manifest["evidence_logs"].items():
        if file_hash(Path(path)) != digest:
            raise ValueError("KTO evidence log changed: " + path)
    for info in manifest["sources"].values():
        case = json.loads(Path(info["source_case"]).read_text())
        if case.get("formal_sft_audit"):
            audited_evidence(case)
    plan = manifest["split_plan"]
    seen, counts = set(), {s: Counter() for s in ("train", "validation")}
    categories = {s: Counter() for s in counts}
    by_design = {s: set() for s in counts}
    with ExitStack() as stack:
        handles = {s: stack.enter_context((output / (s + ".jsonl")).open()) for s in counts}
        for line in (output / "examples.jsonl").open():
            example = json.loads(line)
            split = example["split"]
            if split not in counts or example["case_id"] not in plan[split]:
                raise ValueError("KTO example crosses assigned design split")
            if example["id"] in seen:
                raise ValueError("Duplicate KTO example ID")
            seen.add(example["id"])
            kind = example["category"]
            if kind == "proven_non_implication":
                if (example["label"] is not True or example.get("vacuous") is not None
                        or example["reason"] != "outside_scope_non_implication"):
                    raise ValueError("Non-implication must be desirable and outside vacuity scope")
            elif kind in ("proven_non_vacuous_implication", "proven_vacuous_implication"):
                if (example["label"] is not (kind == "proven_non_vacuous_implication")
                        or example.get("vacuous") is not (kind == "proven_vacuous_implication")):
                    raise ValueError("Implication desirability conflicts with vacuity label")
            else:
                raise ValueError("Unknown KTO category")
            if example.get("assertion_status") != "proven":
                raise ValueError("KTO source property is not proven")
            row = json.loads(next(handles[split]))
            if set(row) != {"prompt", "completion", "label"} or type(row["label"]) is not bool:
                raise ValueError("Invalid KTO trainer schema")
            if row != kto_record(example):
                raise ValueError("KTO response or input differs from canonical example")
            if [m["role"] for m in row["prompt"]] != ["system", "user"] or [m["role"] for m in row["completion"]] != ["assistant"]:
                raise ValueError("Reference response leaked into KTO prompt")
            by_design[split].add(example["case_id"])
            counts[split]["desirable" if row["label"] else "undesirable"] += 1
            categories[split][kind] += 1
        if any(next(handle, None) is not None for handle in handles.values()):
            raise ValueError("Extra KTO trainer rows")
    for split in counts:
        if dict(counts[split]) != manifest["splits"][split]["labels"]:
            raise ValueError("KTO label counts differ from manifest")
        if by_design[split] != set(plan[split]):
            raise ValueError("Missing design in KTO split")
        if dict(categories[split]) != manifest["splits"][split]["categories"]:
            raise ValueError("KTO category counts differ from manifest")
    return {"status": "PASS", "examples": len(seen), "splits": {s: dict(c) for s, c in counts.items()},
            "categories": {s: dict(c) for s, c in categories.items()},
            "test_examples_exported": 0, "formal_jobs_run": 0, "model_calls": 0}


def export(sft_dir, output):
    if output.exists():
        raise ValueError("Output exists; preserve the snapshot and choose a fresh --out")
    sft_manifest_path = sft_dir / "manifest.json"
    sft = json.loads(sft_manifest_path.read_text())
    plan = sft["split_plan"]
    assigned = [cid for split in ("train", "validation", "test") for cid in plan[split]]
    if len(assigned) != len(set(assigned)) or len(plan["train"]) != 25 or len(plan["validation"]) != 5:
        raise ValueError("Expected the saved, disjoint 25/5/5 SFT design split")
    for split in ("train", "validation"):
        if file_hash(sft_dir / (split + ".jsonl")) != sft["files"][split + ".jsonl"]["sha256"]:
            raise ValueError("Saved SFT chat file changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".kto_export_", dir=output.parent))
    manifest = {"version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": file_hash(Path(__file__)), "sft_manifest": str(sft_manifest_path),
                "sft_manifest_sha256": file_hash(sft_manifest_path), "split_plan": plan,
                "sources": {}, "splits": {}, "formal_reduction_applied": False,
                "selection": "Desirable: proven non-vacuous implications with conclusive related-cover evidence plus proven non-implications. Undesirable: proven vacuous implications. Corrected clock audits take precedence.",
                "input_policy": "Exact SFT system/user format; one original assertion plus required support per completion; labels and reasons are metadata only.",
                "deduplication": "Resolved property tokens within design, spec/RTL hashes, and formal clock/reset context; original code retained; conflicting groups quarantined.",
                "sampling": "All eligible unique examples; no class balancing, oversampling, or formal reduction applied.",
                "quality_scope": "Binary labels express desirability under the selected proof/vacuity policy. Non-implications remain outside vacuity scope. Proof does not guarantee specification relevance.",
                "test_examples_exported": 0}
    cache = {}
    try:
        with ExitStack() as stack:
            examples_out = stack.enter_context((temporary / "examples.jsonl").open("w"))
            conflicts_out = stack.enter_context((temporary / "quarantined.jsonl").open("w"))
            exclusions_out = stack.enter_context((temporary / "excluded_rows.jsonl").open("w"))
            for split in ("train", "validation"):
                chat_out = stack.enter_context((temporary / (split + ".jsonl")).open("w"))
                labels, categories = Counter(), Counter()
                for cid in plan[split]:
                    stats = sft["designs"][cid]
                    source, signal_path = Path(stats["source_case"]), Path(stats["source_signals"])
                    if file_hash(source) != stats["case_sha256"] or file_hash(signal_path) != stats["signals_sha256"]:
                        raise ValueError("SFT source snapshot changed: " + cid)
                    case = json.loads(source.read_text())
                    if case.get("status") != "done":
                        raise ValueError("Source case is not complete")
                    audit = audited_evidence(case)
                    exclusions = []
                    examples, counts, conflicts = collect_case(case, json.loads(signal_path.read_text()),
                        str(source), audit, Path(sft["source_cases_root"]).parent, cache, exclusions)
                    for excluded in exclusions:
                        excluded["split"] = split
                        exclusions_out.write(json.dumps(excluded, ensure_ascii=False) + "\n")
                    if not examples:
                        raise ValueError("Design has no usable KTO examples: " + cid)
                    counts.update(split=split, source_case=str(source), case_sha256=file_hash(source),
                                  source_signals=str(signal_path), signals_sha256=file_hash(signal_path))
                    manifest["sources"][cid] = counts
                    for example in examples:
                        example["split"] = split
                        examples_out.write(json.dumps(example, ensure_ascii=False) + "\n")
                        chat_out.write(json.dumps(kto_record(example), ensure_ascii=False) + "\n")
                        labels["desirable" if example["label"] else "undesirable"] += 1
                        categories[example["category"]] += 1
                    for example in conflicts:
                        conflicts_out.write(json.dumps(example, ensure_ascii=False) + "\n")
                if not labels["desirable"] or not labels["undesirable"]:
                    raise ValueError("KTO split needs both feedback classes")
                manifest["splits"][split] = {"designs": len(plan[split]), "examples": sum(labels.values()),
                    "labels": dict(labels), "desirable_to_undesirable_ratio": labels["desirable"] / labels["undesirable"]}
                manifest["splits"][split]["categories"] = dict(categories)
        manifest["evidence_logs"] = {str(path): info["sha256"] for path, info in cache.items() if info}
        manifest["files"] = {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
                             for p in temporary.glob("*.jsonl")}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        result = verify(temporary)
        (temporary / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-dir", type=Path, default=Path("output/fixme_svagen/sft_before_reduction_v2_complete"))
    parser.add_argument("--out", type=Path, default=Path("output/fixme_svagen/kto_before_reduction_v2"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify(args.out)
    else:
        manifest = export(args.sft_dir, args.out)
        result = {"output": str(args.out), "splits": manifest["splits"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
