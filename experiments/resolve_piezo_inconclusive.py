"""Resolve only Piezo_Driver's 18 missing related-cover vacuity labels.

Writes labels to the existing case.json and dataset.jsonl. Original assertion
logs and statuses remain intact; follow-up logs live beside the Piezo case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from fixme_svagen.jasper import JasperRunner, parse_log
from fixme_svagen.sva import Assertion


CASE_ID = "L1/Piezo_Driver"
MISSING_REASON = "related_cover_inconclusive_or_unsupported"
COMPOUND_ORIGINAL = "assert property (reset_holds_state);"
COMPOUND_PROBE = (
    "assert property (@(posedge clock) reset |=> "
    "((reset && (count == 0) && (compare == 0) && (enabled == 0) "
    "&& (Piezo == 0) && (Ack == 0)) || (!reset)));"
)


def make_probe(row: dict) -> tuple[str, str]:
    original = row["assertion"]
    if original == COMPOUND_ORIGINAL:
        definition = re.search(
            r"\bproperty\s+reset_holds_state\s*;(.*?)\bendproperty\b",
            row["support_code"], re.S,
        )
        if not definition:
            raise ValueError("reset_holds_state definition missing")
        body = re.sub(r"\s+", "", definition.group(1))
        expected = (
            "@(posedgeclock)reset|=>(reset&&(count==0)&&(compare==0)"
            "&&(enabled==0)&&(Piezo==0)&&(Ack==0))or(!reset);"
        )
        if body != expected:
            raise ValueError("reset_holds_state changed; inspect its SVA grouping")
        return COMPOUND_PROBE, "single_cycle_property_or_to_boolean_or"

    match = re.fullmatch(
        r"assert property \(@\(posedge clock\) (?P<ante>.+?) \|-> ##1 (?P<cons>.+)\);",
        original, re.S,
    )
    if not match or "|->" in match["cons"] or "|=>" in match["cons"]:
        raise ValueError(f"Unsupported assertion form: {original}")
    return original.replace("|-> ##1", "|=>", 1), "one_cycle_delay_to_nonoverlap"


def atomic_text(path: Path, data: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(data)
    temp.replace(path)


def classify(result) -> tuple[bool | None, str]:
    if result.assertion_status != "proven" or result.precondition_status != "covered":
        return None, MISSING_REASON
    if result.contrapositive_status == "covered":
        return False, "neither_targeted_vacuity_reason"
    if result.contrapositive_status == "unreachable":
        return True, "consequent_always_true"
    return None, MISSING_REASON


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path,
                        default=Path("output/fixme_svagen/cases/L1/Piezo_Driver/case.json"))
    parser.add_argument("--dataset", type=Path, default=Path("output/fixme_svagen/dataset.jsonl"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--saved-results", type=Path,
                        help="Reuse previously run proof logs when migrating existing results")
    args = parser.parse_args()

    case = json.loads(args.case.read_text())
    if case.get("case_id") != CASE_ID or case.get("status") != "done":
        raise ValueError("Expected the completed Piezo_Driver case")
    output_root = args.case.parents[3]
    log_dir = args.case.parent / "final_logs_reset_none"
    log_dir.mkdir(exist_ok=True)
    selected = [row for row in case["rows"] if row.get("reason") == MISSING_REASON
                or row.get("original_reason") == MISSING_REASON]
    if len(selected) != 18 or len({row["assertion"] for row in selected}) != 18:
        raise ValueError(f"Expected exactly 18 target assertions, got {len(selected)}")
    for row in selected:
        if (row.get("assertion_status") != "proven"
                or row.get("precondition_status") != "covered"
                or row.get("contrapositive_status") is not None):
            raise ValueError("Original assertion evidence changed")

    saved = None
    if args.saved_results:
        saved_rows = [json.loads(line) for line in args.saved_results.read_text().splitlines()]
        saved = {item["original_assertion"]: item for item in saved_rows}
        if len(saved) != 18 or set(saved) != {row["assertion"] for row in selected}:
            raise ValueError("Saved results do not match the 18 target assertions")

    updated = {}
    runner = None
    try:
        for row in selected:
            probe, method = make_probe(row)
            digest = hashlib.sha256(row["assertion"].encode()).hexdigest()[:20]
            log_path = log_dir / f"followup_{digest}.log"
            if saved is not None:
                item = saved[row["assertion"]]
                if item["probe_assertion"] != probe or item["probe_method"] != method:
                    raise ValueError("Saved probe differs from the expected equivalent assertion")
                shutil.copyfile(item["followup_jasper_log"], log_path)
            elif not log_path.exists():
                if runner is None:
                    runner = JasperRunner(args.env, timeout=args.timeout)
                result = runner.check(
                    row["rtl"], row["top"], Assertion(probe, row["support_code"], True),
                    row["clock"], None,
                )
                log_path.write_text(result.log)

            result = parse_log(log_path.read_text(errors="replace"), 0)
            vacuous, reason = classify(result)
            revised = dict(row)
            revised.setdefault("original_vacuous", row["vacuous"])
            revised.setdefault("original_reason", row["reason"])
            revised["followup_probe_assertion"] = probe
            revised["followup_probe_method"] = method
            revised["followup_jasper_log"] = str(log_path.relative_to(output_root))
            revised["followup_assertion_status"] = result.assertion_status
            revised["followup_precondition_status"] = result.precondition_status
            revised["followup_contrapositive_status"] = result.contrapositive_status
            if vacuous is not None:
                revised["vacuous"] = vacuous
                revised["reason"] = reason
            updated[row["assertion"]] = revised
    finally:
        if runner is not None:
            runner.close()

    # Validate the dataset before writing either file. Keep every unrelated
    # JSONL line byte-for-byte unchanged.
    new_lines = []
    matched = set()
    with args.dataset.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("case_id") == CASE_ID and row.get("assertion") in updated:
                if (row.get("assertion_status") != "proven"
                        or row.get("precondition_status") != "covered"
                        or row.get("contrapositive_status") is not None
                        or row.get("reason") not in (MISSING_REASON,
                                                     updated[row["assertion"]]["reason"])):
                    raise ValueError("Dataset evidence differs from the saved Piezo case")
                line = json.dumps(updated[row["assertion"]], ensure_ascii=False) + "\n"
                matched.add(row["assertion"])
            new_lines.append(line)
    if matched != set(updated):
        raise ValueError("The dataset lacks one or more target assertions")

    case["rows"] = [updated.get(row["assertion"], row) for row in case["rows"]]
    atomic_text(args.case, json.dumps(case, indent=2, ensure_ascii=False) + "\n")
    atomic_text(args.dataset, "".join(new_lines))
    counts = Counter(str(row["vacuous"]) for row in updated.values())
    print(f"Updated {len(updated)} Piezo assertions in {args.case} and {args.dataset}")
    print(f"Follow-up labels: {dict(counts)}")
    print(f"Follow-up logs: {log_dir}")


if __name__ == "__main__":
    main()
