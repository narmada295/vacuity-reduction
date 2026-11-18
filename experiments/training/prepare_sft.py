"""Export proof-labeled, exactly deduplicated SFT examples before formal reduction.

No model requests or formal jobs are run. Original cases and signals are read only.
Run from the repository root: python3 -m experiments.training.prepare_sft
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from experiments.assertion_extraction.extract_assertions import (
    DECL_RE, remove_sv_comments, resolve_property,
)
from experiments.assertion_extraction.remove_exact_duplicates import tokenize
from experiments.assertion_extraction.formal_clocking import clock_commands


SPLITS = ("train", "validation", "test")
SYSTEM_PROMPT = (
    "You are a SystemVerilog verification engineer. Generate one specification-grounded "
    "assertion for the target signal using the supplied specification and RTL. "
    "Aim for correct, meaningful checks with reachable triggering conditions; avoid "
    "trivial consequents. Non-implication assertions are allowed when appropriate. "
    "Use existing design signals and preserve the stated clock and reset semantics. "
    "Return only a systemverilog code block containing one assert property statement "
    "and any declarations required by that assertion."
)
IDENTIFIER = re.compile(r"[A-Za-z_$][\w$]*")


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def identifiers(text: str) -> set[str]:
    text = remove_sv_comments(text)
    text = re.sub(r'"(?:\\.|[^"\\])*"', ' ', text)
    return set(IDENTIFIER.findall(text))


def category(row: dict) -> str | None:
    if row.get("assertion_status") != "proven":
        return None
    if row.get("reason") == "outside_scope_non_implication" and row.get("vacuous") is None:
        return "proven_non_implication"
    if row.get("vacuous") is False and row.get("reason") == "neither_targeted_vacuity_reason":
        return "proven_non_vacuous_implication"
    return None


def checker_clocks(row: dict, context: dict | None = None) -> list[str]:
    """Reproduce JasperRunner.check's clock declarations for the saved source.

    Include the original support, not the reduced training support: unused
    property definitions can introduce clocks in the original runner.
    """
    if context is not None:
        clock_commands([], context)
        return [declaration["signal"] for declaration in context["declarations"]]
    clocks = list(dict.fromkeys(re.findall(
        r"@\s*\(\s*posedge\s+([A-Za-z_][A-Za-z_0-9]*)",
        (row.get("support_code") or "") + "\n" + row["assertion"],
    )))
    return clocks or ([row["clock"]] if row.get("clock") else [])


def minimal_support(row: dict) -> str:
    """Keep original property/constant declarations transitively used by this SVA.

    Reject unknown support constructs rather than export a potentially incomplete
    training target. The current eligible pool contains properties and localparams.
    This is a deliberately bounded extractor, not a general SystemVerilog parser.
    """
    support = row.get("support_code") or ""
    declarations = []
    remainder = list(support)
    for match in DECL_RE.finditer(support):
        declarations.append((match.start(), match.group(1), match.group(0)))
        remainder[match.start():match.end()] = " " * (match.end() - match.start())
    residual = "".join(remainder)
    for match in re.finditer(r"\b(?:localparam|parameter)\b[^;]*;", residual):
        declaration = match.group(0)
        # Only a single declarator is supported; $bits/$clog2 expressions are fine.
        left, separator, _ = declaration.partition("=")
        names = IDENTIFIER.findall(left)
        if not separator or not names or "," in left:
            raise ValueError("Unsupported support parameter declaration")
        name = names[-1]
        declarations.append((match.start(), name, declaration))
        remainder[match.start():match.end()] = " " * (match.end() - match.start())
    if remove_sv_comments("".join(remainder)).strip():
        raise ValueError("Unsupported support constructs; manual review required")
    by_name = {}
    for _, name, text in declarations:
        if name in by_name and tokenize(by_name[name]) != tokenize(text):
            raise ValueError("Conflicting support declarations: " + name)
        by_name[name] = text
    needed = set()
    pending = identifiers(row["assertion"]) & by_name.keys()
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        needed.add(name)
        pending.update((identifiers(by_name[name]) & by_name.keys()) - needed)
    result = []
    emitted = set()
    for _, name, text in sorted(declarations):
        if name in needed and name not in emitted:
            result.append(text)
            emitted.add(name)
    return "\n".join(result)


def load_split(path: Path) -> tuple[dict, dict[str, str]]:
    plan = json.loads(path.read_text())
    assignments = {}
    for split in SPLITS:
        for case_id in plan[split]:
            if not re.fullmatch(r"L[123]/[A-Za-z_0-9]+", case_id):
                raise ValueError("Invalid case ID: " + case_id)
            if case_id in assignments or case_id in plan.get("excluded", {}):
                raise ValueError("Overlapping/excluded split assignment: " + case_id)
            assignments[case_id] = split
    if not set(plan.get("pending_allowed", [])).issubset(assignments):
        raise ValueError("Pending designs must have an assigned split")
    if not set(plan.get("source_overrides", {})).issubset(assignments):
        raise ValueError("Source overrides must have an assigned split")
    quotas = plan.get("design_quotas")
    if quotas is not None:
        if set(quotas) != set(SPLITS):
            raise ValueError("Design quotas must specify train, validation, and test")
        for split in SPLITS:
            required = quotas[split]
            if not isinstance(required, dict) or set(required) != {"L1", "L2", "L3"}:
                raise ValueError("Design quotas must specify L1, L2, and L3: " + split)
            if any(type(count) is not int or count < 0 for count in required.values()):
                raise ValueError("Design quotas must be nonnegative integers: " + split)
            actual = Counter(case_id.split("/")[0] for case_id in plan[split])
            if any(actual[level] != count for level, count in required.items()):
                raise ValueError(f"Design quota mismatch for {split}: required {required}, got {dict(actual)}")
        if any(assignments[case_id] != "test" for case_id in plan.get("pending_allowed", [])):
            raise ValueError("With design quotas, only test designs may be pending")
    return plan, assignments


def source_reference(row: dict, row_number: int, expression: str, clocking=None) -> dict:
    result = {
        "row_number": row_number,
        "assertion": row["assertion"],
        "support_code_sha256": sha256(row.get("support_code") or ""),
        "property_expression_sha256": sha256(expression),
        "signals": row["signals"],
        "assertion_status": row["assertion_status"],
        "vacuous": row.get("vacuous"),
        "reason": row.get("reason"),
        "precondition_status": row.get("precondition_status"),
        "contrapositive_status": row.get("contrapositive_status"),
        "jasper_log": row.get("jasper_log"),
        "model": row.get("model"),
        "checker_clocks": checker_clocks(row, clocking),
    }
    if row.get("sft_formal_recheck") is not None:
        result["sft_formal_recheck"] = row["sft_formal_recheck"]
    return result


def examples_for_case(case: dict, signals: list[dict], source_path: str) -> tuple[list[dict], dict]:
    case_id = case["case_id"]
    signal_map = {}
    for item in signals:
        name, information = item.get("name"), item.get("information")
        if not isinstance(name, str) or not isinstance(information, str) or not information.strip():
            raise ValueError(case_id + ": missing saved signal description")
        if name in signal_map:
            raise ValueError(case_id + ": duplicate saved signal " + name)
        signal_map[name] = {"name": name, "information": information}
    selected = Counter()
    skipped = Counter()
    groups = {}
    design_identity = None
    for index, row in enumerate(case["rows"], 1):
        kind = category(row)
        if kind is None:
            skipped[str(row.get("assertion_status")) + "/" + str(row.get("reason"))] += 1
            continue
        selected[kind] += 1
        if row.get("case_id") != case_id:
            raise ValueError(case_id + ": mismatched row case ID")
        spec, rtl = row["spec"], row["rtl"]
        if not spec.strip() or not rtl.strip():
            raise ValueError(case_id + ": missing full specification or RTL")
        identity = (sha256(spec), sha256(rtl))
        if any(row.get(field) != digest for field, digest in zip(("spec_sha256", "rtl_sha256"), identity)):
            raise ValueError(case_id + ": source text hash mismatch")
        if design_identity is not None and design_identity != identity:
            raise ValueError(case_id + ": different spec/RTL revisions within one case")
        design_identity = identity
        origins = row.get("signals")
        if not origins or any(name not in signal_map for name in origins):
            raise ValueError(case_id + ": missing origin signal mapping")
        _, expression, _, status, error = resolve_property(row)
        if status != "ok":
            raise ValueError(f"{case_id} row {index}: {status}: {error}")
        has_implication = any(t in ("|->", "|=>") for t in tokenize(expression))
        if has_implication != (kind == "proven_non_vacuous_implication"):
            raise ValueError(f"{case_id} row {index}: implication scope conflicts with label")
        support = minimal_support(row)
        clocking = row.get("formal_clocking", case.get("formal_clocking"))
        if clocking is not None and not row.get("sft_formal_recheck"):
            raise ValueError(case_id + ": explicit clock context needs matching SFT proof/vacuity evidence")
        context = {
            "clock": row.get("clock"),
            "checker_clocks": checker_clocks(row, clocking),
            "reset_signal": row.get("reset"),
            "formal_reset_policy": row["formal_reset_policy"],
            "formal_clocking": clocking,
        }
        # Same comparison as pre-reduction exact dedup: resolved property tokens,
        # with support constants expanded. Distinct clocks/environments stay apart.
        key = (identity, json.dumps(context, sort_keys=True), tokenize(expression))
        reference = source_reference(row, index, expression, clocking)
        if key in groups:
            example = groups[key]
            if example["category"] != kind:
                raise ValueError(case_id + ": contradictory duplicate labels")
            example["provenance"]["sources"].append(reference)
            continue
        target = origins[0]
        completion = ((support + "\n") if support else "") + row["assertion"]
        digest = sha256(json.dumps([case_id, key], sort_keys=True))
        groups[key] = {
            "id": case_id.replace("/", "__") + "__" + digest[:20],
            "case_id": case_id,
            "level": case_id.split("/")[0],
            "spec": spec,
            "rtl": rtl,
            "target_signal": target,
            "signal_info": signal_map[target],
            "assertion": completion,
            "category": kind,
            "assertion_status": "proven",
            "vacuous": row.get("vacuous"),
            "reason": row.get("reason"),
            "formal_context": context,
            "provenance": {
                "source_case": source_path,
                "source_assertion": row["assertion"],
                "source_support_code_sha256": sha256(row.get("support_code") or ""),
                "selected_support_code": support,
                "property_expression": expression,
                "benchmark_revision": row.get("benchmark_revision"),
                "formal_timeout": row.get("formal_timeout"),
                "refinement_iterations": row.get("refinement_iterations"),
                "spec_sha256": identity[0],
                "rtl_sha256": identity[1],
                "sources": [reference],
            },
        }
    examples = list(groups.values())
    return examples, {
        "source_rows": len(case["rows"]),
        "selected_before_deduplication": sum(selected.values()),
        "selected_categories": dict(selected),
        "unique_examples": len(examples),
        "unique_categories": dict(Counter(x["category"] for x in examples)),
        "exact_duplicates_removed": sum(selected.values()) - len(examples),
        "excluded_rows": dict(skipped),
    }


def chat_record(example: dict) -> dict:
    info = example["signal_info"]
    context = example["formal_context"]
    if context["formal_reset_policy"] != "unconstrained":
        raise ValueError("Add an explicit prompt policy for the recorded reset semantics")
    user = (
        "Full specification:\n" + example["spec"]
        + "\n\nFull RTL:\n```systemverilog\n" + example["rtl"] + "\n```"
        + "\n\nTarget signal: " + info["name"]
        + "\nSaved specification-based signal description: " + info["information"]
        + "\n\nFormal checking context:\n"
        + "Checker clock declarations:\n"
        + clock_commands(context["checker_clocks"], context.get("formal_clocking"))
        + "\nReset policy: reset -none. No global reset sequence or extra input assumptions "
          "were imposed for final labeling. Assertion-local clock events and disable iff "
          "conditions retain their own meaning."
        + "\nGenerate one assertion for this signal."
    )
    return {
        "id": example["id"], "case_id": example["case_id"], "category": example["category"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "```systemverilog\n" + example["assertion"] + "\n```"},
        ],
    }


def json_line(handle, value: dict) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def export(cases_root: Path, split_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Output already exists; choose a fresh --out to preserve the snapshot")
    plan, assignments = load_split(split_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".sft_export_", dir=output.parent))
    manifest = {
        "version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "split_plan": plan, "split_plan_sha256": sha256(split_path.read_bytes()),
        "source_cases_root": str(cases_root), "formal_reduction_applied": False,
        "selection": "proven AND (conclusive non-vacuous implication OR outside-scope non-implication)",
        "deduplication": "Within each design and formal context: tokenize resolved property bodies with support constants expanded; retain original representative code and all source references.",
        "support_policy": "Only original property/parameter declarations transitively referenced by the representative assertion are retained. No assertion body is rewritten.",
        "target_signal_policy": "First saved generation-origin signal of the first duplicate representative; descriptions are copied from signals.json, never inferred from the expected assertion.",
        "proof_policy": "Existing saved proof/vacuity labels; no new JasperGold checks or semantic-relevance claims.",
        "clock_policy": "Reconstruct the saved JasperRunner clock declarations from each original support block and assertion, with inferred design-clock fallback. Preserve these declarations explicitly when rechecking the reduced-support target.",
        "tokenization": "Not tokenized or truncated; apply the selected model chat template and measure lengths before training.",
        "pending": [], "designs": {}, "splits": {},
    }
    totals = Counter()
    split_counts = {s: Counter() for s in SPLITS}
    categories = {s: Counter() for s in SPLITS}
    ids = set()
    identities = {}
    try:
        with ExitStack() as stack:
            canonical = stack.enter_context((temporary / "examples.jsonl").open("w"))
            designs = stack.enter_context((temporary / "designs.jsonl").open("w"))
            handles = {s: stack.enter_context((temporary / (s + ".jsonl")).open("w")) for s in SPLITS}
            for case_id, split in assignments.items():
                case_path = Path(plan.get("source_overrides", {}).get(case_id,
                                 cases_root / case_id / "case.json"))
                if not case_path.exists():
                    case = None
                else:
                    raw = case_path.read_bytes()
                    case = json.loads(raw)
                if case is None or case.get("status") != "done":
                    if case_id not in plan.get("pending_allowed", []):
                        raise ValueError("Required completed design missing: " + case_id)
                    manifest["pending"].append({"case_id": case_id, "split": split,
                        "status": case.get("status") if case else "no_completed_case"})
                    continue
                signal_path = cases_root / case_id / "signals.json"
                signal_bytes = signal_path.read_bytes()
                signals = json.loads(signal_bytes)
                examples, stats = examples_for_case(case, signals, str(case_path))
                if not examples:
                    raise ValueError("Assigned design has no eligible examples: " + case_id)
                # Prevent exact copies of a design/specification across split boundaries.
                for text_name in ("spec", "rtl"):
                    identity = (text_name, sha256(examples[0][text_name]))
                    if identity in identities and identities[identity] != split:
                        raise ValueError("Identical " + text_name + " appears across splits")
                    identities[identity] = split
                stats.update(split=split, case_sha256=sha256(raw),
                             signals_sha256=sha256(signal_bytes),
                             source_case=str(case_path), source_signals=str(signal_path))
                manifest["designs"][case_id] = stats
                for field in ("source_rows", "selected_before_deduplication", "unique_examples", "exact_duplicates_removed"):
                    totals[field] += stats[field]
                split_counts[split]["designs"] += 1
                split_counts[split][case_id.split("/")[0] + "_designs"] += 1
                split_counts[split]["examples"] += len(examples)
                categories[split].update(x["category"] for x in examples)
                json_line(designs, {"case_id": case_id, "split": split,
                    "spec": examples[0]["spec"], "rtl": examples[0]["rtl"],
                    "signal_info": signals, "source_case_sha256": sha256(raw),
                    "source_signals_sha256": sha256(signal_bytes)})
                for example in examples:
                    if example["id"] in ids:
                        raise ValueError("Duplicate example ID")
                    ids.add(example["id"])
                    example["split"] = split
                    json_line(canonical, example)
                    json_line(handles[split], chat_record(example))
        # Abort if concurrent source updates occurred during export.
        for stats in manifest["designs"].values():
            for name, digest in (("source_case", "case_sha256"), ("source_signals", "signals_sha256")):
                if sha256(Path(stats[name]).read_bytes()) != stats[digest]:
                    raise ValueError("Source changed during export: " + stats[name])
        manifest["totals"] = dict(totals, designs=len(manifest["designs"]))
        for split in SPLITS:
            manifest["splits"][split] = dict(split_counts[split], categories=dict(categories[split]),
                planned_designs=len(plan[split]))
        manifest["files"] = {p.name: {"sha256": sha256(p.read_bytes()), "bytes": p.stat().st_size}
                             for p in sorted(temporary.glob("*.jsonl"))}
        (temporary / "split_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, default=Path("output/fixme_svagen/cases"))
    parser.add_argument("--split-plan", type=Path, default=Path(__file__).with_name("sft_split_v1.json"))
    parser.add_argument("--out", type=Path, default=Path("output/fixme_svagen/sft_before_reduction_v1"))
    args = parser.parse_args()
    manifest = export(args.cases_root, args.split_plan, args.out)
    print(json.dumps({"output": str(args.out), "totals": manifest["totals"],
                      "splits": manifest["splits"], "pending": manifest["pending"]}, indent=2))


if __name__ == "__main__":
    main()
