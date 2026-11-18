"""Validated records for the opt-in exploration workflow.

These records express model hypotheses. Validation of references and JSON does
not establish that a natural-language specification has been understood correctly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .core import Config, digest


FAMILIES = ("data", "control", "state", "timing", "ordering_priority",
            "reset_recovery", "interface_protocol")
EXPOSURES = ("immediate", "state_dependent", "sequence_dependent", "interaction_dependent")
PURPOSES = ("trigger", "control", "boundary", "interaction", "recovery",
            "adversarial", "history_control")
OPERATORS = ("==", "!=", "<", "<=", ">", ">=")
REFERENCE = re.compile(r"([A-Za-z_]\w*)(?:\[(\d+)(?::(\d+))?\])?\Z")


def normalize_numeric_json(source: str) -> str:
    """Normalize unquoted hex/SV numeric values, preserving every JSON string.

    Models sometimes emit data:25'h1000001 instead of a quoted literal. This
    scanner only converts recognized numeric tokens to decimal JSON integers;
    it does not repair prose, evaluate expressions, or infer missing structure.
    """
    token = re.compile(r"\d+'[sS]?([bBoOdDhH])([0-9a-fA-F_]+)|0[xX][0-9a-fA-F_]+|0[bB][01_]+")
    result, index, quoted, escaped = [], 0, False, False
    while index < len(source):
        char = source[index]
        if quoted:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            index += 1
            continue
        if char == '"':
            quoted = True
        match = token.match(source, index) if char.isdigit() else None
        if match:
            raw = match[0].replace("_", "")
            if "'" in raw:
                size, digits = raw.split("'", 1)
                digits = digits.lstrip("sS")
                number = int(digits[1:], {"b": 2, "o": 8, "d": 10, "h": 16}[digits[0].lower()])
                if number.bit_length() > int(size):
                    raise ValueError("Numeric literal exceeds its declared size")
            else:
                number = int(raw, 0)
            result.append(str(number))
            index = match.end()
        else:
            result.append(char)
            index += 1
    return "".join(result)


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def strings(value: Any, label: str, allowed: set[str] | None = None,
            minimum: int = 0) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"{label} must be a list with at least {minimum} entries")
    result = [text(item, label) for item in value]
    if len(set(result)) != len(result) or (allowed is not None and not set(result) <= allowed):
        raise ValueError(f"{label} contains duplicates or unsupported references")
    return result


def integer(value: Any, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{label} must be an integer in {minimum}..{maximum or 'unbounded'}")
    return value


def quote(value: Any, specification: str) -> str:
    if isinstance(value, dict):
        lines = specification.splitlines(keepends=True)
        start = integer(value.get("start_line"), "spec start_line", 1, len(lines))
        end = integer(value.get("end_line"), "spec end_line", start, len(lines))
        result = "".join(lines[start - 1:end]).strip()
        if len(result) < 12:
            raise ValueError("Referenced spec lines contain fewer than 12 characters")
        return result
    result = text(value, "spec_quote")
    if len(result) >= 12 and " ".join(result.split()) in " ".join(specification.split()):
        return result
    # Models frequently omit Markdown emphasis/code delimiters. Match only those
    # formatting differences and case/whitespace, then copy source-owned text.
    # This is NOT fuzzy semantic matching or paraphrase acceptance.
    def canonical(source):
        chars, positions = [], []
        for index, char in enumerate(source):
            if char in "`*":
                continue
            if char.isspace():
                if chars and chars[-1] != " ":
                    chars.append(" ")
                    positions.append(index)
            else:
                for folded in char.casefold():
                    chars.append(folded)
                    positions.append(index)
        return "".join(chars).rstrip(), positions
    source, positions = canonical(specification)
    needle, _ = canonical(result)
    start = source.find(needle) if len(needle) >= 12 else -1
    if start < 0:
        raise ValueError("spec_quote is not source-grounded: " + repr(result[:160]) +
                         "; use an exact excerpt or {start_line,end_line}")
    return specification[positions[start]:positions[start + len(needle) - 1] + 1]


def predicates(value: Any, signals: set[str], widths: dict[str, int],
               minimum: int = 0) -> list[dict]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f"predicates needs at least {minimum} items")
    result = []
    for item in value:
        if isinstance(item, str):
            # Lossless normalization of a simple comparison, never eval/code execution.
            match = re.fullmatch(r"\s*([A-Za-z_]\w*(?:\[\d+(?::\d+)?\])?)\s*"
                                 r"(==|!=|<=|>=|<|>)\s*"
                                 r"([A-Za-z_]\w*(?:\[\d+(?::\d+)?\])?|\d+)\s*", item)
            if not match:
                raise ValueError("Predicate string must be one supported comparison; use structured JSON")
            item = {"left": match[1], "op": match[2], "right": match[3], "sample": "pre"}
        if not isinstance(item, dict) or not {"left", "op", "right"} <= item.keys():
            raise ValueError("predicate needs left, op, right, and optional sample=pre/post")
        left, right, op = item["left"], item["right"], item["op"]
        sample = item.get("sample", "pre")
        if op not in OPERATORS or sample not in ("pre", "post"):
            raise ValueError("Unsupported predicate operator or sample phase")
        for label, operand in (("left", left), ("right", right)):
            if label == "right" and type(operand) is int and operand >= 0:
                continue
            if label == "right" and isinstance(operand, str) and operand.isdigit():
                continue
            match = REFERENCE.fullmatch(operand) if isinstance(operand, str) else None
            if not match or match[1] not in signals:
                raise ValueError(f"Unknown predicate {label}: {operand!r}")
            if match[2] is not None and match[1] in widths and int(match[2]) >= widths[match[1]]:
                raise ValueError("Predicate bit select exceeds signal width")
            if match[3] is not None and (int(match[3]) > int(match[2]) or
                                        match[1] in widths and int(match[3]) >= widths[match[1]]):
                raise ValueError("Invalid predicate part select")
        result.append({"left": left, "op": op, "right": right, "sample": sample})
    return result


def _objects(value: Any, label: str, minimum: int = 0) -> list[dict]:
    if not isinstance(value, list) or len(value) < minimum or any(not isinstance(x, dict) for x in value):
        raise ValueError(f"{label} must be an object list with at least {minimum} entries")
    return value


def _unique_ids(items: list[dict], label: str) -> set[str]:
    ids = [text(item.get("id"), label + " id") for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate {label} IDs")
    return set(ids)


def validate_analysis(raw: dict, config: Config, specification: str,
                      signals: set[str], widths: dict[str, int],
                      input_signals: set[str] | None = None) -> dict:
    summary = text(raw.get("summary"), "summary")
    behaviors = _objects(raw.get("behaviors"), "behaviors", 1)
    regions = _objects(raw.get("regions"), "regions", 1)
    behavior_ids, region_ids = _unique_ids(behaviors, "behavior"), _unique_ids(regions, "region")
    for behavior in behaviors:
        for field in ("description", "trigger", "expected", "timing"):
            behavior[field] = text(behavior.get(field), field)
        behavior["spec_quote"] = quote(behavior.get("spec_quote"), specification)
        behavior["signals"] = strings(behavior.get("signals"), "signals", signals, 1)
        for field in ("state_dependencies", "exceptions"):
            behavior[field] = strings(behavior.get(field, []), field)
        behavior["related_behaviors"] = strings(behavior.get("related_behaviors", []),
                                                 "related_behaviors", behavior_ids)
        if behavior.get("confidence") not in ("explicit", "inferred", "ambiguous"):
            raise ValueError("Behavior confidence must be explicit, inferred, or ambiguous")
    for region in regions:
        for field in ("name", "description"):
            region[field] = text(region.get(field), field)
        region["signals"] = strings(region.get("signals"), "region signals", signals, 1)
        region["behavior_ids"] = strings(region.get("behavior_ids"), "behavior_ids", behavior_ids, 1)
        region["related_regions"] = strings(region.get("related_regions", []), "related_regions", region_ids)
        region["families"] = _objects(region.get("families"), "families")
        seen = set()
        for item in region["families"]:
            if item.get("family") not in FAMILIES or item["family"] in seen:
                raise ValueError("Invalid or repeated applicable family")
            seen.add(item["family"])
            text(item.get("reason"), "family applicability reason")
            if item.get("confidence") not in ("supported", "uncertain"):
                raise ValueError("Applicability confidence must be supported or uncertain")
        for location in _objects(region.get("source_locations", []), "source_locations"):
            path = Path(text(location.get("file"), "source file"))
            if path not in config.sources:
                raise ValueError("Region source is not configured RTL")
            count = len((config.design_root / path).read_text().splitlines())
            start = integer(location.get("start_line"), "start_line", 1, count)
            integer(location.get("end_line"), "end_line", start, count)
    constraints = _objects(raw.get("input_constraints", []), "input_constraints")
    input_signals = (input_signals or set()) - {config.clock}
    _unique_ids(constraints, "constraint")
    for constraint in constraints:
        text(constraint.get("description"), "constraint description")
        if constraint.get("status") not in ("established", "assumed", "unspecified"):
            raise ValueError("Constraint status must be established, assumed, or unspecified")
        if constraint["status"] == "established":
            constraint["spec_quote"] = quote(constraint.get("spec_quote"), specification)
        constraint["when"] = predicates(constraint.get("when", []), signals, widths)
        constraint["require"] = predicates(constraint.get("require", []), signals, widths)
        referenced = {REFERENCE.fullmatch(p[k])[1] for p in constraint["require"]
                      for k in ("left", "right") if isinstance(p[k], str) and REFERENCE.fullmatch(p[k])}
        if not referenced <= input_signals:
            raise ValueError("Input constraint require clauses must restrict external inputs, not DUT output/state "
                             "requirements. Move response rules to behaviors; use input_constraints=[] "
                             "when no environment restriction is specified. Invalid references: " +
                             ", ".join(sorted(referenced - input_signals)))
        # Empty require means a documented natural-language constraint, not an executable check.
    mapped = {s for region in regions for s in region["signals"]}
    warnings = ["Region coverage is a model-generated inventory, not a completeness proof."]
    missing = sorted(set(widths) - {config.clock, config.reset} - mapped)
    if missing:
        warnings.append("Ports/state signals not assigned to any region: " + ", ".join(missing))
    return {"summary": summary, "behaviors": behaviors, "regions": regions,
            "input_constraints": constraints, "coverage_warnings": warnings,
            "ambiguities": strings(raw.get("ambiguities", []), "ambiguities")}


def validate_plan(raw: dict, analysis: dict, focus: dict, signals: set[str],
                  widths: dict[str, int]) -> dict:
    result = dict(raw)
    for field in ("mechanism", "golden_behavior", "faulty_behavior", "trigger",
                  "observable_consequence", "realism"):
        result[field] = text(raw.get(field), field)
    if raw.get("family") != focus["family"]:
        raise ValueError("Plan family differs from requested family")
    result["secondary_families"] = strings(raw.get("secondary_families", []),
                                             "secondary_families", set(FAMILIES))
    result["region_ids"] = strings(raw.get("region_ids"), "region_ids",
                                    {x["id"] for x in analysis["regions"]}, 1)
    if not set(focus["region_ids"]) <= set(result["region_ids"]):
        raise ValueError("Plan must address every region in the selected focus")
    result["behavior_ids"] = strings(raw.get("behavior_ids"), "behavior_ids",
                                      {x["id"] for x in analysis["behaviors"]}, 1)
    result["target_signals"] = strings(raw.get("target_signals"), "target_signals", signals, 1)
    result['target_effect'] = raw.get('target_effect', 'value_change')
    if result['target_effect'] not in ('value_change', 'update_order'):
        raise ValueError('target_effect must be value_change or update_order')
    exposure = raw.get("exposure")
    result["exposure"] = strings([exposure] if isinstance(exposure, str) else exposure,
                                   "exposure", set(EXPOSURES), 1)
    result["required_history"] = strings(raw.get("required_history", []), "required_history")
    result["trigger_predicates"] = predicates(raw.get("trigger_predicates"), signals, widths, 1)
    latency = raw.get("latency", {})
    lo = integer(latency.get("min"), "minimum trigger-to-consequence latency")
    hi = integer(latency.get("max"), "maximum trigger-to-consequence latency", lo)
    result["latency"] = {"min": lo, "max": hi}
    result["controls"] = _objects(raw.get("controls"), "controls", 1)
    _unique_ids(result["controls"], "control")
    for control in result["controls"]:
        text(control.get("description"), "control description")
        control["predicates"] = predicates(control.get("predicates"), signals, widths, 1)
    return result


def mechanism_key(plan: dict) -> str:
    """Exact normalized mechanism filter; semantic similarity is assessed separately."""
    normalize = lambda s: " ".join(s.casefold().split())
    return digest(json.dumps([normalize(plan["mechanism"]), sorted(plan["target_signals"]),
                              sorted(plan["behavior_ids"])]).encode())


def exploration_slots(analysis: dict, cross_region: bool, global_exploration: bool) -> list[dict]:
    slots = []
    regions = {x["id"]: x for x in analysis["regions"]}
    for region in regions.values():
        for family in region["families"]:
            slots.append({"id": region["id"] + ":" + family["family"],
                          "region_ids": [region["id"]], "family": family["family"], "kind": "region"})
    pairs = {tuple(sorted((region["id"], other))) for region in regions.values()
             for other in region["related_regions"] if other != region["id"]}
    if cross_region:
        for pair in sorted(pairs):
            families = {f["family"] for name in pair for f in regions[name]["families"]}
            for family in FAMILIES:
                if family in families:
                    slots.append({"id": "+".join(pair) + ":" + family,
                                  "region_ids": list(pair), "family": family, "kind": "interaction"})
    if global_exploration and len(regions) > 1:
        families = {f["family"] for r in regions.values() for f in r["families"]}
        for family in FAMILIES:
            if family in families:
                slots.append({"id": "global:" + family, "region_ids": sorted(regions),
                              "family": family, "kind": "global"})
    return [{**slot, "rounds": 0, "attempts": 0, "proposed": 0, "accepted": 0,
             "empty_rounds": 0, "queue": [], "outcomes": {}} for slot in slots]
