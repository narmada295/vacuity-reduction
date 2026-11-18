#!/usr/bin/env python3
"""Extract SVA properties and JasperGold metadata from pipeline JSON/JSONL.

Supported inputs:
    1. JSON object with a `rows` array (the supplied pipeline format)
    2. JSON array of assertion objects
    3. JSONL with one assertion object (or a `rows` wrapper) per line

By default, this tool extracts ALL assertions and separately exports eligible
ones: proven AND (vacuous is false OR reason == outside_scope_non_implication).
It does not decide logical redundancy or run JasperGold.

Usage:
    python extract_assertions.py results.json --out results_extracted
    python extract_assertions.py design1.json design2.json --out results_extracted
    python extract_assertions.py my_results_folder --out results_extracted
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import List

# Named property declarations: property some_name; ... endproperty
# The body is extracted from the support_code stored WITH that row.
DECL_RE = re.compile(
    r"\bproperty\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*;(?P<body>.*?)\bendproperty\b"
    r"(?:\s*:\s*[A-Za-z_$][A-Za-z0-9_$]*)?",
    flags=re.DOTALL,
)
ASSERT_RE = re.compile(
    r"\bassert\s+property\s*\(", flags=re.IGNORECASE
)
SIMPLE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")

FIELDS = [
    "source_file", "row_number", "case_id", "design", "rtl_sha256",
    "property_name", "property_expression", "assertion", "property_definition",
    "clock", "reset", "formal_reset_policy",
    "assertion_status", "vacuous", "reason", "selected", "signals",
    "jasper_log", "extraction_status", "extraction_error",
]


def read_records(path: Path):
    raw = path.read_text(encoding="utf-8-sig")
    try:
        doc = json.loads(raw)
        docs = [doc]
    except json.JSONDecodeError:
        docs = []
        for line_num, line in enumerate(raw.splitlines(), start=1):
            if line.strip():
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_num}: invalid JSONL: {exc}") from exc

    for doc in docs:
        if isinstance(doc, list):
            rows, common = doc, {}
        elif isinstance(doc, dict) and isinstance(doc.get("rows"), list):
            rows, common = doc["rows"], doc
        elif isinstance(doc, dict) and "assertion" in doc:
            rows, common = [doc], {}
        else:
            raise ValueError(f"{path}: expected a rows wrapper, assertion list, or JSONL records")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{path}: row must be an object, got {type(row).__name__}")
            yield row, common


def remove_sv_comments(s: str) -> str:
    """Comment removal for recognizing an identifier, not for rewriting SV code."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", " ", s, flags=re.DOTALL)


def remove_final_semicolon(s: str) -> str:
    """Remove one trailing semicolon (str.removesuffix requires Python 3.9)."""
    return (s[:-1] if s.endswith(";") else s).strip()


def expand_support_parameters(row: dict, expression: str) -> str:
    """Retain constant meanings when a property is detached from support_code.

    For example, localparam W = $bits(signal) must not become an undeclared
    identifier (or a new free symbolic signal) in the downstream harness.
    Parenthesized substitution preserves the constant expression and precedence.
    Original assertion text and its property declaration remain unchanged.
    """
    support = remove_sv_comments(str(row.get('support_code') or ''))
    declarations = {}
    for match in re.finditer(
        r'\b(?:localparam|parameter)\s+'
        r'(?:(?:int|integer|logic|bit)\s+(?:unsigned\s+|signed\s+)?)?'
        r'([A-Za-z_$][\w$]*)\s*=\s*([^;]+);', support
    ):
        name, value = match.group(1), match.group(2).strip()
        if name in declarations and declarations[name] != value:
            raise ValueError('Conflicting support parameter: ' + name)
        declarations[name] = value

    def expand(text, active):
        def replace(match):
            name = match.group(0)
            if name not in declarations:
                return name
            if name in active:
                raise ValueError('Cyclic support parameter: ' + name)
            return '(' + expand(declarations[name], active | {name}) + ')'
        # Skip literals/comments so parameter names inside them are not edited.
        return re.sub(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*|[A-Za-z_$][\w$]*',
                      lambda m: m.group(0) if m.group(0).startswith(('"', '/*', '//')) else replace(m),
                      text, flags=re.S)
    return expand(expression, set())


def property_argument(assertion: str) -> str:
    """Return text inside assert property (...), retaining nested SV expressions.

    String literals and comments are skipped for bracket counting. Does NOT
    attempt full SystemVerilog parsing; unsupported constructs are reported.
    """
    match = ASSERT_RE.search(assertion)
    if not match:
        raise ValueError("no 'assert property (' found")
    start = match.end()
    depth = 1
    i = start
    state = "normal"
    while i < len(assertion):
        c = assertion[i]
        next_char = assertion[i + 1] if i + 1 < len(assertion) else ""
        if state == "line_comment":
            if c == "\n":
                state = "normal"
        elif state == "block_comment":
            if c == "*" and next_char == "/":
                state = "normal"
                i += 1
        elif state == "string":
            if c == "\\":
                i += 1
            elif c == '"':
                state = "normal"
        elif c == '"':
            state = "string"
        elif c == "/" and next_char == "/":
            state = "line_comment"
            i += 1
        elif c == "/" and next_char == "*":
            state = "block_comment"
            i += 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                if not assertion[i + 1 :].strip().startswith(";"):
                    raise ValueError("expected ';' after assert property parentheses")
                return assertion[start:i].strip()
        i += 1
    raise ValueError("unbalanced parentheses in assert property")


def resolve_property(row: dict):
    assertion = str(row.get("assertion") or "")
    content = property_argument(assertion)
    if SIMPLE_IDENTIFIER_RE.fullmatch(remove_sv_comments(content).strip()):
        prop_name = remove_sv_comments(content).strip()
        support_code = str(row.get("support_code") or "")
        matches = [(m.group("body").strip(), m.group(0).strip())
                   for m in DECL_RE.finditer(support_code) if m.group(1) == prop_name]
        if not matches:
            return prop_name, "", "", "missing_definition", (
                f"named property '{prop_name}' not found in support_code"
            )
        if len({pair[0] for pair in matches}) > 1:
            return prop_name, "", "", "ambiguous_definition", (
                f"conflicting definitions for '{prop_name}' in support_code"
            )
        body, declaration = matches[0]
        return prop_name, expand_support_parameters(row, remove_final_semicolon(body)), declaration, "ok", ""
    return "", expand_support_parameters(row, remove_final_semicolon(content)), "", "ok", ""


def is_selected(row: dict) -> bool:
    # Only use the existing dataset classifications; no new vacuity claims.
    status = str(row.get("assertion_status") or "").strip().lower()
    if status != "proven":
        return False
    nonvacuous = row.get("vacuous") is False
    out_of_scope = row.get("reason") == "outside_scope_non_implication"
    return nonvacuous or out_of_scope


def iter_inputs(inputs: List[str]) -> List[Path]:
    paths = []
    for entry in inputs:
        p = Path(entry)
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            paths.extend(sorted(q for q in p.rglob("*")
                                if q.is_file() and q.suffix.lower() in {".json", ".jsonl", ".txt"}))
        else:
            raise FileNotFoundError(entry)
    # Dedupe paths without changing first-seen order
    return list(dict.fromkeys(p.resolve() for p in paths))


def write_csv(path: Path, records: List[dict]):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for rec in records:
            out = dict(rec)
            out["signals"] = json.dumps(rec["signals"], ensure_ascii=False)
            writer.writerow(out)


def write_jsonl(path: Path, records: List[dict]):
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="JSON/JSONL results files (or folder containing them)")
    parser.add_argument("--out", default="extracted_assertions", help="Output directory")
    args = parser.parse_args()

    paths = iter_inputs(args.inputs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for path in paths:
        for n, (row, common) in enumerate(read_records(path), start=1):
            try:
                name, expression, declaration, result, error = resolve_property(row)
            except ValueError as exc:
                name, expression, declaration, result, error = "", "", "", "parse_error", str(exc)
            records.append({
                "source_file": str(path),
                "row_number": n,
                "case_id": row.get("case_id") or common.get("case_id") or "",
                "design": row.get("top") or common.get("top") or "",
                "rtl_sha256": row.get("rtl_sha256") or common.get("rtl_sha256") or "",
                "property_name": name,
                "property_expression": expression,
                "assertion": row.get("assertion") or "",
                "property_definition": declaration,
                "clock": row.get("clock") or common.get("clock") or "",
                "reset": row.get("reset") or common.get("reset") or "",
                "formal_reset_policy": row.get("formal_reset_policy") or common.get("formal_reset_policy") or "",
                "assertion_status": row.get("assertion_status") or "",
                "vacuous": row.get("vacuous"),
                "reason": row.get("reason") or "",
                "selected": is_selected(row),
                "signals": row.get("signals") or [],
                "jasper_log": row.get("jasper_log") or "",
                "extraction_status": result,
                "extraction_error": error,
            })

    selected = [r for r in records if r["selected"]]
    errors = [r for r in records if r["extraction_status"] != "ok"]
    for stem, rows in (("all_assertions", records), ("selected_assertions", selected), ("extraction_issues", errors)):
        write_csv(out / f"{stem}.csv", rows)
        write_jsonl(out / f"{stem}.jsonl", rows)

    print(f"Read {len(paths)} files, {len(records)} assertions")
    print(f"Selected {len(selected)}: proven + (non-vacuous OR out-of-scope)")
    print(f"Named/inline: {sum(bool(r['property_name']) for r in records)}/{sum(not r['property_name'] for r in records)}")
    print(f"Extraction issues: {len(errors)}")
    print(f"Files written to: {out.resolve()}")
    if errors:
        print("WARNING: some properties could not be resolved; check extraction_issues.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
