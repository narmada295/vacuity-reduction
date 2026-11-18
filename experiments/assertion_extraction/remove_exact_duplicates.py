#!/usr/bin/env python3
"""Remove exact textual duplicates from extract_assertions.py JSONL output.

Comparison ignores cosmetic whitespace, not names, operators, or logic.
Duplicates are only compared within the SAME source file and design/context.
This is syntactic deduplication, not full logical equivalence checking.

Usage:
    python remove_exact_duplicates.py extracted_assertions/selected_assertions.jsonl \
        --out deduplicated_assertions
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Tuple

# This lexical splitter keeps identifiers, string literals, operators, and
# punctuation separate. Thus `ab` != `a b`, unlike removing all whitespace.
TOKEN_RE = re.compile(
    r'''"(?:\\.|[^"\\])*"|\\[^\s]+|[A-Za-z_$][A-Za-z0-9_$]*'''
    r'''|\d+'[sS]?[bBoOdDhH][a-zA-Z0-9_xXzZ?]+'''
    r'''|\d+|\|->|\|=>|===|!==|==\?|!=\?|<<<|>>>'''
    r'''|##|&&|\|\||==|!=|<=|>=|<<|>>|\*\*|->|=>|::|\+\+|--'''
    r'''|\S'''
)


def tokenize(s: str) -> Tuple[str, ...]:
    # Intentionally conservative: comments not normalized, no semantic rewrites.
    return tuple(TOKEN_RE.findall(s.strip()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="selected_assertions.jsonl from the extractor")
    ap.add_argument("--out", type=Path, default=Path("deduplicated_assertions"))
    args = ap.parse_args()
    rows = [json.loads(ln) for ln in args.input.read_text(encoding="utf-8").splitlines() if ln.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    unresolved = []
    for row in rows:
        if row.get("extraction_status") != "ok" or not row.get("property_expression", "").strip():
            unresolved.append(row)
            continue
        context = (
            row.get("source_file", ""), row.get("case_id", ""),
            row.get("design", ""), row.get("rtl_sha256", ""),
            row.get("formal_reset_policy", ""),
        )
        groups[context, tokenize(row["property_expression"])].append(row)

    unique = [group[0] for group in groups.values()] + unresolved
    # Preserve original record order in exported data.
    order = {id(row): n for n, row in enumerate(rows)}
    unique.sort(key=lambda r: order[id(r)])
    duplicates = []
    for group in groups.values():
        kept = group[0]
        for removed in group[1:]:
            duplicates.append({
                "design": kept.get("design"),
                "kept_row_number": kept.get("row_number"),
                "kept_property": kept.get("property_name") or kept.get("assertion"),
                "removed_row_number": removed.get("row_number"),
                "removed_property": removed.get("property_name") or removed.get("assertion"),
            })

    with (args.out / "unique_assertions.jsonl").open("w", encoding="utf-8") as fh:
        for row in unique:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out / "duplicate_groups.csv").open("w", encoding="utf-8", newline="") as fh:
        cols = ["design", "kept_row_number", "kept_property", "removed_row_number", "removed_property"]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(duplicates)
    print(f"Input: {len(rows)} | Unique: {len(unique)} | Removed exact duplicates: {len(duplicates)}")
    if unresolved:
        print(f"WARNING: retained {len(unresolved)} assertions with unresolved property bodies")
    print(f"Outputs: {args.out.resolve()}")


if __name__ == "__main__":
    main()
