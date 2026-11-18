"""Load an assertion set without exposing it to the mutation planner."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .core import Config


@dataclass(frozen=True)
class AssertionSet:
    labels: tuple[str, ...]
    metadata: tuple[dict, ...]
    inline_code: str | None
    checker_file: Path | None


def load_assertions(path: Path, limit: int = 0) -> AssertionSet:
    """Accept SANGAM JSONL/text, raw SVA snippets, or a named bind checker."""
    path = path.resolve()
    source = path.read_text()
    if path.suffix.lower() != ".jsonl" and re.search(r"\bbind\b", source):
        labels = tuple(re.findall(r"(?m)^\s*([A-Za-z_]\w*)\s*:\s*assert\s+property\b", source))
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("Bind checker needs unique named assert property labels")
        if limit:
            raise ValueError("--max-assertions applies only to JSONL sets")
        return AssertionSet(labels, tuple({"label": label} for label in labels), None, path)
    if path.suffix.lower() == ".jsonl":
        lines = [json.loads(line) for line in source.splitlines() if line.strip()]
    else:
        chunks = re.split(r"(?m)^\[\d+\]\s+Signals:[^\n]*\n", source)
        chunks = chunks[1:] if len(chunks) > 1 else [source]
        lines = []
        for chunk in chunks:
            definitions = {
                match.group(1): match.group(0)
                for match in re.finditer(r"\bproperty\s+([A-Za-z_]\w*)\s*;.*?\bendproperty\b",
                                         chunk, flags=re.S)
            }
            for assertion in _scan_assertions(chunk):
                reference = re.fullmatch(r"assert\s+property\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;",
                                         assertion, flags=re.S)
                name = reference.group(1) if reference else ""
                lines.append({"assertion": assertion, "property_name": name,
                              "property_definition": definitions.get(name, "")})
    if limit:
        lines = lines[:limit]
    if not lines:
        raise ValueError("Assertion file has no parseable assert property statements")
    return _from_rows(lines)


def _scan_assertions(source: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"\bassert\s+property\s*\(", source):
        depth = 1
        index = match.end()
        while index < len(source) and depth:
            depth += (source[index] == "(") - (source[index] == ")")
            index += 1
        if depth == 0 and re.match(r"\s*;", source[index:]):
            end = index + re.match(r"\s*;", source[index:]).end()
            found.append(source[match.start():end].strip())
    return found


def _from_rows(lines: list[dict]) -> AssertionSet:
    parts: list[str] = []
    metadata: list[dict] = []
    definitions: dict[str, str] = {}
    for index, row in enumerate(lines, start=1):
        assertion = str(row.get("assertion", "")).strip()
        if not re.match(r"^assert\s+property\s*\(", assertion, re.S) or not assertion.endswith(";"):
            raise ValueError(f"JSONL row {index} is not an assert property statement")
        definition = str(row.get("property_definition") or row.get("support_code") or "").strip()
        if definition:
            match = re.search(r"\bproperty\s+([A-Za-z_]\w*)\s*[;(]", definition)
            if match:
                name = match.group(1)
                if name in definitions and definitions[name] != definition:
                    raise ValueError(f"Conflicting property definitions for {name}")
                if name not in definitions:
                    definitions[name] = definition
                    parts.append(definition)
            else:
                parts.append(definition)
        label = f"ML_ASSERT_{index:04d}"
        parts.append(f"{label}: {assertion}")
        metadata.append({"label": label, "row_number": row.get("row_number", index),
                         "property_name": row.get("property_name", ""),
                         "assertion": assertion})
    return AssertionSet(tuple(item["label"] for item in metadata), tuple(metadata),
                        "\n".join(parts) + "\n", None)


def inject_in_top(config: Config, directory: Path, code: str) -> None:
    """Place checks only in an evaluation job copy, never in saved mutant RTL."""
    hits: list[tuple[Path, str, re.Match[str]]] = []
    pattern = re.compile(r"\bmodule\s+" + re.escape(config.top) + r"\b")
    for source in config.sources:
        path = directory / "rtl" / source
        text = path.read_text()
        hits.extend((path, text, match) for match in pattern.finditer(text))
    if len(hits) != 1:
        raise ValueError(f"Expected one module {config.top} in source list; found {len(hits)}")
    path, text, match = hits[0]
    end = re.search(r"\bendmodule\b", text[match.end():])
    if not end:
        raise ValueError(f"No endmodule after {config.top}")
    position = match.end() + end.start()
    path.write_text(text[:position] + "\n" + code + "\n" + text[position:])
