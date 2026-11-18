"""Extract individual concurrent assertions from SANGAM's final answer."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Assertion:
    text: str
    support: str
    implication: bool


def code_from_response(response: str) -> str:
    blocks = re.findall(r"```(?:systemverilog|verilog|sv)?\s*\n?(.*?)```", response, re.I | re.S)
    code = "\n".join(blocks) if blocks else response.split("[SVAs]")[-1]
    return re.sub(r"(?m)^\s*\[(?:width|connectivity|functionality|function)\]\s*$", "", code)


def extract_assertions(response: str) -> list[Assertion]:
    """Keep declarations with each assertion; do not modify the final assertion text."""
    code = code_from_response(response)
    matches: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\bassert\s+property\s*\(", code, re.I):
        depth = 1
        i = match.end()
        while i < len(code) and depth:
            if code[i] == "(":
                depth += 1
            elif code[i] == ")":
                depth -= 1
            i += 1
        if depth or not re.match(r"\s*;", code[i:]):
            continue
        end = i + re.match(r"\s*;", code[i:]).end()
        matches.append((match.start(), end, code[match.start():end].strip()))

    if not matches:
        return []
    remaining = code
    for start, end, _ in reversed(matches):
        remaining = remaining[:start] + remaining[end:]
    # Only retain declarations that can be shared by individually checked SVAs.
    support_parts = re.findall(r"\b(?:localparam|parameter|logic|wire|reg)\b[^;]*;|\bassign\b[^;]*;|\bproperty\b.*?\bendproperty\b", remaining, re.I | re.S)
    support = "\n".join(support_parts)
    result: list[Assertion] = []
    seen: set[str] = set()
    for _, _, statement in matches:
        if statement in seen:
            continue
        seen.add(statement)
        body = statement
        if not re.search(r"\|->|\|=>", body):
            reference = re.search(r"\bassert\s+property\s*\(\s*([A-Za-z_][A-Za-z_0-9]*)\s*\)", statement, re.I)
            if reference:
                definition = re.search(
                    r"\bproperty\s+" + re.escape(reference.group(1)) + r"\s*;(?P<body>.*?)\bendproperty\b",
                    support,
                    re.I | re.S,
                )
                if definition:
                    body = definition.group("body")
        result.append(Assertion(statement, support, bool(re.search(r"\|->|\|=>", body))))
    return result


def named_assertion(assertion: Assertion, name: str = "sangam_assert") -> str:
    return f"{assertion.support}\n{name}: {assertion.text}\n"


def insert_in_module(rtl: str, top: str, snippet: str) -> str:
    pattern = re.compile(r"\bmodule\s+" + re.escape(top) + r"\b")
    match = pattern.search(rtl)
    if not match:
        raise ValueError(f"module {top!r} not found in RTL")
    end = re.search(r"\bendmodule\b", rtl[match.end():])
    if not end:
        raise ValueError(f"endmodule for {top!r} not found")
    pos = match.end() + end.start()
    return rtl[:pos] + "\n" + snippet + "\n" + rtl[pos:]


def module_name(rtl: str, preferred: str) -> str:
    names = re.findall(r"\bmodule\s+([A-Za-z_][A-Za-z_0-9]*)", rtl)
    if not names:
        raise ValueError("no module declaration in RTL")
    return preferred if preferred in names else names[0]


def clock_reset(rtl: str) -> tuple[str | None, str | None]:
    clocks = re.findall(r"@\s*\(\s*posedge\s+([A-Za-z_][A-Za-z_0-9]*)", rtl)
    clock = next((x for x in clocks if "clk" in x.lower() or "clock" in x.lower()), None)
    if not clock:
        clock = clocks[0] if clocks else None
    inputs = []
    for declaration in re.findall(r"\binput\b(.*?)(?=\b(?:input|output|inout)\b|[;)])", rtl, re.S):
        ranges = re.findall(r"\[\s*([^:\]]+)\s*:\s*([^\]]+)\]", declaration)
        if any(left.strip() != right.strip() for left, right in ranges):
            # Data buses such as arburst can contain "rst" in their names.
            # Only a scalar input can be selected as the reset expression.
            continue
        declaration = re.sub(r"\[[^]]*\]", "", declaration)
        for part in declaration.split(","):
            part = part.split("=", 1)[0]
            match = re.search(r"([A-Za-z_][A-Za-z_0-9]*)\s*$", part)
            if match:
                inputs.append(match.group(1))
    resets = inputs
    reset = next((x for x in resets if "rst" in x.lower() or "reset" in x.lower()), None)
    return clock, reset
