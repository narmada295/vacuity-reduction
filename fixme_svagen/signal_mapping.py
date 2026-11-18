"""Specification-grounded signal mapping for new evaluation designs.

Uses the same DeepSeek client as the FIXME SANGAM adaptation. Exact declaration
and quotation checks validate provenance, not the semantics of model summaries.
No assertions, teacher outputs, or formal outcomes are inputs to this stage.
"""

import hashlib
import json
from pathlib import Path
import re

from .generate import extract_json_object


SYSTEM = "You are SANGAM's specification signal mapper. Return JSON only, without assertions."
POLICY = "documented_ports_and_module_internals_v1"


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def mapping_prompt(spec, rtl, candidates):
    return (
        "Map the listed candidate signals to the supplied original specification. "
        "Read the RTL to identify which declared signal implements documented behavior, "
        "even if the specification does not mention its exact identifier. "
        "Only evaluate candidates in this list; there are other batches.\n"
        "Return {\"signals\":[{\"name\":\"exact candidate name\",\"selected\":true,"
        "\"information\":\"short factual description\",\"spec_quote\":\"verbatim supporting passage\","
        "\"reason\":\"why selected or excluded\"}]}. Include exactly one entry per candidate.\n"
        "Select every port, including clock and reset. Select internal signals when they "
        "implement a documented function, such as a counter, state, FIFO status, data path, "
        "interrupt condition, timing setting, or control condition. Exclude incidental "
        "implementation bookkeeping without a documented behavioral role.\n"
        "For every selected internal signal, copy a supporting contiguous passage of "
        "20 to 400 characters from the specification into spec_quote. It may describe "
        "the function without naming the signal. A shared passage may support related "
        "signals. Do not invent or paraphrase quotations. Ports may have an empty quote "
        "when no explicit requirement is documented; describe their factual RTL role.\n"
        "Keep information concise (one or two sentences). Distinguish specification "
        "requirements from implementation observations. Do not invent timing, reset "
        "guarantees, environmental assumptions, or behavioral requirements. Do not infer "
        "a new requirement merely because an RTL expression exists. Respect the selected "
        "module boundary, not functions only implemented by its parent or children. "
        "Do not emit SVAs or formal outcome labels.\n\n"
        "Candidates (compiler-validated original declarations; parameters excluded):\n"
        + json.dumps(candidates, ensure_ascii=False)
        + "\n\nOriginal specification:\n" + spec
        + "\n\nRTL dependency closure:\n```systemverilog\n" + rtl + "\n```"
    )


def validate_mapping(payload, candidates, spec, filter_unsupported=False):
    entries = payload.get("signals")
    if not isinstance(entries, list):
        raise ValueError("Mapper must return a signals list")
    expected = {item["name"]: item for item in candidates}
    actual = [item.get("name") for item in entries if isinstance(item, dict)]
    if len(actual) != len(entries) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("Mapper must classify every supplied candidate exactly once")
    result = []
    spec_text = normalized(spec)
    for original in entries:
        entry = dict(original)
        candidate = expected[entry["name"]]
        if not isinstance(entry.get("selected"), bool):
            raise ValueError("Selected must be Boolean: " + entry["name"])
        if candidate["kind"] == "port" and not entry["selected"]:
            raise ValueError("A declared port was excluded: " + entry["name"])
        if not isinstance(entry.get("reason"), str) or not entry["reason"].strip():
            raise ValueError("Missing selection reason: " + entry["name"])
        quote = entry.get("spec_quote", "")
        quote_error = None
        if not isinstance(quote, str) or (quote and normalized(quote) not in spec_text):
            quote_error = "Supporting passage is not in the specification: " + entry["name"]
        if entry["selected"]:
            if not isinstance(entry.get("information"), str) or not entry["information"].strip():
                raise ValueError("Selected signal lacks a description: " + entry["name"])
            # Short author-written HJSON descriptions such as "RX enable" are
            # sufficient provenance; the requested longer quote is preferred.
            if candidate["kind"] == "internal" and (not isinstance(quote, str) or not 8 <= len(quote) <= 400):
                quote_error = "Selected internal signal lacks a supporting passage: " + entry["name"]
        if quote_error:
            if not filter_unsupported:
                raise ValueError(quote_error)
            entry['proposed_spec_quote'] = quote
            entry['spec_quote'] = ''
            entry['evidence_validation_error'] = quote_error
            if candidate['kind'] == 'internal' and entry['selected']:
                entry['proposed_selected'] = True
                entry['selected'] = False
                entry['selection_filter_reason'] = 'Excluded: no valid supporting specification passage.'
        result.append(dict(entry, **{k: v for k, v in candidate.items() if k != "name"}))
    return sorted(result, key=lambda item: list(expected).index(item["name"]))


def map_documented_signals(llm, spec, rtl, candidates, directory, batch_size=40):
    """Save each request before calling the model and reuse only matching batches."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    accepted = []
    for number, start in enumerate(range(0, len(candidates), batch_size)):
        batch = candidates[start:start + batch_size]
        user = mapping_prompt(spec, rtl, batch)
        request = {"system": SYSTEM, "user": user, "model": llm.model, "policy": POLICY}
        fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        folder = directory / ("batch_" + str(number).zfill(2))
        folder.mkdir(exist_ok=True)
        request_file, result_file = folder / "request.json", folder / "result.json"
        if request_file.exists() and json.loads(request_file.read_text()) != request:
            raise ValueError("Saved mapping batch has different inputs: " + str(folder))
        request_file.write_text(json.dumps(request, indent=2) + "\n")
        if result_file.exists():
            saved = json.loads(result_file.read_text())
            if saved["fingerprint"] != fingerprint:
                raise ValueError("Saved mapping fingerprint differs")
            entries = validate_mapping({"signals": saved["signals"]}, batch, spec, filter_unsupported=True)
        else:
            # Preserve and reuse an already returned, now-validated artifact
            # after interruption rather than calling the endpoint again.
            cached = None
            for response_file in sorted(folder.glob("response_*.txt")):
                try:
                    cached = validate_mapping(extract_json_object(response_file.read_text()), batch, spec, filter_unsupported=True)
                    break
                except (ValueError, KeyError, TypeError):
                    pass
            correction = ""
            for attempt in range(3) if cached is None else ():
                print("MAPPING", directory.name, "batch", number + 1,
                      "candidates", len(batch), "attempt", attempt + 1, flush=True)
                response = llm.chat(SYSTEM, user + correction)
                (folder / ("response_" + str(attempt) + ".txt")).write_text(response)
                try:
                    entries = validate_mapping(extract_json_object(response), batch, spec, filter_unsupported=True)
                    break
                except (ValueError, KeyError, TypeError) as error:
                    (folder / ("validation_error_" + str(attempt) + ".txt")).write_text(str(error))
                    if attempt == 2:
                        raise
                    correction = "\n\nCorrect this validation error and return the entire batch again: " + str(error)
            if cached is not None:
                entries = cached
            result_file.write_text(json.dumps({"fingerprint": fingerprint, "model": llm.model,
                                               "signals": entries}, indent=2) + "\n")
        accepted.extend(entries)
    if len(accepted) != len(candidates):
        raise ValueError("Incomplete candidate mapping")
    return accepted
