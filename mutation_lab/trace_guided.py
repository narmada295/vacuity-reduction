"""Trace-grounded RTL mutation prototype; the legacy campaign is unchanged.

Templates identify possible faults. Golden execution supplies the event history;
golden/mutant execution supplies every reported value and divergence cycle.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .campaign import direct_config
from .core import Config, Edit, apply_edits, digest, read_spec, rtl_fingerprint, write_json
from .formal import lint
from .planner import LLMPlanner
from .scenario_validation import _ports, _simulate, _testbench


ASSIGNMENT = re.compile(r"(?m)^[ \t]*([A-Za-z_]\w*)[ \t]*<=[ \t]*([^;\n]+);")


def _zero_literal(value: str) -> bool:
    compact = value.strip().lower().replace("_", "")
    return compact in ("0", "'0") or bool(re.fullmatch(r"\d+'[s]?[bhod]0+", compact))


@dataclass(frozen=True)
class Idea:
    id: str
    category: str
    template: str
    event: str
    target: str
    source_file: str
    description: str
    edits: tuple[Edit, ...]


@dataclass(frozen=True)
class Seed:
    id: str
    event: str
    first_event_cycle: int
    second_event_cycle: int
    cycles: tuple[dict[str, int], ...]


def _top_body(config: Config, file: Path) -> tuple[str, str] | None:
    source = file.read_text()
    match = re.search(r"\bmodule\s+" + re.escape(config.top) +
                      r"\b.*?\bendmodule\b", source, flags=re.DOTALL)
    return (source, match.group()) if match else None


def discover_ideas(config: Config, ports: list[dict], internal: set[str]) -> list[Idea]:
    """Instantiate conservative templates on simple top-level clocked updates."""
    inputs = {p["name"] for p in ports if p["direction"] == "input"}
    visible = internal | {p["name"] for p in ports if p["direction"] == "output"}
    events = {p["name"] for p in ports if p["direction"] == "input" and p["width"] == 1
              and p["name"] not in (config.clock, config.reset)}
    ideas: list[Idea] = []

    def add(category: str, template: str, event: str, target: str,
            source_name: str, description: str, edits: tuple[Edit, ...]) -> None:
        ideas.append(Idea(f"c{len(ideas) + 1:04d}", category, template, event,
                          target, source_name, description, edits))

    for source_name in config.sources:
        result = _top_body(config, config.design_root / source_name)
        if result is None:
            continue
        source, body = result
        statements = [(match.group(0), match.group(1), match.group(2).strip())
                      for match in ASSIGNMENT.finditer(body)
                      if match.group(1) in visible]
        counters: list[tuple[str, str, str, str]] = []
        updates: list[tuple[str, str, str]] = []
        for old, target, rhs in statements:
            counter = re.fullmatch(r"\((.+)\)\s*\?\s*([^:]+)\s*:\s*" +
                                   re.escape(target) + r"\s*\+\s*1\s*", rhs)
            if counter and config.reset and re.search(r"\b" + re.escape(config.reset) +
                                                      r"\b", counter.group(1)):
                counters.append((old, target, counter.group(1).strip(),
                                 counter.group(2).strip()))
            for event in sorted(events):
                if (re.search(r"\(\s*" + re.escape(event) + r"\s*\)\s*\?", rhs)
                        and re.search(r":\s*" + re.escape(target) + r"\s*\)\s*$", rhs)):
                    updates.append((old, target, event))
        if not updates:
            continue
        header = re.search(r"\bmodule\s+" + re.escape(config.top) +
                           r"\b[^;]*;", body, flags=re.DOTALL)
        for update_old, update_target, event in updates:
            if event not in inputs:
                continue
            for counter_old, counter_target, guard, reset_value in counters:
                if counter_target == update_target:
                    continue
                counter_new = counter_old.replace("(" + guard + ")",
                                                  "(" + guard + " | " + event + ")", 1)
                if counter_new != counter_old:
                    add("interaction", "restart_on_update", event, counter_target,
                        str(source_name),
                        f"A {event} update while {counter_target} is running restarts {counter_target}",
                        (Edit(str(source_name), counter_old, counter_new),))
                event_group = re.search(r"\(\s*" + re.escape(event) + r"\s*\)\s*\?",
                                        update_old)
                if event_group and _zero_literal(reset_value):
                    event_text = event_group.group(0)
                    active_guard = f"({event} && ({counter_target} == {reset_value})) ?"
                    update_new = update_old.replace(event_text, active_guard, 1)
                    add("interaction", "ignore_update_while_active", event, update_target,
                        str(source_name),
                        f"A {event} update is ignored while {counter_target} is active",
                        (Edit(str(source_name), update_old, update_new),))
            if header:
                seen_name = "mutlab_seen_" + event.lower() + "_" + update_target.lower()
                if not re.search(r"\b" + re.escape(seen_name) + r"\b", source):
                    header_old = header.group(0)
                    reset_active = config.reset or ""
                    seen_code = (f"\nreg {seen_name};\n"
                                 f"always @(posedge {config.clock}) begin\n"
                                 f"  if ({reset_active}) {seen_name} <= 1'b0;\n"
                                 f"  else if ({event}) {seen_name} <= 1'b1;\nend\n")
                    event_group = re.search(r"\(\s*" + re.escape(event) + r"\s*\)\s*\?",
                                            update_old)
                    if event_group and config.reset:
                        update_new = update_old.replace(
                            event_group.group(0), f"({event} && !{seen_name}) ?", 1)
                        add("state_sequence", "ignore_repeated_update", event, update_target,
                            str(source_name),
                            f"The first {event} updates {update_target}; later {event} events do not",
                            (Edit(str(source_name), header_old, header_old + seen_code),
                             Edit(str(source_name), update_old, update_new)))
    distinct: list[Idea] = []
    seen_edits: set[tuple[tuple[str, str, str], ...]] = set()
    for idea in ideas:
        key = tuple((edit.source_file, edit.old, edit.new) for edit in idea.edits)
        if key in seen_edits:
            continue
        seen_edits.add(key)
        distinct.append(idea)
    return distinct


def make_seeds(config: Config, ports: list[dict], events: set[str],
               max_seeds: int = 18) -> list[Seed]:
    """Exercise a first event and a reconfiguration after reachable state evolves."""
    inputs = [p for p in ports if p["direction"] == "input"
              and p["name"] not in (config.clock, config.reset)]
    wide = [p for p in inputs if p["width"] > 1]
    reset_active = 0 if config.reset and config.reset.lower().endswith(("_n", "_b")) else 1
    reset_inactive = 1 - reset_active
    seeds: list[Seed] = []
    for event in sorted(events):
        data_port = wide[0] if wide else None
        width = data_port["width"] if data_port else 0
        high = (1 << (width - 1)) if width > 2 else 0
        mask = (1 << width) - 1 if width else 0
        pairs = [(high | (3 & mask), high | (5 & mask)),
                 (high | (3 & mask), 5 & mask),
                 (3 & mask, high | (5 & mask)),
                 (3 & mask, 5 & mask)] if data_port else [(0, 0)]
        for first, second in pairs:
            for gap in (2, 3, 4):
                second_event = 1 + gap
                initial = {p["name"]: 0 for p in inputs}
                initial[config.reset] = reset_active
                cycles = [initial]
                for cycle in range(1, max(12, second_event + 8)):
                    values: dict[str, int] = {}
                    if cycle == 1:
                        values[config.reset] = reset_inactive
                    if cycle in (1, second_event):
                        values[event] = 1
                        if data_port:
                            values[data_port["name"]] = first if cycle == 1 else second
                    elif cycle in (2, second_event + 1):
                        values[event] = 0
                    cycles.append(values)
                seeds.append(Seed(f"t{len(seeds) + 1:04d}", event, 1, second_event,
                                  tuple(cycles)))
        # A later reconfiguration at a counter boundary separates "ignore while
        # active" from "ignore every later write" on designs that support it.
        if data_port:
            for second, third_cycle, third in (
                    (high | (5 & mask), 6, high | (7 & mask)),
                    (5 & mask, 8, 7 & mask)):
                values = {p["name"]: 0 for p in inputs}
                values[config.reset] = reset_active
                cycles = [values]
                for cycle in range(1, 17):
                    values = {}
                    if cycle == 1:
                        values[config.reset] = reset_inactive
                    if cycle in (1, 3, third_cycle):
                        values[event] = 1
                        values[data_port["name"]] = (
                            high | (3 & mask) if cycle == 1 else
                            second if cycle == 3 else third)
                    elif cycle in (2, 4, third_cycle + 1):
                        values[event] = 0
                    cycles.append(values)
                seeds.append(Seed(f"t{len(seeds) + 1:04d}", event, 1, 3,
                                  tuple(cycles)))
    if len(seeds) <= max_seeds:
        return seeds
    groups = {event: [seed for seed in seeds if seed.event == event] for event in sorted(events)}
    selected: list[Seed] = []
    while len(selected) < max_seeds and any(groups.values()):
        for event in sorted(events):
            if groups[event] and len(selected) < max_seeds:
                selected.append(groups[event].pop(0))
    return selected


def _capture(config: Config, source_root: Path, ports: list[dict],
             internal: set[str], seed: Seed, watched: set[str], work: Path) -> dict[tuple[int, str], str]:
    observations = [{"cycle": cycle, "signal": signal}
                    for cycle in range(len(seed.cycles)) for signal in sorted(watched & internal)]
    bench = work / "trace.sv"
    bench.write_text(_testbench(config, ports, internal, list(seed.cycles), observations))
    return _simulate(config, source_root, bench, work)


def _values(trace: dict[tuple[int, str], str], seed: Seed, signals: set[str]) -> list[dict[str, str]]:
    return [{signal: trace.get((cycle, signal), "missing") for signal in sorted(signals)}
            for cycle in range(len(seed.cycles))]


def _first_difference(golden: dict, mutant: dict, seed: Seed,
                      signals: set[str]) -> int | None:
    for cycle in range(len(seed.cycles)):
        for signal in signals:
            left, right = golden.get((cycle, signal)), mutant.get((cycle, signal))
            if left is None or right is None or any(x in left + right for x in "xz"):
                continue
            if int(left, 16) != int(right, 16):
                return cycle
    return None


def _select_trace(config: Config, candidate: Idea, directory: Path,
                  seeds: list[Seed], golden: dict[str, dict], ports: list[dict],
                  internal: set[str], work: Path) -> dict[str, Any] | None:
    outputs = {p["name"] for p in ports if p["direction"] == "output"}
    watched = outputs | {candidate.target}
    selected: dict[str, Any] | None = None
    corpus: list[dict[str, Any]] = []
    for seed in seeds:
        if seed.event != candidate.event:
            continue
        mutant = _capture(config, directory / "rtl", ports, internal, seed, watched, work)
        gold = golden[seed.id]
        corpus.append({"seed_id": seed.id, "outputs": _values(mutant, seed, outputs)})
        target_cycle = _first_difference(gold, mutant, seed, {candidate.target})
        output_cycle = _first_difference(gold, mutant, seed, outputs)
        if selected is not None or target_cycle is None or output_cycle is None:
            continue
        if (target_cycle < seed.second_event_cycle or output_cycle < seed.second_event_cycle
                or output_cycle < target_cycle):
            continue
        selected = {"seed_id": seed.id, "inputs": list(seed.cycles),
                "first_event_cycle": seed.first_event_cycle,
                "second_event_cycle": seed.second_event_cycle,
                "first_target_difference_cycle": target_cycle,
                "first_output_difference_cycle": output_cycle,
                "propagation_cycles": output_cycle - target_cycle,
                "golden": _values(gold, seed, watched),
                "mutant": _values(mutant, seed, watched)}
    if selected is not None:
        selected["observed_behavior_hash"] = digest(json.dumps(corpus, sort_keys=True).encode())
        selected["behavior_seeds"] = len(corpus)
    return selected


SELECT_SYSTEM = """Select promising RTL fault mechanisms from verified golden traces.
The input lists a specification, actual golden signal values, and applicable mutation
ideas. Return JSON with ranked_ids (a subset of idea IDs in priority order), reasons
(object mapping selected IDs to short causal explanations), and optional novel_ideas.
Each novel idea needs id beginning n, category, event (one-bit input), target (direct
internal register or output), source_file, description, and edits (1-3 exact old/new
RTL spans). Include its ID in ranked_ids to try it. Select only ideas supported by
the specification and trace. Prefer selective, multi-event faults. Do not invent
signal values or assertions. Omit unsupported ideas.
"""


def _rank_with_llm(config: Config, ideas: list[Idea], seeds: list[Seed],
                   golden: dict[str, dict], outputs: set[str], internal: set[str],
                   ports: list[dict]) -> tuple[list[Idea], dict]:
    planner = LLMPlanner(config, require_witness=False)
    seed_by_event = {seed.event: seed for seed in reversed(seeds)}
    listings = []
    for idea in ideas:
        seed = seed_by_event[idea.event]
        listings.append({"id": idea.id, "category": idea.category,
                         "template": idea.template, "description": idea.description,
                         "target": idea.target, "source_file": idea.source_file,
                         "event": idea.event,
                         "rtl_anchor": [edit.old for edit in idea.edits],
                         "input_sequence": list(seed.cycles),
                         "golden_trace": _values(golden[seed.id], seed,
                                                 outputs | {idea.target})})
    focus = config.planner.get("context_sources", [str(source) for source in config.sources])
    sources = "\n\n".join(f"RTL FILE {name}:\n" + (config.design_root / name).read_text()
                           for name in focus)
    prompt = ("SPECIFICATION:\n" + read_spec(config.spec_file) +
              "\n\nGOLDEN RTL:\n" + sources +
              "\n\nVERIFIED IDEAS AND GOLDEN TRACES:\n" + json.dumps(listings))
    if len(prompt) > int(config.planner.get("max_context_chars", 100000)):
        raise ValueError("Trace selection context exceeds --max-context-chars")
    response = planner._request(SELECT_SYSTEM, prompt)
    ids = response.get("ranked_ids")
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        raise ValueError("LLM ranking needs ranked_ids as a list of idea IDs")
    lookup = {idea.id: idea for idea in ideas}
    events = {p["name"] for p in ports if p["direction"] == "input" and p["width"] == 1
              and p["name"] not in (config.clock, config.reset)}
    raw_novel = response.get("novel_ideas", [])
    if not isinstance(raw_novel, list) or len(raw_novel) > 5:
        raise ValueError("novel_ideas must be a list of at most five ideas")
    for raw in raw_novel:
        if not isinstance(raw, dict) or any(not isinstance(raw.get(field), str) or
                                             not raw[field].strip() for field in
                                             ("id", "category", "event", "target", "source_file", "description")):
            raise ValueError("Each novel idea needs id, category, event, target, source_file, description")
        if (not re.fullmatch(r"n\d{1,4}", raw["id"]) or raw["id"] in lookup or
                raw["event"] not in events or raw["event"] not in seed_by_event or
                raw["target"] not in (internal | outputs) or
                raw["category"] not in ("simple", "timing", "state_sequence", "interaction")):
            raise ValueError("Novel idea has invalid ID, event, target, or category")
        raw_edits = raw.get("edits")
        if not isinstance(raw_edits, list) or not 1 <= len(raw_edits) <= 3:
            raise ValueError("Novel idea needs one to three exact edits")
        edits = tuple(Edit.parse(item, config) for item in raw_edits)
        if any(edit.source_file != raw["source_file"] for edit in edits):
            raise ValueError("Novel idea edits must use its declared source_file")
        lookup[raw["id"]] = Idea(raw["id"], raw["category"], "llm_constrained",
                                  raw["event"], raw["target"], raw["source_file"],
                                  raw["description"], edits)
    if any(item not in lookup for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("LLM ranking contains unknown or repeated idea IDs")
    return [lookup[item] for item in ids], response


def run_trace_campaign(args: Any) -> None:
    """Generate measured mutants while leaving evaluation assertions untouched."""
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use a fresh --out directory: {output}")
    if not args.reset:
        raise ValueError("Trace-guided generation requires --reset for reachable traces")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.direct.json", direct_config(args))
    config = Config.load(output / "config.direct.json")
    with tempfile.TemporaryDirectory(prefix="mutlab_trace_") as temporary:
        work = Path(temporary)
        ports, internal = _ports(config, config.design_root, work)
        if config.reset not in {p["name"] for p in ports if p["direction"] == "input"}:
            raise ValueError("Configured reset must be a top-level input")
        ideas = discover_ideas(config, ports, internal)
        if not ideas:
            raise ValueError("No applicable top-level sequential templates found; see trace-guided limits")
        discovered_count = len(ideas)
        seeds = make_seeds(config, ports, {idea.event for idea in ideas}, args.max_seeds)
        exercised = {seed.event for seed in seeds}
        ideas = [idea for idea in ideas if idea.event in exercised]
        watched = {idea.target for idea in ideas}
        outputs = {p["name"] for p in ports if p["direction"] == "output"}
        golden = {seed.id: _capture(config, config.design_root, ports, internal,
                                    seed, outputs | watched, work) for seed in seeds}
        write_json(output / "golden_traces.json", [
            {"seed": asdict(seed), "values": _values(golden[seed.id], seed,
                                                      outputs | watched)} for seed in seeds])
        selection: dict[str, Any] = {"mode": args.planner, "ideas": len(ideas)}
        if args.planner == "llm":
            try:
                ideas, response = _rank_with_llm(config, ideas, seeds, golden,
                                                 outputs, internal, ports)
                selection["response"] = response
                selection["selected_ids"] = [idea.id for idea in ideas]
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                selection["error"] = str(exc)
                selection["fallback"] = "template_order"
        write_json(output / "selection.json", selection)
        seen_rtl = {rtl_fingerprint(config, config.design_root): "golden"}
        clusters: dict[str, str] = {}
        records = []
        accepted = 0
        for idea in ideas:
            if accepted >= args.count:
                break
            identifier = f"m{len(records) + 1:04d}"
            directory = output / "mutants" / identifier
            directory.mkdir(parents=True)
            record: dict[str, Any] = {"id": identifier, "category": idea.category,
                                      "status": "pending", "idea": {
                                          **{key: value for key, value in asdict(idea).items()
                                             if key != "edits"},
                                          "edits": [asdict(edit) for edit in idea.edits]}}
            try:
                record["edit"] = apply_edits(config, idea.edits, directory)
                fingerprint = rtl_fingerprint(config, directory / "rtl")
                record["rtl_fingerprint"] = fingerprint
                if fingerprint in seen_rtl:
                    record.update(status="duplicate_rtl", duplicate_of=seen_rtl[fingerprint])
                else:
                    seen_rtl[fingerprint] = identifier
                    record["lint"] = lint(config, directory)
                    if record["lint"]["status"] != "passed":
                        record.update(status="lint_rejected",
                                      reason=record["lint"].get("detail", ""))
                    else:
                        trace = _select_trace(config, idea, directory, seeds, golden,
                                              ports, internal, work)
                        if trace is None:
                            record.update(status="no_verified_trace",
                                          reason="No reset-reachable seed showed a second-event state difference followed by an output difference")
                        else:
                            record["measured_trace"] = trace
                            signature = trace["observed_behavior_hash"]
                            prior = clusters.get(signature)
                            record["behavior_cluster"] = prior or identifier
                            if prior:
                                record.update(status="duplicate_behavior", duplicate_of=prior,
                                              reason="Same output traces on all generated seeds")
                            else:
                                if args.formal_compare:
                                    from .differential_formal import compare_outputs
                                    record["formal_compare"] = compare_outputs(
                                        config, config.design_root, directory / "rtl", ports,
                                        directory / "formal_compare")
                                if record.get("formal_compare", {}).get("status") == "equivalent":
                                    record["status"] = "formal_equivalent_conflict"
                                else:
                                    record["status"] = "generated_unqualified"
                                    accepted += 1
                                    clusters[signature] = identifier
            except (OSError, ValueError, RuntimeError) as exc:
                record.update(status="generation_error", reason=str(exc)[:1200])
            write_json(directory / "record.json", record)
            records.append(record)
            print(f"{identifier}: {record['status']} ({idea.template})")
    write_json(output / "summary.json", {"config": str(output / "config.direct.json"),
                                        "planner_mode": args.planner,
                                        "counts": {status: sum(r["status"] == status for r in records)
                                                   for status in sorted({r["status"] for r in records})},
                                        "candidate_ideas": discovered_count,
                                        "selected_ideas": len(ideas),
                                        "attempts": len(records), "accepted": accepted,
                                        "requested": args.count,
                                        "mutants": [r["id"] for r in records]})
    print(f"Results: {output / 'summary.json'}")
