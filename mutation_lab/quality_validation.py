"""Small, design-independent simulation checks for candidate RTL mutants.

These probes are a quality filter, not an exhaustive equivalence or spec proof.
They use port widths and direct register names discovered from the RTL.
"""

from __future__ import annotations

import json
import itertools
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import Config, Edit, digest
from .scenario_validation import _ports, _simulate, _testbench


@dataclass(frozen=True)
class Probe:
    id: str
    kind: str
    event: tuple[str, ...] | None
    cycles: tuple[dict[str, int], ...]
    reset_overlap_cycle: int | None = None


def _reset_active(name: str | None) -> int:
    if name and (name.lower().endswith(("_n", "_b")) or
                 name.lower().startswith(("nreset", "nrst"))):
        return 0
    return 1


def make_probes(config: Config, ports: list[dict], limit: int = 32) -> list[Probe]:
    inputs = [p for p in ports if p["direction"] == "input"
              and p["name"] not in (config.clock, config.reset)]
    events = [p["name"] for p in inputs if p["width"] == 1]
    wide = [p for p in inputs if p["width"] > 1]
    probes: list[Probe] = []

    def add(kind: str, event: tuple[str, ...] | None, data: dict | None, first: int = 0,
            second_cycle: int | None = None, second: int = 0,
            reset_overlap: bool = False) -> None:
        active = _reset_active(config.reset)
        initial = {p["name"]: 0 for p in inputs}
        if config.reset:
            initial[config.reset] = active
        cycles: list[dict[str, int]] = [initial]
        length = max(14, (second_cycle or 1) + 9)
        for cycle in range(1, length):
            values: dict[str, int] = {}
            if config.reset and cycle == 1:
                values[config.reset] = 1 - active
            if cycle in (1, second_cycle):
                if event:
                    values.update({name: 1 for name in event})
                if data:
                    values[data["name"]] = first if cycle == 1 else second
                if reset_overlap and cycle == second_cycle and config.reset:
                    values[config.reset] = active
            elif cycle in (2, (second_cycle + 1) if second_cycle else -1):
                if event:
                    values.update({name: 0 for name in event})
                if reset_overlap and cycle == second_cycle + 1 and config.reset:
                    values[config.reset] = 1 - active
            cycles.append(values)
        probes.append(Probe(f"p{len(probes) + 1:03d}", kind, event, tuple(cycles),
                            second_cycle if reset_overlap else None))

    event_groups = [(name,) for name in events]
    if len(events) <= 8:
        event_groups.extend(itertools.combinations(events, 2))
    elif len(events) > 8:
        event_groups.extend((events[index], events[(index + 1) % len(events)])
                            for index in range(len(events)))
    stimuli = ([(event, data) for event in event_groups for data in wide]
               if event_groups and wide else
               [(event, None) for event in event_groups] if event_groups else
               [(None, data) for data in wide] if wide else [(None, None)])
    groups: dict[tuple[tuple[str, ...] | None, str | None], list[Probe]] = {}
    for event, data in stimuli:
        first_index = len(probes)
        key = (event, data["name"] if data else None)
        width = data["width"] if data else 0
        mask = (1 << width) - 1 if width else 0
        high = (1 << (width - 1)) if width > 1 else 0
        if not event and not data:
            add("idle", None, None)
            groups[key] = probes[first_index:]
            continue
        first_values = [0, 1, high | (3 & mask), high | (5 & mask)] if data else [0]
        for value in dict.fromkeys(first_values):
            add("first", event, data, value)
        # Cover both directions of a data-bit transition. This exposes state
        # faults such as accepting an initial enable but ignoring a later
        # disable, without knowing what any particular bit means to the DUT.
        pairs = ([(high | (3 & mask), 3 & mask),
                  (high | (3 & mask), high | (5 & mask)),
                  (3 & mask, high | (5 & mask)),
                  (high | (5 & mask), 0)] if data else [(0, 0)])
        for first, second in pairs:
            for gap in (2, 3, 4, 5):
                add("repeat", event, data, first, 1 + gap, second)
        if config.reset:
            add("reset_overlap", event, data, high | (3 & mask), 6,
                high | (5 & mask), True)
        group = probes[first_index:]
        groups[key] = ([p for p in group if p.kind == "first"][:1] +
                       [p for p in group if p.kind == "reset_overlap"] +
                       [p for p in group if p.kind == "repeat"] +
                       [p for p in group if p.kind == "first"][1:])
    # Round-robin across input combinations keeps a small cap from covering
    # only the first data bus or first one-bit event.
    if len(probes) <= limit:
        return probes
    selected: list[Probe] = []
    while len(selected) < limit and any(groups.values()):
        for key in groups:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
    return selected


def _first_diff(golden: dict, mutant: dict, cycles: int, signals: set[str]) -> int | None:
    for cycle in range(cycles):
        for signal in signals:
            left, right = golden.get((cycle, signal)), mutant.get((cycle, signal))
            if not left or not right or any(x in left + right for x in "xz"):
                continue
            if int(left, 16) != int(right, 16):
                return cycle
    return None


def _literal(value: str) -> bool:
    return bool(re.fullmatch(r"(?:\d+'[sS]?[bBoOdDhH])?[0-9a-fA-F_]+", value.strip()))


def obvious_constant_edit(edits: tuple[Edit, ...]) -> bool:
    """Catch a replaced assignment whose whole new value is constant.

    A conditional override added after the original assignment is different:
    it may be a narrow fault and must be judged from behavior, not this regex.
    """
    assignment = re.compile(r"\b([A-Za-z_]\w*)\s*<=\s*([^;]+);")
    for edit in edits:
        old_assignments = assignment.findall(edit.old)
        new_assignments = assignment.findall(edit.new)
        for name, rhs in new_assignments:
            old_values = [value for target, value in old_assignments if target == name]
            new_values = [value for target, value in new_assignments if target == name]
            if len(old_values) != 1 or len(new_values) != 1:
                continue
            compact = rhs.strip()
            constant = _literal(compact)
            ternary = re.fullmatch(r".+\?\s*([^?:]+)\s*:\s*([^?:]+)", compact, flags=re.S)
            if ternary:
                left, right = ternary.group(1).strip(), ternary.group(2).strip()
                constant = _literal(left) and _literal(right) and left == right
            if constant and not _literal(old_values[0]):
                return True
    return False


SIGNAL_REF = re.compile(r"([A-Za-z_]\w*)(?:\[(\d+)\])?\Z")


def _valid_ref(value: str, known: set[str]) -> bool:
    if value.isdigit():
        return True
    match = SIGNAL_REF.fullmatch(value)
    return bool(match and match.group(1) in known)


def _conditions(trigger: str, known: set[str]) -> list[tuple[str, str, str]]:
    operand = r"[A-Za-z_]\w*(?:\[\d+\])?|\d+"
    pattern = re.compile(rf"(?<!\w)({operand})\s*(==|!=|>=|<=|>|<|=)\s*({operand})(?![\w\[])")
    return [(left, op, right) for left, op, right in pattern.findall(trigger)
            if _valid_ref(left, known) and _valid_ref(right, known)]


def _plan_conditions(plan: Any, known: set[str]) -> list[tuple[str, str, str]]:
    structured = getattr(plan, "trigger_predicates", ())
    if structured:
        return [(item["left"], item["op"], item["right"])
                for item in structured if _valid_ref(item["left"], known)
                and _valid_ref(item["right"], known)]
    return _conditions(plan.trigger, known)


def _condition_holds(conditions: list[tuple[str, str, str]], probe: Probe,
                     cycle: int, golden: dict, inputs: set[str]) -> bool | None:
    values: dict[str, int] = {}
    for step in probe.cycles[:cycle + 1]:
        values.update(step)
    operations = {"==": lambda a, b: a == b, "=": lambda a, b: a == b,
                  "!=": lambda a, b: a != b, ">=": lambda a, b: a >= b,
                  "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
                  "<": lambda a, b: a < b}
    for left, op, right in conditions:
        def value(name: str) -> int | None:
            if name.isdigit():
                return int(name)
            match = SIGNAL_REF.fullmatch(name)
            if match is None:
                return None
            signal, bit = match.groups()
            if signal in inputs:
                result = values.get(signal, 0)
            else:
                raw = golden.get((cycle - 1, signal)) if cycle > 0 else None
                if not raw or any(x in raw for x in "xz"):
                    return None
                result = int(raw, 16)
            return (result >> int(bit)) & 1 if bit is not None else result
        a, b = value(left), value(right)
        if a is None or b is None:
            return None
        if not operations[op](a, b):
            return False
    return True


def _trace_excerpt(probe: Probe, golden: dict, mutant: dict,
                   signals: set[str], through: int) -> list[dict]:
    """Show a short replayable waveform around an observed difference."""
    values: dict[str, int] = {}
    rows = []
    for cycle in range(min(len(probe.cycles), through + 3)):
        values.update(probe.cycles[cycle])
        rows.append({"cycle": cycle, "inputs": dict(values),
                     "golden": {name: golden.get((cycle, name)) for name in sorted(signals)},
                     "mutant": {name: mutant.get((cycle, name)) for name in sorted(signals)}})
    return rows


class BehaviorAuditor:
    def __init__(self, config: Config, max_probes: int = 32):
        missing = [name for name in ("verilator", "iverilog", "vvp") if not shutil.which(name)]
        if missing:
            raise RuntimeError("Behavior audit needs tools on PATH: " + ", ".join(missing))
        if max_probes < 1:
            raise ValueError("Behavior audit needs at least one probe")
        self.config = config
        self.temp = tempfile.TemporaryDirectory(prefix="mutlab_quality_")
        self.work = Path(self.temp.name)
        self.ports, self.internal = _ports(config, config.design_root, self.work)
        self.inputs = {p["name"] for p in self.ports if p["direction"] == "input"}
        self.outputs = {p["name"] for p in self.ports if p["direction"] == "output"}
        if not self.outputs:
            raise ValueError("Behavior audit needs at least one top-level output")
        self.probes = make_probes(config, self.ports, max_probes)
        self.watched = self.outputs | self.internal
        self.golden = {probe.id: self._run(config.design_root, probe)
                       for probe in self.probes}
        self.signatures: dict[str, str] = {}

    def close(self) -> None:
        self.temp.cleanup()

    def __enter__(self) -> "BehaviorAuditor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run(self, source_root: Path, probe: Probe) -> dict:
        observations = [{"cycle": cycle, "signal": signal}
                        for cycle in range(len(probe.cycles))
                        for signal in sorted(self.internal)]
        bench = self.work / "quality_probe.sv"
        bench.write_text(_testbench(self.config, self.ports, self.internal,
                                   list(probe.cycles), observations))
        return _simulate(self.config, source_root, bench, self.work)

    def assess(self, mutant_root: Path, plan: Any, edits: tuple[Edit, ...]) -> dict[str, Any]:
        constant_rhs_edit = obvious_constant_edit(edits)
        target = plan.target_signal
        watched_target = {target} if target in self.watched else set()
        conditions = _plan_conditions(plan, self.watched | self.inputs)
        structured = getattr(plan, "trigger_predicates", ())
        if structured and len(conditions) != len(structured):
            return {"status": "uncheckable_trigger",
                    "reason": "At least one planned trigger predicate names a signal unavailable to the simulator",
                    "trigger_predicates": structured}
        raw_controls = getattr(plan, "control_predicates", ())
        controls = [[(item["left"], item["op"], item["right"])
                     for item in group] for group in raw_controls]
        if any(not _valid_ref(ref, self.watched | self.inputs)
               for group in controls for left, _, right in group for ref in (left, right)):
            return {"status": "uncheckable_control",
                    "reason": "At least one control predicate names a signal unavailable to the simulator",
                    "control_predicates": raw_controls}
        signature: list[dict] = []
        differences: list[dict] = []
        ordinary_difference = None
        reset_side_effect = None
        unknown_result = None
        off_trigger: list[dict] = []
        control_mismatch: list[dict] = []
        control_hits = [0] * len(controls)
        trigger_unknown = False
        trace_excerpt = None
        golden_target_values: set[int] = set()
        mutant_target_values: set[int] = set()
        aligned = False
        checked_condition = False
        for probe in self.probes:
            mutant = self._run(mutant_root, probe)
            golden = self.golden[probe.id]
            if watched_target:
                for cycle in range(len(probe.cycles)):
                    for source, values in ((golden, golden_target_values),
                                           (mutant, mutant_target_values)):
                        raw = source.get((cycle, target))
                        if raw and not any(char in raw for char in "xz"):
                            values.add(int(raw, 16))
            if unknown_result is None:
                for cycle in range(len(probe.cycles)):
                    for signal in self.outputs | watched_target:
                        reference = golden.get((cycle, signal))
                        candidate = mutant.get((cycle, signal))
                        if reference and not any(x in reference for x in "xz") and (
                                candidate is None or any(x in candidate for x in "xz")):
                            unknown_result = {"probe": probe.id, "cycle": cycle,
                                              "signal": signal,
                                              "golden": reference, "mutant": candidate}
                            break
                    if unknown_result:
                        break
            target_cycle = _first_diff(golden, mutant, len(probe.cycles), watched_target)
            output_cycle = _first_diff(golden, mutant, len(probe.cycles), self.outputs)
            signature.append({"probe": probe.id,
                              "signals": [{name: mutant.get((cycle, name), "missing")
                                           for name in sorted(self.watched)}
                                          for cycle in range(len(probe.cycles))]})
            if target_cycle is None and output_cycle is None:
                continue
            entry = {"probe": probe.id, "kind": probe.kind,
                     "first_target_difference": target_cycle,
                     "first_output_difference": output_cycle}
            if output_cycle is not None:
                entry["output_values"] = {
                    name: {"golden": golden.get((output_cycle, name)),
                           "mutant": mutant.get((output_cycle, name))}
                    for name in sorted(self.outputs)
                    if golden.get((output_cycle, name)) != mutant.get((output_cycle, name))}
            if target_cycle is not None:
                entry["target_values"] = {
                    "golden": golden.get((target_cycle, target)),
                    "mutant": mutant.get((target_cycle, target))}
            differences.append(entry)
            if probe.kind == "first" and ordinary_difference is None:
                ordinary_difference = entry
            if probe.reset_overlap_cycle is not None and output_cycle is not None:
                cycle = probe.reset_overlap_cycle
                if any(golden.get((cycle, name)) != mutant.get((cycle, name))
                       for name in self.outputs):
                    reset_side_effect = entry
            event_cycle = min(cycle for cycle in (target_cycle, output_cycle)
                              if cycle is not None)
            if trace_excerpt is None:
                named = {SIGNAL_REF.fullmatch(ref).group(1)
                         for left, _, right in conditions + [item for group in controls for item in group]
                         for ref in (left, right)
                         if SIGNAL_REF.fullmatch(ref)}
                observed = self.outputs | watched_target | (named & self.internal)
                rows = _trace_excerpt(probe, golden, mutant, observed,
                                      max(cycle for cycle in
                                          (target_cycle, output_cycle)
                                          if cycle is not None))
                transitions = {}
                if watched_target:
                    for side in ("golden", "mutant"):
                        transitions[side] = [row["cycle"] for previous, row in
                            zip(rows, rows[1:]) if previous[side].get(target) != row[side].get(target)]
                trace_excerpt = {"probe": probe.id,
                                 "rows": rows,
                                 "target_transition_cycles": transitions}
            if conditions and event_cycle is not None:
                # A plan may explicitly permit a bounded latency from cause
                # to first state difference. Do not let an unrelated trigger
                # many cycles earlier justify an off-trigger mutation.
                window = getattr(plan, "trigger_window_cycles", 0)
                start = max(0, event_cycle - window)
                if self.config.reset and event_cycle > 0:
                    start = max(start, 1)
                outcomes = [_condition_holds(conditions, probe, cycle,
                                             golden, self.inputs)
                            for cycle in range(start, event_cycle + 1)]
                checked_condition |= any(item is not None for item in outcomes)
                if any(item is True for item in outcomes):
                    aligned = True
                elif any(item is None for item in outcomes):
                    trigger_unknown = True
                else:
                    off_trigger.append(entry)
            for index, control in enumerate(controls):
                if _condition_holds(control, probe, event_cycle, golden, self.inputs):
                    control_mismatch.append({"control": index + 1, **entry})
        for probe in self.probes:
            golden = self.golden[probe.id]
            for index, control in enumerate(controls):
                if any(_condition_holds(control, probe, cycle, golden, self.inputs)
                       for cycle in range(len(probe.cycles))):
                    control_hits[index] += 1
        fingerprint = digest(json.dumps(signature, sort_keys=True).encode())
        observed_stuck_constant = (len(golden_target_values) > 1 and
                                   len(mutant_target_values) == 1)
        witness_check = None
        if differences:
            first = differences[0]
            target_cycle = first["first_target_difference"]
            output_cycle = first["first_output_difference"]
            if target_cycle is not None and (output_cycle is None or target_cycle <= output_cycle):
                values = first["target_values"]
                witness_check = {"probe": first["probe"], "cycle": target_cycle,
                                 "signal": target, **values}
            elif first.get("output_values"):
                signal = sorted(first["output_values"])[0]
                witness_check = {"probe": first["probe"], "cycle": output_cycle,
                                 "signal": signal, **first["output_values"][signal]}
        elif unknown_result:
            witness_check = {"probe": unknown_result["probe"],
                             "cycle": unknown_result["cycle"],
                             "signal": unknown_result["signal"],
                             "golden": unknown_result["golden"],
                             "mutant": unknown_result["mutant"]}
        result: dict[str, Any] = {"status": "passed", "probes": len(self.probes),
                                  "different_probes": len(differences),
                                  "difference_fraction": round(len(differences) / len(self.probes), 3),
                                  "constant_rhs_edit": constant_rhs_edit,
                                  "observed_stuck_constant": observed_stuck_constant,
                                  "golden_target_values": len(golden_target_values),
                                  "mutant_target_values": len(mutant_target_values),
                                  "different_first_probes": sum(
                                      item["kind"] == "first" for item in differences),
                                  "total_first_probes": sum(
                                      probe.kind == "first" for probe in self.probes),
                                  "target_observed": bool(watched_target),
                                  "output_witness": any(item["first_output_difference"] is not None
                                                        for item in differences),
                                  "evidence": differences[:8],
                                  "witness_inputs": next((list(probe.cycles)
                                      for probe in self.probes if differences and
                                      probe.id == differences[0]["probe"]), None),
                                  "ordinary_difference": ordinary_difference,
                                  "reset_side_effect": reset_side_effect,
                                  "unknown_result": unknown_result,
                                  "off_trigger_evidence": off_trigger[:4],
                                  "control_mismatch_evidence": control_mismatch[:4],
                                  "control_probe_hits": control_hits,
                                  "unobserved_controls": [index + 1 for index, hits in
                                                          enumerate(control_hits) if hits == 0],
                                  "trigger_unknown": trigger_unknown,
                                  "trace_excerpt": trace_excerpt,
                                  "witness_check": witness_check,
                                  "trigger_conditions": conditions,
                                  "trigger_source": "structured" if structured else "text",
                                  "trigger_checked": checked_condition,
                                  "trigger_aligned": aligned if checked_condition else None,
                                  "behavior_signature": fingerprint}
        if unknown_result and unknown_result["mutant"] is None:
            result.update(status="missing_observation",
                          reason="Simulator did not produce a target or output observation")
        elif observed_stuck_constant:
            result.update(status="stuck_constant",
                          reason="The target stays at one value across probes where golden RTL varies")
        elif ordinary_difference and getattr(plan, "preserve_first_operation", False):
            result.update(status="control_operation_changed",
                          reason="The plan says the first ordinary operation must work, but it differs")
        elif reset_side_effect and not re.search(r"\breset\b", plan.trigger, re.I):
            result.update(status="unexpected_reset_effect",
                          reason="The mutant changes behavior on reset with an input event, outside the plan")
        elif control_mismatch:
            result.update(status="control_mismatch",
                          reason="The mutant differs in a scenario the plan says must remain correct")
        elif off_trigger:
            result.update(status="trigger_mismatch",
                          reason="At least one first divergence occurs before or outside the plan's trigger")
        elif (watched_target and differences and not any(
                item["first_target_difference"] is not None for item in differences)
                and not (unknown_result and unknown_result["signal"] == target)):
            result.update(status="target_unchanged",
                          reason="The planned target signal did not differ on any probe")
        elif not differences and not unknown_result:
            result.update(status="unobserved",
                          reason="No reset-starting probe showed a state or output difference")
        elif not differences and unknown_result:
            result["validation_level"] = "unknown_only"
            result["note"] = "Mutant has an unknown value where golden RTL is known"
        elif not result["output_witness"]:
            result["validation_level"] = "internal_state_only"
            result["note"] = "Only a target-state difference was observed; output effect is unverified"
        else:
            result["validation_level"] = "output_difference"
        if unknown_result and differences:
            result["note"] = "Mutant also has an unknown value where golden RTL is known"
        if result["status"] == "passed" and fingerprint in self.signatures:
            result.update(status="duplicate_behavior",
                          reason=f"Observed state and output behavior matches {self.signatures[fingerprint]} on every probe",
                          duplicate_of=self.signatures[fingerprint])
        return result

    def accept(self, signature: str, identifier: str) -> None:
        self.signatures[signature] = identifier
