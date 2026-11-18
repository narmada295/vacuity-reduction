"""Execute model-proposed stimuli and characterize measured RTL trajectories.

No stimulus patterns are invented here. Python validates and expands the model's
JSON, drives both designs identically, and checks explicit observations.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from .core import Config, digest, write_json
from .exploration_schema import (PURPOSES, REFERENCE, integer, text)
from .scenario_validation import NAME, _number, _simulate


def _probe_number(value):
    number = _number(value)
    if isinstance(value, str):
        match = re.fullmatch(r"(\d+)'[sS]?[bBoOdDhH][0-9a-fA-F]+", value.replace("_", ""))
        if match and number.bit_length() > int(match[1]):
            raise ValueError("Probe literal exceeds its declared size; implicit truncation is not allowed")
    return number


def discover_interface(config: Config, work: Path) -> tuple[list[dict], dict[str, int]]:
    """Discover fixed-width top-level ports/state, including clockless designs."""
    for tool in ("verilator", "iverilog", "vvp"):
        if not shutil.which(tool):
            raise RuntimeError(f"explore-campaign requires {tool} on PATH")
    work.mkdir(parents=True, exist_ok=True)
    xml_dir = work / "xml"
    command = ["verilator", "--xml-only", "-Wno-fatal", "--top-module", config.top,
               "--Mdir", str(xml_dir), *config.lint_settings.get("extra_args", [])]
    command.extend("-I" + str(config.design_root / p) for p in config.include_dirs)
    command.extend(str(config.design_root / p) for p in config.sources)
    run = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if run.returncode:
        raise RuntimeError("Verilator discovery failed: " + (run.stdout + run.stderr)[-2000:])
    root = ET.parse(xml_dir / f"V{config.top}.xml")
    module = root.find(".//netlist/module[@topModule='1']")
    if module is None:
        raise RuntimeError("No elaborated top module")
    dtypes = {x.get("id"): x for x in root.findall(".//typetable/*")}
    ports, state = [], {}
    for var in module.findall("var"):
        name, direction = var.get("name", ""), var.get("dir")
        if not NAME.fullmatch(name):
            continue
        dtype = dtypes.get(var.get("dtype_id"))
        if dtype is None or dtype.tag != "basicdtype":
            if direction:
                raise ValueError(f"Unsupported top-level port type: {name}")
            continue
        width = abs(int(dtype.get("left", "0")) - int(dtype.get("right", "0"))) + 1
        if direction:
            if direction not in ("input", "output"):
                raise ValueError(f"Unsupported {direction} port: {name}")
            ports.append({"name": name, "direction": direction, "width": width})
        else:
            state[name] = width
    inputs = {p["name"]: p["width"] for p in ports if p["direction"] == "input"}
    if config.clock and inputs.get(config.clock) != 1:
        raise ValueError("Configured clock must be a one-bit top-level input")
    if config.reset and inputs.get(config.reset) != 1:
        raise ValueError("Configured reset must be a one-bit top-level input")
    if not any(p["direction"] == "output" for p in ports):
        raise ValueError("Need at least one top-level output")
    return ports, state


def validate_probe(raw: dict, config: Config, ports: list[dict], state: dict[str, int],
                   plan: dict, max_cycles: int, reset_active: int) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Probe must be an object")
    identifier = text(raw.get("id"), "probe id")
    purpose = raw.get("purpose")
    if purpose not in PURPOSES:
        raise ValueError("Unsupported probe purpose")
    inputs = {p["name"]: p["width"] for p in ports
              if p["direction"] == "input" and p["name"] != config.clock}
    observed = {p["name"]: p["width"] for p in ports} | state
    steps = raw.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Probe needs input steps")
    cycles, current = [], {}
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not isinstance(step.get("inputs"), dict):
            raise ValueError("Each step needs inputs and optional repeat")
        if not set(step["inputs"]) <= inputs.keys():
            raise ValueError("Probe drives an unknown/non-input signal or the clock")
        values = {name: _probe_number(value) for name, value in step["inputs"].items()}
        for name, value in values.items():
            if value >= 1 << inputs[name]:
                raise ValueError(f"Probe {identifier} input {name}={value} exceeds {inputs[name]} bits; "
                                 f"valid range 0..{(1 << inputs[name]) - 1}. Use a fitting hex/Verilog literal.")
        if index == 0 and set(values) != inputs.keys():
            raise ValueError("First step must explicitly initialize every non-clock input")
        repeat = integer(step.get("repeat", 1), "step repeat", 1, max_cycles)
        if len(cycles) + repeat > max_cycles:
            raise ValueError("Probe exceeds --max-probe-cycles")
        current.update(values)
        cycles.extend(dict(current) for _ in range(repeat))
    if config.reset and cycles[0][config.reset] != reset_active:
        raise ValueError("Probe must start with configured reset asserted")
    if config.reset and not any(c[config.reset] != reset_active for c in cycles[1:]):
        raise ValueError("Probe never releases reset")
    observations = raw.get("observations", [])
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    parsed = []
    for item in observations:
        if not isinstance(item, dict) or not {"cycle", "signal", "golden", "mutant"} <= item.keys():
            raise ValueError("Observation needs cycle, signal, golden, mutant")
        cycle = integer(item["cycle"], "observation cycle", 0, len(cycles) - 1)
        signal = item["signal"]
        if signal not in observed or signal == config.clock:
            raise ValueError("Observation signal is not available")
        g, m = _probe_number(item["golden"]), _probe_number(item["mutant"])
        if max(g, m) >= 1 << observed[signal]:
            raise ValueError("Observation value exceeds signal width")
        parsed.append({"cycle": cycle, "signal": signal, "golden": g, "mutant": m})
    if len({(x["cycle"], x["signal"]) for x in parsed}) != len(parsed):
        raise ValueError("Duplicate observation")
    checks = raw.get("control_checks", [])
    if not isinstance(checks, list):
        raise ValueError("control_checks must be a list")
    controls = {c["id"] for c in plan["controls"]}
    parsed_checks = []
    for check in checks:
        if not isinstance(check, dict) or check.get("control_id") not in controls:
            raise ValueError("Control check references an unknown plan control")
        cycle = integer(check.get("cycle"), "control check cycle", 0, len(cycles) - 1)
        start = integer(check.get("start_cycle", cycle), "control start_cycle", 0, cycle)
        end = integer(check.get("end_cycle", len(cycles) - 1), "control end_cycle", cycle, len(cycles) - 1)
        parsed_checks.append({**check, "start_cycle": start, "end_cycle": end})
    if purpose in ("control", "history_control") and not checks:
        raise ValueError("Control probe needs a plan control check at a specified cycle")
    legality = raw.get("legality")
    if not isinstance(legality, dict) or legality.get("status") not in ("supported", "uncertain"):
        raise ValueError("Probe needs legality status=supported/uncertain, reason, and spec_quote")
    text(legality.get("reason"), "probe legality reason")
    if raw.get("history_of") is not None and not isinstance(raw["history_of"], str):
        raise ValueError("history_of must be a probe ID string or null")
    # Quote validation against the actual spec is performed by the controller.
    return {"id": identifier, "purpose": purpose,
            "description": text(raw.get("description"), "probe description"),
            "steps": steps, "cycles": cycles, "observations": parsed,
            "control_checks": parsed_checks, "legality": legality,
            "history_of": raw.get("history_of")}


def _known(raw: str | None) -> int | None:
    return int(raw, 16) if raw and not any(c in raw for c in "xz") else None


def predicate_holds(predicates: list[dict], cycle: int, cycles: list[dict],
                    trace: dict) -> bool | None:
    def operand(value, sample):
        if type(value) is int or isinstance(value, str) and value.isdigit():
            return int(value)
        match = REFERENCE.fullmatch(value)
        name, bit, low = match[1], match[2], match[3]
        if name in cycles[cycle]:
            number = cycles[cycle][name]
        else:
            index = cycle - 1 if sample == "pre" else cycle
            number = _known(trace.get((index, name))) if index >= 0 else None
        if number is not None and bit is not None:
            low_index = int(low) if low is not None else int(bit)
            return (number >> low_index) & ((1 << (int(bit) - low_index + 1)) - 1)
        return number
    unknown = False
    for item in predicates:
        a, b = operand(item["left"], item["sample"]), operand(item["right"], item["sample"])
        if a is None or b is None:
            unknown = True
            continue
        outcomes = {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b,
                    ">": a > b, ">=": a >= b}
        if not outcomes[item["op"]]:
            return False
    return None if unknown else True


def _bench(config: Config, ports: list[dict], state: dict[str, int], cycles: list[dict]) -> str:
    lines = ["module mutation_trace_tb;"]
    for port in ports:
        vector = f" [{port['width'] - 1}:0]" if port["width"] > 1 else ""
        lines.append(f"  {'logic' if port['direction'] == 'input' else 'wire'}{vector} {port['name']};")
    lines.append(f"  {config.top} dut(" + ", ".join(f".{p['name']}({p['name']})" for p in ports) + ");")
    lines.append("  initial begin")
    if config.clock:
        lines.append(f"    {config.clock} = 0;")
    watched = {p["name"] for p in ports if p["name"] != config.clock} | state.keys()
    for cycle, inputs in enumerate(cycles):
        for name, value in inputs.items():
            width = next(p["width"] for p in ports if p["name"] == name)
            lines.append(f"    {name} = {width}'h{value:x};")
        lines.append(f"    #2 {config.clock} = 1;" if config.clock else "    #2;")
        lines.append("    #1;")
        for signal in sorted(watched):
            ref = "dut." + signal if signal in state else signal
            lines.append(f'    $display("MLTRACE|{cycle}|{signal}|%h", {ref});')
        if config.clock:
            lines.append(f"    #1 {config.clock} = 0;")
    return "\n".join(lines + ["    $finish;", "  end", "endmodule", ""])


class ProbeExecutor:
    def __init__(self, config: Config, ports: list[dict], state: dict[str, int], work: Path):
        self.config, self.ports, self.state, self.work = config, ports, state, work
        self.outputs = {p["name"] for p in ports if p["direction"] == "output"}
        self.golden_cache: dict[str, dict] = {}

    def execute(self, root: Path, probe: dict, directory: Path) -> tuple[dict, dict]:
        directory.mkdir(parents=True, exist_ok=True)
        bench = directory / "probe.sv"
        bench.write_text(_bench(self.config, self.ports, self.state, probe["cycles"]))
        signature = digest(json.dumps(probe["cycles"], sort_keys=True).encode())
        if signature not in self.golden_cache:
            self.golden_cache[signature] = _simulate(self.config, self.config.design_root, bench, directory)
        golden = self.golden_cache[signature]
        mutant = _simulate(self.config, root, bench, directory)
        write_json(directory / "traces.json", [
            {"cycle": cycle, "inputs": values,
             "golden": {name: value for (index, name), value in golden.items() if index == cycle},
             "mutant": {name: value for (index, name), value in mutant.items() if index == cycle}}
            for cycle, values in enumerate(probe["cycles"])])
        return golden, mutant

    def characterize(self, probe: dict, golden: dict, mutant: dict,
                     plan: dict, analysis: dict) -> dict:
        cycles = probe["cycles"]
        outputs, differences, unknowns, missing = self.outputs, [], [], []
        watched = outputs | (set(plan["target_signals"]) & self.state.keys())
        for cycle in range(len(cycles)):
            for signal in sorted(watched):
                a, b = golden.get((cycle, signal)), mutant.get((cycle, signal))
                av, bv = _known(a), _known(b)
                if a is None or b is None:
                    missing.append({"cycle": cycle, "signal": signal})
                elif av is not None and bv is not None and av != bv:
                    differences.append({"cycle": cycle, "signal": signal,
                                        "golden": av, "mutant": bv,
                                        "output": signal in outputs})
                elif av is not None and bv is None:
                    unknowns.append({"cycle": cycle, "signal": signal, "golden": a, "mutant": b})
        predictions = []
        for item in probe["observations"]:
            actual = {side: _known(trace.get((item["cycle"], item["signal"])))
                      for side, trace in (("golden", golden), ("mutant", mutant))}
            predictions.append({**item, "actual": actual,
                                "matched": all(actual[s] == item[s] for s in ("golden", "mutant"))})
        trigger_cycles = [cycle for cycle in range(len(cycles))
                          if predicate_holds(plan["trigger_predicates"], cycle, cycles, golden) is True]
        lo, hi = plan["latency"]["min"], plan["latency"]["max"]
        first = differences[0] if differences else None
        # A measured trigger must precede the *first* watched divergence. Otherwise
        # a later trigger could falsely explain a pre-existing unrelated failure.
        aligned = bool(first and any(t <= first["cycle"] for t in trigger_cycles))
        target_differences = [d for d in differences if d["signal"] in plan["target_signals"]]
        first_target = target_differences[0] if target_differences else None
        # Observed effects are independent of predicted values and latency estimates.
        # Update-order faults can leave identical sampled target values while
        # changing downstream outputs (e.g. blocking vs nonblocking assignment).
        target_edge = first_target or (first if plan.get('target_effect') == 'update_order' else None)
        intended = [d for d in differences if aligned and target_edge
                    and d["cycle"] >= target_edge["cycle"]
                    and any(t <= target_edge["cycle"] for t in trigger_cycles)]
        latencies = []
        for kind, difference in (("target", first_target), ("output", next((d for d in intended if d["output"]), None))):
            if difference and aligned:
                preceding = [t for t in trigger_cycles if t <= difference["cycle"]]
                if preceding:
                    # Repeated triggers make attribution ambiguous: retain all candidate delays.
                    delays = [difference["cycle"] - t for t in preceding]
                    latencies.append({"kind": kind, "cycle": difference["cycle"],
                                      "candidate_trigger_cycles": preceding,
                                      "observed_delays": delays,
                                      "unique_attribution": len(preceding) == 1,
                                      "estimate_matches_any": any(lo <= delay <= hi for delay in delays)})
        controls = {c["id"]: c for c in plan["controls"]}
        checks = []
        for check in probe["control_checks"]:
            cycle = check["cycle"]
            activated = predicate_holds(controls[check["control_id"]]["predicates"], cycle, cycles, golden)
            start, end = check.get("start_cycle", cycle), check.get("end_cycle", len(cycles) - 1)
            known_at_check = all(_known(golden.get((c, s))) is not None and
                                 _known(mutant.get((c, s))) is not None
                                 for c in range(start, end + 1) for s in watched)
            window_differences = [d for d in differences if start <= d["cycle"] <= end]
            outside = [d for d in differences if not start <= d["cycle"] <= end]
            checks.append({**check, "activated": activated, "known_at_check": known_at_check,
                           "start_cycle": start, "end_cycle": end,
                           "window_differences": window_differences,
                           "outside_window_differences": outside,
                           "preserved": activated is True and known_at_check
                           and not window_differences})
        violations, assumptions, unchecked = [], [], []
        for constraint in analysis["input_constraints"]:
            if constraint["status"] != "established":
                assumptions.append(constraint["id"])
                continue
            if not constraint["require"]:
                unchecked.append(constraint["id"])
                continue
            for cycle in range(len(cycles)):
                when = predicate_holds(constraint["when"], cycle, cycles, golden)
                require = predicate_holds(constraint["require"], cycle, cycles, golden)
                if when is True and require is False:
                    violations.append({"constraint": constraint["id"], "cycle": cycle})
                elif when is None or when is True and require is None:
                    unchecked.append(constraint["id"])
        predicted_cells = {(x['cycle'], x['signal']) for x in predictions
                           if x['matched'] and x['golden'] != x['mutant']}
        # Later divergences may be consequences of the original fault, not independent bugs.
        additional = [x for x in differences if (x["cycle"], x["signal"]) not in predicted_cells]
        known_output = any(_known(golden.get((c, s))) is not None and
                           _known(mutant.get((c, s))) is not None
                           for c in range(len(cycles)) for s in outputs)
        return {"id": probe["id"], "purpose": probe["purpose"], "cycles": len(cycles),
                "first_difference": first, "differences": differences,
                "additional_differences": additional,
                "output_difference": any(x["output"] for x in differences),
                "unknowns": unknowns, "missing_observations": missing,
                "predictions": predictions, "trigger_cycles": trigger_cycles,
                "probe_verification": {
                    "trigger_reached": bool(trigger_cycles),
                    "target_changed": bool(first_target),
                    "target_effect": plan.get('target_effect', 'value_change'),
                    "first_target_difference": first_target,
                    "first_divergence_after_trigger": aligned,
                    "reachability_scope": "this executed input sequence only",
                    "causality_proven": False},
                "measured_latencies": latencies,
                "prediction_mismatches": [p for p in predictions if not p["matched"]],
                "first_difference_trigger_aligned": aligned,
                "intended_observations": intended,
                "control_checks": checks, "constraint_violations": violations,
                "unchecked_constraints": sorted(set(unchecked)), "assumptions": assumptions,
                "legality": {**probe["legality"], "machine_check": "violated" if violations else
                             "no_recorded_constraint_violation", "complete_protocol_proof": False},
                "adversarial_exercised": probe["purpose"] == "adversarial" and known_output,
                "input_signature": digest(json.dumps(cycles, sort_keys=True).encode()),
                "trace_signature": digest(json.dumps([
                    [cycle, signal, mutant.get((cycle, signal))]
                    for cycle in range(len(cycles)) for signal in sorted(outputs)
                ]).encode())}


def summarize_evidence(results: list[dict], plan: dict) -> dict:
    eligible = [r for r in results if not r["constraint_violations"]
                and r["legality"]["status"] == "supported" and not r["missing_observations"]]
    intended = [r for r in eligible if r["purpose"] == "trigger" and r["intended_observations"]
                and r["first_difference_trigger_aligned"]]
    output_intended = [r for r in intended if any(x["signal"] in {
        d["signal"] for d in r["differences"] if d["output"]}
        for x in r["intended_observations"])]
    required = {c["id"] for c in plan["controls"]}
    exercised = {c["control_id"] for r in eligible for c in r["control_checks"] if c["preserved"]}
    mismatches = [{"probe": r["id"], **c} for r in eligible for c in r["control_checks"]
                  if c["activated"] is True and not c["preserved"]]
    first = next((r["first_difference"] for r in output_intended), None)
    other_inputs = {r["input_signature"] for r in results if r["purpose"] != "adversarial"}
    adversarial = [r for r in eligible if r["adversarial_exercised"]
                   and r["input_signature"] not in other_inputs]
    fact = {"probes": len(results), "output_witnesses": len(output_intended),
            "controls_preserved": sorted(exercised), "controls_missing": sorted(required - exercised),
            "control_mismatches": len(mismatches), "first_intended_difference": first,
            "adversarial_probes_exercised": len({r["input_signature"] for r in adversarial})}
    gaps = []
    if not output_intended:
        gaps.append("No measured output consequence after an exercised trigger and planned target divergence")
    if required - exercised:
        gaps.append("Control cases not exercised and preserved: " + ", ".join(sorted(required - exercised)))
    if mismatches:
        gaps.append("A claimed preserved control has a differing trajectory")
    if not fact["adversarial_probes_exercised"]:
        gaps.append("No distinct supported adversarial probe was executed with known output observations")
    evidence_level = ("observed_output_difference" if any(r["output_difference"] for r in eligible)
                      else "internal_difference_only" if any(r["differences"] for r in eligible)
                      else "unknown_only" if any(r["unknowns"] for r in eligible) else "unobserved")
    return {"ready_for_review": not gaps, "gaps": gaps, "fact_check": fact,
            "probe_verification": {r["id"]: r.get("probe_verification", {}) for r in results},
            "timing": {"source": "simulation", "plan_estimate": plan["latency"],
                       "measured": {r["id"]: r.get("measured_latencies", []) for r in results},
                       "prediction_mismatch_count": sum(not p["matched"] for r in results for p in r["predictions"]),
                       "prediction_errors_are_acceptance_gates": False},
            "evidence_level": evidence_level, "control_mismatches": mismatches,
            "predicted_exposure": plan["exposure"],
            "exposure_evidence": {
                "trigger_probe_cycles": {r["id"]: r["trigger_cycles"] for r in results},
                "history_requirement_proven": False,
                "minimum_exposure_depth_proven": False},
            "specification_proof": False,
            "note": "Sampled trajectories and model-supported legality; not an exhaustive protocol or specification proof"}
