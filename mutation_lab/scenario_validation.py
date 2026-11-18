"""Check a concrete scenario witness on golden and mutated RTL by simulation."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .core import Config


NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
TRACE = re.compile(r"(?m)^MLTRACE\|(\d+)\|([A-Za-z_][A-Za-z_0-9]*)\|([0-9a-fA-FxXzZ]+)$")
SV_LITERAL = re.compile(r"\d+'[sS]?([bBoOdDhH])([0-9a-fA-F_]+)\Z")


def _known(value: str | None) -> bool:
    return value is not None and "x" not in value and "z" not in value


def _number(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Witness values must be integers, not booleans")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        raw = value.replace("_", "")
        match = SV_LITERAL.fullmatch(raw)
        if match:
            result = int(match.group(2), {"b": 2, "o": 8, "d": 10, "h": 16}[match.group(1).lower()])
        elif re.fullmatch(r"0[xX][0-9a-fA-F]+|0[bB][01]+|[0-9]+", raw):
            result = int(raw, 0) if raw.lower().startswith(("0x", "0b")) else int(raw)
        else:
            raise ValueError(f"Unsupported witness value: {value!r}")
    else:
        raise ValueError(f"Unsupported witness value: {value!r}")
    if result < 0:
        raise ValueError("Witness values must be nonnegative")
    return result


def _ports(config: Config, source_root: Path, work: Path) -> tuple[list[dict], set[str]]:
    xml_dir = work / "xml"
    command = [config.lint_settings.get("executable", "verilator"), "--xml-only",
               "-Wno-fatal", "--top-module", config.top, "--Mdir", str(xml_dir)]
    command.extend(config.lint_settings.get("extra_args", []))
    command.extend("-I" + str(source_root / item) for item in config.include_dirs)
    command.extend(str(source_root / item) for item in config.sources)
    result = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError("Verilator port discovery failed: " + (result.stdout + result.stderr)[-1200:])
    xml = ET.parse(xml_dir / ("V" + config.top + ".xml"))
    module = xml.find(".//netlist/module[@topModule='1']")
    if module is None:
        raise RuntimeError("Could not find elaborated top module in Verilator XML")
    dtypes = {node.get("id"): node for node in xml.findall(".//typetable/*")}
    ports = []
    internal = set()
    for var in module.findall("var"):
        name = var.get("name", "")
        if not NAME.fullmatch(name):
            continue
        direction = var.get("dir")
        if direction is None:
            internal.add(name)
            continue
        if direction not in ("input", "output"):
            raise RuntimeError(f"Witness simulator does not support {direction} port {name}")
        dtype = dtypes.get(var.get("dtype_id"))
        if dtype is None or dtype.tag != "basicdtype":
            raise RuntimeError(f"Witness simulator does not support port type for {name}")
        try:
            width = abs(int(dtype.get("left", "0")) - int(dtype.get("right", "0"))) + 1
        except ValueError as exc:
            raise RuntimeError(f"Witness simulator needs a fixed width for {name}") from exc
        ports.append({"name": name, "direction": direction, "width": width})
    if not ports or config.clock not in {p["name"] for p in ports if p["direction"] == "input"}:
        raise RuntimeError("Witness simulator needs the configured top-level input clock")
    return ports, internal


def _parse_witness(config: Config, witness: dict, ports: list[dict],
                   internal: set[str], require_difference: bool = True
                   ) -> tuple[list[dict[str, int]], list[dict]]:
    inputs = {p["name"]: p["width"] for p in ports
              if p["direction"] == "input" and p["name"] != config.clock}
    outputs = {p["name"] for p in ports if p["direction"] == "output"}
    raw_cycles = witness.get("cycles")
    raw_observations = witness.get("observations")
    if not isinstance(raw_cycles, list) or not 2 <= len(raw_cycles) <= 128:
        raise ValueError("Witness needs 2 to 128 input cycles")
    if not isinstance(raw_observations, list) or not 2 <= len(raw_observations) <= 32:
        raise ValueError("Witness needs 2 to 32 observations")
    cycles: list[dict[str, int]] = []
    for index, item in enumerate(raw_cycles):
        if not isinstance(item, dict) or not set(item).issubset(inputs):
            raise ValueError(f"Witness cycle {index} contains an unknown input")
        values = {name: _number(value) for name, value in item.items()}
        for name, value in values.items():
            if value >= 1 << inputs[name]:
                raise ValueError(f"Witness cycle {index} value exceeds width of {name}")
        cycles.append(values)
    if config.reset and config.reset in inputs and config.reset not in cycles[0]:
        raise ValueError("Witness first cycle must set the configured reset input")
    observations = []
    seen = set()
    for item in raw_observations:
        if not isinstance(item, dict) or set(item) != {"cycle", "signal", "golden", "mutant"}:
            raise ValueError("Each witness observation needs cycle, signal, golden, mutant")
        cycle, signal = item["cycle"], item["signal"]
        if (isinstance(cycle, bool) or not isinstance(cycle, int)
                or not 0 <= cycle < len(cycles)):
            raise ValueError("Witness observation cycle is out of range")
        if not isinstance(signal, str) or signal not in outputs | internal:
            raise ValueError(f"Witness signal is not a top-level output or direct register: {signal!r}")
        if (cycle, signal) in seen:
            raise ValueError("Witness observation repeats the same cycle and signal")
        seen.add((cycle, signal))
        observations.append({"cycle": cycle, "signal": signal,
                             "golden": _number(item["golden"]),
                             "mutant": _number(item["mutant"])})
    differences = [item for item in observations if item["golden"] != item["mutant"]]
    if require_difference and not differences:
        raise ValueError("Trigger witness must predict a golden/mutant difference")
    if not require_difference and differences:
        raise ValueError("Control observations must predict identical golden/mutant values")
    return cycles, observations


def _testbench(config: Config, ports: list[dict], internal: set[str],
               cycles: list[dict[str, int]], observations: list[dict]) -> str:
    lines = ["module mutation_trace_tb;"]
    for port in ports:
        width = port["width"]
        dtype = "logic" if port["direction"] == "input" else "wire"
        vector = f" [{width - 1}:0]" if width > 1 else ""
        lines.append(f"  {dtype}{vector} {port['name']};")
    connections = ", ".join(f".{p['name']}({p['name']})" for p in ports)
    lines.append(f"  {config.top} dut({connections});")
    lines.append("  initial begin")
    lines.append(f"    {config.clock} = 1'b0;")
    for port in ports:
        if port["direction"] == "input" and port["name"] != config.clock:
            lines.append(f"    {port['name']} = '0;")
    for index, values in enumerate(cycles):
        for name, value in values.items():
            width = next(p["width"] for p in ports if p["name"] == name)
            lines.append(f"    {name} = {width}'h{value:x};")
        lines.append(f"    #2 {config.clock} = 1'b1;")
        lines.append("    #1;")
        observed = {item["signal"] for item in observations if item["cycle"] == index}
        observed.update(p["name"] for p in ports if p["direction"] == "output")
        for signal in sorted(observed):
            reference = signal if signal not in internal else "dut." + signal
            lines.append(f'    $display("MLTRACE|{index}|{signal}|%h", {reference});')
        lines.append(f"    #1 {config.clock} = 1'b0;")
    lines.extend(["    $finish;", "  end", "endmodule", ""])
    return "\n".join(lines)


def _simulate(config: Config, source_root: Path, testbench: Path, work: Path) -> dict:
    binary = work / "simulation.vvp"
    command = ["iverilog", "-g2012", "-s", "mutation_trace_tb", "-o", str(binary)]
    command.extend("-I" + str(source_root / item) for item in config.include_dirs)
    command.extend(str(source_root / item) for item in config.sources)
    command.append(str(testbench))
    compile_result = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=60)
    if compile_result.returncode:
        raise RuntimeError("Icarus compile failed: " + compile_result.stderr[-1200:])
    run = subprocess.run(["vvp", str(binary)], cwd=work, capture_output=True,
                         text=True, timeout=30)
    if run.returncode:
        raise RuntimeError("Icarus simulation failed: " + (run.stdout + run.stderr)[-1200:])
    return {(int(cycle), signal): value.lower() for cycle, signal, value in TRACE.findall(run.stdout)}


def validate_scenario(config: Config, mutant_dir: Path, witness: dict,
                      category: str | None = None, require_controls: bool = False) -> dict:
    """Require a reachable trace matching both the planned golden and mutant values."""
    if not all(shutil.which(tool) for tool in ("verilator", "iverilog", "vvp")):
        return {"status": "validation_unavailable", "reason": "Need Verilator, Icarus, and vvp"}
    try:
        with tempfile.TemporaryDirectory(prefix="mutlab_scenario_") as name:
            work = Path(name)
            ports, internal = _ports(config, config.design_root, work)
            cycles, observations = _parse_witness(config, witness, ports, internal)
            controls = witness.get("controls", [])
            needed = 2 if category == "interaction" else 1
            if (not isinstance(controls, list) or len(controls) > 3
                    or (require_controls and len(controls) < needed)):
                raise ValueError(f"Scenario needs {needed} to 3 control traces")
            if len({json.dumps(item, sort_keys=True) for item in controls}) != len(controls):
                raise ValueError("Control traces must be distinct")
            first_difference = min(item["cycle"] for item in observations
                                   if item["golden"] != item["mutant"])
            if require_controls and not any(
                    item["cycle"] < first_difference and item["golden"] == item["mutant"]
                    for item in observations):
                raise ValueError("Trigger needs an agreeing observation before the first fault")
            testbench = work / "mutation_trace_tb.sv"
            testbench.write_text(_testbench(config, ports, internal, cycles, observations))
            golden = _simulate(config, config.design_root, testbench, work)
            for item in observations:
                key = (item["cycle"], item["signal"])
                actual = golden.get(key, "missing")
                if actual == "missing" or "x" in actual or "z" in actual or int(actual, 16) != item["golden"]:
                    return {"status": "bad_witness", "reason":
                            f"Golden {key} was {actual}, expected {item['golden']:x}"}
            mutant = _simulate(config, mutant_dir / "rtl", testbench, work)
            for item in observations:
                key = (item["cycle"], item["signal"])
                actual = mutant.get(key, "missing")
                if actual == "missing" or "x" in actual or "z" in actual or int(actual, 16) != item["mutant"]:
                    return {"status": "scenario_mismatch", "reason":
                            f"Mutant {key} was {actual}, expected {item['mutant']:x}"}
            outputs = [port["name"] for port in ports if port["direction"] == "output"]
            for cycle in range(first_difference):
                for signal in outputs:
                    key = (cycle, signal)
                    if golden.get(key) != mutant.get(key):
                        return {"status": "early_divergence", "reason":
                                f"Output {signal} differs at cycle {cycle}, before the planned fault at cycle {first_difference}"}
            for control_index, control in enumerate(controls):
                if not isinstance(control, dict):
                    raise ValueError(f"Control {control_index} must be a JSON object")
                control_cycles, control_observations = _parse_witness(
                    config, control, ports, internal, require_difference=False)
                testbench.write_text(_testbench(config, ports, internal,
                                               control_cycles, control_observations))
                control_golden = _simulate(config, config.design_root, testbench, work)
                control_mutant = _simulate(config, mutant_dir / "rtl", testbench, work)
                for item in control_observations:
                    key = (item["cycle"], item["signal"])
                    actual = control_golden.get(key, "missing")
                    if (actual == "missing" or "x" in actual or "z" in actual
                            or int(actual, 16) != item["golden"]):
                        return {"status": "bad_witness", "reason":
                                f"Control {control_index} golden {key} was {actual}, expected {item['golden']:x}"}
                    mutant_actual = control_mutant.get(key, "missing")
                    if (mutant_actual == "missing" or "x" in mutant_actual or "z" in mutant_actual
                            or int(mutant_actual, 16) != item["mutant"]):
                        return {"status": "control_divergence", "reason":
                                f"Control {control_index} mutant {key} was {mutant_actual}, expected {item['mutant']:x}"}
                for cycle in range(len(control_cycles)):
                    for signal in outputs:
                        key = (cycle, signal)
                        if control_golden.get(key) != control_mutant.get(key):
                            return {"status": "control_divergence", "reason":
                                    f"Control {control_index} output {signal} differs at cycle {cycle}"}
                active = any(
                    _known(control_golden.get((cycle, signal)))
                    and _known(control_golden.get((cycle - 1, signal)))
                    and control_golden[(cycle, signal)] != control_golden[(cycle - 1, signal)]
                    for cycle in range(1, len(control_cycles)) for signal in outputs)
                if not active:
                    return {"status": "weak_control", "reason":
                            f"Control {control_index} exercises no known output transition"}
            return {"status": "passed", "cycles": len(cycles),
                    "observations": len(observations), "controls": len(controls),
                    "first_difference_cycle": first_difference}
    except ValueError as exc:
        return {"status": "bad_witness", "reason": str(exc)[:1200]}
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ET.ParseError) as exc:
        return {"status": "validation_unavailable", "reason": str(exc)[:1200]}
