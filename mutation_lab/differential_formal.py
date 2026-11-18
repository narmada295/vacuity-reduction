"""Optional reset-constrained output comparison for single-module RTL.

The miter shares top-level inputs and compares the two outputs on every clock.
It establishes behavioral difference, never specification correctness.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .core import Config, stage_sources, write_json
from .formal import named_status, run_check


def build_miter(config: Config, golden_root: Path, mutant_root: Path,
                ports: list[dict], destination: Path) -> Config:
    if len(config.sources) != 1:
        raise ValueError("Formal comparison currently supports one RTL source file")
    source = config.sources[0]
    golden = (golden_root / source).read_text()
    mutant = (mutant_root / source).read_text()
    module_pattern = re.compile(r"\bmodule\s+([A-Za-z_]\w*)\b")
    for text in (golden, mutant):
        modules = module_pattern.findall(text)
        if modules != [config.top]:
            raise ValueError("Formal comparison needs one top module per RTL source")
    destination.mkdir(parents=True, exist_ok=True)
    design = destination / "design"
    design.mkdir()
    renamed = []
    for label, text in (("golden", golden), ("mutant", mutant)):
        module_name = f"mutlab_{label}_{config.top}"
        path = design / f"{label}.sv"
        path.write_text(module_pattern.sub("module " + module_name, text, count=1))
        renamed.append(module_name)
    inputs = [port for port in ports if port["direction"] == "input"]
    outputs = [port for port in ports if port["direction"] == "output"]
    if not outputs:
        raise ValueError("Formal comparison needs at least one top-level output")
    declarations = []
    for port in inputs:
        width = f" [{port['width'] - 1}:0]" if port["width"] > 1 else ""
        declarations.append(f"input logic{width} {port['name']}")
    lines = ["module mutlab_miter(", "  " + ",\n  ".join(declarations), ");"]
    for label, module in zip(("golden", "mutant"), renamed):
        for port in outputs:
            width = f" [{port['width'] - 1}:0]" if port["width"] > 1 else ""
            lines.append(f"wire{width} {label}_{port['name']};")
        wires = [f".{port['name']}({port['name']})" for port in inputs]
        wires.extend(f".{port['name']}({label}_{port['name']})" for port in outputs)
        lines.append(f"{module} {label}_dut(" + ", ".join(wires) + ");")
    active_low = config.reset.lower().endswith(("_n", "_b")) or config.reset.lower().startswith(("nreset", "nrst"))
    reset_condition = f"!{config.reset}" if active_low else config.reset
    equality = " && ".join(f"(golden_{p['name']} == mutant_{p['name']})" for p in outputs)
    lines.extend([f"MUTLAB_OUTPUT_EQUIV: assert property (@(posedge {config.clock}) "
                  f"disable iff ({reset_condition}) ({equality}));", "endmodule", ""])
    (design / "miter.sv").write_text("\n".join(lines))
    raw = json.loads(config.path.read_text())
    raw.update({"design_root": str(design), "sources": ["golden.sv", "mutant.sv", "miter.sv"],
                "include_dirs": ["."], "top": "mutlab_miter", "clock": config.clock,
                "reset": config.reset, "oracle_file": None, "evaluation_file": None})
    write_json(destination / "config.json", raw)
    return Config.load(destination / "config.json")


def compare_outputs(config: Config, golden_root: Path, mutant_root: Path,
                    ports: list[dict], destination: Path) -> dict:
    try:
        miter = build_miter(config, golden_root, mutant_root, ports, destination)
        job = destination / "job"
        stage_sources(miter, job)
        empty = destination / "empty_checker.sv"
        empty.write_text("// Miter assertion is in miter.sv.\n")
        result = run_check(miter, job, empty)
        property_status = named_status(result["properties"], "MUTLAB_OUTPUT_EQUIV")
        return {"status": ("unavailable" if result["status"] == "unavailable" else
                           "different" if result["status"] == "completed" and property_status == "cex"
                           else "equivalent" if result["status"] == "completed" and property_status == "proven"
                           else "inconclusive"), "property_status": property_status,
                "formal_run": result}
    except (OSError, ValueError, RuntimeError) as exc:
        return {"status": "unavailable", "reason": str(exc)}
