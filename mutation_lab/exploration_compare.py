"""Compare fixed and targeted probes on the same saved candidate pool."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from .core import Config, Edit, write_json
from .quality_validation import BehaviorAuditor


def compare_probes(output: Path, max_probes: int = 48) -> dict:
    output = output.resolve()
    config = Config.load(output / "config.direct.json")
    frozen = output / "golden" / "rtl"
    if frozen.is_dir():
        config = replace(config, design_root=frozen)
    if config.clock is None:
        raise ValueError("The legacy fixed-probe audit requires a clock; this comparison is not applicable")
    rows = []
    with BehaviorAuditor(config, max_probes) as auditor:
        for file in sorted((output / "mutants").glob("*/record.json")):
            record = json.loads(file.read_text())
            if "family" not in record.get("plan", {}):
                continue
            revisions = [r for r in record["revisions"] if r.get("validation")]
            if not revisions:
                continue
            revision = revisions[-1]
            root = file.parent / "attempts" / f"a{revision['revision'] + 1}" / "rtl"
            plan = record["plan"]
            legacy_predicates = lambda values: [{"left": p["left"], "op": p["op"],
                                                  "right": str(p["right"])} for p in values]
            adapter = SimpleNamespace(target_signal=plan["target_signals"][0], trigger=plan["trigger"],
                trigger_predicates=legacy_predicates(plan["trigger_predicates"]),
                control_predicates=[legacy_predicates(c["predicates"]) for c in plan["controls"]],
                trigger_window_cycles=plan["latency"]["max"], preserve_first_operation=False)
            all_predicates = plan["trigger_predicates"] + [p for c in plan["controls"] for p in c["predicates"]]
            part_selects = any(":" in p["left"] or ":" in str(p["right"]) for p in all_predicates)
            if part_selects:
                # The raw output-exposure comparison remains useful even when
                # the old semantic predicate language cannot express the plan.
                adapter.trigger_predicates, adapter.control_predicates = (), ()
                adapter.trigger = "Raw output exposure comparison"
            try:
                baseline = auditor.assess(root, adapter, tuple(Edit(**e) for e in revision["edits"]))
                rows.append({"mutant": record["id"], "saved_status": record["status"],
                    "legacy_probes": baseline["probes"], "legacy_audit_status": baseline["status"],
                    "legacy_output_difference": baseline.get("output_witness", False),
                    "legacy_evidence": baseline.get("evidence", []),
                    "targeted_evidence": revision["validation"],
                    "targeted_probe_count": revision["validation"]["fact_check"]["probes"],
                    "targeted_output_difference": revision["validation"]["evidence_level"] == "observed_output_difference",
                    "phase_adapter_warning": part_selects or any(p.get("sample") == "post" for p in all_predicates)})
            except (ValueError, RuntimeError, OSError) as exc:
                rows.append({"mutant": record["id"], "error": str(exc)})
    report = {"same_saved_candidates": True, "llm_calls": 0, "mutants": rows,
              "note": "Compare output exposure separately from acceptance. Legacy adapters do not implement new post-phase predicates or exact consequence checks. Neither sampled audit proves equivalence or spec correctness."}
    index = 1
    destination = output / "probe_comparison.json"
    while destination.exists():
        index += 1
        destination = output / f"probe_comparison_{index}.json"
    write_json(destination, report)
    print(f"Probe comparison: {destination}")
    return report
