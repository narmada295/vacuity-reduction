"""Opt-in full-context, family-guided mutation exploration.

The legacy refine workflow is intentionally untouched. No evaluation assertion
file or kill result is read here. Simulation evidence is finite and model-based
spec interpretation remains explicitly unqualified.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .campaign import direct_config
from .core import (Config, apply_edits, digest, read_spec, rtl_fingerprint, write_json)
from .formal import lint
from .planner import LLMPlanner, _json_object
from .quality_campaign import _parse_edits, _top_header
from .quality_validation import _reset_active
from .exploration_probes import (ProbeExecutor, discover_interface, summarize_evidence,
                                 validate_probe)
from .exploration_schema import (EXPOSURES, FAMILIES, exploration_slots, mechanism_key,
                                 normalize_numeric_json, quote, text, validate_analysis, validate_plan)
from . import exploration_prompts as prompts
from .exploration_checks import semantic_review, static_plan_checks


VERSION = "explore-v2.1"

PLAN_CHECKS = ['golden_behavior', 'actual_change', 'trigger', 'controls', 'consequence']
EDIT_CHECKS = ['mechanism', 'trigger', 'controls', 'single_defect']


class BudgetExhausted(RuntimeError):
    pass


class CandidateCheckFailed(Exception):
    def __init__(self, status: str, reason: str):
        self.status = status
        super().__init__(reason)


class StageResponseError(RuntimeError):
    def __init__(self, stage: str, reason: str):
        super().__init__(stage + ": " + reason)
        self.stage = stage


class Requests:
    """Bound and persist every attempted call, including malformed responses."""
    def __init__(self, config: Config, args: Namespace, base: str, client=None):
        self.args, self.base, self.calls, self.raw_plans = args, base, 0, 0
        self.client = client or LLMPlanner(config, require_witness=False)

    def ask(self, stage: str, system: str, payload: dict,
            validate: Callable[[dict], Any] = lambda x: x):
        feedback = ""
        for retry in range(self.args.response_retries + 1):
            if self.calls >= self.args.max_llm_calls:
                raise BudgetExhausted("max_llm_calls")
            prompt = self.base + "\n\nSTAGE INPUT:\n" + json.dumps(payload, ensure_ascii=False) + feedback
            if len(prompt) > self.args.max_context_chars:
                raise BudgetExhausted(f"max_context_chars ({len(prompt)} characters; raise --max-context-chars)")
            self.calls += 1
            print(f"LLM c{self.calls:05d}: {stage}", flush=True)
            directory = self.args.out / "llm_calls" / f"c{self.calls:05d}"
            directory.mkdir(parents=True)
            write_json(directory / "request.json", {"stage": stage, "system": system, "prompt": prompt})
            try:
                array_keys = {"fault_plans": "plans", "mechanism_similarity": "assessments",
                              "rtl_edit": "edits", "probes": "probes", "probe_refinement": "probes"}
                if isinstance(self.client, LLMPlanner):
                    self.client.config.planner["raw_response_file"] = str(directory / "raw_response.txt")
                try:
                    raw = self.client._request(system, prompt, **(
                        {"array_key": array_keys[stage]} if stage in array_keys else {}))
                except ValueError as exc:
                    raw_file = directory / "raw_response.txt"
                    if stage not in ("probes", "probe_refinement") or not raw_file.exists() or "truncated" in str(exc):
                        raise
                    raw = _json_object(normalize_numeric_json(raw_file.read_text()), array_key="probes")
                    write_json(directory / "normalization.json", {
                        "operation": "Converted unquoted hex/Verilog numeric tokens to JSON integers; no expressions evaluated"})
                write_json(directory / "response.json", raw)
                if stage == "fault_plans" and isinstance(raw.get("plans"), list):
                    self.raw_plans += len(raw["plans"])
                result = validate(raw)
                write_json(directory / "result.json", {"status": "valid"})
                return result
            except (ValueError, RuntimeError, OSError) as exc:
                write_json(directory / "result.json", {"status": "failed", "reason": str(exc)})
                if retry == self.args.response_retries:
                    raise StageResponseError(stage, str(exc)) from exc
                feedback = "\n\nPREVIOUS RESPONSE ERROR: " + str(exc) + "\nCorrect the structured response."
            finally:
                if isinstance(self.client, LLMPlanner):
                    self.client.config.planner.pop("raw_response_file", None)
        raise AssertionError("unreachable")


def _sources(config: Config, root: Path) -> str:
    return "\n\n".join(f"RTL FILE {p}:\n" + (root / p).read_text() for p in config.sources)


def _snapshot(config: Config, output: Path) -> Path:
    """Freeze include/source trees without copying a nested output into itself."""
    files = set(config.sources)
    for include in config.include_dirs:
        for directory, children, names in os.walk(config.design_root / include, followlinks=False):
            base = Path(directory)
            children[:] = [name for name in children if not name.startswith(".")
                            and name != "__pycache__" and not (base / name).is_symlink()
                            and not (base / name / "config.direct.json").is_file()
                            and not (base / name).resolve().is_relative_to(output)]
            files.update((base / name).relative_to(config.design_root) for name in names
                         if not name.startswith(".") and not (base / name).is_symlink())
    root = output / "golden" / "rtl"
    root.mkdir(parents=True)
    for relative in sorted(files):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.design_root / relative, target)
    return root


def _config(args: Namespace) -> dict:
    # direct_config expects a clock; use a temporary explicit name for clockless
    # designs, then save null. Existing commands retain their previous inference.
    content = "\n".join(p.read_text() for p in args.rtl)
    without_comments = re.sub(r"//[^\n]*|/\*.*?\*/", "", content, flags=re.S)
    edges = set(re.findall(r"\b(posedge|negedge)\s+([A-Za-z_]\w*)", without_comments))
    clock_edges = {(edge, name) for edge, name in edges if name != args.reset}
    names = {name for _, name in clock_edges}
    clock = args.clock or (next(iter(names)) if len(names) == 1 else None)
    if len(names) > 1 or any(edge == "negedge" and name == clock for edge, name in clock_edges):
        raise ValueError("explore-campaign currently supports one rising-edge clock or a combinational design")
    if clock_edges and clock is None:
        raise ValueError("Pass --clock for the sequential design")
    copied = Namespace(**vars(args))
    copied.clock = clock or "__mutlab_clockless__"
    raw = direct_config(copied)
    raw["clock"] = clock
    raw["planner"]["context_sources"] = raw["sources"]  # Always full golden RTL.
    raw["exploration"] = {"version": VERSION, "families": list(FAMILIES),
                          "exposures": list(EXPOSURES),
                          "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                      if k not in ("rtl", "include_dir", "focus_rtl")}}
    return raw


def _compact_archive(archive: list[dict]) -> list[dict]:
    return [{"id": x["id"], "family": x["family"], "mechanism": x["mechanism"],
             "target_signals": x["target_signals"], "behavior_ids": x["behavior_ids"],
             "trigger": x["trigger"], "required_history": x["required_history"],
             "observable_consequence": x["observable_consequence"],
             "outcome": x.get("outcome", "pending"),
             "observed_behavior": x.get("observed_behavior", "not observed/reviewed")} for x in archive]


def _plan_batch(raw: dict, analysis: dict, focus: dict, signals: set[str],
                widths: dict[str, int], limit: int) -> dict:
    plans = raw.get("plans")
    if not isinstance(plans, list) or len(plans) > limit:
        raise ValueError("Return a plans list within the requested batch size")
    errors, valid = [], 0
    for plan in plans:
        try:
            validate_plan(plan, analysis, focus, signals, widths)
            valid += 1
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append(str(exc))
    if plans and not valid:
        raise ValueError("No valid plans in batch. Correct the schema: " + "; ".join(dict.fromkeys(errors)))
    return raw


def _similarity(raw: dict, plans: list[dict], archive: list[dict]) -> dict[str, dict]:
    rows = raw.get("assessments")
    if not isinstance(rows, list):
        raise ValueError("Similarity review needs assessments")
    requested, known = {p["id"] for p in plans}, {p["id"] for p in plans + archive}
    result = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("plan_id") not in requested or row["plan_id"] in result:
            raise ValueError("Similarity review references a repeated/unknown new plan")
        relation, similar = row.get("relationship"), row.get("similar_to")
        if relation not in ("novel", "related", "redundant") or not isinstance(similar, list):
            raise ValueError("Invalid similarity assessment")
        if any(not isinstance(x, str) or x not in known or x == row["plan_id"] for x in similar):
            raise ValueError("Similarity references an unknown plan or itself")
        if relation != "novel" and not similar:
            raise ValueError("Related/redundant judgment needs a comparison plan")
        text(row.get("reason"), "similarity reason")
        result[row["plan_id"]] = row
    if set(result) != requested:
        raise ValueError("Similarity review must assess every new plan")
    return result


def _review(raw: dict, evidence: dict, specification: str) -> dict:
    if raw.get("verdict") not in ("accept", "revise", "reject"):
        raise ValueError("Review verdict must be accept, revise, or reject")
    if raw.get("measured_fact_check") != evidence["fact_check"]:
        raise ValueError("Review must copy measured_fact_check exactly from evidence.fact_check")
    for field in ("reason", "implemented_behavior", "observed_behavior"):
        text(raw.get(field), field)
    for field in ("issues", "unexpected_effects", "unverified_claims"):
        if not isinstance(raw.get(field), list) or any(not isinstance(x, str) for x in raw[field]):
            raise ValueError(f"Review {field} must be a string list")
    realism, spec = raw.get("realism"), raw.get("specification")
    if not isinstance(realism, dict) or realism.get("assessment") not in ("plausible", "uncertain", "implausible"):
        raise ValueError("Review needs realism assessment")
    if not isinstance(spec, dict) or spec.get("assessment") not in ("supported", "ambiguous", "unsupported"):
        raise ValueError("Review needs specification assessment")
    text(realism.get("reason"), "realism reason")
    text(spec.get("reason"), "specification reason")
    if spec["assessment"] == "supported":
        spec["spec_quote"] = quote(spec.get("spec_quote"), specification)
    if raw.get("plan_match") not in ("supported", "mismatch", "uncertain"):
        raise ValueError("Review needs plan_match")
    return raw


def _synthesis(config: Config, root: Path) -> dict:
    executable = shutil.which("yosys")
    if not executable:
        raise RuntimeError("--synthesis-check requires yosys on PATH")
    script = "read_verilog -sv " + " ".join("-I" + json.dumps(str(root / p)) for p in config.include_dirs)
    script += " " + " ".join(json.dumps(str(root / p)) for p in config.sources)
    script += "; hierarchy -check -top " + config.top + "; proc; check -assert"
    run = subprocess.run([executable, "-Q", "-T", "-p", script], capture_output=True, text=True, timeout=90)
    result = {"status": "passed" if run.returncode == 0 else "failed",
              "kind": "yosys_frontend_proc_check", "detail": (run.stdout + run.stderr)[-5000:]}
    return result


def _probe_batch(raw: dict, config: Config, ports: list[dict], state: dict,
                 plan: dict, args: Namespace, specification: str, used: set[str]) -> list[dict]:
    probes = raw.get("probes")
    if not isinstance(probes, list) or not 1 <= len(probes) <= args.probes_per_round:
        raise ValueError("Probe batch needs 1..--probes-per-round probes")
    parsed, seen = [], set(used)
    for item in probes:
        probe = validate_probe(item, config, ports, state, plan, args.max_probe_cycles, args.reset_active)
        if probe["id"] in seen:
            raise ValueError("Probe ID already used; use new IDs in refinement")
        seen.add(probe["id"])
        if probe["legality"]["status"] == "supported":
            probe["legality"]["spec_quote"] = quote(probe["legality"].get("spec_quote"), specification)
        parsed.append(probe)
    return parsed


def _history_pairs(probes: list[dict], results: list[dict]) -> list[dict]:
    """Record controlled same-suffix comparisons; never claim minimal history."""
    by_id, reports = {p["id"]: p for p in probes}, {r["id"]: r for r in results}
    pairs = []
    for probe in probes:
        original = by_id.get(probe.get("history_of"))
        if original is None:
            continue
        suffix = 0
        for a, b in zip(reversed(original["cycles"]), reversed(probe["cycles"])):
            if a != b:
                break
            suffix += 1
        original_report, control = reports.get(original["id"]), reports.get(probe["id"])
        supported = bool(suffix and original_report and control
                         and any(d["output"] and d["cycle"] >= len(original["cycles"]) - suffix
                                 for d in original_report["differences"])
                         and not control["differences"] and not control["unknowns"]
                         and any(c["preserved"] for c in control["control_checks"]))
        pairs.append({"probe": original["id"], "history_control": probe["id"],
                      "matching_suffix_cycles": suffix, "history_effect_supported": supported,
                      "necessity_proven": False})
    return pairs


def _result_context(results: list[dict]) -> list[dict]:
    """Keep measured summaries bounded; full measurements and traces stay on disk."""
    compact = []
    for result in results:
        row = dict(result)
        for field in ("differences", "additional_differences", "unknowns", "trigger_cycles", "intended_observations"):
            values = row[field]
            row[field + "_count"] = len(values)
            if len(values) > 24:
                row[field] = values[:16] + values[-8:]
                row[field + "_excerpted"] = True
        row['control_checks'] = []
        for check in result['control_checks']:
            compact_check = dict(check)
            for field in ('window_differences', 'outside_window_differences'):
                values = check.get(field, [])
                compact_check[field + '_count'] = len(values)
                compact_check[field] = values if len(values) <= 24 else values[:16] + values[-8:]
            row['control_checks'].append(compact_check)
        compact.append(row)
    return compact


def _shared_replay(config: Config, executor: ProbeExecutor, records: list[dict],
                   bank: list[dict], args: Namespace) -> None:
    """Use the same final discovered-probe panel for every accepted candidate."""
    signatures = {}
    for record in records:
        if record["status"] != "generated_unqualified":
            continue
        directory = args.out / "mutants" / record["id"]
        rows = []
        for index, item in enumerate(bank[:args.replay_probes]):
            # Source-mutant control IDs/expectations do not belong to this plan.
            probe = {**item["probe"], "control_checks": []}
            golden, mutant = executor.execute(directory / "rtl", probe,
                                               directory / "shared_replay" / f"p{index + 1:04d}")
            result = executor.characterize(probe, golden, mutant, record["plan"],
                                            {"input_constraints": []})
            rows.append({"bank_id": item["id"], "output_difference": result["output_difference"],
                         "trace_signature": result["trace_signature"]})
        comparable = json.dumps(rows, sort_keys=True)
        related = signatures.get(comparable, []) if rows else []
        record["behavioral_comparison"] = {"probes": rows, "similar_to": list(related),
                                            "equivalence_proven": False}
        if rows:
            signatures.setdefault(comparable, []).append(record["id"])
        write_json(directory / "shared_replay.json", record["behavioral_comparison"])
        write_json(directory / "record.json", record)


def _validate_mutant(config: Config, root: Path, directory: Path, plan: dict, analysis: dict,
                     requests: Requests, executor: ProbeExecutor, args: Namespace,
                     specification: str, feedback: str) -> tuple[dict, list[dict], list[dict]]:
    probes, results, history = [], [], []
    mutant_rtl = _sources(config, root)
    for round_index in range(args.probe_refinements + 1):
        payload = {"analysis": analysis, "plan": plan, "actual_mutant_rtl": mutant_rtl,
                   "interface": [{**p, "max_unsigned": hex((1 << p["width"]) - 1),
                                  "highest_bit_value": hex(1 << (p["width"] - 1))} for p in executor.ports],
                   "observable_state": executor.state,
                   "clock": config.clock, "reset": config.reset, "reset_active": args.reset_active,
                   "max_probes_this_round": args.probes_per_round,
                   "max_cycles_per_probe": args.max_probe_cycles,
                   "previous_probes": [{k: v for k, v in p.items() if k != "cycles"} for p in probes],
                   "previous_results": _result_context(results),
                   "repair_feedback": feedback,
                   "instruction": "Initial trigger, control, and adversarial probes" if round_index == 0
                   else "Refine probes to address the measured evidence gaps; keep prior evidence"}
        if history:
            payload["evidence_gaps"] = history[-1]["evidence"]["gaps"]
        batch = requests.ask("probes" if round_index == 0 else "probe_refinement", prompts.PROBES,
                             payload, lambda raw: _probe_batch(raw, config, executor.ports,
                             executor.state, plan, args, specification, {p["id"] for p in probes}))
        for probe in batch:
            # Filenames are controller-owned; model probe IDs never become filesystem paths.
            probe_dir = directory / "probes" / f"p{len(probes) + 1:04d}"
            write_json(probe_dir / "probe.json", probe)
            golden, mutant = executor.execute(root, probe, probe_dir)
            result = executor.characterize(probe, golden, mutant, plan, analysis)
            result["trace_file"] = str((probe_dir / "traces.json").relative_to(args.out))
            write_json(probe_dir / "result.json", result)
            probes.append(probe)
            results.append(result)
        evidence = summarize_evidence(results, plan)
        evidence["exposure_evidence"]["history_comparisons"] = _history_pairs(probes, results)
        history.append({"round": round_index, "new_probes": len(batch), "evidence": evidence})
        write_json(directory / "probe_refinements.json", history)
        write_json(directory / "results.json", results)
        write_json(directory / "validation.json", evidence)
        if evidence["ready_for_review"]:
            break
    return evidence, probes, results


def _report(output: Path, summary: dict, records: list[dict]) -> None:
    fields = ["id", "status", "family", "exposure", "mechanism", "evidence_level", "specification_evidence"]
    with (output / "mutant_catalog.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            plan = record["plan"]
            writer.writerow({"id": record["id"], "status": record["status"], "family": plan["family"],
                             "exposure": ",".join(plan["exposure"]), "mechanism": plan["mechanism"],
                             "evidence_level": record.get("validation", {}).get("evidence_level", "not_run"),
                             "specification_evidence": record.get("review", {}).get("specification", {}).get("assessment", "unreviewed")})
    (output / "report.md").write_text(
        "# RTL mutation exploration\n\n" +
        f"Attempted {summary['attempts']}; accepted {summary['accepted']}; "
        f"proposed {summary['plans_proposed']} plans. Stop: {summary['stop_reason']}.\n\n"
        "Accepted means structurally checked, targeted output evidence and exercised controls, "
        "plus model-supported specification relevance. It does not mean formal specification proof.\n\n"
        "- `design_analysis.json`: full-context behavioral catalog and ambiguities.\n"
        "- `functional_regions.json`: overlapping exploration regions.\n"
        "- `exploration_tracker.json`: queues, repeated rounds, and outcomes.\n"
        "- `fault_plans.json`: proposed plans, including unimplemented plans.\n"
        "- `mechanism_archive.json`: similarity judgments and outcomes.\n"
        "- `mutant_catalog.csv`: candidate index.\n"
        "- `mutants/mNNNN/record.json`: intended plan, edits, review, and uncertainties.\n"
        "- `mutants/mNNNN/plan_validation.json`: source-grounded semantic plan assessment.\n"
        "- `mutants/mNNNN/attempts/aN/plan_rtl_verification.json`: actual-edit assessment.\n"
        "- `mutants/mNNNN/attempts/aN/probes/pNNNN/`: stimuli, testbench, full traces, measurements.\n\n"
        "Held-out assertions were not used. Run the existing `score` command separately.\n")


def run_exploration(args: Namespace, client=None) -> dict:
    args.out = args.out.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a fresh --out directory; existing campaigns are never overwritten")
    raw = _config(args)
    args.out.mkdir(parents=True, exist_ok=True)
    # Preserve an explicit simulation reset polarity when the existing score
    # command later constructs Jasper jobs, without changing legacy inference.
    polarity = _reset_active(args.reset) if args.reset_active == "auto" else int(args.reset_active)
    template = args.out / "jasper_exploration.tcl.template"
    reset_command = ("reset {~" + args.reset + "}" if polarity == 0 else "reset " + args.reset) if args.reset else "reset -none"
    template.write_text("\n".join(["{{ANALYZE_COMMAND}}", "elaborate -top {{TOP}}",
        "clock " + raw["clock"] if raw["clock"] else "clock -none", reset_command,
        "prove -all", "report -summary -force -result", "exit -force", ""]))
    raw["jasper"]["tcl_template"] = str(template)
    write_json(args.out / "config.direct.json", raw)
    original_config = Config.load(args.out / "config.direct.json")
    frozen = _snapshot(original_config, args.out)
    config = replace(original_config, design_root=frozen)
    specification, rtl = read_spec(config.spec_file), _sources(config, config.design_root)
    frozen_spec = args.out / "golden" / "spec.txt"
    frozen_spec.write_text(specification)
    # Scoring later uses the immutable campaign inputs, even if the working
    # source directory has subsequently changed.
    raw["design_root"], raw["spec_file"] = str(frozen), str(frozen_spec)
    write_json(args.out / "config.direct.json", raw)
    config = replace(config, spec_file=frozen_spec)
    ports, state = discover_interface(config, args.out / "interface")
    widths = {p["name"]: p["width"] for p in ports} | state
    signals = set(widths)
    input_signals = {p["name"] for p in ports if p["direction"] == "input"}
    if args.reset_active == "auto":
        args.reset_active = _reset_active(config.reset)
    else:
        args.reset_active = int(args.reset_active)
    source_lines = "\n".join(f"{i}: {line}" for i, line in enumerate(specification.splitlines(), 1))
    rtl_index = '\n\n'.join('RTL SOURCE LINE INDEX ' + str(p) + ':\n' +
                '\n'.join(f'{i}: {line}' for i, line in enumerate((config.design_root / p).read_text().splitlines(), 1))
                for p in config.sources)
    base = ("FULL SPECIFICATION:\n" + specification +
            "\n\nSPEC SOURCE LINE INDEX (references only, not exploration boundaries):\n" + source_lines +
            "\n\nFULL GOLDEN RTL:\n" + rtl + '\n\n' + rtl_index)
    requests = Requests(config, args, base, client)
    executor = ProbeExecutor(config, ports, state, args.out)
    source_hashes = {str(p): digest((config.design_root / p).read_bytes()) for p in config.sources}
    manifest = {"version": VERSION, "spec_sha256": digest(specification.encode()),
                "original_inputs": {"design_root": str(original_config.design_root),
                                    "spec_file": str(original_config.spec_file)},
                "source_hashes": source_hashes, "model": config.planner["model"],
                "planner_settings": config.planner, "clock": config.clock,
                "reset": config.reset, "reset_active": args.reset_active,
                "assertions_used_for_generation": False,
                "implementation_hashes": {p.name: digest(p.read_bytes())
                                          for p in Path(__file__).parent.glob("*.py")}}
    implementation = args.out / "implementation"
    implementation.mkdir()
    for name in manifest["implementation_hashes"]:
        shutil.copy2(Path(__file__).with_name(name), implementation / name)
    write_json(args.out / "manifest.json", manifest)
    analysis_key = digest(json.dumps([VERSION, manifest["spec_sha256"], source_hashes,
        config.top, config.clock, config.reset, args.reset_active, config.planner["model"],
        config.planner["extra_body"], manifest["implementation_hashes"]], sort_keys=True).encode())
    analysis = None
    if args.analysis_cache and args.analysis_cache.exists():
        cached = json.loads(args.analysis_cache.read_text())
        if cached.get("fingerprint") == analysis_key:
            analysis = validate_analysis(cached["analysis"], config, specification, signals, widths, input_signals)
    records, archive, slots, batches = [], [], [], []
    region_effort, family_effort = {}, {}
    accepted, planning_calls, stop_reason = 0, 0, "running"
    seen_rtl = {rtl_fingerprint(config, config.design_root): "golden"}
    seen_mechanisms: dict[str, str] = {}
    probe_bank: list[dict] = []

    def persist():
        counts = {status: sum(r["status"] == status for r in records)
                  for status in sorted({r["status"] for r in records})}
        summary = {"planner_mode": "family_explore", "config": str(original_config.path),
                   "requested": args.count, "attempts": len(records), "accepted": accepted,
                   "plans_proposed": len(archive), "planning_calls": planning_calls,
                   "raw_plans_returned": requests.raw_plans,
                   "unique_valid_plans": len(archive),
                   "llm_calls": requests.calls, "stop_reason": stop_reason,
                   "counts": counts, "assertions_used_for_generation": False,
                   "formal_qualification_performed": False,
                   "metrics": {
                       "lint_passing_candidates": sum(any(v.get("lint", {}).get("status") == "passed"
                                                            for v in r["revisions"]) for r in records),
                       "output_observed_candidates": sum(r.get("validation", {}).get("evidence_level") ==
                                                          "observed_output_difference" for r in records),
                       "review_spec_supported_candidates": sum(r.get("review", {}).get("specification", {}).get(
                           "assessment") == "supported" for r in records),
                       'plan_supported_candidates': sum(r.get('plan_validation', {}).get('verdict') == 'supported' for r in records),
                       'edit_supported_candidates': sum(r.get('plan_rtl_verification', {}).get('verdict') == 'supported' for r in records),
                       'prediction_mismatches': sum(r.get('validation', {}).get('timing', {}).get('prediction_mismatch_count', 0) for r in records),
                       "with_missing_controls": sum(bool(r.get("validation", {}).get("fact_check", {}).get(
                           "controls_missing")) for r in records),
                       "with_reported_unexpected_effects": sum(bool(r.get("unexpected_effects")) for r in records),
                       "predicted_exposure_counts": {tag: sum(tag in r["plan"]["exposure"] for r in records)
                                                     for tag in EXPOSURES},
                       "formal_history_necessity_proofs": 0,
                   },
                   "quality_note": "Model-supported spec relevance and sampled evidence, not specification proof",
                   "by_family": {f: {"attempts": sum(r["plan"]["family"] == f for r in records),
                                      "accepted": sum(r["plan"]["family"] == f and r["status"] == "generated_unqualified"
                                                      for r in records)} for f in FAMILIES}}
        write_json(args.out / "summary.json", summary)
        write_json(args.out / "exploration_tracker.json", slots)
        write_json(args.out / "fault_plans.json", batches)
        write_json(args.out / "mechanism_archive.json", archive)
        write_json(args.out / "probe_bank.json", probe_bank)
        _report(args.out, summary, records)
        return summary

    try:
        if analysis is None:
            analysis = requests.ask("design_analysis", prompts.ANALYZE,
                {"interface": ports, "observable_state": state, "clock": config.clock,
                 "reset": config.reset, "reset_active": args.reset_active},
                lambda x: validate_analysis(x, config, specification, signals, widths, input_signals))
            if args.analysis_cache:
                write_json(args.analysis_cache, {"fingerprint": analysis_key, "analysis": analysis})
        write_json(args.out / "design_analysis.json", analysis)
        write_json(args.out / "functional_regions.json", analysis["regions"])
        slots = exploration_slots(analysis, args.cross_region, args.global_exploration)
        for slot in slots:
            slot["dispatches"] = 0
        if args.synthesis_check and not shutil.which("yosys"):
            raise RuntimeError("--synthesis-check requires yosys")
        while len(records) < args.max_attempts:
            if args.count and accepted >= args.count:
                stop_reason = "requested_count"
                break
            if requests.calls >= args.max_llm_calls:
                raise BudgetExhausted("max_llm_calls")
            eligible = [s for s in slots if s["queue"] or (
                (not args.region_rounds or s["rounds"] < args.region_rounds)
                and (not args.empty_rounds or s["empty_rounds"] < args.empty_rounds)
                and planning_calls < args.max_planning_calls)]
            if not eligible:
                stop_reason = ("no_applicable_families" if not slots else "exploration_budget_or_saturation")
                break
            cold_regions = [s for s in eligible if s["kind"] == "region"
                            and all(region_effort.get(r, 0) == 0 for r in s["region_ids"])]
            if cold_regions:
                eligible = cold_regions
            slot = min(eligible, key=lambda s: (
                sum(region_effort.get(r, 0) for r in s["region_ids"]) / len(s["region_ids"]),
                s["dispatches"], family_effort.get(s["family"], 0), s["id"]))
            slot["dispatches"] += 1
            for region_id in slot["region_ids"]:
                region_effort[region_id] = region_effort.get(region_id, 0) + 1
            family_effort[slot["family"]] = family_effort.get(slot["family"], 0) + 1
            if not slot["queue"]:
                planning_calls += 1
                slot["rounds"] += 1
                batch_record = {"focus": slot["id"], "round": slot["rounds"], "plans": [], "invalid": []}
                batches.append(batch_record)
                try:
                    response = requests.ask("fault_plans", prompts.PLAN,
                        {"analysis": analysis, "focus": {k: slot[k] for k in ("region_ids", "family", "kind")},
                         "batch_size": args.plans_per_batch, "round": slot["rounds"],
                         "archive": _compact_archive(archive),
                         "previous_batch_feedback": [b for b in batches[:-1] if b["focus"] == slot["id"]][-2:]},
                        lambda x: _plan_batch(x, analysis, slot, signals, widths, args.plans_per_batch))
                    batch_record["exploration_note"] = response.get("exploration_note", "")
                    new = []
                    for raw_plan in response["plans"]:
                        try:
                            plan = validate_plan(raw_plan, analysis, slot, signals, widths)
                            key = mechanism_key(plan)
                            if key in seen_mechanisms:
                                batch_record["invalid"].append({"reason": "exact_repeated_mechanism",
                                                                 "duplicate_of": seen_mechanisms[key], "plan": plan})
                                continue
                            plan["id"] = f"P{len(archive) + 1:05d}"
                            plan["focus"] = slot["id"]
                            seen_mechanisms[key] = plan["id"]
                            archive.append(plan)
                            new.append(plan)
                        except (ValueError, TypeError, AttributeError) as exc:
                            batch_record["invalid"].append({"reason": str(exc), "plan": raw_plan})
                    if new:
                        batch_record["plans"] = [p["id"] for p in new]
                        try:
                            comparisons = requests.ask("mechanism_similarity", prompts.SIMILARITY,
                                {"analysis": analysis, "proposed": new,
                                 "archive": _compact_archive([p for p in archive if p not in new])},
                                lambda x: _similarity(x, new, [p for p in archive if p not in new]))
                            for plan in new:
                                plan["similarity"] = comparisons[plan["id"]]
                        except (ValueError, RuntimeError, OSError) as exc:
                            if isinstance(exc, BudgetExhausted):
                                raise
                            for plan in new:
                                plan["similarity"] = {"relationship": "unresolved", "reason": str(exc)}
                        new.sort(key=lambda p: {"novel": 0, "related": 1, "unresolved": 2,
                                                "redundant": 3}[p["similarity"]["relationship"]])
                        slot["queue"].extend(p["id"] for p in new)
                        slot["proposed"] += len(new)
                        slot["empty_rounds"] = 0
                    else:
                        slot["empty_rounds"] += 1
                    batch_record["plans"] = [p["id"] for p in new]
                except (ValueError, RuntimeError, OSError) as exc:
                    if isinstance(exc, BudgetExhausted):
                        raise
                    slot["empty_rounds"] += 1
                    batch_record["error"] = str(exc)
                persist()
                if not slot["queue"]:
                    continue
            plan_id = slot["queue"].pop(0)
            plan = next(p for p in archive if p["id"] == plan_id)
            slot["attempts"] += 1
            identifier = f"m{len(records) + 1:04d}"
            directory = args.out / "mutants" / identifier
            directory.mkdir(parents=True)
            record = {"id": identifier, "category": plan["family"], "plan": plan,
                      "status": "pending", "revisions": [], "remaining_uncertainty": []}
            records.append(record)
            feedback = ""
            try:
                static = static_plan_checks(plan, widths, input_signals)
                record['plan_static_checks'] = static
                write_json(directory / 'plan_static_checks.json', static)
                if static['status'] != 'passed':
                    raise CandidateCheckFailed('plan_rejected', '; '.join(static['issues']))
                plan_review = requests.ask('plan_validation', prompts.PLAN_CHECK,
                    {'analysis': analysis, 'plan': plan, 'static_checks': static,
                     'interface': ports, 'clock': config.clock, 'reset': config.reset,
                     'reset_active': args.reset_active,
                     'review_subject': 'proposed_mutation',
                     'design_roles': {'golden': 'Unmodified baseline; golden_behavior must match source',
                                      'proposed': 'Hypothetical variant after faulty_behavior; not yet implemented'},
                     'validation_question': 'Does the proposed change turn the golden behavior into a distinct plausible faulty behavior? Compare both under identical inputs. Do not require the fault to already exist in golden RTL.'},
                    lambda raw: semantic_review(raw, PLAN_CHECKS, config, config.design_root, plan))
                record['plan_validation'] = plan_review
                write_json(directory / 'plan_validation.json', plan_review)
                if plan_review['verdict'] != 'supported':
                    raise CandidateCheckFailed('plan_rejected' if plan_review['verdict'] == 'mismatch'
                                               else 'unresolved_plan_validation', plan_review['reason'])
                for revision in range(args.max_repairs + 1):
                    attempt = directory / "attempts" / f"a{revision + 1}"
                    attempt.mkdir(parents=True)
                    entry = {"revision": revision, "status": "pending"}
                    record["revisions"].append(entry)
                    record['latest_attempt'] = f'a{revision + 1}'
                    # Prior evidence stays in revisions; top-level evidence must
                    # never describe an earlier source revision as the current one.
                    for field in ('validation', 'review', 'plan_rtl_verification'):
                        record.pop(field, None)
                    try:
                        edits = requests.ask("rtl_edit", prompts.EDIT,
                            {"analysis": analysis, "plan": plan, "plan_validation": plan_review, "feedback": feedback,
                             "previous_edits": [r.get("edits", []) for r in record["revisions"][:-1]]},
                            lambda x: _parse_edits(x, config))
                        entry["edits"] = [vars(e) for e in edits]
                        applied = apply_edits(config, edits, attempt)
                        root = attempt / "rtl"
                        if _top_header(config, root) != _top_header(config, config.design_root):
                            raise ValueError("Top module declaration/interface changed")
                        fingerprint = rtl_fingerprint(config, root)
                        if fingerprint in seen_rtl and seen_rtl[fingerprint] != identifier:
                            raise ValueError("Resulting RTL duplicates " + seen_rtl[fingerprint])
                        entry["lint"] = lint(config, attempt)
                        if entry["lint"]["status"] != "passed":
                            raise ValueError("Verilator lint failed: " + entry["lint"].get("detail", ""))
                        mutated_ports, _ = discover_interface(replace(config, design_root=root), attempt / "interface")
                        if mutated_ports != ports:
                            raise ValueError("Elaborated input/output names, directions, or widths changed")
                        # Keep even unresolved/rejected compiled candidates in the archive.
                        # Same-candidate repairs may retain the source; other candidates cannot.
                        seen_rtl[fingerprint] = identifier
                        entry["synthesis"] = _synthesis(config, root) if args.synthesis_check else {"status": "not_run"}
                        if entry["synthesis"]["status"] == "failed":
                            raise ValueError("Yosys synthesis frontend check failed: " + entry["synthesis"]["detail"])
                        mutant_index = '\n\n'.join('MUTANT SOURCE LINE INDEX ' + str(p) + ':\n' +
                            '\n'.join(f'{i}: {line}' for i, line in enumerate((root / p).read_text().splitlines(), 1))
                            for p in config.sources)
                        edit_review = requests.ask('plan_rtl_verification', prompts.EDIT_CHECK,
                            {'plan': plan, 'plan_validation': plan_review, 'actual_edits': entry['edits'],
                             'actual_mutant_rtl': _sources(config, root), 'mutant_source_index': mutant_index,
                             'interface': ports, 'clock': config.clock, 'reset': config.reset},
                            lambda raw: semantic_review(raw, EDIT_CHECKS, config, root, plan, edit=True))
                        entry['plan_rtl_verification'] = edit_review
                        record['plan_rtl_verification'] = edit_review
                        write_json(attempt / 'plan_rtl_verification.json', edit_review)
                        if edit_review['verdict'] != 'supported':
                            entry['status'] = 'plan_mismatch' if edit_review['verdict'] == 'mismatch' else 'unresolved_plan_match'
                            feedback = json.dumps({'same_mechanism_repair': True, 'verification': edit_review})
                            if edit_review['verdict'] == 'uncertain':
                                raise CandidateCheckFailed('unresolved_plan_match', edit_review['reason'])
                            continue
                        evidence, probes, results = _validate_mutant(config, root, attempt, plan, analysis,
                            requests, executor, args, specification, feedback)
                        entry["validation"] = evidence
                        record["validation"] = evidence
                        # Limited shared bank makes sampled behavior comparable without fixed stimuli.
                        replay = []
                        for number, bank in enumerate(probe_bank[:args.replay_probes]):
                            golden, mutant = executor.execute(root, bank["probe"], attempt / "replay" / f"p{number + 1:04d}")
                            measured = executor.characterize({**bank["probe"], "control_checks": []},
                                                               golden, mutant, plan, analysis)
                            replay.append({"bank_id": bank["id"], "output_difference": measured["output_difference"],
                                           "trace_signature": measured["trace_signature"]})
                        write_json(attempt / "replay.json", replay)
                        record["behavioral_comparison"] = {"probes": replay,
                            "similar_to": [r["id"] for r in records[:-1]
                                if replay and r.get("behavioral_comparison", {}).get("probes") == replay],
                            "equivalence_proven": False}
                        if (not evidence["ready_for_review"] and
                                evidence["evidence_level"] in ("unobserved", "unknown_only", "internal_difference_only")):
                            entry["status"] = "unresolved_behavior"
                            feedback = json.dumps({"evidence_gaps": evidence["gaps"],
                                                   "results": _result_context(results)}, ensure_ascii=False)
                            # No exposure alone is not grounds to change a plausible implementation.
                            break
                        trace_excerpts = []
                        for result in results:
                            rows = json.loads((args.out / result["trace_file"]).read_text())
                            divergence = (result["first_difference"] or {}).get("cycle", 0)
                            selected = sorted(set(range(min(3, len(rows)))) |
                                set(range(max(0, divergence - 2), min(len(rows), divergence + 8))) |
                                {p["cycle"] for p in result["predictions"]} |
                                {c["cycle"] for c in result["control_checks"]} |
                                {len(rows) - 1})
                            trace_excerpts.append({"probe": result["id"], "rows": [rows[i] for i in selected],
                                                   "full_trace_file": result["trace_file"]})
                        review = requests.ask("behavior_review", prompts.REVIEW,
                            {"analysis": analysis, "plan": plan, "plan_validation": plan_review,
                             "plan_rtl_verification": edit_review, "actual_mutant_rtl": _sources(config, root),
                             "actual_edits": entry["edits"], "evidence": evidence,
                             "results": _result_context(results), "trace_excerpts": trace_excerpts,
                             "mechanism_similarity": plan.get("similarity"), "cross_mutant_replay": replay},
                            lambda x: _review(x, evidence, specification))
                        entry["review"] = review
                        record["review"] = review
                        acceptable = (evidence["ready_for_review"] and
                            review["verdict"] == "accept" and review["plan_match"] == "supported"
                            and review["realism"]["assessment"] == "plausible"
                            and review["specification"]["assessment"] == "supported")
                        if acceptable:
                            shutil.copytree(root, directory / "rtl")
                            record.update(status="generated_unqualified", edit=applied,
                                edits=entry["edits"], rtl_fingerprint=fingerprint, lint=entry["lint"],
                                implemented_behavior=review["implemented_behavior"],
                                observed_behavior=review["observed_behavior"],
                                remaining_uncertainty=review["unverified_claims"],
                                unexpected_effects=review["unexpected_effects"],
                                specification_evidence={"status": "model_supported", "formal_proof": False})
                            seen_rtl[fingerprint] = identifier
                            accepted += 1
                            slot["accepted"] += 1
                            entry["status"] = "accepted"
                            existing = {digest(json.dumps(b["probe"]["cycles"], sort_keys=True).encode()) for b in probe_bank}
                            for probe, result in zip(probes, results):
                                key = digest(json.dumps(probe["cycles"], sort_keys=True).encode())
                                if result["output_difference"] and not result["constraint_violations"] and key not in existing:
                                    probe_bank.append({"id": f"T{len(probe_bank) + 1:05d}",
                                                       "source_mutant": identifier, "probe": probe})
                                    existing.add(key)
                            break
                        if review["verdict"] == "accept" and not evidence["ready_for_review"]:
                            entry["status"] = "unresolved_behavior"
                            feedback = "Controller evidence gaps override review acceptance: " + "; ".join(evidence["gaps"])
                            break
                        entry["status"] = "review_rejected" if review["verdict"] == "reject" else "repair_requested"
                        feedback = json.dumps(review)
                        if review["verdict"] == "reject":
                            break
                        if review['plan_match'] != 'mismatch':
                            # Missing evidence/prediction mistakes do not justify changing the bug.
                            entry['status'] = 'unresolved_behavior'
                            break
                    except StageResponseError as exc:
                        # Bad probe JSON/service failure does not invalidate the RTL.
                        entry.update(status="unresolved_" + exc.stage, reason=str(exc))
                        record.update(status="unresolved", reason=str(exc))
                        if (attempt / 'validation.json').exists():
                            record['validation'] = json.loads((attempt / 'validation.json').read_text())
                            entry['validation'] = record['validation']
                        break
                    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                        if isinstance(exc, BudgetExhausted):
                            raise
                        entry.update(status="implementation_error", reason=str(exc))
                        feedback = str(exc)
                if record["status"] == "pending":
                    record.update(status=("unresolved" if any(r["status"] == "unresolved_behavior" for r in record["revisions"])
                        else 'plan_mismatch' if record['revisions'][-1]['status'] == 'plan_mismatch'
                        else "review_rejected" if record.get("review") else "implementation_rejected"), reason=feedback)
            except CandidateCheckFailed as exc:
                record.update(status=exc.status, reason=str(exc))
            except StageResponseError as exc:
                record.update(status='unresolved_' + exc.stage, reason=str(exc))
            except BudgetExhausted:
                record.update(status="unresolved_budget", reason="LLM call budget reached")
                if record['revisions'] and record['revisions'][-1]['status'] == 'pending':
                    record['revisions'][-1]['status'] = 'unresolved_budget'
                raise
            finally:
                plan["outcome"] = record["status"]
                plan["observed_behavior"] = record.get("observed_behavior", "not observed/reviewed")
                slot["outcomes"][record["status"]] = slot["outcomes"].get(record["status"], 0) + 1
                write_json(directory / "record.json", record)
                persist()
                print(f"{identifier}: {record['status']} ({plan['family']}, {plan_id})", flush=True)
        else:
            stop_reason = "requested_count" if args.count and accepted >= args.count else "max_attempts"
    except BudgetExhausted as exc:
        stop_reason = str(exc)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        stop_reason = "setup_or_controller_error"
        write_json(args.out / "setup_error.json", {"reason": str(exc)})
        persist()
        raise
    # Replaying a common final panel also backfills early mutants that predated the bank.
    try:
        _shared_replay(config, executor, records, probe_bank, args)
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        write_json(args.out / "replay_error.json", {"reason": str(exc),
                                                    "note": "Generation decisions were preserved"})
    summary = persist()
    print(f"Results: {args.out / 'summary.json'}", flush=True)
    return summary
