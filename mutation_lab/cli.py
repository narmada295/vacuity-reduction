"""Command-line entry points for proposing, qualifying, and scoring RTL mutants."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .core import (SCENARIO_CATEGORIES, Config, ScenarioMutation, apply_proposal,
                   apply_scenario, rtl_fingerprint, stage_sources, write_json)
from .formal import lint, named_status, run_check
from .planner import FilePlanner, LLMPlanner
from .scenario_validation import validate_scenario


def _require_fresh(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Output directory already contains files: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


def mutation_schedule(count: int, mix: str | None) -> list[str]:
    """Interleave exact per-category quotas; --count retains its legacy meaning."""
    if not mix:
        return ["simple"] * count
    remaining: dict[str, int] = {}
    for part in mix.split(","):
        name, separator, raw_number = part.strip().partition("=")
        name, raw_number = name.strip(), raw_number.strip()
        if (not separator or name not in ("simple", *SCENARIO_CATEGORIES)
                or name in remaining or not raw_number.isdigit() or int(raw_number) < 1):
            raise ValueError("--mix needs unique positive category counts, e.g. "
                             "simple=3,timing=2,state_sequence=1,interaction=1")
        remaining[name] = int(raw_number)
    schedule = []
    while any(remaining.values()):
        for name in remaining:
            if remaining[name]:
                schedule.append(name)
                remaining[name] -= 1
    return schedule


def _baseline(config: Config, output: Path) -> dict:
    if not config.requirements or not config.oracle_file:
        raise ValueError("Qualification needs requirements and oracle_file in the config")
    directory = output / "golden"
    stage_sources(config, directory)
    lint_result = lint(config, directory)
    record = {"lint": lint_result}
    if lint_result["status"] == "failed":
        write_json(output / "golden.json", record)
        raise RuntimeError("Golden RTL fails lint; see golden.json")
    formal = run_check(config, directory, config.oracle_file)
    record["formal"] = formal
    record["requirements"] = {
        requirement.id: {
            "oracle": named_status(formal["properties"], requirement.oracle),
            "reach": named_status(formal["properties"], requirement.reach),
        } for requirement in config.requirements
    }
    write_json(output / "golden.json", record)
    invalid = [name for name, result in record["requirements"].items()
               if result != {"oracle": "proven", "reach": "covered"}]
    if formal["status"] != "completed" or invalid:
        raise RuntimeError("Golden oracle proof/reachability incomplete; see golden.json "
                           f"(formal={formal['status']}, requirements={invalid})")
    return record


def run_generation(args: argparse.Namespace) -> None:
    config = Config.load(args.config)
    output = args.out.resolve()
    _require_fresh(output)
    (output / "config.snapshot.json").write_bytes(config.path.read_bytes())
    if not args.generate_only:
        _baseline(config, output)
    validate_scenarios = (bool(config.planner.get("require_witness", True))
                          and not getattr(args, "no_scenario_validation", False))
    planner = (FilePlanner(config, args.proposals.resolve()) if args.proposals else
               LLMPlanner(config, getattr(args, "scenario_mode", None),
                          require_witness=validate_scenarios))
    planner_mode = "offline" if args.proposals else planner.scenario_mode
    feedback: list[dict] = []
    seen: set[str] = set()
    seen_rtl: dict[str, str] = {}
    golden_rtl = rtl_fingerprint(config, config.design_root)
    schedule = mutation_schedule(args.count, getattr(args, "mix", None))
    requested = Counter(schedule)
    accepted: Counter[str] = Counter()
    fill_quotas = getattr(args, "fill_quotas", False)
    max_attempts = getattr(args, "max_attempts", None)
    if fill_quotas and (max_attempts is None or max_attempts < len(schedule)):
        raise ValueError("--fill-quotas needs --max-attempts at least as large as "
                         "the requested total")
    if not fill_quotas:
        max_attempts = len(schedule)
    category_order = tuple(dict.fromkeys(schedule))
    category_index = 0
    index = 0
    while index < max_attempts:
        if fill_quotas:
            if all(accepted[name] >= target for name, target in requested.items()):
                break
            while True:
                category = category_order[category_index % len(category_order)]
                category_index += 1
                if accepted[category] < requested[category]:
                    break
        else:
            category = schedule[index]
        index += 1
        identifier = f"m{index:04d}"
        directory = output / "mutants" / identifier
        directory.mkdir(parents=True)
        try:
            proposal = planner.propose(feedback, category)
        except StopIteration:
            directory.rmdir()
            break
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            record = {"id": identifier, "category": category, "planner_mode": planner_mode,
                      "status": "proposal_error", "reason": str(exc)}
            focus_target = getattr(planner, "last_focus_target", None)
            if focus_target:
                record["focus_target"] = focus_target
            fault_angle = getattr(planner, "last_fault_angle", None)
            if fault_angle:
                record["fault_angle"] = fault_angle
            write_json(directory / "record.json", record)
            feedback.append({"category": category, "result": "proposal_error",
                             "reason": str(exc), "focus_target": focus_target,
                             "fault_angle": fault_angle})
            print(f"{identifier}: proposal_error")
            continue
        record = {"id": identifier, "category": category,
                  "planner_mode": planner_mode, "status": "pending"}
        focus_target = getattr(planner, "last_focus_target", None)
        if focus_target:
            record["focus_target"] = focus_target
        fault_angle = getattr(planner, "last_fault_angle", None)
        if fault_angle:
            record["fault_angle"] = fault_angle
        if isinstance(proposal, ScenarioMutation):
            record["scenario"] = asdict(proposal.scenario)
            record["edits"] = [asdict(edit) for edit in proposal.edits]
        else:
            record["proposal"] = asdict(proposal)
        key = proposal.key()
        if key in seen:
            record.update(status="duplicate", reason="Identical edit already proposed")
        else:
            seen.add(key)
            try:
                record["edit"] = (apply_scenario(config, proposal, directory)
                                  if isinstance(proposal, ScenarioMutation)
                                  else apply_proposal(config, proposal, directory))
            except (OSError, ValueError) as exc:
                record.update(status="edit_rejected", reason=str(exc))
            else:
                rtl_key = rtl_fingerprint(config, directory / "rtl")
                record["rtl_fingerprint"] = rtl_key
                if rtl_key == golden_rtl:
                    record.update(status="no_token_change",
                                  reason="Edited RTL has the same tokens as golden RTL")
                elif rtl_key in seen_rtl:
                    record.update(status="duplicate_rtl", duplicate_of=seen_rtl[rtl_key],
                                  reason=f"Resulting RTL matches {seen_rtl[rtl_key]}")
                else:
                    seen_rtl[rtl_key] = identifier
                    lint_result = lint(config, directory)
                    record["lint"] = lint_result
                    if lint_result["status"] == "failed":
                        record.update(status="lint_rejected", reason=lint_result["detail"][-1000:])
                    elif (isinstance(proposal, ScenarioMutation) and validate_scenarios
                          and proposal.scenario.witness is not None):
                        validation = validate_scenario(config, directory,
                                                       proposal.scenario.witness,
                                                       category=category,
                                                       require_controls=not bool(args.proposals))
                        record["scenario_validation"] = validation
                        if validation["status"] != "passed":
                            record.update(status=validation["status"],
                                          reason=validation.get("reason", ""))
                    if record["status"] == "pending" and args.generate_only:
                        record["status"] = "generated_unqualified"
                    elif record["status"] == "pending":
                        formal = run_check(config, directory, config.oracle_file)
                        requirement_id = (proposal.scenario.requirement_id
                                          if isinstance(proposal, ScenarioMutation)
                                          else proposal.requirement_id)
                        requirement = config.requirement(requirement_id)
                        oracle_status = named_status(formal["properties"], requirement.oracle)
                        record["formal"] = formal
                        record["oracle_status"] = oracle_status
                        if formal["status"] == "completed" and oracle_status == "cex":
                            record["status"] = "qualified"
                        elif formal["status"] == "completed" and oracle_status == "proven":
                            record.update(status="rejected_no_spec_violation",
                                          reason="Mutant still satisfies this requirement")
                        else:
                            record.update(status="inconclusive",
                                          reason=f"formal={formal['status']}; oracle={oracle_status}")
        write_json(directory / "record.json", record)
        if record["status"] in ("generated_unqualified", "qualified"):
            accepted[category] += 1
        edits = (proposal.edits if isinstance(proposal, ScenarioMutation)
                 else (proposal,))
        feedback.append({"category": category, "result": record["status"],
                         "focus_target": focus_target,
                         "fault_angle": fault_angle,
                         "reason": record.get("reason", ""),
                         "duplicate_of": record.get("duplicate_of"),
                         "fault": (proposal.scenario.faulty_behavior
                                   if isinstance(proposal, ScenarioMutation)
                                   else proposal.failure)[:300],
                         "edits": [{"source_file": edit.source_file,
                                    "old": edit.old[:240], "new": edit.new[:240]}
                                   for edit in edits]})
        print(f"{identifier}: {record['status']}")
    records = [json.loads(path.read_text()) for path in sorted((output / "mutants").glob("*/record.json"))]
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    categories: dict[str, dict[str, int]] = {}
    for record in records:
        group = categories.setdefault(record.get("category", "simple"), {})
        group[record["status"]] = group.get(record["status"], 0) + 1
    write_json(output / "summary.json", {"config": str(config.path),
                                         "planner_mode": planner_mode, "counts": counts,
                                         "categories": categories,
                                         "requested_mix": dict(requested),
                                         "accepted_mix": dict(accepted),
                                         "attempts": len(records),
                                         "fill_quotas": fill_quotas,
                                         "max_attempts": max_attempts,
                                         "mutants": [item["id"] for item in records]})
    print(f"Results: {output / 'summary.json'}")


def run_evaluation(args: argparse.Namespace) -> None:
    config = Config.load(args.config)
    if not config.evaluation_file or not config.evaluation_names:
        raise ValueError("Evaluation needs evaluation_file and evaluation_names in the config")
    output = args.out.resolve()
    records = [json.loads(path.read_text()) for path in sorted((output / "mutants").glob("*/record.json"))]
    qualified = [item for item in records if item["status"] == "qualified"]
    if not qualified:
        raise ValueError("No qualified mutants to evaluate")
    snapshot = config.source_snapshot()
    if any(item.get("edit", {}).get("source_snapshot") != snapshot for item in qualified):
        raise ValueError("Golden source files changed since generation; use the original config and RTL")
    evaluation = output / "evaluation"
    _require_fresh(evaluation)
    golden = evaluation / "golden"
    stage_sources(config, golden)
    golden_run = run_check(config, golden, config.evaluation_file)
    baseline = {name: named_status(golden_run["properties"], name)
                for name in config.evaluation_names}
    if golden_run["status"] != "completed":
        raise RuntimeError("Golden assertion evaluation failed; see evaluation/golden/jasper.log")
    results = []
    for item in qualified:
        identifier = item["id"]
        destination = evaluation / identifier
        destination.mkdir()
        shutil.copytree(output / "mutants" / identifier / "rtl", destination / "rtl")
        run = run_check(config, destination, config.evaluation_file)
        statuses = {name: named_status(run["properties"], name)
                    for name in config.evaluation_names}
        labels = {name: ("invalid_on_golden" if baseline[name] != "proven" else
                         "inconclusive" if run["status"] != "completed" else
                         "killed" if statuses[name] == "cex" else
                         "survived" if statuses[name] == "proven" else "inconclusive")
                  for name in config.evaluation_names}
        results.append({"mutant": identifier, "formal_status": run["status"],
                        "assertion_statuses": statuses, "labels": labels})
        print(f"{identifier}: {labels}")
    write_json(evaluation / "results.json", {"golden_assertions": baseline, "mutants": results})


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated specification-guided RTL mutation prototype")
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run", help="Propose, generate, and qualify one mutant at a time")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--count", type=int, default=1, help="Number of proposals to request")
    run.add_argument("--mix", help="Exact category quotas, e.g. simple=3,timing=2; overrides --count")
    run.add_argument("--fill-quotas", action="store_true",
                     help="Retry rejected attempts until each category has its requested candidates")
    run.add_argument("--max-attempts", type=int,
                     help="Required with --fill-quotas; caps total LLM proposals")
    run.add_argument("--scenario-mode", choices=("two-call", "one-call"),
                     help="LLM requests per non-simple scenario; default is two-call")
    run.add_argument("--no-scenario-validation", action="store_true",
                     help="Skip concrete witness simulation for scenario proposals")
    run.add_argument("--proposals", type=Path, help="Offline JSONL proposals instead of an LLM")
    run.add_argument("--generate-only", action="store_true", help="Generate RTL without formal qualification")
    evaluate = subparsers.add_parser("evaluate", help="Run frozen evaluation assertions on qualified mutants")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    campaign = subparsers.add_parser("campaign", help="Run from spec, RTL, and assertion files directly")
    campaign.add_argument("--spec", type=Path, required=True)
    campaign.add_argument("--rtl", type=Path, nargs="+", required=True)
    campaign.add_argument("--assertions", type=Path, help="SANGAM-style JSONL or named bind SV checker")
    campaign.add_argument("--out", type=Path, required=True)
    campaign.add_argument("--count", type=int, default=1)
    campaign.add_argument("--mix", help="Exact category quotas, e.g. simple=3,timing=2; overrides --count")
    campaign.add_argument("--fill-quotas", action="store_true",
                          help="Retry rejected attempts until each category has its requested candidates")
    campaign.add_argument("--max-attempts", type=int,
                          help="Required with --fill-quotas; caps total LLM proposals")
    campaign.add_argument("--scenario-mode", choices=("two-call", "one-call"),
                          default="two-call", help="LLM requests per non-simple scenario")
    campaign.add_argument("--no-scenario-validation", action="store_true",
                          help="Skip concrete witness simulation for scenario proposals")
    campaign.add_argument("--proposals", type=Path, help="Offline JSONL proposal list instead of LLM")
    campaign.add_argument("--generate-only", action="store_true")
    campaign.add_argument("--max-assertions", type=int, default=0,
                          help="Use first N JSONL assertions for a small pilot; 0 means all")
    campaign.add_argument("--design-root", type=Path)
    campaign.add_argument("--include-dir", type=Path, action="append", default=[])
    campaign.add_argument("--focus-rtl", type=Path, action="append", default=[])
    campaign.add_argument("--top")
    campaign.add_argument("--clock")
    campaign.add_argument("--reset", help="Optional formal reset; omitted means unconstrained")
    campaign.add_argument("--jg-mode", choices=("ssh", "local"), default="ssh")
    campaign.add_argument("--jg-timeout", type=int, default=180)
    campaign.add_argument("--jg-executable", default="jg", help="Local JasperGold command")
    campaign.add_argument("--env-file", type=Path, default=Path(".env"))
    campaign.add_argument("--endpoint", default="https://api.tokenfactory.nebius.com/v1/chat/completions")
    campaign.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    campaign.add_argument("--key-env", default="NEBIUS_API_KEY1")
    campaign.add_argument("--max-context-chars", type=int, default=100000)
    campaign.add_argument("--planner-max-tokens", type=int, default=4096,
                          help="Maximum model output tokens per planning or edit request")
    campaign.add_argument("--lint-arg", action="append", default=[],
                          help="Extra Verilator option, e.g. --lint-arg=--timing")
    trace = subparsers.add_parser("trace-campaign", help="Generate mutants from measured golden traces and RTL templates")
    trace.add_argument("--spec", type=Path, required=True)
    trace.add_argument("--rtl", type=Path, nargs="+", required=True)
    trace.add_argument("--out", type=Path, required=True)
    trace.add_argument("--reset", required=True, help="Top-level reset input for reachable traces")
    trace.add_argument("--count", type=int, default=5, help="Maximum measured mutants to keep")
    trace.add_argument("--max-seeds", type=int, default=18)
    trace.add_argument("--planner", choices=("offline", "llm"), default="offline",
                       help="Rank discovered mechanisms with one LLM call, or use template order")
    trace.add_argument("--formal-compare", action="store_true",
                       help="Run a reset-constrained JasperGold output miter on each measured mutant")
    trace.add_argument("--design-root", type=Path)
    trace.add_argument("--include-dir", type=Path, action="append", default=[])
    trace.add_argument("--focus-rtl", type=Path, action="append", default=[])
    trace.add_argument("--top")
    trace.add_argument("--clock")
    trace.add_argument("--jg-mode", choices=("ssh", "local"), default="ssh")
    trace.add_argument("--jg-timeout", type=int, default=180)
    trace.add_argument("--jg-executable", default="jg")
    trace.add_argument("--env-file", type=Path, default=Path(".env"))
    trace.add_argument("--endpoint", default="https://api.tokenfactory.nebius.com/v1/chat/completions")
    trace.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    trace.add_argument("--key-env", default="NEBIUS_API_KEY1")
    trace.add_argument("--max-context-chars", type=int, default=100000)
    trace.add_argument("--planner-max-tokens", type=int, default=4096)
    trace.add_argument("--lint-arg", action="append", default=[])
    quality = subparsers.add_parser("refine-campaign", help="Generate spec-grounded RTL mutants with LLM critique and repair")
    quality.add_argument("--spec", type=Path, required=True)
    quality.add_argument("--rtl", type=Path, nargs="+", required=True)
    quality.add_argument("--out", type=Path, required=True)
    quality.add_argument("--count", type=int, default=5, help="Target number of accepted mutants")
    quality.add_argument("--max-attempts", type=int,
                         help="Maximum fault plans; defaults to three times --count")
    quality.add_argument("--max-repairs", type=int, default=1,
                         help="Additional edit attempts for one fault plan")
    quality.add_argument("--max-requirements", type=int, default=12)
    quality.add_argument("--categories", default="simple,timing,state_sequence,interaction",
                         help="Comma-separated fault categories to rotate through")
    quality.add_argument("--quality-threshold", type=int, default=19,
                         help="Minimum LLM critic score out of 25, with every dimension at least 3")
    quality.add_argument("--behavior-check", choices=("required", "warn", "off"),
                         default="required",
                         help="Require sampled RTL behavior evidence, record warnings, or disable probes")
    quality.add_argument("--max-probes", type=int, default=48,
                         help="Maximum design-independent input traces per mutant")
    quality.add_argument("--design-root", type=Path)
    quality.add_argument("--include-dir", type=Path, action="append", default=[])
    quality.add_argument("--focus-rtl", type=Path, action="append", default=[])
    quality.add_argument("--top")
    quality.add_argument("--clock")
    quality.add_argument("--reset")
    quality.add_argument("--jg-mode", choices=("ssh", "local"), default="ssh")
    quality.add_argument("--jg-timeout", type=int, default=180)
    quality.add_argument("--jg-executable", default="jg")
    quality.add_argument("--env-file", type=Path, default=Path(".env"))
    quality.add_argument("--endpoint", default="https://api.tokenfactory.nebius.com/v1/chat/completions")
    quality.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    quality.add_argument("--key-env", default="NEBIUS_API_KEY1")
    quality.add_argument("--max-context-chars", type=int, default=100000)
    quality.add_argument("--planner-max-tokens", type=int, default=4096)
    quality.add_argument("--planner-extra-body", type=Path,
                         help="JSON file of provider request settings; replaces default thinking settings")
    quality.add_argument("--planner-timeout", type=int, default=120,
                         help="Timeout in seconds for each model request")
    quality.add_argument("--lint-arg", action="append", default=[])
    audit = subparsers.add_parser("audit-refine", help="Replay saved refine mutants through sampled behavior checks")
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--max-probes", type=int, default=48)
    score = subparsers.add_parser("score", help="Score saved candidate mutants; safe to retry after license waits")
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--assertions", type=Path, required=True)
    score.add_argument("--max-assertions", type=int, default=0)
    score.add_argument("--jg-timeout", type=int, help="Override saved JasperGold timeout")
    from .exploration_cli import register_exploration
    register_exploration(subparsers)
    args = parser.parse_args()
    if args.action == "run":
        if args.count < 1:
            parser.error("--count must be positive")
        try:
            total = len(mutation_schedule(args.count, args.mix))
        except ValueError as exc:
            parser.error(str(exc))
        if args.fill_quotas and (args.max_attempts is None or args.max_attempts < total):
            parser.error("--fill-quotas requires --max-attempts >= requested total")
        if args.max_attempts is not None and args.max_attempts < 1:
            parser.error("--max-attempts must be positive")
        run_generation(args)
    elif args.action == "evaluate":
        run_evaluation(args)
    elif args.action == "campaign":
        if args.count < 1 or args.max_assertions < 0 or args.planner_max_tokens < 1:
            parser.error("--count and --planner-max-tokens must be positive; "
                         "--max-assertions must be nonnegative")
        try:
            total = len(mutation_schedule(args.count, args.mix))
        except ValueError as exc:
            parser.error(str(exc))
        if args.fill_quotas and (args.max_attempts is None or args.max_attempts < total):
            parser.error("--fill-quotas requires --max-attempts >= requested total")
        if args.max_attempts is not None and args.max_attempts < 1:
            parser.error("--max-attempts must be positive")
        from .campaign import run_campaign
        run_campaign(args)
    elif args.action == "trace-campaign":
        if args.count < 1 or args.max_seeds < 1 or args.planner_max_tokens < 1:
            parser.error("--count, --max-seeds, and --planner-max-tokens must be positive")
        from .trace_guided import run_trace_campaign
        run_trace_campaign(args)
    elif args.action == "compare-probes":
        if args.max_probes < 1:
            parser.error("--max-probes must be positive")
        from .exploration_compare import compare_probes
        compare_probes(args.out, args.max_probes)
    elif args.action == "explore-campaign":
        from .exploration_cli import validate_arguments
        from .exploration import run_exploration
        validate_arguments(args, parser)
        run_exploration(args)
    elif args.action == "refine-campaign":
        from .quality_campaign import CATEGORIES, run_quality_campaign
        if (args.count < 1 or args.max_repairs < 0 or args.max_requirements < 1
                or args.planner_max_tokens < 1 or args.max_probes < 1 or args.planner_timeout < 1):
            parser.error("--count, --max-requirements, and --planner-max-tokens must be positive; "
                         "--max-probes and --planner-timeout must be positive; --max-repairs must be nonnegative")
        if args.max_attempts is None:
            args.max_attempts = args.count * 3
        if args.max_attempts < 1 or not 15 <= args.quality_threshold <= 25:
            parser.error("--max-attempts must be positive; --quality-threshold must be 15..25")
        categories = [part.strip() for part in args.categories.split(",")]
        if not categories or any(part not in CATEGORIES for part in categories) or len(set(categories)) != len(categories):
            parser.error("--categories needs unique names from simple,timing,state_sequence,interaction")
        run_quality_campaign(args)
    elif args.action == "audit-refine":
        if args.max_probes < 1:
            parser.error("--max-probes must be positive")
        from .quality_campaign import audit_saved_candidates
        audit_saved_candidates(args.out, args.max_probes)
    else:
        if args.max_assertions < 0:
            parser.error("--max-assertions must be nonnegative")
        from .campaign import retry_scoring
        retry_scoring(args)
