"""Direct spec + RTL + assertion-set campaign, with no required oracle file."""

from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path

from .assertions import AssertionSet, inject_in_top, load_assertions
from .cli import run_generation
from .core import Config, digest, stage_sources, write_json
from .formal import named_status, run_check


def _infer_top(paths: list[Path]) -> str:
    names = re.findall(r"\bmodule\s+([A-Za-z_]\w*)\b",
                       "\n".join(path.read_text(errors="replace") for path in paths))
    if len(names) != 1:
        raise ValueError(f"Found {len(names)} RTL modules; pass --top explicitly")
    return names[0]


def _infer_clock(paths: list[Path]) -> str:
    names = set(re.findall(r"@\s*\(\s*posedge\s+([A-Za-z_]\w*)",
                           "\n".join(path.read_text(errors="replace") for path in paths)))
    if len(names) != 1:
        raise ValueError(f"Found clock candidates {sorted(names)}; pass --clock explicitly")
    return names.pop()


def _within(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError as exc:
        raise ValueError(f"Path must be inside design root {root}: {path}") from exc


def direct_config(args: Namespace) -> dict:
    paths = [path.resolve() for path in args.rtl]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("Every --rtl path must be an existing file")
    root = args.design_root.resolve() if args.design_root else paths[0].parent
    sources = [_within(root, path) for path in paths]
    includes = ["."] + [_within(root, path) for path in args.include_dir]
    focused = [_within(root, path) for path in args.focus_rtl] if args.focus_rtl else sources
    if any(path not in sources for path in focused):
        raise ValueError("Every --focus-rtl path must also be listed with --rtl")
    spec = args.spec.resolve()
    if not spec.is_file():
        raise ValueError(f"Specification not found: {spec}")
    extra_body = ({"chat_template_kwargs": {"thinking": False}}
                  if "nebius.com" in args.endpoint else {})
    options_file = getattr(args, "planner_extra_body", None)
    if options_file is not None:
        extra_body = json.loads(options_file.read_text())
        if not isinstance(extra_body, dict):
            raise ValueError("--planner-extra-body must contain a JSON object")
        reserved = {"model", "messages", "max_tokens", "stream"} & extra_body.keys()
        if reserved:
            raise ValueError("Planner extra body cannot override: " + ", ".join(sorted(reserved)))
    return {
        "design_root": str(root),
        "sources": sources,
        "include_dirs": list(dict.fromkeys(includes)),
        "top": args.top or _infer_top(paths),
        "clock": args.clock or _infer_clock(paths),
        "spec_file": str(spec),
        "requirements": [{"id": "SPEC", "text": "Choose one concrete functional requirement "
                          "from the supplied specification for each proposed mutant.",
                          "oracle": "ORACLE_UNUSED", "reach": "REACH_UNUSED"}],
        "planner": {"endpoint": args.endpoint, "model": args.model,
                    "scenario_mode": getattr(args, "scenario_mode", "two-call"),
                    "require_witness": not getattr(args, "no_scenario_validation", False),
                    "key_env": args.key_env, "env_file": str(args.env_file.resolve()),
                    "context_sources": focused, "max_context_chars": args.max_context_chars,
                    "max_tokens": getattr(args, "planner_max_tokens", 4096),
                    "timeout_seconds": getattr(args, "planner_timeout", 120),
                    "request_log": str((args.out / "model_requests.jsonl").resolve()),
                    "extra_body": extra_body},
        "jasper": {"mode": args.jg_mode, "timeout_seconds": args.jg_timeout,
                   "executable": args.jg_executable,
                   "env_file": str(args.env_file.resolve())},
        "lint": {"extra_args": args.lint_arg},
        **({"reset": args.reset} if args.reset else {}),
    }


def _prepare_evaluation(config: Config, directory: Path, assertion_set: AssertionSet) -> Path:
    if assertion_set.inline_code is not None:
        inject_in_top(config, directory, assertion_set.inline_code)
        empty = directory / "empty_checker.sv"
        empty.write_text("// Assertions are embedded in the isolated RTL copy.\n")
        return empty
    assert assertion_set.checker_file is not None
    return assertion_set.checker_file


def evaluate_candidates(config: Config, output: Path, assertions: Path, limit: int,
                        evaluation_name: str = "evaluation") -> dict:
    assertion_set = load_assertions(assertions, limit)
    records = [json.loads(path.read_text())
               for path in sorted((output / "mutants").glob("*/record.json"))]
    candidates = [item for item in records if item["status"] == "generated_unqualified"]
    if not candidates:
        raise ValueError("No generated, lint-passing mutants to evaluate")
    snapshot = config.source_snapshot()
    if any(item.get("edit", {}).get("source_snapshot") != snapshot for item in candidates):
        raise ValueError("Golden RTL changed since mutant generation; use the original source files")
    base = output / evaluation_name
    base.mkdir(exist_ok=False)
    golden_dir = base / "golden"
    stage_sources(config, golden_dir)
    checker = _prepare_evaluation(config, golden_dir, assertion_set)
    golden_run = run_check(config, golden_dir, checker)
    golden = {label: named_status(golden_run["properties"], label)
              for label in assertion_set.labels}
    if golden_run["status"] != "completed":
        write_json(base / "results.json", {"golden_formal": golden_run, "golden_assertions": golden})
        raise RuntimeError(f"Golden assertion run: {golden_run['status']}; see {base / 'golden' / 'jasper.log'}")
    valid = {label for label, status in golden.items() if status == "proven"}
    if not valid:
        write_json(base / "results.json", {"golden_formal": golden_run, "golden_assertions": golden})
        raise RuntimeError(f"No supplied assertion proved on golden RTL; see {base / 'results.json'}")
    results: list[dict] = []
    for item in candidates:
        identifier = item["id"]
        directory = base / identifier
        directory.mkdir()
        shutil.copytree(output / "mutants" / identifier / "rtl", directory / "rtl")
        checker = _prepare_evaluation(config, directory, assertion_set)
        run = run_check(config, directory, checker)
        statuses = {label: named_status(run["properties"], label)
                    for label in assertion_set.labels}
        labels = {label: ("invalid_on_golden" if label not in valid else
                          "inconclusive" if run["status"] != "completed" else
                          "killed" if statuses[label] == "cex" else
                          "survived" if statuses[label] == "proven" else "inconclusive")
                  for label in assertion_set.labels}
        valid_labels = [labels[label] for label in valid]
        verdict = ("killed" if "killed" in valid_labels else
                   "inconclusive" if "inconclusive" in valid_labels else "survived")
        results.append({"mutant": identifier,
                        "category": item.get("category", "simple"),
                        "scenario": item.get("scenario"),
                        "verdict": verdict,
                        "formal_status": run["status"], "assertion_statuses": statuses,
                        "labels": labels})
        print(f"{identifier}: {verdict} ({valid_labels.count('killed')} assertions)")
    kill_counts = {label: sum(item["labels"][label] == "killed" for item in results)
                   for label in assertion_set.labels if label in valid}
    summary = {
        "candidate_mutants": len(candidates),
        "valid_assertions_on_golden": len(valid),
        "assertions_killing_any_mutant": sum(count > 0 for count in kill_counts.values()),
        "killed_mutants": sum(item["verdict"] == "killed" for item in results),
        "surviving_mutants": sum(item["verdict"] == "survived" for item in results),
        "inconclusive_mutants": sum(item["verdict"] == "inconclusive" for item in results),
        "spec_confirmed_bugs": False,
    }
    by_category: dict[str, dict[str, int]] = {}
    for item in results:
        group = by_category.setdefault(item["category"],
                                       {"candidate_mutants": 0, "killed": 0,
                                        "survived": 0, "inconclusive": 0})
        group["candidate_mutants"] += 1
        group[item["verdict"]] += 1
    summary["by_category"] = by_category
    report = {"summary": summary, "assertion_source": str(assertions.resolve()),
              "assertion_sha256": digest(assertions.read_bytes()),
              "assertions": assertion_set.metadata, "golden_assertions": golden,
              "assertion_kill_counts": kill_counts, "mutants": results}
    write_json(base / "results.json", report)
    with (base / "kill_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mutant", "category", "verdict", *assertion_set.labels])
        for item in results:
            writer.writerow([item["mutant"], item["category"], item["verdict"],
                             *(item["labels"][label] for label in assertion_set.labels)])
    return report


def run_campaign(args: Namespace) -> None:
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Use a fresh --out directory: {output}")
    if not args.generate_only and not args.assertions:
        raise ValueError("Provide --assertions or use --generate-only")
    raw = direct_config(args)
    with tempfile.TemporaryDirectory(prefix="sangam_mutlab_config_") as temporary:
        config_path = Path(temporary) / "config.json"
        write_json(config_path, raw)
        run_generation(Namespace(config=config_path, out=output, count=args.count,
                                 mix=getattr(args, "mix", None),
                                 scenario_mode=getattr(args, "scenario_mode", "two-call"),
                                 no_scenario_validation=getattr(args, "no_scenario_validation", False),
                                 fill_quotas=getattr(args, "fill_quotas", False),
                                 max_attempts=getattr(args, "max_attempts", None),
                                 proposals=args.proposals, generate_only=True))
    saved = output / "config.direct.json"
    write_json(saved, raw)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["config"] = str(saved)
    write_json(summary_path, summary)
    if args.generate_only:
        return
    config = Config.load(saved)
    report = evaluate_candidates(config, output, args.assertions, args.max_assertions)
    print(f"Kill matrix: {output / 'evaluation' / 'kill_matrix.csv'}")
    print(f"Killed {report['summary']['killed_mutants']} of {report['summary']['candidate_mutants']} "
          "candidate mutants")


def retry_scoring(args: Namespace) -> None:
    output = args.out.resolve()
    config = Config.load(output / "config.direct.json")
    if args.jg_timeout:
        from dataclasses import replace
        config = replace(config, jasper={**config.jasper, "timeout_seconds": args.jg_timeout})
    index = 1
    name = "evaluation"
    while (output / name).exists():
        index += 1
        name = f"evaluation_{index}"
    report = evaluate_candidates(config, output, args.assertions, args.max_assertions, name)
    print(f"Kill matrix: {output / name / 'kill_matrix.csv'}")
    print(f"Killed {report['summary']['killed_mutants']} of {report['summary']['candidate_mutants']} "
          "candidate mutants")
