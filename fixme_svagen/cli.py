"""CLI for building <spec, RTL, final SANGAM assertion, vacuity, reason> rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .generate import GENERATION_CONFIG, MODEL, NebiusClient, SangamGenerator, map_signals
from .jasper import JGResult, JasperRunner
from .sva import Assertion, clock_reset, module_name

FORMAL_RESET_POLICY = "unconstrained"


@dataclass(frozen=True)
class Case:
    case_id: str
    level: str
    spec_path: Path
    rtl_path: Path


def discover_cases(fixme_root: Path) -> list[Case]:
    base = fixme_root / "data" / "SVAGen"
    cases = []
    for level in ("L1", "L2", "L3"):
        for folder in sorted((base / level / "Text").iterdir()):
            if not folder.is_dir():
                continue
            specs, rtls = list(folder.glob("*.txt")), list(folder.glob("*.v"))
            if len(specs) != 1 or len(rtls) != 1:
                raise ValueError(f"Expected one spec and RTL in {folder}")
            cases.append(Case(f"{level}/{folder.name}", level, specs[0], rtls[0]))
    if len(cases) != 40:
        raise ValueError(f"Expected 40 FIXME SVAGen Text cases; found {len(cases)}")
    return cases


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def final_fingerprint(case: Case, benchmark_sha: str, model: str, iterations: int,
                      timeout: int, rtl: str, top: str, clock: str | None,
                      reset: str | None, assertion: Assertion) -> str:
    fields = (case.case_id, benchmark_sha, model, GENERATION_CONFIG, iterations,
              timeout, rtl, top, clock, reset, assertion.text, assertion.support,
              assertion.implication)
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


def load_final_result(path: Path, log_path: Path, fingerprint: str) -> JGResult | None:
    if not path.exists() or not log_path.exists():
        return None
    saved = json.loads(path.read_text())
    log = log_path.read_text(errors="replace")
    if saved.get("fingerprint") != fingerprint or saved.get("log_sha256") != hashlib.sha256(log.encode()).hexdigest():
        return None
    return JGResult(saved["assertion_status"], saved["precondition_status"],
                    saved["contrapositive_status"], log, saved["exit_code"])


def save_final_result(path: Path, log_path: Path, fingerprint: str, result: JGResult) -> None:
    log_path.write_text(result.log)
    write_json(path, {
        "fingerprint": fingerprint,
        "log_sha256": hashlib.sha256(result.log.encode()).hexdigest(),
        "assertion_status": result.assertion_status,
        "precondition_status": result.precondition_status,
        "contrapositive_status": result.contrapositive_status,
        "exit_code": result.exit_code,
    })


def process_case(case: Case, output: Path, llm: NebiusClient, jasper: JasperRunner,
                 benchmark_sha: str, iterations: int, max_signals: int,
                 selected_signals: set[str] | None = None,
                 reuse_models: set[str] | None = None) -> dict:
    accepted_models = {llm.model} | (reuse_models or set())
    workdir = output / "cases" / case.case_id
    completed = workdir / "case.json"
    if completed.exists():
        saved = json.loads(completed.read_text())
        if (saved.get("status") in ("done", "no_final_assertions") and saved.get("benchmark_sha") == benchmark_sha
                and saved.get("model") in accepted_models
                and set(saved.get("generation_models", [saved.get("model")])) <= accepted_models
                and saved.get("formal_timeout") == jasper.timeout
                and saved.get("refinement_iterations") == iterations
                and saved.get("formal_reset_policy") == FORMAL_RESET_POLICY
                and saved.get("generation_config") == GENERATION_CONFIG):
            return saved
    workdir.mkdir(parents=True, exist_ok=True)
    spec, rtl = case.spec_path.read_text(errors="replace"), case.rtl_path.read_text(errors="replace")
    top = module_name(rtl, case.rtl_path.stem)
    clock, reset = clock_reset(rtl)
    map_file = workdir / "signals.json"
    if map_file.exists():
        signals = json.loads(map_file.read_text())
    else:
        signals = map_signals(llm, spec, rtl)
        write_json(map_file, signals)
    selected = [(i, signal) for i, signal in enumerate(signals)
                if selected_signals is None or signal["name"] in selected_signals]
    if max_signals:
        selected = selected[:max_signals]
    if selected_signals and not selected:
        raise ValueError(f"Requested signal(s) not mapped for {case.case_id}: {sorted(selected_signals)}")
    final: dict[str, tuple[Assertion, list[str]]] = {}
    assertion_models: dict[str, list[str]] = {}
    for index, signal in selected:
        generation_dir = f"iterations_{iterations}_{GENERATION_CONFIG}"
        if clock is None:
            generation_dir += "_clockless_v1"
        signal_dir = workdir / "signals" / f"{index:03d}_{signal['name']}" / generation_dir
        # Legacy completed signals have no metadata: they were generated by MODEL.
        # Isolate new-model work so an interrupted legacy search is preserved.
        legacy_dir = signal_dir
        legacy_model = MODEL if MODEL in (reuse_models or set()) else llm.model
        if reuse_models and llm.model != MODEL:
            signal_dir = legacy_dir.with_name(generation_dir + "_model_"
                + hashlib.sha256(llm.model.encode()).hexdigest()[:12])
            if (legacy_dir / "final.json").exists() and legacy_model in accepted_models:
                signal_dir = legacy_dir
        signal_file = signal_dir / "final.json"
        metadata_file = signal_dir / "generation_metadata.json"
        if signal_file.exists():
            signal_model = (json.loads(metadata_file.read_text())["model"]
                            if metadata_file.exists() else legacy_model)
            if signal_model not in accepted_models:
                raise ValueError("Signal checkpoint model is not authorized: " + signal_model)
            entries = json.loads(signal_file.read_text())
        else:
            signal_model = llm.model
            generator = SangamGenerator(llm, jasper, rtl, top, clock, reset, signal_dir, iterations)
            assertions, trace = generator.run_signal(signal, spec)
            write_json(signal_dir / "trace.json", trace)
            entries = [a.__dict__ for a in assertions]
            write_json(metadata_file, {"model": signal_model, "generation_config": GENERATION_CONFIG})
            write_json(signal_file, entries)
        for entry in entries:
            assertion = Assertion(**entry)
            models = assertion_models.setdefault(assertion.text, [])
            if signal_model not in models:
                models.append(signal_model)
            if assertion.text in final:
                final[assertion.text][1].append(signal["name"])
            else:
                final[assertion.text] = (assertion, [signal["name"]])
    rows = []
    logs = workdir / "final_logs_reset_none"
    logs.mkdir(exist_ok=True)
    checkpoints = workdir / "final_results_reset_none"
    checkpoints.mkdir(exist_ok=True)
    for index, (assertion, names) in enumerate(final.values(), start=1):
        digest = hashlib.sha256((assertion.support + assertion.text).encode()).hexdigest()[:20]
        log_path = logs / f"{digest}.log"
        result_path = checkpoints / f"{digest}.json"
        source_models = assertion_models[assertion.text]
        fingerprint = final_fingerprint(case, benchmark_sha, source_models[0], iterations,
                                        jasper.timeout, rtl, top, clock, None, assertion)
        result = load_final_result(result_path, log_path, fingerprint)
        if result is None:
            result = jasper.check(rtl, top, assertion, clock, None)
            if result.assertion_status == "license_unavailable":
                raise RuntimeError("JasperGold license unavailable during final checks")
            save_final_result(result_path, log_path, fingerprint, result)
            print(f"  final checks {index}/{len(final)}: {result.assertion_status}", flush=True)
        vacuous, reason = result.label(assertion.implication)
        rows.append({
            "spec": spec,
            "rtl": rtl,
            "assertion": assertion.text,
            "vacuous": vacuous,
            "reason": reason,
            "case_id": case.case_id,
            "level": case.level,
            "signals": names,
            "support_code": assertion.support,
            "benchmark_revision": benchmark_sha,
            "model": source_models[0],
            "generation_models": source_models,
            "model_provider": "Nebius",
            "model_thinking_request": "chat_template_kwargs.thinking=false",
            "generation_config": GENERATION_CONFIG,
            "spec_sha256": hashlib.sha256(spec.encode()).hexdigest(),
            "rtl_sha256": hashlib.sha256(rtl.encode()).hexdigest(),
            "top": top,
            "clock": clock,
            "reset": reset,
            "formal_reset_policy": FORMAL_RESET_POLICY,
            "formal_timeout": jasper.timeout,
            "refinement_iterations": iterations,
            "assertion_status": result.assertion_status,
            "precondition_status": result.precondition_status,
            "contrapositive_status": result.contrapositive_status,
            "jasper_log": str(log_path.relative_to(output)),
        })
    status = "partial" if len(selected) < len(signals) else ("done" if final else "no_final_assertions")
    failure_reason = None
    if status == "no_final_assertions":
        candidate_logs = workdir.glob("signals/**/candidate_logs/*.log")
        if any("multiple event control statements is not supported for synthesis"
               in path.read_text(errors="replace") for path in candidate_logs):
            failure_reason = "JasperGold cannot elaborate benchmark RTL with multiple event controls"
        else:
            failure_reason = "No candidate assertion passed JasperGold syntax and elaboration checks"
    record = {
        "status": status,
        "case_id": case.case_id,
        "benchmark_sha": benchmark_sha,
        "model": llm.model,
        "generation_models": sorted({m for models in assertion_models.values() for m in models}),
        "formal_timeout": jasper.timeout,
        "formal_reset_policy": FORMAL_RESET_POLICY,
        "refinement_iterations": iterations,
        "generation_config": GENERATION_CONFIG,
        "signal_count": len(signals),
        "processed_signals": len(selected),
        "rows": rows,
    }
    if failure_reason:
        record["failure_reason"] = failure_reason
    write_json(completed, record)
    return record


def rebuild_jsonl(output: Path, benchmark_sha: str, model: str, iterations: int,
                  reuse_models: set[str] | None = None) -> int:
    accepted_models = {model} | (reuse_models or set())
    rows = []
    for path in sorted((output / "cases").glob("L*/**/case.json")):
        record = json.loads(path.read_text())
        if (record.get("status") == "done" and record.get("benchmark_sha") == benchmark_sha
                and record.get("model") in accepted_models
                and set(record.get("generation_models", [record.get("model")])) <= accepted_models
                and record.get("refinement_iterations") == iterations
                and record.get("formal_reset_policy") == FORMAL_RESET_POLICY
                and record.get("generation_config") == GENERATION_CONFIG):
            rows.extend(record["rows"])
    output.mkdir(parents=True, exist_ok=True)
    temp = output / "dataset.jsonl.tmp"
    with temp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(output / "dataset.jsonl")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixme-root", required=True, type=Path, help="FIXME checkout at a pinned commit")
    parser.add_argument("--output", type=Path, default=Path("output/fixme_svagen"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reuse-model", action="append", default=[],
                        help="Preserve completed cases/signals from this earlier model; repeatable")
    parser.add_argument("--case", action="append", help="Case ID such as L1/PIC_8259; repeatable")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-signals", type=int, default=0)
    parser.add_argument("--signal", action="append", help="Generate a named signal only; repeatable")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--jg-timeout", type=int, default=120, help="Hard timeout in seconds per JasperGold assertion")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    root = args.fixme_root.resolve()
    cases = discover_cases(root)
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case.case_id in wanted]
        if len(cases) != len(wanted):
            raise SystemExit("Unknown case ID in --case")
    if args.max_cases:
        cases = cases[:args.max_cases]
    if args.list_cases:
        for case in cases:
            print(case.case_id, case.spec_path.name, case.rtl_path.name)
        return
    benchmark_sha = revision(root)
    output = args.output.resolve()
    llm = NebiusClient(args.env_file, args.model)
    with JasperRunner(args.env_file, timeout=args.jg_timeout) as jasper:
        print("Checking JasperGold license and related-cover support...", flush=True)
        try:
            jasper.preflight()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        for case in cases:
            print("Processing", case.case_id, flush=True)
            record = process_case(case, output, llm, jasper, benchmark_sha, args.iterations,
                                  args.max_signals, set(args.signal) if args.signal else None,
                                  set(args.reuse_model))
            print("  status", record["status"], "rows", len(record["rows"]), flush=True)
            rebuild_jsonl(output, benchmark_sha, args.model, args.iterations, set(args.reuse_model))
    count = rebuild_jsonl(output, benchmark_sha, args.model, args.iterations, set(args.reuse_model))
    print("Wrote", count, "rows to", output / "dataset.jsonl")


if __name__ == "__main__":
    main()
