#!/usr/bin/env python3
"""Run prepared SANGAM reduction cases sequentially on the JasperGold host.

The plan is a JSON object with ``remote_base`` and a ``designs`` list. Each
design supplies ``case_id``, ``case``, and ``out``. The existing pipeline owns
all proof checkpoints; rerunning this driver resumes unfinished designs.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


LICENSE_FAILURE = re.compile(
    r"license checkout failed|waiting for license|FLEXnet ERROR|"
    r"cannot connect to license server|license_unavailable|"
    r"ERROR \(ELC004\)|ERROR \(LMC-",
    re.IGNORECASE,
)
NON_LICENSE_FAILURE = re.compile(r"ERROR \((?:ECK|EPF)\d+\)", re.IGNORECASE)


def is_license_failure(evidence):
    # Jasper can log unsuccessful feature probes before successfully checking
    # out another license. A subsequent clock/proof error is not a license retry.
    return bool(LICENSE_FAILURE.search(evidence) and not NON_LICENSE_FAILURE.search(evidence))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tail(path, limit=32768):
    path = Path(path)
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - limit))
        return handle.read().decode("utf8", errors="replace")


def selected_count(item):
    path = Path(item["out"]) / "extracted" / "selected_assertions.jsonl"
    if not path.exists():
        return 10 ** 9
    with path.open(encoding="utf8") as handle:
        return sum(bool(line.strip()) for line in handle)


def validated(item):
    out = Path(item["out"])
    summary = out / "summary.json"
    metadata = out / "input_metadata.json"
    snapshot = out / "input_case.json"
    if not all(path.exists() for path in (summary, metadata, snapshot)):
        return False
    fingerprint = sha256(item["case"])
    return (read_json(summary).get("status") == "VALIDATED"
            and read_json(metadata).get("sha256") == fingerprint
            and sha256(snapshot) == fingerprint)


def failure_evidence(item):
    out = Path(item["out"])
    batch = tail(out / "batch.log")
    if "=== attempt " in batch:
        batch = batch.rsplit("=== attempt ", 1)[-1]
    paths = [out / stage / filename for stage in ('consistency', 'validation')
             for filename in ('jg.log', 'jgproject/jg_console.log')]
    pruning = out / "pruning" / "jobs"
    if pruning.exists():
        jobs = list(pruning.glob("*/jg.log"))
        if jobs:
            job = max(jobs, key=lambda path: path.stat().st_mtime).parent
            paths.extend((job / "jg.log", job / "runner_error.txt",
                          job / "jgproject" / "jg_console.log"))
    return batch + "\n" + "\n".join(tail(path) for path in paths)


def clear_failed_license_project(item):
    """Archive stale failed projects, including consistency/validation stages.

    Keep successful stages and projects with a live Jasper console owner.
    The console log retains license evidence even when a later launch reports
    only 'Cannot obtain ownership of project directory'.
    """
    root = Path(item['out'])
    stages = [root / 'consistency', root / 'validation']
    jobs = list((root / 'pruning/jobs').glob('*/jg.log'))
    if jobs:
        stages.append(max(jobs, key=lambda path: path.stat().st_mtime).parent)
    for stage in stages:
        project = stage / 'jgproject'
        evidence = tail(project / 'jg_console.log') + tail(stage / 'jg.log')
        if ((stage / 'verified.json').exists() or not project.is_dir()
                or project.is_symlink() or not is_license_failure(evidence)):
            continue
        active = False
        for lock in project.glob('*.lock'):
            match = re.search(r'\.(\d+)\.lock$', lock.name)
            if match:
                proc = Path('/proc') / match.group(1)
                try:
                    active |= ('jg_console' in (proc / 'comm').read_text()
                               and (proc / 'cwd').resolve() == stage.resolve())
                except FileNotFoundError:
                    pass
                except PermissionError:
                    active = True
        if active:
            print('Preserving project with live owner: ' + str(project), flush=True)
            continue
        archived = project.with_name('jgproject_license_failed_' + str(time.time_ns()
                            if hasattr(time, 'time_ns') else int(time.time() * 1000000)))
        project.rename(archived)
        print('Archived stale license project: ' + str(archived), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--jg", default="jg")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retry-delay", type=int, default=180)
    parser.add_argument("--license-retries", type=int, default=240)
    args = parser.parse_args()

    plan = read_json(args.plan)
    base = Path(plan["remote_base"])
    results_root = base / "results"
    status_path = results_root / plan.get("status_name", "reduction_25_status.json")
    items = sorted(plan["designs"], key=lambda item: (selected_count(item), item["case_id"]))
    status = {"state": "running", "pid": os.getpid(),
              "selection": plan.get("selection"),
              "order": [item["case_id"] for item in items], "cases": {}}
    write_json(status_path, status)

    for index, item in enumerate(items, 1):
        case_id = item["case_id"]
        out = Path(item["out"])
        out.mkdir(parents=True, exist_ok=True)
        record = {"out": str(out), "selected": selected_count(item), "attempts": 0}
        status["cases"][case_id] = record
        status["current"] = case_id
        if validated(item):
            record["status"] = "validated"
            record["summary"] = read_json(out / "summary.json")
            write_json(status_path, status)
            print("{}/{} {}: already VALIDATED".format(index, len(items), case_id), flush=True)
            continue

        while True:
            record["attempts"] += 1
            record["status"] = "running"
            write_json(status_path, status)
            command = [sys.executable, str(base / "scripts" / "run_pipeline.py"),
                       "--case", item["case"], "--out", str(out),
                       "--jg", args.jg, "--timeout", str(args.timeout)]
            print("{}/{} {}: attempt {} ({} selected)".format(
                index, len(items), case_id, record["attempts"], record["selected"]), flush=True)
            try:
                with (out / "batch.log").open("a", encoding="utf8") as log:
                    log.write("\n=== attempt {} ===\n".format(record["attempts"]))
                    log.flush()
                    code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
            except OSError as error:
                code = -1
                record["last_error"] = str(error)
            if code == 0 and validated(item):
                record["status"] = "validated"
                record["summary"] = read_json(out / "summary.json")
                write_json(status_path, status)
                print("{}/{} {}: VALIDATED".format(index, len(items), case_id), flush=True)
                break

            evidence = failure_evidence(item)
            record["last_error"] = evidence[-1500:]
            license_failure = is_license_failure(evidence)
            if license_failure:
                clear_failed_license_project(item)
            retry_limit = args.license_retries if license_failure else 2
            if record["attempts"] >= retry_limit:
                record["status"] = "failed"
                record["exit_code"] = code
                write_json(status_path, status)
                print("{}/{} {}: FAILED after {} attempts; inspect {}".format(
                    index, len(items), case_id, record["attempts"], out / "batch.log"), flush=True)
                break
            record["status"] = "retrying"
            record["exit_code"] = code
            write_json(status_path, status)
            print("{}/{} {}: retrying after {} seconds".format(
                index, len(items), case_id, args.retry_delay), flush=True)
            time.sleep(args.retry_delay)

    status.pop("current", None)
    failures = [case_id for case_id, record in status["cases"].items()
                if record["status"] != "validated"]
    status["state"] = "finished_with_failures" if failures else "complete"
    status["failed_cases"] = failures
    write_json(status_path, status)
    print("Finished: {} validated, {} failed".format(len(items) - len(failures), len(failures)), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
