"""Recheck the SFT candidates under L3/3's audited clock context.

Original case files are read only. A single combined harness checks assertions
and all generated related covers without assuming any assertion as a premise.
"""

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex

from experiments.assertion_extraction.extract_assertions import resolve_property
from experiments.assertion_extraction.formal_clocking import clock_commands
from experiments.assertion_extraction.remove_exact_duplicates import tokenize
from experiments.fixme_cohort import connect, command, put
from experiments.training.prepare_sft import category, sha256
from fixme_svagen.sva import insert_in_module

ROOT = Path("output/fixme_svagen")
AUDIT = ROOT / "audits/L3_3_sft_vacuity_v2"
REMOTE = "/storage/ckarfa/narmada/experiments/results/L3/3/sft_vacuity_v2"


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def assess(states, label, implication):
    """Require every generated related cover in each family to be covered."""
    main = states.get(label, "missing")
    covers = {
        family: {name: status for name, status in states.items()
                 if re.fullmatch(re.escape(label) + ":" + family + r"\d+", name)}
        for family in ("precondition", "contrapositive")
    }
    if not implication:
        vacuous, reason = None, "outside_scope_non_implication"
    elif main == "cex":
        vacuous, reason = False, "assertion_falsified"
    elif main != "proven":
        vacuous, reason = None, "formal_inconclusive_or_error"
    elif any(status == "unreachable" for family in covers.values() for status in family.values()):
        vacuous, reason = True, "related_cover_unreachable_in_audited_context"
    elif all(family and all(status == "covered" for status in family.values())
             for family in covers.values()):
        vacuous, reason = False, "neither_targeted_vacuity_reason"
    else:
        vacuous, reason = None, "related_cover_inconclusive_or_unsupported"
    return {"assertion_status": main, "vacuous": vacuous, "reason": reason,
            "related_covers": covers}


def read_states(path):
    result = {}
    for line in path.read_text().splitlines():
        name, status = line.split("\t")
        name = name.rsplit(".", 1)[-1]
        if name in result:
            raise ValueError("Duplicate property status: " + name)
        result[name] = status.strip().strip("{}").lower()
    return result


def prepare():
    if AUDIT.exists():
        raise ValueError("Audit already exists; inspect it instead of replacing evidence")
    source = ROOT / "cases/L3/3/case.json"
    raw = source.read_bytes()
    case = json.loads(raw)
    if case.get("formal_clock_recheck", {}).get("status") != "confirmed":
        raise ValueError("Clock audit must be confirmed first")
    context = case["formal_clocking"]
    groups = {}
    for number, row in enumerate(case["rows"], 1):
        if category(row) is None:
            continue
        _, expression, _, status, error = resolve_property(row)
        if status != "ok":
            raise ValueError(error)
        key = tokenize(expression)
        if key not in groups:
            groups[key] = {"label": "SFT_CHECK_%04d" % number, "expression": expression,
                           "expression_sha256": sha256(json.dumps(key)),
                           "implication": category(row) == "proven_non_vacuous_implication",
                           "source_rows": []}
        groups[key]["source_rows"].append(number)
    items = list(groups.values())
    snippet = "\n\n".join(item["label"] + ": assert property (\n" + item["expression"] + "\n);"
                             for item in items)
    snippet += "\nSFT_TRACE_EXISTS: cover property (@(posedge forever_cpuclk) 1'b1);\n"
    first = case["rows"][0]
    rtl = insert_in_module(first["rtl"], first["top"], snippet)
    tcl = "\n".join([
        "analyze -sv12 dut.sv",
        "elaborate -top " + first["top"] + " -create_related_covers {precondition contrapositive}",
        clock_commands([], context), "reset -none", "prove -all",
        "report -summary -force -result", 'set fh [open "status.tsv" w]',
        "foreach p [lsort [get_property_list]] {",
        '  if {[catch {get_property_info -list status $p} state]} {set state QUERY_ERROR}',
        '  puts $fh "$p\\t$state"', "}", "close $fh", "exit -force", "",
    ])
    AUDIT.mkdir(parents=True)
    (AUDIT / "input_case.json").write_bytes(raw)
    (AUDIT / "dut.sv").write_text(rtl)
    (AUDIT / "run.tcl").write_text(tcl)
    request = {"case_id": "L3/3", "created_utc": datetime.now(timezone.utc).isoformat(),
               "source_case": str(source), "source_case_sha256": sha256(raw),
               "formal_clocking": context, "remote": REMOTE, "properties": items,
               "dut_sha256": sha256(rtl), "tcl_sha256": sha256(tcl),
               "all_related_covers_required": True}
    write_json(AUDIT / "request.json", request)
    print(json.dumps({"unique_properties": len(items),
                      "categories": dict(Counter("implication" if p["implication"] else "non_implication"
                                                 for p in items)), "source_rows": sum(len(p["source_rows"]) for p in items)}))


def launch():
    request = json.loads((AUDIT / "request.json").read_text())
    if sha256((AUDIT / "input_case.json").read_bytes()) != request["source_case_sha256"]:
        raise ValueError("Audit input changed")
    driver = '''import fcntl,json,sys,traceback
from pathlib import Path
sys.path.insert(0, '/storage/ckarfa/narmada/experiments/scripts')
from run_pipeline import run_jg
p=Path(REMOTE_REPLACE)
lock=(p/'audit.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (p/'complete.json').exists():
 print('Audit already complete',flush=True)
 raise SystemExit(0)
try:
 run_jg(p,'/home/Cadence_tools/jasper_2022.12/bin/jg',1800)
 result={'status':'complete'}
except Exception as exc:
 traceback.print_exc()
 result={'status':'failed','error':type(exc).__name__,'message':str(exc)}
(p/'complete.json').write_text(json.dumps(result))
print(json.dumps(result),flush=True)
'''.replace("REMOTE_REPLACE", repr(REMOTE))
    (AUDIT / "driver.py").write_text(driver)
    ssh, sftp = connect()
    try:
        command(ssh, ["mkdir", "-p", REMOTE])
        for name in ("dut.sv", "run.tcl", "request.json", "driver.py"):
            put(ssh, sftp, AUDIT / name, REMOTE + "/" + name, True)
        shell = "source ~/cad_cshrc; exec python3 -u " + shlex.quote(REMOTE + "/driver.py")
        code = ("import subprocess; from pathlib import Path; p=Path(" + repr(REMOTE) + "); "
                "log=(p/'driver.log').open('ab',buffering=0); "
                "child=subprocess.Popen(['csh','-c'," + repr(shell) + "],stdin=subprocess.DEVNULL,"
                "stdout=log,stderr=subprocess.STDOUT,start_new_session=True); print('Audit PID',child.pid)")
        print(command(ssh, ["python3", "-c", code]))
    finally:
        sftp.close()
        ssh.close()


def sync():
    ssh, sftp = connect()
    try:
        for name in ("complete.json", "status.tsv", "jg.log", "driver.log"):
            try:
                with sftp.open(REMOTE + "/" + name, "rb") as handle:
                    (AUDIT / name).write_bytes(handle.read())
            except IOError:
                pass
    finally:
        sftp.close()
        ssh.close()
    complete = AUDIT / "complete.json"
    if not complete.exists():
        print("Audit still running.")
        print((AUDIT / "driver.log").read_text(errors="replace")[-1000:])
        return
    done = json.loads(complete.read_text())
    if done["status"] != "complete":
        raise RuntimeError(done)
    request = json.loads((AUDIT / "request.json").read_text())
    states = read_states(AUDIT / "status.tsv")
    if states.get("SFT_TRACE_EXISTS") != "covered":
        raise ValueError("The audited environment has no covered trace")
    assessments = {}
    for item in request["properties"]:
        if item["label"] not in states:
            raise ValueError("Missing main assertion result: " + item["label"])
        assessments[item["label"]] = dict(item, **assess(states, item["label"], item["implication"]))
    counts = Counter(category(item) or "excluded" for item in assessments.values())
    result = {"status": "complete", "formal_clocking": request["formal_clocking"],
              "source_case_sha256": request["source_case_sha256"],
              "request_sha256": sha256((AUDIT / "request.json").read_bytes()),
              "status_tsv_sha256": sha256((AUDIT / "status.tsv").read_bytes()),
              "log_sha256": sha256((AUDIT / "jg.log").read_bytes()),
              "trace_status": "covered", "categories": dict(counts),
              "properties": assessments, "formal_reduction_applied": False}
    write_json(AUDIT / "assessment.json", result)
    print(json.dumps({"status": "complete", "categories": dict(counts),
                      "main_statuses": dict(Counter(p["assertion_status"] for p in assessments.values())),
                      "trace_status": "covered"}, indent=2))


def overlay():
    """Create a separate audited case for SFT; never edit generation sources."""
    output = AUDIT / "case.sft_rechecked.json"
    if output.exists():
        raise ValueError("Audited case already exists; preserve it")
    result = json.loads((AUDIT / "assessment.json").read_text())
    request = json.loads((AUDIT / "request.json").read_text())
    if result["status"] != "complete" or result["trace_status"] != "covered":
        raise ValueError("Audit must be complete with a covered trace")
    hashes = {
        "input_case.json": request["source_case_sha256"],
        "request.json": result["request_sha256"],
        "status.tsv": result["status_tsv_sha256"],
        "jg.log": result["log_sha256"],
    }
    for name, digest in hashes.items():
        if sha256((AUDIT / name).read_bytes()) != digest:
            raise ValueError("Audit file changed: " + name)
    case = json.loads((AUDIT / "input_case.json").read_text())
    digest = sha256((AUDIT / "assessment.json").read_bytes())
    for label, assessment in result["properties"].items():
        for number in assessment["source_rows"]:
            row = case["rows"][number - 1]
            _, expression, _, status, _ = resolve_property(row)
            if status != "ok" or sha256(json.dumps(tokenize(expression))) != assessment["expression_sha256"]:
                raise ValueError("Audited expression differs from source")
            previous = {name: row.get(name) for name in (
                "assertion_status", "vacuous", "reason", "jasper_log",
                "precondition_status", "contrapositive_status")}
            row.update({name: assessment[name] for name in ("assertion_status", "vacuous", "reason")})
            row["precondition_status"] = ("covered" if assessment["implication"]
                and category(assessment) == "proven_non_vacuous_implication" else None)
            row["contrapositive_status"] = row["precondition_status"]
            row["jasper_log"] = str((AUDIT / "jg.log").relative_to(ROOT))
            row["sft_formal_recheck"] = {"label": label, "assessment_sha256": digest,
                "original_result": previous, "related_covers": assessment["related_covers"]}
    case["formal_sft_audit"] = {
        "version": 1, "formal_clocking": copy.deepcopy(result["formal_clocking"]),
        "files": {name: {"path": str(AUDIT / name), "sha256": expected}
                  for name, expected in dict(hashes, **{"assessment.json": digest}).items()},
    }
    case["formal_clocking"] = result["formal_clocking"]
    write_json(output, case)
    print("Created separate audited SFT source:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "sync", "overlay"))
    args = parser.parse_args()
    {"prepare": prepare, "launch": launch, "sync": sync, "overlay": overlay}[args.action]()


if __name__ == "__main__":
    main()
