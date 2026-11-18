"""Detached, resumable JasperGold worker; compatible with remote Python 3.6."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def statuses(log, tsv=None):
    states = dict(re.findall(r'(?m)^\[\d+\]\s+(\S+)\s+'
        r'(proven|cex|covered|unreachable|undetermined|unknown|error|bounded_proven|bounded_unreachable)\b', log))
    if tsv and tsv.exists():
        for line in tsv.read_text().splitlines():
            name, value = line.split('\t', 1)
            states[name] = value.strip().strip('{}').lower()
    return states


def assess(states, implication):
    # Summary reports use top.property while get_property_list can return
    # <embedded>::top.property. These are aliases of the same physical goal.
    canonical = {}
    for name, value in states.items():
        name = name.removeprefix('<embedded>::') if hasattr(name, 'removeprefix') else (
            name[len('<embedded>::'):] if name.startswith('<embedded>::') else name)
        if name in canonical and canonical[name] != value:
            canonical[name] = 'conflicting_status'
        else:
            canonical[name] = value
    states = canonical
    main = [s for n, s in states.items() if n == 'eval_assert' or n.endswith('.eval_assert')]
    main = main[0] if len(main) == 1 else 'missing'
    covers = {family: {n: s for n, s in states.items()
        if re.search(r'(?:^|\.)eval_assert:' + family + r'\d+$', n)}
        for family in ('precondition', 'contrapositive')}
    trace = [s for n, s in states.items() if n == 'eval_trace' or n.endswith('.eval_trace')]
    result = {'assertion_status': main, 'related_covers': covers,
              'trace_status': trace[0] if len(trace) == 1 else 'missing',
              'vacuous': None, 'reason': 'formal_inconclusive_or_error'}
    if implication is False:
        result['reason'] = 'outside_scope_non_implication'
    elif main == 'cex':
        result['reason'] = 'assertion_falsified'
    elif implication is None:
        result['reason'] = 'implication_scope_unresolved'
    elif main == 'proven' and result['trace_status'] == 'covered':
        reasons = []
        for family, reason in (('precondition', 'antecedent_never_true'),
                               ('contrapositive', 'consequent_always_true')):
            if 'unreachable' in covers[family].values():
                reasons.append(reason)
        if reasons:
            result.update(vacuous=True, reason='; '.join(reasons))
        elif all(family and all(s == 'covered' for s in family.values()) for family in covers.values()):
            result.update(vacuous=False, reason='neither_targeted_vacuity_reason')
        else:
            result['reason'] = 'related_cover_inconclusive_or_unsupported'
    elif main == 'proven':
        result['reason'] = 'formal_context_trace_not_confirmed'
    return result


def license_unavailable(log, timed_out):
    # Jasper can reject one feature (e.g. jasper_fpv, -5), successfully
    # check out another (jasper_fao), and then report an ordinary syntax error.
    if re.search(r'\[ERROR \(VERI-|ERROR \((?:ENL|ECK)\d+', log):
        return False
    failures = list(re.finditer(r'FLEXnet ERROR\(-(15|16|4|5|96)\b|Cannot connect to license server|License checkout failed', log, re.I))
    successes = list(re.finditer(r'checkout succeeded', log, re.I))
    if failures and (not successes or failures[-1].start() > successes[-1].start()):
        return True
    return bool(timed_out and re.search(r'Waiting for license', log, re.I) and not successes)


def run_tool(folder, executable, timeout):
    tsv = folder / 'status.tsv'
    if tsv.exists():
        tsv.unlink()
    started = time.monotonic()
    timed_out = False
    with (folder / 'jg.log').open('w') as log:
        process = subprocess.Popen([executable, '-batch', 'run.tcl'], cwd=str(folder),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except ProcessLookupError:
                process.wait()
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    text = (folder / 'jg.log').read_text(errors='replace')
    console = folder / 'jgproject/jg_console.log'
    full_log = text + ('\n' + console.read_text(errors='replace') if console.exists() else '')
    return text, full_log, process.returncode, timed_out, time.monotonic() - started


def run(root, executable):
    lock = (root / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes.decode())
    if sha(Path(__file__).read_bytes()) != manifest['worker_sha256']:
        raise ValueError('Remote worker differs from the frozen evaluation')
    jobs = [json.loads(line) for line in (root / 'jobs.jsonl').open() if line.strip()]
    if sha((root / 'jobs.jsonl').read_bytes()) != manifest['jobs_sha256']:
        raise ValueError('Remote job list differs from the frozen evaluation')
    completed = {}
    state = {'state': 'running', 'pid': os.getpid(), 'jobs_total': len(jobs),
             'manifest_sha256': sha(manifest_bytes)}

    def update(**fields):
        state.update(fields)
        state.update(jobs_complete=len(completed), jobs_remaining=len(jobs) - len(completed),
                     outcomes=dict(__import__('collections').Counter(r['assertion_status'] for r in completed.values())),
                     updated_utc=datetime.datetime.utcnow().isoformat() + 'Z')
        write_json(root / 'status.json', state)

    for job in jobs:
        path = root / 'jobs' / job['job_id'] / 'result.json'
        if path.exists():
            result = json.loads(path.read_text())
            if result['job_fingerprint'] != job['fingerprint']:
                raise ValueError('Saved formal result has a different fingerprint')
            completed[job['job_id']] = result
    update()
    try:
        for job in jobs:
            if job['job_id'] in completed:
                continue
            folder = root / 'jobs' / job['job_id']
            for name in ('dut.sv', 'run.tcl'):
                if sha((folder / name).read_bytes()) != job['files'][name]:
                    raise ValueError('Remote formal input changed: ' + str(folder / name))
            while True:
                update(state='running', current_job=job['job_id'], current_origin=job['first_origin'])
                log, full_log, code, timed_out, elapsed = run_tool(folder, executable, manifest['timeout_seconds'])
                states = statuses(log, folder / 'status.tsv')
                result = assess(states, job['implication'])
                # A completed named proof is stronger evidence than an earlier
                # failed license-feature attempt in the console.
                if result['assertion_status'] == 'missing' and license_unavailable(full_log, timed_out):
                    attempt = folder / ('infrastructure_' + str(int(time.time())) + '.log')
                    attempt.write_text(full_log)
                    update(state='waiting_for_license', infrastructure_log=str(attempt))
                    print('License unavailable; saved work retained. Retrying in 60 seconds.', flush=True)
                    time.sleep(60)
                    continue
                if result['assertion_status'] == 'missing':
                    if timed_out:
                        result['assertion_status'] = 'timeout'
                    elif re.search(r'ECK\d+', full_log):
                        result['assertion_status'] = 'clock_context_error'
                    elif re.search(r'ERROR.*(?:syntax|parse)|VERI-1137|VERI-1209', full_log, re.I):
                        result['assertion_status'] = 'syntax_error'
                    else:
                        result['assertion_status'] = 'analysis_or_elaboration_error' if code else 'tool_error'
                result.update(job_id=job['job_id'], job_fingerprint=job['fingerprint'], exit_code=code,
                    timed_out=timed_out, elapsed_seconds=elapsed, states=states,
                    jg_log_sha256=sha((folder / 'jg.log').read_bytes()))
                write_json(folder / 'result.json', result)
                completed[job['job_id']] = result
                update()
                print('FORMAL', len(completed), '/', len(jobs), job['first_origin'],
                      result['assertion_status'], result['reason'], flush=True)
                break
        state.pop('current_job', None)
        state.pop('current_origin', None)
        update(state='complete')
    except BaseException as error:
        update(state='failed_or_interrupted', error=type(error).__name__ + ': ' + str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--jg', default='/home/Cadence_tools/jasper_2022.12/bin/jg')
    args = parser.parse_args()
    run(args.root, args.jg)


if __name__ == '__main__':
    main()
