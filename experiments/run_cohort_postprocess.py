#!/usr/bin/env python3
"""Run the existing reduction and checker-coverage tools on a remote cohort.

Portable to the JasperGold host's Python 3.6. A file lock prevents duplicate
workers; the underlying tools own their proof checkpoints and retry policies.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--jg', default='/home/Cadence_tools/jasper_2022.12/bin/jg')
    ap.add_argument('--coverage-ready', action='store_true',
                    help='Run coverage for validated new designs while other reductions are blocked')
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    base = Path(plan['remote_base'])
    tag = plan.get('batch_name', str(len(plan['case_ids'])))
    lock = (base / ('postprocess_' + tag + '.lock')).open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another cohort worker is active.', flush=True)
        return 0
    status_path = base / ('postprocess_' + tag + '_status.json')
    state = {'state': 'running', 'pid': os.getpid(), 'case_ids': plan['case_ids'],
             'new_case_ids': plan['new_case_ids'], 'stages': {}}

    def run(name, command):
        state['current'] = name
        state['updated_utc'] = datetime.utcnow().isoformat() + 'Z'
        state['stages'][name] = 'running'
        write(status_path, state)
        print('START ' + name, flush=True)
        code = subprocess.call(command)
        if code:
            state['stages'][name] = 'failed'
            state['state'] = 'failed'
            state['exit_code'] = code
            write(status_path, state)
            raise RuntimeError('{} exited {}'.format(name, code))
        state['stages'][name] = 'complete'
        write(status_path, state)
        print('DONE ' + name, flush=True)

    deferred = []
    if args.coverage_ready:
        state['stages']['reduction'] = 'deferred'
    else:
        run('reduction', [sys.executable, '-u', str(base / 'results/run_batch.py'),
                         '--plan', args.plan, '--jg', args.jg, '--timeout', '600'])
    for cid in plan['new_case_ids']:
        if args.coverage_ready:
            item = next(item for item in plan['designs'] if item['case_id'] == cid)
            summary = Path(item['out']) / 'summary.json'
            if not summary.exists() or json.loads(summary.read_text()).get('status') != 'VALIDATED':
                deferred.append(cid)
                state['deferred_reduction_cases'] = deferred
                write(status_path, state)
                print('DEFER coverage pending validated reduction: ' + cid, flush=True)
                continue
        for selection in ('reduced', 'unreduced'):
            run(cid + '/' + selection,
                [sys.executable, '-u', str(base / 'checker_coverage/run_batch.py'),
                 '--plan', args.plan, '--jg', args.jg, '--timeout', '900',
                 '--batch-name', tag, '--case', cid, '--assertion-set', selection])
    state['state'] = 'failed' if deferred else 'complete'
    state.pop('current', None)
    state['updated_utc'] = datetime.utcnow().isoformat() + 'Z'
    write(status_path, state)
    if deferred:
        state['current'] = 'deferred_reduction'
        write(status_path, state)
        print('Coverage finished for validated designs; reductions still pending: '
              + ', '.join(deferred), flush=True)
        return 1
    print('All remote reduction and coverage stages complete.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
