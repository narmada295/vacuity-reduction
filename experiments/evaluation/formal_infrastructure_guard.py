"""Recover dead Jasper project locks during license retries (remote Python 3.6)."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import time


def worker_matches(root, pid, proc):
    try:
        arguments = (proc / str(pid) / 'cmdline').read_bytes().split(b'\0')
        return (str(root / 'formal_worker.py').encode() in arguments
                and b'--root' in arguments and str(root).encode() in arguments)
    except OSError:
        return False


def recover(root, proc=Path('/proc')):
    state = json.loads((root / 'status.json').read_text())
    if state.get('state') != 'waiting_for_license':
        return None
    pid = state['pid']
    job_id = state.get('current_job', '')
    if not re.fullmatch(r'[0-9a-f]{24}', job_id) or not worker_matches(root, pid, proc):
        return None
    folder = root / 'jobs' / job_id
    project = folder / 'jgproject'
    if not project.is_dir() or project.is_symlink() or (folder / 'result.json').exists():
        return None
    # Stop only this verified worker. Re-read status while it cannot start a job.
    os.kill(pid, signal.SIGSTOP)
    try:
        stopped = False
        for _ in range(20):
            text = (proc / str(pid) / 'status').read_text()
            if re.search(r'^State:\s+T', text, re.M):
                stopped = True
                break
            time.sleep(.05)
        if not stopped:
            return None
        current = json.loads((root / 'status.json').read_text())
        if (current.get('state') != 'waiting_for_license' or current.get('current_job') != job_id
                or current.get('pid') != pid or (folder / 'result.json').exists()):
            return None
        locks = list(project.glob('*.lock'))
        if not locks:
            return None
        owners = []
        for lock in locks:
            match = re.search(r'\.(\d+)\.lock$', lock.name)
            if not match or (proc / match.group(1)).exists():
                return None
            owners.append(int(match.group(1)))
        # A live process using this job is an additional reason to leave it alone.
        for process in proc.iterdir():
            if not process.name.isdigit():
                continue
            try:
                cwd = (process / 'cwd').resolve(strict=True)
            except OSError:
                continue
            if cwd == folder.resolve() or folder.resolve() in cwd.parents:
                return None
        stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')
        archive = root / 'infrastructure' / (stamp + '_' + job_id)
        archive.mkdir(parents=True)
        if (folder / 'jg.log').exists():
            shutil.copyfile(str(folder / 'jg.log'), str(archive / 'jg_launch.log'))
        project.rename(archive / 'jgproject')
        audit = {'action': 'archive_dead_project_for_license_retry', 'job_id': job_id,
                 'worker_pid': pid, 'dead_lock_owners': owners, 'archive': str(archive),
                 'original_status': current, 'assertion_or_formal_inputs_changed': False,
                 'completed_results_changed': False, 'utc': stamp}
        (archive / 'recovery.json').write_text(json.dumps(audit, indent=2) + '\n')
        return audit
    finally:
        if worker_matches(root, pid, proc):
            os.kill(pid, signal.SIGCONT)


def run(root, interval):
    lock = (root / 'infrastructure_guard.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root / 'manifest.json').read_text())
    if (str(root) != manifest['remote'] or hashlib.sha256(
            (root / 'formal_worker.py').read_bytes()).hexdigest() != manifest['worker_sha256']):
        raise ValueError('Infrastructure guard requires the exact deployed formal snapshot')
    while True:
        try:
            state = json.loads((root / 'status.json').read_text())
            if state.get('state') == 'complete':
                return
            if not worker_matches(root, state.get('pid', 0), Path('/proc')):
                return
            audit = recover(root)
            if audit:
                print(json.dumps(audit), flush=True)
        except Exception as error:
            print('Infrastructure check:', type(error).__name__, str(error), flush=True)
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=20)
    arguments = parser.parse_args()
    run(arguments.root, arguments.interval)
