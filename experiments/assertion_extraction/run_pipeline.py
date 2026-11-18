#!/usr/bin/env python3
"""One-command, resumable SVA redundancy pipeline (Python 3.6+).

Example, on the JasperGold machine after sourcing the Cadence environment:
  python3 run_pipeline.py --case case.json --out piezo_run
  python3 run_pipeline.py --out piezo_run --status

Stages: extract -> exact dedup -> RTL consistency gate -> [optional audit]
        -> sequential free-trace pruning -> independent free-trace validation.
All results are written to --out. The input file and RTL are never changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, data):
    dest = Path(path)
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf8')
    tmp.replace(dest)


def show_process(cmd, cwd=None, logfile=None, timeout=None):
    print('>>> ' + ' '.join(str(x) for x in cmd), flush=True)
    if logfile:
        with Path(logfile).open('w', encoding='utf8') as log:
            proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Stop the Jasper session as well as its console on timeout.
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=5)
                except ProcessLookupError:
                    proc.wait()
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise
        if proc.returncode:
            raise RuntimeError('Command failed (exit {}). Inspect {}'.format(proc.returncode, logfile))
    else:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, timeout=timeout)
        if proc.returncode:
            raise RuntimeError('Command failed (exit {}): {}'.format(proc.returncode, cmd))


def py(script, *args):
    return [sys.executable, str(HERE / script)] + [str(x) for x in args]


def parse_status_tsv(path):
    if not path.exists():
        raise RuntimeError('Missing {}. Inspect JasperGold log.'.format(path))
    result = {}
    for line in path.read_text(encoding='utf8').splitlines():
        parts = line.split('\t')
        if len(parts) != 2 or not parts[0] or parts[0] in result:
            raise RuntimeError('Malformed/duplicate status in {}: {!r}'.format(path, line))
        result[parts[0]] = parts[1].strip().strip('{}').lower()
    return result


def run_jg(directory, executable, timeout):
    directory = Path(directory)
    status = directory / 'status.tsv'
    if status.exists():
        status.unlink()  # Never reuse a stale successful result.
    show_process([executable, '-batch', 'run.tcl'], cwd=directory,
                 logfile=directory / 'jg.log', timeout=timeout)


def check_consistency(folder):
    meta = read_json(folder / 'metadata.json')
    states = parse_status_tsv(folder / 'status.tsv')
    # Use precisely the labels the generated RTL witness contains; prevents a
    # short status.tsv from silently producing a false PASS.
    rtl_text = (folder / 'rtl_witness.sv').read_text(encoding='utf8')
    property_names = set(re.findall(r'\b(RTL_CHECK_\d+):\s+assert\s+property', rtl_text))
    if len(property_names) != meta['count']:
        raise RuntimeError('Mismatch in consistency witness assertion count')
    expected = property_names | {'RTL_TRACE_EXISTS'}
    if set(states) != expected:
        raise RuntimeError('Consistency results incomplete or unexpected: {}'.format(sorted(expected - set(states))))
    if any(states[k] != 'proven' for k in property_names) or states['RTL_TRACE_EXISTS'] != 'covered':
        raise RuntimeError('RTL premise consistency NOT established: {}'.format(
            {k: v for k, v in states.items() if v not in ('proven', 'covered')}))
    print('PASS: RTL premise consistency ({} proven, trace covered)'.format(meta['count']))
    return {'checks': meta['count'], 'status': 'PASS', 'trace_cover': 'covered',
            'formal_clocking': meta.get('formal_clocking')}


def check_final(folder):
    manifest = read_json(folder / 'validation_manifest.json')
    statuses = parse_status_tsv(folder / 'status.tsv')
    expected = {'RESTORE_{:04d}'.format(int(x)) for x in manifest['removed_checked']}
    if set(statuses) != expected:
        raise RuntimeError('Final validation missing/unexpected statuses: {}'.format(sorted(expected - set(statuses))))
    bad = {key: value for key, value in statuses.items() if value != 'proven'}
    if bad:
        raise RuntimeError('Final validation FAILED: {}'.format(bad))
    print('PASS: final free-trace proof ({} removed assertions restored)'.format(len(expected)))
    return {'status': 'PASS', 'targets_proven': len(expected)}


def check_completed_formal(folder):
    state = read_json(folder / 'state.json')
    eligible = {x for x in state['order'] if not state['properties'][x]['excluded_liveness']}
    processed = [x['target_id'] for x in state['processed']]
    if set(processed) != eligible or len(processed) != len(eligible):
        raise RuntimeError('Formal checks incomplete: {} of {}. See {}/jobs/*/jg.log'.format(
            len(set(processed)), len(eligible), folder))
    return state


def export_final(folder, state):
    rows = []
    for ln in (folder / 'pruning' / 'retained_assertions.jsonl').read_text(encoding='utf8').splitlines():
        if ln.strip():
            rows.append(json.loads(ln))
    sv = ['// Retained properties. Insert inside the RTL module where signals are in scope.',
          '// Do not treat this include as an independent free-trace proof harness.',
          '// Liveness properties (if any) were retained unchanged.', '']
    for row in rows:
        sv.extend(['// Original row {}: {}'.format(row['row_number'], row.get('property_name') or 'inline'),
                   'KEEP_{:04d}: assert property ('.format(int(row['row_number'])),
                   row['property_expression'], ');', ''])
    (folder / 'retained_assertions.svh').write_text('\n'.join(sv) + '\n', encoding='utf8')
    shutil.copyfile(str(folder / 'pruning' / 'retained_assertions.jsonl'),
                    str(folder / 'retained_assertions.jsonl'))
    original_records = sum(1 for line in (folder / 'extracted' / 'all_assertions.jsonl').open(encoding='utf8') if line.strip())
    selected_records = sum(1 for line in (folder / 'extracted' / 'selected_assertions.jsonl').open(encoding='utf8') if line.strip())
    unique_records = len(state['order'])
    removed_formal = len(state['removed'])
    summary = {
        'design': state.get('design'), 'original_records': original_records,
        'selected_proven_nonvacuous_or_out_of_scope': selected_records,
        'exact_duplicate_copies_removed': selected_records - unique_records,
        'unique_after_deduplication': unique_records,
        'formal_removed': removed_formal,
        'retained': len(rows),
        'retained_liveness': sum(1 for x in state['remaining'] if state['properties'][x]['excluded_liveness']),
        'status': 'VALIDATED',
        'formal_clocking': state.get('formal_clocking'),
        'retained_jsonl': str(folder / 'retained_assertions.jsonl'),
        'retained_svh': str(folder / 'retained_assertions.svh'),
        'proof_results_csv': str(folder / 'pruning' / 'results.csv'),
        'final_validation': str(folder / 'validation' / 'status.tsv') if removed_formal else 'no_formal_removals'
    }
    write_json(folder / 'summary.json', summary)
    print('\nDONE: {} selected -> {} unique -> {} retained ({} formal removals).'.format(
        selected_records, unique_records, len(rows), removed_formal))
    print('Verified outputs:', folder / 'retained_assertions.jsonl', folder / 'retained_assertions.svh')
    return summary


def run_pipeline(args):
    if not args.case:
        raise ValueError('--case is required except for --status')
    original = Path(args.case).expanduser().resolve()
    if not original.is_file():
        raise ValueError('No such case file: {}'.format(original))
    root = Path(args.out).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    meta_file = root / 'input_metadata.json'
    snapshot = root / 'input_case.json'
    fingerprint = sha256(original)
    if meta_file.exists():
        meta = read_json(meta_file)
        if meta['sha256'] != fingerprint or not snapshot.exists() or sha256(snapshot) != fingerprint:
            raise RuntimeError('Output directory belongs to a different case.json. Choose a new --out.')
    else:
        if snapshot.exists():
            raise RuntimeError('Existing input_case.json with no input_metadata.json. Choose a fresh --out.')
        shutil.copyfile(str(original), str(snapshot))
        write_json(meta_file, {'source': str(original), 'sha256': fingerprint,
                               'python': sys.version.split()[0]})
    if (root / 'summary.json').exists():
        summary = read_json(root / 'summary.json')
        if summary.get('status') == 'VALIDATED':
            print('Already completed and validated. Use --status to view outputs.')
            return

    if args.prepare_only:
        print('PREPARE-ONLY: no JasperGold or deletion performed.')

    extracted = root / 'extracted'
    selected = extracted / 'selected_assertions.jsonl'
    if not selected.exists():
        show_process(py('extract_assertions.py', snapshot, '--out', extracted))
    issues = [json.loads(ln) for ln in (extracted / 'extraction_issues.jsonl').read_text(encoding='utf8').splitlines() if ln.strip()]
    bad_selected = [x for x in issues if x.get('selected')]
    if bad_selected:
        raise RuntimeError('{} selected assertions could not be parsed. Inspect extraction_issues.csv'.format(len(bad_selected)))
    if not selected.read_text(encoding='utf8').strip():
        raise RuntimeError('No eligible assertions in {}'.format(original))

    dedup = root / 'deduplicated'
    unique = dedup / 'unique_assertions.jsonl'
    if not unique.exists():
        show_process(py('remove_exact_duplicates.py', selected, '--out', dedup))

    consistency = root / 'consistency'
    consistency_ok = consistency / 'verified.json'
    if not consistency_ok.exists():
        if not (consistency / 'run.tcl').exists():
            show_process(py('check_premise_consistency.py', 'prepare', '--input', unique,
                            '--case', snapshot, '--out', consistency))
        if args.prepare_only:
            print('Prepared consistency job at {}'.format(consistency))
            return
        run_jg(consistency, args.jg, args.timeout)
        proof = check_consistency(consistency)
        write_json(consistency_ok, proof)
    else:
        check_consistency(consistency)  # Revalidate recorded evidence.

    if args.audit:
        audit = root / 'audit'
        if not (audit / 'state.json').exists():
            cmd = py('formal_redundancy.py', 'init', '--input', unique, '--case', snapshot,
                     '--out', audit)
            if args.decls: cmd += ['--decls', args.decls]
            show_process(cmd)
        if args.prepare_only:
            print('Prepared audit job at {}'.format(audit))
            return
        audit_state = read_json(audit / 'state.json')
        show_process(py('formal_redundancy.py', 'run', '--out', audit,
                        '--max-jobs', len(audit_state['order']), '--timeout', args.timeout,
                        '--jg', args.jg))
        check_completed_formal(audit)
        print('Audit completed. No assertions removed in audit stage.')

    pruning = root / 'pruning'
    if not (pruning / 'state.json').exists():
        cmd = py('formal_redundancy.py', 'init', '--input', unique, '--case', snapshot,
                 '--out', pruning)
        if args.decls: cmd += ['--decls', args.decls]
        show_process(cmd)
    if args.prepare_only:
        print('Prepared first pruning job at {}'.format(pruning))
        return
    current = read_json(pruning / 'state.json')
    show_process(py('formal_redundancy.py', 'run', '--out', pruning,
                    '--max-jobs', len(current['order']), '--timeout', args.timeout,
                    '--jg', args.jg, '--auto-prune'))
    state = check_completed_formal(pruning)
    print('Pruning: {} removed, {} retained.'.format(len(state['removed']), len(state['remaining'])))

    if state['removed']:
        validation = root / 'validation'
        if not (validation / 'validation_manifest.json').exists():
            show_process(py('validate_reduced_set.py', 'prepare', '--run', pruning, '--out', validation))
        run_jg(validation, args.jg, args.timeout)
        proof = check_final(validation)
        write_json(validation / 'verified.json', proof)
    export_final(root, state)


def status(args):
    root = Path(args.out).expanduser().resolve()
    if (root / 'summary.json').exists():
        print(json.dumps(read_json(root / 'summary.json'), indent=2))
        return
    print('Not yet finalized:', root)
    for label in ('extracted', 'deduplicated', 'consistency', 'audit', 'pruning', 'validation'):
        p = root / label
        if p.exists():
            print(' - {}: {}'.format(label, 'present'))
    p = root / 'pruning' / 'state.json'
    if p.exists():
        s = read_json(p)
        print('   checked={}, removed={}, retained={}'.format(len(s['processed']), len(s['removed']), len(s['remaining'])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', help='Input design case.json (one design per run)')
    ap.add_argument('--out', required=True, help='Output folder; resumable on rerun')
    ap.add_argument('--jg', default='jg', help='JasperGold executable (must be in PATH or absolute path)')
    ap.add_argument('--timeout', type=int, default=600, help='Timeout in seconds per JasperGold job (default 600)')
    ap.add_argument('--decls', help='Optional explicit symbol width declarations for complex RTL')
    ap.add_argument('--audit', action='store_true', help='Also run all-others audit without pruning (extra formal runs)')
    ap.add_argument('--prepare-only', action='store_true', help='Prepare files but do not launch JasperGold')
    ap.add_argument('--status', action='store_true', help='Show final or partial progress')
    args = ap.parse_args()
    try:
        if args.status:
            status(args)
        else:
            if not args.prepare_only and not shutil.which(args.jg):
                raise RuntimeError("Cannot find JasperGold '{}'. Start csh; source ~/cad_cshrc; rerun from that shell.".format(args.jg))
            run_pipeline(args)
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as err:
        print('ERROR: {}'.format(err), file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
