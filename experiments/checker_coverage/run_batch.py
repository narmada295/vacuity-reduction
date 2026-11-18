#!/usr/bin/env python3
"""Measure JasperGold checker coverage for the 25-design assertion sets.

Runs on the JasperGold host against reduction_25_plan.json. Each design gets an
instrumented copy of its RTL, separate combined COI/proof and proof-only Jasper
projects, raw logs, and a machine-readable plus human-readable summary.
"""

import argparse
import collections
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
try:
    from experiments.assertion_extraction.formal_clocking import clock_commands
except ImportError:
    # On the Jasper host this file lives in experiments/checker_coverage/;
    # the reduction tools and their shared helper live in experiments/scripts/.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from formal_clocking import clock_commands


CLOCK_RE = re.compile(r'@\s*\(\s*(?:posedge|negedge)\s+([A-Za-z_$][\w$]*)\s*\)')
MODULE_RE = re.compile(r'(?m)^\s*module\s+(?:automatic\s+)?([A-Za-z_$][\w$]*)\b')
IDENT_RE = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*\Z')
LICENSE_RE = re.compile(r'FLEXnet ERROR\(-15\b|Cannot connect to license server|Waiting for license|license checkout failed', re.I)
METRICS = {
    'coi_statement': ('combined', 'COI STATEMENT COVERAGE'),
    'coi_branch': ('combined', 'COI BRANCH COVERAGE'),
    'default_statement': ('combined', 'DEFAULT CHECKER STATEMENT COVERAGE'),
    'default_branch': ('combined', 'DEFAULT CHECKER BRANCH COVERAGE'),
    'proof_core_statement': ('proof', 'PROOF CORE STATEMENT COVERAGE'),
    'proof_core_branch': ('proof', 'PROOF CORE BRANCH COVERAGE'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf8')
    tmp.replace(path)


def safe_ident(name):
    if not IDENT_RE.fullmatch(name):
        raise ValueError('Invalid Verilog identifier: {!r}'.format(name))
    return name


def source_rtl(case_path):
    files = sorted(path for path in case_path.parent.iterdir() if path.suffix.lower() in ('.v', '.sv'))
    if len(files) != 1:
        raise ValueError('Expected one Verilog source beside {}: {}'.format(case_path, files))
    return files[0]


def insert_checks(rtl, top, rows):
    modules = list(MODULE_RE.finditer(rtl))
    matches = [m for m in modules if m.group(1) == top]
    if len(matches) != 1:
        raise ValueError('Expected one module {!r}, found {}'.format(top, [m.group(1) for m in modules]))
    start = matches[0].end()
    following = next((m.start() for m in modules if m.start() > start), len(rtl))
    ends = list(re.finditer(r'(?m)^\s*endmodule\b', rtl[start:following]))
    if len(ends) != 1:
        raise ValueError('Expected one endmodule for {}'.format(top))
    labels = []
    checks = []
    for row in rows:
        expression = row.get('property_expression', '').strip().rstrip(';')
        if not expression:
            raise ValueError('Empty property_expression in row {}'.format(row.get('row_number')))
        label = 'KEEP_{:04d}'.format(int(row['row_number']))
        if label in labels:
            raise ValueError('Duplicate assertion label {}'.format(label))
        labels.append(label)
        checks.append('{}: assert property (\n  {}\n);'.format(label, expression))
    insert_at = start + ends[0].start()
    checked = rtl[:insert_at] + '\n// Retained assertions (generated)\n' + '\n\n'.join(checks) + '\n' + rtl[insert_at:]
    return checked, labels


def tcl_text(top, clocks, proof_only, context=None):
    commands = clock_commands(clocks, context)
    prefix = [
        'check_cov -init -model {{statement branch}} -type {} -enable_prove_based_proof_core'.format(
            'proof' if proof_only else '{coi proof}'),
        'analyze -sv12 ../checked.sv',
        'elaborate -top {}'.format(safe_ident(top)),
        commands,
        'reset -none',
        'prove -all',
    ]
    if proof_only:
        prefix += [
            'check_cov -measure -type proof -all',
            'puts "=== PROOF CORE STATEMENT COVERAGE ==="',
            'check_cov -report -type checker -model statement -no_return',
            'puts "=== PROOF CORE BRANCH COVERAGE ==="',
            'check_cov -report -type checker -model branch -no_return',
        ]
    else:
        prefix += [
            'check_cov -measure -type {coi proof} -all',
            'puts "=== COI STATEMENT COVERAGE ==="',
            'check_cov -report -type checker -checker_mode coi -model statement -no_return',
            'puts "=== COI BRANCH COVERAGE ==="',
            'check_cov -report -type checker -checker_mode coi -model branch -no_return',
            'puts "=== DEFAULT CHECKER STATEMENT COVERAGE ==="',
            'check_cov -report -type checker -model statement -no_return',
            'puts "=== DEFAULT CHECKER BRANCH COVERAGE ==="',
            'check_cov -report -type checker -model branch -no_return',
            'puts "=== ASSERTION RESULTS ==="',
            'report -summary -force -result',
            'puts "=== COVERAGE ANALYSIS COMPLETED ==="',
        ]
    return '\n'.join(prefix + ['exit -force', ''])


def run_name(assertion_set, batch_name='25'):
    return 'run_{}_{}'.format(batch_name, 'final' if assertion_set == 'reduced' else 'unreduced')


def prepare(item, root, assertion_set='reduced', batch_name='25'):
    case_id = item['case_id']
    case_path = Path(item['case'])
    reduction = Path(item['out'])
    output = root / case_id / run_name(assertion_set, batch_name)
    reduction_summary = reduction / 'summary.json'
    validated = load_json(reduction_summary) if reduction_summary.exists() else {}
    if assertion_set == 'reduced' and validated.get('status') != 'VALIDATED':
        raise ValueError('Reduction is not validated for {}'.format(case_id))
    if assertion_set == 'reduced':
        assertion_path = reduction / 'retained_assertions.jsonl'
    else:
        assertion_path = reduction / 'extracted' / 'selected_assertions.jsonl'
    rows = load_jsonl(assertion_path)
    if assertion_set == 'reduced' and len(rows) != validated.get('retained'):
        raise ValueError('Retained count disagrees with reduction summary for {}'.format(case_id))
    if assertion_set == 'unreduced':
        for row in rows:
            if (row.get('selected') is not True or
                    str(row.get('assertion_status', '')).strip().lower() != 'proven' or
                    not (row.get('vacuous') is False or row.get('reason') == 'outside_scope_non_implication')):
                raise ValueError('Unselected or unproven assertion in {}: {}'.format(case_id, row.get('row_number')))
    case = load_json(case_path)
    context = case.get('formal_clocking')
    case_rows = case['rows']
    if not case_rows:
        raise ValueError('No case rows for {}'.format(case_id))
    top = safe_ident(case_rows[0]['top'])
    source = source_rtl(case_path)
    clocks = list(dict.fromkeys(CLOCK_RE.findall('\n'.join(r.get('property_expression', '') for r in rows))))
    default_clock = case_rows[0].get('clock')
    if default_clock and not clocks:
        clocks = [default_clock]
    clocks = [safe_ident(c) for c in clocks]
    manifest = {
        'schema_version': 3,
        'case_id': case_id, 'top': top, 'source_rtl': str(source),
        'source_sha256': sha256(source), 'assertion_set': assertion_set,
        'batch_name': batch_name,
        'assertion_source': str(assertion_path), 'assertion_sha256': sha256(assertion_path),
        'assertion_count': len(rows), 'reduction_summary': str(reduction_summary) if reduction_summary.exists() else None,
        'reduction_summary_sha256': sha256(reduction_summary) if reduction_summary.exists() else None,
        'clocks': [d['signal'] for d in context['declarations']] if context else clocks,
        'property_event_signals': clocks,
        'formal_clocking': context,
        'reset_policy': 'none', 'coverage_models': ['statement', 'branch'],
        'coverage_types': ['coi', 'proof'],
    }
    if assertion_set == 'reduced':
        manifest.update(retained_assertions=str(assertion_path),
                        retained_sha256=sha256(assertion_path), retained_count=len(rows))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'manifest.json', manifest)
    shutil.copyfile(source, output / 'source.v')
    shutil.copyfile(assertion_path, output / ('retained_assertions.jsonl' if assertion_set == 'reduced' else 'selected_assertions.jsonl'))
    if rows:
        checked, labels = insert_checks(source.read_text(encoding='utf8'), top, rows)
        (output / 'checked.sv').write_text(checked, encoding='utf8')
        for stage, proof_only in (('combined', False), ('proof', True)):
            stage_dir = output / stage
            stage_dir.mkdir(exist_ok=True)
            (stage_dir / 'run.tcl').write_text(tcl_text(top, clocks, proof_only, context), encoding='utf8')
    else:
        labels = []
    manifest['labels'] = labels
    write_json(output / 'manifest.json', manifest)
    return output, manifest


def metric_from_log(log, marker, top):
    match = re.search(r'^=== {} ===\s*$'.format(re.escape(marker)), log, flags=re.M)
    if not match:
        raise ValueError('Missing coverage report: {}'.format(marker))
    section = log[match.end():]
    next_marker = re.search(r'^=== .* ===\s*$', section, flags=re.M)
    if next_marker:
        section = section[:next_marker.start()]
    total_match = re.search(r'Total Cover Items:\s*(\d+)', section)
    if not total_match:
        raise ValueError('Missing cover-item denominator for {}'.format(marker))
    total = int(total_match.group(1))
    row_pattern = r'^\|\s*{}\s*\|\s*([0-9]+(?:\.[0-9]+)?)%\s*\((\d+)/(\d+)\)'.format(re.escape(top))
    row = re.search(row_pattern, section, flags=re.M)
    if row:
        covered, denominator = int(row.group(2)), int(row.group(3))
        if denominator != total:
            raise ValueError('Report denominator disagrees with total cover items: {}'.format(marker))
        reported = float(row.group(1))
        return {'covered': covered, 'total': denominator,
                'percent': reported if denominator else None,
                'jasper_reported_percent': reported}
    if total == 0:
        return {'covered': 0, 'total': 0, 'percent': None}
    raise ValueError('Missing top-level checker coverage row for {}'.format(marker))


def property_results(log, labels):
    result = {}
    preconditions = {}
    for line in log.splitlines():
        match = re.match(r'^\[\d+\]\s+\S*\.(KEEP_\d+)(:precondition\d+)?\s+(\w+)\s+', line)
        if not match:
            continue
        label, precondition, status = match.groups()
        if precondition:
            preconditions.setdefault(label, {})[precondition[1:]] = status
        else:
            result[label] = status
    missing = sorted(set(labels) - set(result))
    if missing:
        raise ValueError('Missing {} assertion results: {}'.format(len(missing), missing[:10]))
    return result, preconditions


def summarize(output, manifest):
    count = manifest['assertion_count']
    summary = dict(manifest)
    summary['generated_at_utc'] = now()
    summary['output_dir'] = str(output)
    if count == 0:
        summary.update(status='no_retained_assertions', explanation='No checker coverage is defined for an empty assertion set.',
                       coverage={name: None for name in METRICS}, assertion_results={},
                       assertion_status_counts={}, precondition_status_counts={})
        return summary
    logs = {stage: (output / stage / 'coverage.log').read_text(encoding='utf8', errors='replace')
            for stage in ('combined', 'proof')}
    coverage = {}
    for name, (stage, marker) in METRICS.items():
        coverage[name] = metric_from_log(logs[stage], marker, manifest['top'])
    statuses, preconditions = property_results(logs['combined'], manifest['labels'])
    summary.update(status='complete', coverage=coverage,
                   assertion_results=statuses,
                   assertion_status_counts=dict(collections.Counter(statuses.values())),
                   precondition_results=preconditions,
                   precondition_status_counts=dict(collections.Counter(
                       status for values in preconditions.values() for status in values.values())),
                   logs={'combined': str(output / 'combined' / 'coverage.log'),
                         'proof': str(output / 'proof' / 'coverage.log')})
    return summary


def write_markdown(output, summary):
    lines = ['# Checker coverage: {}'.format(summary['case_id']), '',
             '- Status: `{}`'.format(summary['status']),
             '- Top module: `{}`'.format(summary['top']),
             '- Assertion set: {}'.format(summary['assertion_set']),
             '- Assertions in set: {}'.format(summary['assertion_count']),
             '- Clocks: {}'.format(', '.join('`{}`'.format(c) for c in summary['clocks']) or 'none'),
             '- Reset policy: none', '']
    if summary['status'] == 'complete':
        lines += ['| Metric | Covered | Total | Coverage |', '|---|---:|---:|---:|']
        for name, value in summary['coverage'].items():
            pct = 'N/A' if value['percent'] is None else '{:.2f}%'.format(value['percent'])
            lines.append('| {} | {} | {} | {} |'.format(name, value['covered'], value['total'], pct))
        lines += ['', '## Assertion results', '']
        for status, amount in sorted(summary['assertion_status_counts'].items()):
            lines.append('- {}: {}'.format(status, amount))
        if summary['precondition_status_counts']:
            lines += ['', '## Precondition cover results', '']
            for status, amount in sorted(summary['precondition_status_counts'].items()):
                lines.append('- {}: {}'.format(status, amount))
    else:
        lines += [summary.get('explanation') or summary.get('error', ''), '']
    (output / 'summary.md').write_text('\n'.join(lines).rstrip() + '\n', encoding='utf8')


def license_failure(stage_dir, log_path):
    console = stage_dir / 'jgproject' / 'jg_console.log'
    evidence = log_path.read_text(encoding='utf8', errors='replace')[-32768:]
    if console.exists():
        evidence += console.read_text(encoding='utf8', errors='replace')[-32768:]
    return bool(LICENSE_RE.search(evidence))


def run_stage(output, stage, jg, timeout, retry_delay, license_retries):
    stage_dir = output / stage
    complete_path = stage_dir / 'complete.json'
    fingerprint = hashlib.sha256((sha256(output / 'checked.sv') + sha256(stage_dir / 'run.tcl')).encode()).hexdigest()
    if complete_path.exists() and (stage_dir / 'coverage.log').exists():
        if load_json(complete_path).get('fingerprint') == fingerprint:
            return
    for attempt in range(1, license_retries + 1):
        log_path = stage_dir / 'coverage_attempt_{:03d}.log'.format(attempt)
        command = [jg, '-batch', 'run.tcl']
        try:
            with log_path.open('w', encoding='utf8') as log:
                result = subprocess.run(command, cwd=str(stage_dir), stdout=log,
                                        stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
            error = 'JasperGold exited {}'.format(code)
        except subprocess.TimeoutExpired:
            code = -1
            error = 'JasperGold timed out after {} seconds'.format(timeout)
        if code == 0:
            shutil.copyfile(log_path, stage_dir / 'coverage.log')
            write_json(complete_path, {'fingerprint': fingerprint, 'completed_at_utc': now(),
                                       'attempt': attempt, 'log': str(stage_dir / 'coverage.log')})
            return
        if license_failure(stage_dir, log_path):
            project = stage_dir / 'jgproject'
            if project.is_dir() and not project.is_symlink():
                shutil.rmtree(project)
            print('{} {}: license unavailable, retry {}/{}'.format(output.parent.name, stage, attempt, license_retries), flush=True)
            if attempt < license_retries:
                time.sleep(retry_delay)
                continue
        raise RuntimeError('{} in {} (see {})'.format(error, stage, log_path))


def aggregate(root, plan, states, assertion_set='reduced', batch_name='25'):
    out = {'generated_at_utc': now(), 'assertion_set': assertion_set, 'batch_name': batch_name,
           'designs': {}, 'counts': dict(collections.Counter(states.values())),
           'metrics': {}}
    for item in plan['designs']:
        case_id = item['case_id']
        summary_path = root / case_id / run_name(assertion_set, batch_name) / 'summary.json'
        if summary_path.exists():
            summary = load_json(summary_path)
            out['designs'][case_id] = {'status': summary['status'], 'summary': str(summary_path),
                                       'assertion_count': summary.get('assertion_count', summary.get('retained_count')),
                                       'coverage': summary.get('coverage')}
    for name in METRICS:
        values = [entry['coverage'][name] for entry in out['designs'].values()
                  if entry['status'] == 'complete' and entry['coverage'][name] is not None]
        covered = sum(v['covered'] for v in values)
        total = sum(v['total'] for v in values)
        out['metrics'][name] = {'covered': covered, 'total': total,
                                'percent': round(100 * covered / total, 2) if total else None,
                                'designs_with_metric': len(values)}
    write_json(root / ('coverage_{}_{}.json'.format(batch_name, 'summary' if assertion_set == 'reduced' else 'unreduced_summary')), out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--jg', default='jg')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--retry-delay', type=int, default=180)
    parser.add_argument('--license-retries', type=int, default=240)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--case', action='append')
    parser.add_argument('--assertion-set', choices=('reduced', 'unreduced'), default='reduced')
    parser.add_argument('--batch-name', default='25', help='Name used in run folders and status files')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+', args.batch_name):
        parser.error('--batch-name must contain letters, digits, or underscores only')
    plan = load_json(args.plan)
    root = args.root or Path(plan['remote_base']) / 'checker_coverage'
    root.mkdir(parents=True, exist_ok=True)
    items = [item for item in plan['designs'] if not args.case or item['case_id'] in args.case]
    if args.case and len(items) != len(set(args.case)):
        parser.error('Unknown --case in {}'.format(args.case))
    state_path = root / ('coverage_{}_{}.json'.format(args.batch_name,
                         'status' if args.assertion_set == 'reduced' else 'unreduced_status'))
    states = {}
    write_json(state_path, {'state': 'preparing' if args.prepare_only else 'running', 'cases': states})
    for index, item in enumerate(items, 1):
        case_id = item['case_id']
        output = root / case_id / run_name(args.assertion_set, args.batch_name)
        summary_path = output / 'summary.json'
        try:
            output, manifest = prepare(item, root, args.assertion_set, args.batch_name)
            previous = load_json(summary_path) if summary_path.exists() else {}
            if (previous.get('status') in ('complete', 'no_retained_assertions')
                    and previous.get('schema_version') == manifest['schema_version']
                    and previous.get('source_sha256') == manifest['source_sha256']
                    and previous.get('formal_clocking') == manifest.get('formal_clocking')
                    and previous.get('assertion_sha256') == manifest['assertion_sha256']):
                states[case_id] = previous['status']
                print('{}/{} {}: already {}'.format(index, len(items), case_id, previous['status']), flush=True)
                continue
            if args.prepare_only and manifest['assertion_count']:
                states[case_id] = 'prepared'
                print('{}/{} {}: prepared {} checks'.format(index, len(items), case_id, manifest['assertion_count']), flush=True)
                continue
            if manifest['assertion_count']:
                states[case_id] = 'running_combined'
                write_json(state_path, {'state': 'running', 'current': case_id, 'cases': states})
                run_stage(output, 'combined', args.jg, args.timeout, args.retry_delay, args.license_retries)
                states[case_id] = 'running_proof'
                write_json(state_path, {'state': 'running', 'current': case_id, 'cases': states})
                run_stage(output, 'proof', args.jg, args.timeout, args.retry_delay, args.license_retries)
            summary = summarize(output, manifest)
            write_json(summary_path, summary)
            write_markdown(output, summary)
            states[case_id] = summary['status']
            print('{}/{} {}: {}'.format(index, len(items), case_id, summary['status']), flush=True)
        except Exception as error:
            output.mkdir(parents=True, exist_ok=True)
            failed = {'case_id': case_id, 'status': 'failed', 'error': str(error), 'generated_at_utc': now(),
                      'output_dir': str(output)}
            write_json(summary_path, failed)
            states[case_id] = 'failed'
            print('{}/{} {}: FAILED {}'.format(index, len(items), case_id, error), flush=True)
        write_json(state_path, {'state': 'preparing' if args.prepare_only else 'running',
                                'current': case_id, 'cases': states})
        if not args.case:
            aggregate(root, plan, states, args.assertion_set, args.batch_name)
    final_state = ('prepared' if args.prepare_only else
                   'finished_with_failures' if 'failed' in states.values() else 'complete')
    write_json(state_path, {'state': final_state, 'cases': states})
    if not args.case:
        aggregate(root, plan, states, args.assertion_set, args.batch_name)
    print('Finished: {}'.format(dict(collections.Counter(states.values()))), flush=True)
    return 1 if 'failed' in states.values() else 0


if __name__ == '__main__':
    sys.exit(main())
