"""Evaluate the frozen four-OpenTitan/two-FIXME base/SFT/KTO responses."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex

from experiments.evaluation import formal, formal_test, run_mixed
from experiments.evaluation.generate import write_json
from experiments.evaluation.prepare_mixed import COHORT, DATA, ROOT, task_for_case
from experiments.assertion_extraction.formal_clocking import clock_commands


DEFAULT = ROOT / 'formal_v1'
ARCHIVE = ROOT / 'returned/external_mixed_six_v4_outputs.tar.gz'
REMOTE = '/storage/ckarfa/narmada/experiments/evaluation_one_shot_v4/external_mixed_formal_v1'
GENERATIONS = Path('outputs/eval_one_shot_v4/external_mixed_test')
ALLOWED_ROOTS = (str(GENERATIONS), 'logs/eval_one_shot_v4/external_mixed_test')
WORKER = formal.DEFAULT / 'formal_worker.frozen.py'


def implication_scope(item):
    """Ignore comments/strings only for scope analysis; never rewrite tool input."""
    try:
        _, expression, _, status, _ = formal.resolve_property({
            'assertion': formal.mask_noncode(item['assertion']),
            'support_code': formal.mask_noncode(item.get('selected_support', ''))})
    except ValueError:
        return None
    return any(t in ('|->', '|=>') for t in formal.tokenize(expression)) if status == 'ok' else None


def load_contexts(tasks, designs):
    """Reconstruct each exact prompt from delivered inputs, not teacher answers."""
    by_case = {case['case_id']: case for case in designs}
    if len(by_case) != len(designs):
        raise ValueError('Duplicate design contexts')
    result = {}
    for task in tasks:
        case = by_case[task['case_id']]
        matches = [signal for signal in case['signals'] if signal['name'] == task['target_signal']]
        if len(matches) != 1 or task_for_case(case, matches[0]) != task:
            raise ValueError('Formal source context differs from the original inference prompt')
        context = case['formal_context']
        if context['formal_reset_policy'] != 'unconstrained':
            raise ValueError('Unexpected formal reset policy')
        clocks = clock_commands(context['checker_clocks'], context.get('formal_clocking'))
        result[task['task_id']] = {'rtl': case['rtl'], 'spec': case['spec'], 'top': case['top'],
                                   'clock_tcl': clocks, 'formal_context': context}
    if set(result) != {task['task_id'] for task in tasks}:
        raise ValueError('Missing formal task context')
    return result


def load_returned(out):
    digest = formal_test.extract_returned(ARCHIVE, out / 'returned', allowed_roots=ALLOWED_ROOTS)
    directory = out / 'returned' / GENERATIONS
    protocol = json.loads((directory / 'frozen_protocol.json').read_text())
    local = json.loads((DATA / 'protocol.json').read_text())
    if protocol != local or protocol['cohort'] != COHORT:
        raise ValueError('Returned evaluation differs from the frozen mixed protocol')
    prefix = 'data/external_mixed_v4/'
    for relative, digest_expected in protocol['files'].items():
        if relative.startswith(prefix):
            path = DATA / relative[len(prefix):]
        elif relative.startswith('scripts/eval_one_shot_v4/'):
            path = Path('experiments/evaluation') / Path(relative).name
        else:
            raise ValueError('Unexpected frozen input path: ' + relative)
        if formal.sha(path.read_bytes()) != digest_expected:
            raise ValueError('Frozen input changed: ' + relative)
    with (DATA / 'prompts.jsonl').open() as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    formal_test.validate_tasks(tasks, protocol)
    rows, configs = {}, {}
    for label in run_mixed.LABELS:
        values, config = run_mixed.check_model_output(directory / label, label, tasks, protocol)
        for key in ('model', 'model_revision', 'chat_template_sha256'):
            if config[key] != protocol['expected_training'][key]:
                raise ValueError('Unexpected model/tokenizer identity: ' + label + '/' + key)
        by_task = {row['task_id']: row for row in values}
        rows[label] = [by_task[task['task_id']] for task in tasks]
        configs[label] = config
    run_mixed.check_comparison(rows, configs)
    with (DATA / 'designs.jsonl').open() as handle:
        designs = [json.loads(line) for line in handle if line.strip()]
    contexts = load_contexts(tasks, designs)
    verification = {'status': 'PASS', 'cohort': COHORT, 'split': 'test',
                    'checked_utc': datetime.now(timezone.utc).isoformat(),
                    'archive_sha256': digest, 'signals_per_model': len(tasks),
                    'total_saved_responses': sum(len(values) for values in rows.values()),
                    'signals_by_design': protocol['signals_by_design'],
                    'previous_train_validation_and_test_designs_excluded': True,
                    'same_effective_prompts_seeds_token_lengths_and_settings': True,
                    'formal_context_matches_complete_original_inference_prompt': True,
                    'no_generated_assertion_repair_or_outcome_selection': True,
                    'models': {label: {'generation_config_sha256': formal.sha(
                        (directory / label / 'generation_config.json').read_bytes()),
                        'adapter_files': configs[label]['adapter_files'],
                        'selected_checkpoint': protocol['models'][label].get('selected_checkpoint')}
                               for label in run_mixed.LABELS}}
    write_json(out / 'input_verification.json', verification)
    return (tasks, rows, protocol, digest), contexts


def prepare(out):
    if (out / 'manifest.json').exists():
        raise ValueError('Formal snapshot exists; use deploy, sync or watch')
    loaded, contexts = load_returned(out)
    original = json.loads((formal.DEFAULT / 'manifest.json').read_text())
    if formal.sha(WORKER.read_bytes()) != original['worker_sha256']:
        raise ValueError('Original native formal worker changed')
    manifest = formal.prepare(out, original['timeout_seconds'], loaded=loaded,
                              labels=run_mixed.LABELS, remote=REMOTE, worker_source=WORKER,
                              contexts=contexts)
    manifest.update(cohort=COHORT, mixed_preparer_sha256=formal.sha(Path(__file__).read_bytes()),
                    input_verification_sha256=formal.sha((out / 'input_verification.json').read_bytes()),
                    context_source='Frozen delivered design records; no reference assertions required',
                    context_designs_sha256=formal.sha((DATA / 'designs.jsonl').read_bytes()),
                    external_cached_checks=0)
    write_json(out / 'manifest.json', manifest)
    (out / 'formal_mixed.frozen.py').write_bytes(Path(__file__).read_bytes())
    (out / 'formal_preparer.frozen.py').write_bytes(Path(formal.__file__).read_bytes())
    # A separate six-job infrastructure check exercises the exact RTL and clocks
    # with a trivial proof. Its results never enter model statistics.
    pre_tasks, pre_rows, pre_contexts, seen = [], [], {}, set()
    for task in loaded[0]:
        if task['case_id'] in seen:
            continue
        seen.add(task['case_id'])
        context = contexts[task['task_id']]
        ident = 'preflight__' + task['case_id'].replace('/', '__')
        item = {'task_id': ident, 'case_id': task['case_id'], 'target_signal': '__context_preflight__'}
        pre_tasks.append(item)
        clock = context['formal_context']['checker_clocks'][0]
        pre_rows.append(dict(item, response="assert property (@(posedge " + clock + ") 1'b1);", status='preflight'))
        pre_contexts[ident] = context
    formal.prepare(out / 'preflight', original['timeout_seconds'],
                   loaded=(pre_tasks, {'preflight': pre_rows}, {'signal_count': len(pre_tasks)}, loaded[3]),
                   labels=('preflight',), remote=REMOTE + '/preflight', worker_source=WORKER, contexts=pre_contexts)
    return write_comparison(out, formal.summarize(out, scope_resolver=implication_scope))


def deploy_guard(out):
    from experiments.fixme_cohort import connect, command, put
    manifest = json.loads((out / 'manifest.json').read_text())
    remote = manifest['remote']
    source = Path(__file__).with_name('formal_infrastructure_guard.py')
    ssh, sftp = connect()
    try:
        remote_path = remote + '/formal_infrastructure_guard.py'
        put(ssh, sftp, source, remote_path, refuse_changes=True)
        script = '''import json,subprocess
from pathlib import Path
p=Path(ROOT)
saved=p/'guard_launch.json'
if saved.exists():
 old=json.loads(saved.read_text());cmd=Path('/proc')/str(old['pid'])/'cmdline'
 if cmd.exists() and str(p/'formal_infrastructure_guard.py').encode() in cmd.read_bytes():
  print(json.dumps(dict(old,reused=True)));raise SystemExit(0)
log=(p/'infrastructure_guard.log').open('ab',buffering=0)
child=subprocess.Popen(['python3','-u',str(p/'formal_infrastructure_guard.py'),'--root',str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={'pid':child.pid,'root':str(p)}
saved.write_text(json.dumps(result))
print(json.dumps(result))
'''.replace('ROOT', repr(remote))
        result = json.loads(command(ssh, ['python3', '-c', script]))
        write_json(out / 'guard_launch.json', dict(result, script_sha256=formal.sha(source.read_bytes())))
        (out / 'formal_infrastructure_guard.frozen.py').write_bytes(source.read_bytes())
        return result
    finally:
        sftp.close()
        ssh.close()


def preflight_gate(out):
    summary = formal.sync(out / 'preflight')
    jobs = list(map(json.loads, (out / 'preflight/jobs.jsonl').read_text().splitlines()))
    details = []
    for job in jobs:
        path = out / 'preflight/jobs' / job['job_id'] / 'result.json'
        result = json.loads(path.read_text()) if path.exists() else {}
        checked = formal.assess(result.get('states', {}), False)
        details.append({'origin': job['first_origin'], 'job_id': job['job_id'],
                        'assertion_status': checked['assertion_status'] if checked['assertion_status'] != 'missing'
                                             else result.get('assertion_status', 'pending'),
                        'trace_status': checked['trace_status']})
    passed = len(details) == 6 and all(d['assertion_status'] == 'proven' and d['trace_status'] == 'covered' for d in details)
    report = {'status': 'PASS' if passed else 'PENDING' if any(d['assertion_status'] == 'pending' for d in details) else 'FAIL',
              'cases': details, 'worker': summary.get('worker'),
              'preflight_results_excluded_from_model_metrics': True}
    write_json(out / 'preflight_verification.json', report)
    return report


def deploy(out, preflight=False):
    if preflight:
        formal.deploy(out / 'preflight')
        deploy_guard(out / 'preflight')
        return
    report = preflight_gate(out)
    if report['status'] != 'PASS':
        raise ValueError('All six native context preflights must pass before full deployment: ' + report['status'])
    formal.deploy(out)
    deploy_guard(out)


def write_comparison(out, summary):
    manifest = json.loads((out / 'manifest.json').read_text())
    with (out / 'responses.jsonl').open() as handle:
        responses = [json.loads(line) for line in handle]
    result = {'cohort': COHORT, 'split': 'additional_test',
              'analysis_version': summary.get('analysis_version'),
              'scope_policy': 'Read-only property resolution with comments and strings masked; original native inputs and results preserved.',
              'updated_utc': datetime.now(timezone.utc).isoformat(),
              'signals_per_model': manifest['protocol']['signal_count'],
              'same_generation_and_formal_protocol_verified': True,
              'all_models_formally_evaluated': all(m['metrics_final'] for m in summary['models'].values()),
              'counts_include_duplicate_assertion_occurrences': True,
              'models': {label: dict(counts) for label, counts in summary['models'].items()}, 'per_design': {},
              'worker': summary.get('worker'),
              'native_jobs_total': manifest.get('unique_jobs'),
              'native_jobs_completed_locally': sum(summary.get('tool_jobs_reassessed_outcomes', {}).values()),
              'comparison_reader_sha256': formal.sha(Path(__file__).read_bytes()),
              'formal_reader_sha256': formal.sha(Path(formal.__file__).read_bytes()),
              'source': str(out / 'summary.json'), 'manifest_sha256': formal.sha((out / 'manifest.json').read_bytes())}
    for label, counts in result['models'].items():
        counts.update({key: manifest.get('models', {}).get(label, {}).get(key, 0)
                       for key in ('responses', 'offline_errors', 'empty_responses', 'cap_violations')})
    pending = {label: set() for label in run_mixed.LABELS}
    reasons = {label: Counter() for label in run_mixed.LABELS}
    vacuity_reasons = {label: Counter() for label in run_mixed.LABELS}
    with (out / 'results.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            reasons[row['label']][row['reason']] += 1
            if row['assertion_status'] == 'proven' and row.get('vacuous') is True:
                antecedent = 'antecedent_never_true' in row['reason']
                consequent = 'consequent_always_true' in row['reason']
                for key, present in [('antecedent_never_true', antecedent),
                                     ('consequent_always_true', consequent),
                                     ('both_targeted_reasons', antecedent and consequent)]:
                    vacuity_reasons[row['label']][key] += int(present)
            if row['assertion_status'] == 'pending':
                pending[row['label']].add((row['case_id'], row['task_id']))
    result['outcome_reasons_by_model'] = {label: dict(counts) for label, counts in reasons.items()}
    result['vacuity_reason_counts_by_model'] = {label: dict(counts) for label, counts in vacuity_reasons.items()}
    for cid, signal_count in manifest['protocol']['signals_by_design'].items():
        result['per_design'][cid] = {}
        for label in run_mixed.LABELS:
            counts = dict(summary['per_design'].get(cid, {}).get(label, {}))
            n = counts.get('assertion_occurrences', 0)
            unresolved = sum(c == cid for c, _ in pending[label])
            resolved = counts.get('proved_vacuous', 0) + counts.get('proved_nonvacuous', 0)
            counts.update(signals=signal_count, signals_remaining=unresolved,
                          signals_evaluated=signal_count - unresolved, metrics_final=unresolved == 0,
                          proof_success_rate=counts.get('proved', 0) / n if n and not unresolved else None,
                          nonvacuous_yield=counts.get('proved_nonvacuous', 0) / n if n and not unresolved else None,
                          conditional_vacuity_rate=counts.get('proved_vacuous', 0) / resolved if resolved and not unresolved else None,
                          empty_responses=sum(r['label'] == label and r['case_id'] == cid and r['no_assertions'] for r in responses),
                          cap_violations=sum(r['label'] == label and r['case_id'] == cid and r['cap_violation'] for r in responses))
            result['per_design'][cid][label] = counts
    write_json(ROOT / 'comparison_external_mixed_v4.json', result)
    columns = ('design', 'model', 'signals_evaluated', 'signals_remaining', 'assertion_occurrences',
               'proved', 'proof_cex', 'proved_nonvacuous', 'proved_vacuous', 'proved_non_implication',
               'proved_vacuity_unresolved', 'proof_pending', 'proof_success_rate',
               'conditional_vacuity_rate', 'nonvacuous_yield', 'metrics_final', 'empty_responses', 'cap_violations')
    extra_outcomes = sorted({key for counts in result['models'].values() for key in counts
                             if key.startswith('proof_') and key not in columns})
    columns += tuple(extra_outcomes)
    with (ROOT / 'comparison_external_mixed_v4.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for cid, values in [('ALL', result['models'])] + list(result['per_design'].items()):
            for label in run_mixed.LABELS:
                data = values[label]
                writer.writerow({'design': cid, 'model': label,
                                 **{key: data.get(key, None if key.endswith('rate') or key == 'nonvacuous_yield' else 0)
                                    for key in columns if key not in ('design', 'model')}})
    text = ['# Additional test: four OpenTitan and two FIXME designs', '',
            'Status: **' + ('Complete' if result['all_models_formally_evaluated'] else 'In progress; final rates withheld for unfinished populations') + '**.', '',
            'All generated assertion occurrences are counted, including duplicates and failures. '
            'Non-implication properties are outside the two vacuity tests. '
            'A completed signal means its extracted candidates have outcomes; empty responses are reported separately.', '',
            '| Design | Model | Signals complete | Left | Assertions | Proved | CEX | Nonvacuous | Vacuous | Proved NI | Proof success | Conditional vacuity | NV yield |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cid, values in [('ALL', result['models'])] + list(result['per_design'].items()):
        for label in run_mixed.LABELS:
            counts = values[label]
            rates = [('—' if counts.get(key) is None else '{:.2%}'.format(counts[key]))
                     for key in ('proof_success_rate', 'conditional_vacuity_rate', 'nonvacuous_yield')]
            cells = [cid, label] + [str(counts.get(key, 0)) for key in (
                'signals_evaluated', 'signals_remaining', 'assertion_occurrences', 'proved', 'proof_cex',
                'proved_nonvacuous', 'proved_vacuous', 'proved_non_implication')] + rates
            text.append('| ' + ' | '.join(cells) + ' |')
    text += ['', '## Files', '', '- `formal_v1/results.jsonl`: per-assertion results and reasons.',
             '- `formal_v1/jobs/<id>/`: exact instrumented RTL, Tcl, native logs and named statuses.',
             '- `formal_v1/input_verification.json`: inference integrity and model comparisons.',
             '- `formal_v1/preflight_verification.json`: separate native infrastructure checks.',
             '- `comparison_external_mixed_v4.json` and `.csv`: complete counts and comparisons.', '']
    text += ['## Response accounting', '',
             '| Model | Responses | Empty responses | Responses over requested cap | Offline extraction/support errors |',
             '|---|---:|---:|---:|---:|']
    for label, counts in result['models'].items():
        text.append('| ' + ' | '.join([label] + [str(counts[key]) for key in (
            'responses', 'empty_responses', 'cap_violations', 'offline_errors')]) + ' |')
    text += ['', 'Empty responses are not successful formal results. Offline errors remain in assertion counts. '
             'The CSV and JSON retain individual error and unresolved outcome categories.', '']
    (ROOT / 'comparison_external_mixed_v4.md').write_text('\n'.join(text))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'preflight-status', 'deploy', 'sync', 'summary', 'watch'))
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.out)
    elif args.action == 'preflight':
        deploy(args.out, preflight=True)
        return
    elif args.action == 'preflight-status':
        result = preflight_gate(args.out)
    elif args.action == 'deploy':
        deploy(args.out)
        return
    elif args.action == 'watch':
        formal.watch(args.out, args.interval, on_summary=lambda summary: write_comparison(args.out, summary),
                     scope_resolver=implication_scope)
        return
    else:
        summary = (formal.sync(args.out, scope_resolver=implication_scope) if args.action == 'sync'
                   else formal.summarize(args.out, scope_resolver=implication_scope))
        result = write_comparison(args.out, summary)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
