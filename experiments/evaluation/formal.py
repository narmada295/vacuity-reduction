"""Prepare, deploy and summarize the frozen base/SFT one-shot formal evaluation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import tarfile
import time

from experiments.evaluation.generate import inference_messages, sample_seed, write_json
from experiments.evaluation.run_full import validate_outputs
from experiments.evaluation.formal_worker import assess
from experiments.training.prepare_sft import chat_record, minimal_support
from experiments.assertion_extraction.extract_assertions import resolve_property
from experiments.assertion_extraction.remove_exact_duplicates import tokenize
from experiments.assertion_extraction.formal_clocking import clock_commands
from fixme_svagen.sva import code_from_response, insert_in_module, module_name


BASE = Path('output/fixme_svagen/evaluation_one_shot_v4')
DEFAULT = BASE / 'formal_validation_v1'
REMOTE = '/storage/ckarfa/narmada/experiments/evaluation_one_shot_v4/formal_validation_v1'


def sha(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def mask_noncode(code):
    """Mask comments and strings while preserving source offsets."""
    return re.sub(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*',
                  lambda m: ' ' * len(m.group()), code, flags=re.S)


def extract_occurrences(response):
    code = code_from_response(response)
    masked = mask_noncode(code)
    hits = list(re.finditer(r'\bassert\s+property\b', masked))
    spans, items = [], []
    for n, match in enumerate(hits):
        limit = hits[n + 1].start() if n + 1 < len(hits) else len(code)
        prefix = masked[:match.start()]
        label = re.search(r'(?:^|[;\n])\s*([A-Za-z_$][\w$]*)\s*:\s*$', prefix)
        start = label.start(1) if label else match.start()
        opening = re.match(r'\s*\(', masked[match.end():limit])
        error = None
        end = limit
        if not opening:
            error = 'missing_assert_property_parenthesis'
        else:
            pos = match.end() + opening.end()
            depth = 1
            while pos < limit and depth:
                depth += (masked[pos] == '(') - (masked[pos] == ')')
                pos += 1
            if depth:
                error = 'unbalanced_assert_property_parentheses'
            else:
                tail = re.match(r'\s*;', masked[pos:limit])
                if not tail:
                    error = 'missing_assert_property_semicolon_or_extra_parenthesis'
                    # Do not swallow later property declarations into a bad statement.
                    newline = code.find('\n', pos, limit)
                    end = newline if newline != -1 else limit
                else:
                    end = pos + tail.end()
        spans.append((start, end))
        items.append({'assertion': code[match.start():end].strip(),
                      'source_assertion': code[start:end].strip(), 'extraction_error': error})
    remaining = list(code)
    for start, end in spans:
        remaining[start:end] = ' ' * (end - start)
    support = ''.join(remaining)
    policy = re.search(r'\b(?:assume|restrict|force|release|always(?:_ff|_comb|_latch)?|initial|assign|module|endmodule)\b',
                       mask_noncode(support))
    return items, support, 'forbidden_generated_context' if policy else None


def load_returned(out):
    returned = out / 'returned'
    archive = BASE / 'eval_one_shot_v4_full_validation_outputs.tar.gz'
    if not (returned / 'outputs/eval_one_shot_v4/full_validation/frozen_protocol.json').exists():
        returned.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as handle:
            for item in handle.getmembers():
                path = Path(item.name)
                if (path.is_absolute() or '..' in path.parts or not (item.isfile() or item.isdir())
                        or item.size > 25 * 1024 * 1024
                        or not (item.name.startswith('outputs/eval_one_shot_v4/')
                                or item.name.startswith('logs/eval_one_shot_v4/')
                                or item.name == 'logs/eval_one_shot_v4' and item.isdir())):
                    raise ValueError('Unexpected return archive member: ' + item.name)
            handle.extractall(returned, filter='data')
    folder = returned / 'outputs/eval_one_shot_v4/full_validation'
    protocol = json.loads((folder / 'frozen_protocol.json').read_text())
    if protocol != json.loads((BASE / 'full_validation_protocol.json').read_text()):
        raise ValueError('Returned run differs from the frozen full-validation protocol')
    tasks = [json.loads(line) for line in (BASE / 'validation/prompts.jsonl').open()]
    if sha((BASE / 'validation/prompts.jsonl').read_bytes()) != protocol['files'][protocol['prompts']]:
        raise ValueError('Saved inference inputs changed')
    rows, configs = {}, {}
    for label in ('base', 'sft'):
        rows[label], configs[label] = validate_outputs(folder / label, label, tasks, protocol)
        config = configs[label]
        if sha(json.dumps({k: v for k, v in config.items() if k != 'fingerprint'}, sort_keys=True)) != config['fingerprint']:
            raise ValueError('Generation configuration fingerprint is invalid')
        for row in rows[label]:
            path = folder / label / 'attempts' / (row['task_id'] + '__sample_0.json')
            if json.loads(path.read_text()) != row:
                raise ValueError('Aggregate generation differs from its saved attempt')
    for task, base, sft in zip(tasks, rows['base'], rows['sft']):
        for row in (base, sft):
            if row['task_id'] != task['task_id'] or row['seed'] != sample_seed(task['task_id'], 0, protocol['generation']['seed']):
                raise ValueError('Returned task or sampling seed differs')
            expected = sha(json.dumps(inference_messages(task['messages'], 12), sort_keys=True))
            if row['effective_messages_sha256'] != expected:
                raise ValueError('Returned effective prompt differs from the frozen input')
        if base['prompt_tokens'] != sft['prompt_tokens']:
            raise ValueError('Base and SFT received different tokenized inputs')
    return tasks, rows, protocol, sha(archive.read_bytes())


def load_contexts(tasks, reference_map=None):
    task_map = {task['task_id']: task for task in tasks}
    reference_map = reference_map or BASE / 'validation/reference_map.jsonl'
    with reference_map.open() as handle:
        references = {r['task_id']: r['source_example_ids'] for line in handle
                      if (r := json.loads(line)).get('selected') and r['task_id'] in task_map}
    by_example = {ident: task for task, ids in references.items() for ident in ids}
    contexts = {}
    source = Path('output/fixme_svagen/sft_before_reduction_v2_complete/examples.jsonl')
    for line in source.open():
        example = json.loads(line)
        task_id = by_example.get(example['id'])
        if not task_id or task_id in contexts:
            continue
        if chat_record(example)['messages'][:2] != task_map[task_id]['messages']:
            raise ValueError('Formal source context differs from the inference prompt')
        context = example['formal_context']
        clocks = clock_commands(context['checker_clocks'], context.get('formal_clocking'))
        if context['formal_reset_policy'] != 'unconstrained':
            raise ValueError('Unexpected reset assumptions')
        contexts[task_id] = {'rtl': example['rtl'], 'spec': example['spec'], 'formal_context': context,
                            'clock_tcl': clocks, 'top': module_name(example['rtl'], task_map[task_id]['case_id'].split('/')[-1])}
    if set(contexts) != set(task_map):
        raise ValueError('Formal context missing for an inference task')
    return contexts


def prepare(out, timeout, *, loaded=None, labels=('base', 'sft'), remote=REMOTE,
            worker_source=None, reference_map=None, contexts=None):
    if (out / 'manifest.json').exists():
        raise ValueError('Formal snapshot exists; resume/sync it instead of replacing inputs')
    out.mkdir(parents=True, exist_ok=True)
    tasks, rows, protocol, archive_sha = load_returned(out) if loaded is None else loaded
    if set(rows) != set(labels):
        raise ValueError('Formal population labels differ from the validated generations')
    contexts = load_contexts(tasks, reference_map) if contexts is None else contexts
    if set(contexts) != {task['task_id'] for task in tasks}:
        raise ValueError('Formal context population differs from the validated tasks')
    occurrences, responses, jobs = [], [], {}
    worker_bytes = (worker_source or Path(__file__).with_name('formal_worker.py')).read_bytes()
    worker_hash = sha(worker_bytes)
    for index, task in enumerate(tasks):
        for label in labels:
            row = rows[label][index]
            if any(row[key] != task[key] for key in ('task_id', 'case_id', 'target_signal')):
                raise ValueError('Formal response does not match the assigned inference task')
            context = contexts[task['task_id']]
            items, full_support, policy_error = extract_occurrences(row.get('response', ''))
            response = {key: row.get(key) for key in ('task_id', 'case_id', 'target_signal', 'status', 'new_tokens', 'length_limited')}
            response.update(label=label, assertion_occurrences=len(items), cap_violation=len(items) > 12,
                            no_assertions=len(items) == 0, context_policy_error=policy_error)
            responses.append(response)
            for number, item in enumerate(items, 1):
                entry = dict(item, label=label, task_id=task['task_id'], case_id=task['case_id'],
                    target_signal=task['target_signal'], occurrence=number,
                    occurrence_id=label + '__' + task['task_id'] + '__' + str(number), job_id=None)
                error = item['extraction_error'] or policy_error
                support = full_support
                expression, implication, resolution = None, None, None
                if not error:
                    try:
                        # Drop only unrelated declarations; never repair assertion bodies or names.
                        support = minimal_support({'assertion': item['assertion'], 'support_code': full_support})
                    except ValueError as exc:
                        error = 'unsupported_shared_support: ' + str(exc)
                if not error:
                    try:
                        _, expression, _, resolution, _ = resolve_property({'assertion': item['assertion'], 'support_code': support})
                        if resolution == 'ok':
                            implication = any(t in ('|->', '|=>') for t in tokenize(expression))
                    except ValueError:
                        resolution = 'unresolved'
                    # Named unresolved references still go to JasperGold, unchanged.
                    clock_lines = context['clock_tcl']
                    first_clock = context['formal_context']['checker_clocks']
                    event = '@(posedge ' + first_clock[0] + ') ' if first_clock else ''
                    snippet = support + '\neval_assert: ' + item['assertion'] + '\n'
                    snippet += 'eval_trace: cover property (' + event + "1'b1);\n"
                    dut = insert_in_module(context['rtl'], context['top'], snippet)
                    tcl = '\n'.join(['analyze -sv12 dut.sv',
                        'elaborate -top ' + context['top'] + ' -create_related_covers {precondition contrapositive}',
                        clock_lines, 'reset -none', 'prove -all', 'report -summary -force -result',
                        'set fh [open "status.tsv" w]', 'foreach p [lsort [get_property_list]] {',
                        '  if {[catch {get_property_info -list status $p} state]} {set state QUERY_ERROR}',
                        '  puts $fh "$p\\t$state"', '}', 'close $fh', 'exit -force', ''])
                    identity = {'dut_sha256': sha(dut), 'tcl_sha256': sha(tcl), 'timeout': timeout,
                                'worker_sha256': worker_hash, 'implication': implication}
                    fingerprint = sha(json.dumps(identity, sort_keys=True))
                    job_id = fingerprint[:24]
                    entry.update(job_id=job_id, implication=implication, resolution=resolution,
                                 selected_support=support)
                    if job_id not in jobs:
                        directory = out / 'jobs' / job_id
                        directory.mkdir(parents=True)
                        (directory / 'dut.sv').write_text(dut)
                        (directory / 'run.tcl').write_text(tcl)
                        jobs[job_id] = {'job_id': job_id, 'fingerprint': fingerprint,
                            'implication': implication, 'files': {'dut.sv': sha(dut), 'run.tcl': sha(tcl)},
                            'first_origin': label + ' ' + task['case_id'] + ' ' + task['target_signal'],
                            'occurrences': 0}
                    jobs[job_id]['occurrences'] += 1
                else:
                    entry.update(offline_outcome='syntax_extraction_error' if item['extraction_error'] else 'output_context_error',
                                 offline_reason=error)
                occurrences.append(entry)
    for name, values in (('occurrences.jsonl', occurrences), ('responses.jsonl', responses), ('jobs.jsonl', jobs.values())):
        with (out / name).open('w') as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + '\n')
    manifest = {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_archive_sha256': archive_sha, 'protocol': protocol, 'remote': remote,
        'worker_sha256': worker_hash, 'preparer_sha256': sha(Path(__file__).read_bytes()),
        'timeout_seconds': timeout, 'reset_policy': 'unconstrained',
        'clock_policy': 'Exact recorded task clock declarations, never inferred from model outputs.',
        'support_policy': 'Original required property/parameter declarations only; unrelated declarations dropped, assertion bodies and identifiers never repaired.',
        'vacuity_policy': 'Proved implications: any unreachable related cover establishes the corresponding targeted reason; every precondition/contrapositive cover must be covered for a nonvacuous label; trace cover required.',
        'unique_jobs': len(jobs), 'jobs_sha256': sha((out / 'jobs.jsonl').read_bytes()),
        'models': {label: {'responses': sum(r['label'] == label for r in responses),
            'assertion_occurrences': sum(r['label'] == label for r in occurrences),
            'offline_errors': sum(r['label'] == label and r['job_id'] is None for r in occurrences),
            'empty_responses': sum(r['label'] == label and r['no_assertions'] for r in responses),
            'cap_violations': sum(r['label'] == label and r['cap_violation'] for r in responses)} for label in labels}}
    write_json(out / 'manifest.json', manifest)
    (out / 'formal_worker.frozen.py').write_bytes(worker_bytes)
    return manifest


def deploy(out):
    from experiments.fixme_cohort import connect, command, put
    manifest = json.loads((out / 'manifest.json').read_text())
    remote = manifest['remote']
    payload = out / 'formal_payload.tar.gz'
    with tarfile.open(payload, 'w:gz') as archive:
        for name in ('manifest.json', 'jobs.jsonl'):
            archive.add(out / name, arcname=name)
        frozen_worker = out / 'formal_worker.frozen.py'
        if sha(frozen_worker.read_bytes()) != manifest['worker_sha256']:
            raise ValueError('Frozen worker source differs from the formal snapshot')
        archive.add(frozen_worker, arcname='formal_worker.py')
        for line in (out / 'jobs.jsonl').open():
            job = json.loads(line)
            for name in ('dut.sv', 'run.tcl'):
                path = Path('jobs') / job['job_id'] / name
                archive.add(out / path, arcname=str(path))
            # A new population can reuse a previously completed native check
            # only when its complete input fingerprint and log checksum match.
            folder = out / 'jobs' / job['job_id']
            if (folder / 'result.json').exists():
                result = json.loads((folder / 'result.json').read_text())
                if (result['job_fingerprint'] != job['fingerprint']
                        or sha((folder / 'jg.log').read_bytes()) != result['jg_log_sha256']):
                    raise ValueError('Cached native result differs from the formal input or log')
                for name in ('result.json', 'jg.log', 'status.tsv'):
                    if (folder / name).exists():
                        archive.add(folder / name, arcname=str(Path('jobs') / job['job_id'] / name))
    ssh, sftp = connect()
    try:
        put(ssh, sftp, payload, remote + '/payload.tar.gz')
        launcher = '''import hashlib,json,subprocess,tarfile
from pathlib import Path
p=Path(ROOT)
manifest=p/'manifest.json'
if manifest.exists():
 if hashlib.sha256(manifest.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('Different existing formal snapshot; preserve it')
 status=p/'status.json'
 if status.exists():
  state=json.loads(status.read_text());pid=state.get('pid',0)
  process=Path('/proc')/str(pid)/'cmdline'
  if state.get('state')=='complete' or process.exists() and str(p/'formal_worker.py').encode() in process.read_bytes():
   print(json.dumps({'pid':pid,'remote':str(p),'reused_existing_worker':True}));raise SystemExit(0)
with tarfile.open(str(p/'payload.tar.gz')) as a:a.extractall(str(p))
command=['csh','-c',SHELL]
log=(p/'worker.log').open('ab',buffering=0)
child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':child.pid,'remote':str(p)}))
'''.replace('ROOT', repr(remote)).replace('EXPECTED', repr(sha((out / 'manifest.json').read_bytes()))).replace('SHELL', repr('source ~/cad_cshrc; exec python3 -u '
            + shlex.quote(remote + '/formal_worker.py') + ' --root ' + shlex.quote(remote)))
        result = command(ssh, ['python3', '-c', launcher])
        write_json(out / 'launch.json', json.loads(result))
        print(result, flush=True)
    finally:
        sftp.close()
        ssh.close()


def sync(out, *, scope_resolver=None):
    from experiments.fixme_cohort import connect, command
    manifest = json.loads((out / 'manifest.json').read_text())
    ssh, sftp = connect()
    try:
        remote = manifest['remote']
        for name in ('status.json', 'worker.log'):
            try:
                with sftp.open(remote + '/' + name, 'rb') as handle:
                    (out / name).write_bytes(handle.read())
            except OSError:
                pass
        completed_remote = set(json.loads(command(ssh, ['python3', '-c',
            'import json; from pathlib import Path; p=Path(' + repr(remote)
            + '); print(json.dumps([f.parent.name for f in (p/"jobs").glob("*/result.json")]))'])))
        for line in (out / 'jobs.jsonl').open():
            job = json.loads(line)
            folder = out / 'jobs' / job['job_id']
            if (folder / 'result.json').exists() or job['job_id'] not in completed_remote:
                continue
            try:
                with sftp.open(remote + '/jobs/' + job['job_id'] + '/result.json', 'rb') as handle:
                    result = json.loads(handle.read().decode())
                if result['job_fingerprint'] != job['fingerprint']:
                    raise ValueError('Remote result has a different job fingerprint')
                for name in ('jg.log', 'status.tsv'):
                    try:
                        with sftp.open(remote + '/jobs/' + job['job_id'] + '/' + name, 'rb') as handle:
                            (folder / name).write_bytes(handle.read())
                    except OSError:
                        pass
                if sha((folder / 'jg.log').read_bytes()) != result['jg_log_sha256']:
                    raise ValueError('Remote proof log hash mismatch')
                write_json(folder / 'result.json', result)
            except OSError:
                pass
    finally:
        sftp.close()
        ssh.close()
    return summarize(out, scope_resolver=scope_resolver)


def summarize(out, *, scope_resolver=None):
    manifest = json.loads((out / 'manifest.json').read_text())
    labels = tuple(manifest['models'])
    signal_count = manifest['protocol']['signal_count']
    totals, per_design = {}, {}
    pending_signals = {label: set() for label in labels}
    reanalyzed_jobs = {}
    for label in labels:
        totals[label] = Counter()
    results_temp = out / 'results.jsonl.tmp'
    with results_temp.open('w') as handle, (out / 'occurrences.jsonl').open() as source:
        for line in source:
            item = json.loads(line)
            result_path = out / 'jobs' / str(item['job_id']) / 'result.json'
            if item['job_id'] is None:
                item.update(assertion_status=item['offline_outcome'], vacuous=None, reason=item['offline_reason'])
            elif result_path.exists():
                raw = json.loads(result_path.read_text())
                job_id = item['job_id']
                if job_id not in reanalyzed_jobs:
                    scope = item.get('implication') if scope_resolver is None else scope_resolver(item)
                    checked = assess(raw['states'], scope)
                    interpreted = dict(raw)
                    if checked['assertion_status'] != 'missing':
                        interpreted.update(checked)
                    if interpreted['assertion_status'] != 'proven' and interpreted['reason'] == 'outside_scope_non_implication':
                        interpreted['reason'] = ('assertion_falsified' if interpreted['assertion_status'] == 'cex'
                                                 else 'formal_inconclusive_or_error')
                    interpreted.update(analysis_version=2,
                        original_result_status=raw['assertion_status'], original_result_reason=raw['reason'])
                    if scope_resolver is not None:
                        interpreted.update(analysis_version=3, interpreted_implication=scope,
                                           native_input_implication=item.get('implication'))
                    reanalyzed_jobs[job_id] = interpreted
                item.update(reanalyzed_jobs[job_id])
            else:
                item.update(assertion_status='pending', vacuous=None, reason='formal_pending')
                pending_signals[item['label']].add(item['task_id'])
            counts = totals[item['label']]
            design = per_design.setdefault(item['case_id'], {}).setdefault(item['label'], Counter())
            for c in (counts, design):
                c['assertion_occurrences'] += 1
                c['proof_' + item['assertion_status']] += 1
                if item['assertion_status'] == 'proven':
                    c['proved'] += 1
                    if item['vacuous'] is True:
                        c['proved_vacuous'] += 1
                    elif item['vacuous'] is False:
                        c['proved_nonvacuous'] += 1
                    elif item['reason'] == 'outside_scope_non_implication':
                        c['proved_non_implication'] += 1
                    else:
                        c['proved_vacuity_unresolved'] += 1
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    results_temp.replace(out / 'results.jsonl')
    for label, counts in totals.items():
        counts['signals_remaining'] = len(pending_signals[label])
        counts['signals_evaluated'] = signal_count - counts['signals_remaining']
        n = counts['assertion_occurrences']
        counts['metrics_final'] = counts['proof_pending'] == 0
        counts['proof_success_rate'] = counts['proved'] / n if n and counts['metrics_final'] else None
        counts['nonvacuous_yield'] = counts['proved_nonvacuous'] / n if n and counts['metrics_final'] else None
        resolved = counts['proved_nonvacuous'] + counts['proved_vacuous']
        counts['conditional_vacuity_rate'] = counts['proved_vacuous'] / resolved if resolved and counts['metrics_final'] else None
    summary = {'analysis_version': 2 if scope_resolver is None else 3,
               'result_policy': 'Canonicalize <embedded>:: property aliases when assessing saved named results; original native results and logs preserved.',
               'tool_jobs_reassessed_outcomes': dict(Counter(r['assertion_status'] for r in reanalyzed_jobs.values())),
               'models': {k: dict(v) for k, v in totals.items()},
               'per_design': {cid: {k: dict(v) for k, v in models.items()} for cid, models in per_design.items()}}
    status_path = out / 'status.json'
    if status_path.exists():
        summary['worker'] = json.loads(status_path.read_text())
        summary['worker']['original_worker_outcomes'] = summary['worker'].pop('outcomes', {})
        summary['worker']['outcomes'] = summary['tool_jobs_reassessed_outcomes']
        summary['worker']['outcomes_analysis_version'] = summary['analysis_version']
    write_json(out / 'summary.json', summary)
    return summary


def watch(out, interval, *, on_summary=None, scope_resolver=None):
    import fcntl
    lock = (out / 'sync.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            result = sync(out, scope_resolver=scope_resolver)
            if on_summary is not None:
                on_summary(result)
            print(json.dumps({'analysis_version': result['analysis_version'],
                              'worker': result.get('worker'), 'models': result['models']}), flush=True)
            if result.get('worker', {}).get('state') == 'complete':
                return
        except Exception as error:
            print('Synchronization will retry:', type(error).__name__, str(error), flush=True)
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'deploy', 'sync', 'summary', 'watch'))
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.out, args.timeout)
    elif args.action == 'deploy':
        deploy(args.out)
        return
    elif args.action == 'sync':
        result = sync(args.out)
    elif args.action == 'watch':
        watch(args.out, args.interval)
        return
    else:
        result = summarize(args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
