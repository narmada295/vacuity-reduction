"""Import all frozen TEST responses and evaluate base/SFT/KTO with JasperGold."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile

from experiments.evaluation import formal, run_test
from experiments.evaluation.generate import write_json


BASE = formal.BASE
DEFAULT = BASE / 'formal_test_v1'
ARCHIVE = BASE / 'eval_one_shot_v4_full_test_outputs.tar.gz'
REMOTE = '/storage/ckarfa/narmada/experiments/evaluation_one_shot_v4/formal_test_v1'
GENERATIONS = Path('outputs/eval_one_shot_v4/full_test')
ALLOWED_ROOTS = (str(GENERATIONS), 'logs/eval_one_shot_v4/test')


def extract_returned(archive, destination, *, allowed_roots=None):
    digest = formal.sha(archive.read_bytes())
    marker = destination / 'archive_sha256.json'
    if marker.exists():
        if json.loads(marker.read_text())['sha256'] != digest:
            raise ValueError('TEST return archive changed after extraction')
        return digest
    destination.mkdir(parents=True, exist_ok=True)
    allowed_roots = ALLOWED_ROOTS if allowed_roots is None else allowed_roots
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        if len(members) > 10000 or sum(item.size for item in members) > 512 * 1024 * 1024:
            raise ValueError('Unexpected TEST archive size')
        names = set()
        for item in members:
            path = Path(item.name)
            if (path.is_absolute() or '..' in path.parts or str(path) in names
                    or not (item.isfile() or item.isdir()) or item.size > 25 * 1024 * 1024
                    or not any(str(path) == root or str(path).startswith(root + '/')
                               for root in allowed_roots)):
                raise ValueError('Unexpected TEST archive member: ' + item.name)
            names.add(str(path))
        handle.extractall(destination, filter='data')
    write_json(marker, {'sha256': digest})
    return digest


def validate_tasks(tasks, protocol):
    if protocol['split'] != 'test' or tuple(protocol['models']) != run_test.LABELS:
        raise ValueError('Formal TEST evaluation requires the frozen three-model TEST protocol')
    if (len(tasks) != protocol['signal_count']
            or len({t['task_id'] for t in tasks}) != len(tasks)
            or len({(t['case_id'], t['target_signal']) for t in tasks}) != len(tasks)
            or dict(Counter(t['case_id'] for t in tasks)) != protocol['signals_by_design']):
        raise ValueError('TEST signal population differs from the frozen protocol')
    designs = {t['case_id'] for t in tasks}
    if designs != set(protocol['test_designs']) or designs & set(protocol['excluded_designs']):
        raise ValueError('TEST design membership differs or overlaps training/validation')
    if any([m['role'] for m in t['messages']] != ['system', 'user'] for t in tasks):
        raise ValueError('TEST inference input contains target responses or unexpected roles')


def load_returned(out):
    archive_sha = extract_returned(ARCHIVE, out / 'returned')
    folder = out / 'returned' / GENERATIONS
    protocol = json.loads((folder / 'frozen_protocol.json').read_text())
    if protocol != json.loads((BASE / 'full_test_protocol.json').read_text()):
        raise ValueError('Returned TEST protocol differs from the frozen local protocol')
    inputs = BASE / 'test'
    if formal.sha((inputs / 'prompts.jsonl').read_bytes()) != protocol['files'][protocol['prompts']]:
        raise ValueError('Frozen TEST inference inputs changed')
    if formal.sha((inputs / 'manifest.json').read_bytes()) != protocol['files']['data/eval_one_shot_v4/test/manifest.json']:
        raise ValueError('Frozen TEST input manifest changed')
    input_manifest = json.loads((inputs / 'manifest.json').read_text())
    if formal.sha((inputs / 'reference_map.jsonl').read_bytes()) != input_manifest['files']['reference_map.jsonl']:
        raise ValueError('TEST formal-context reference map changed')
    with (inputs / 'prompts.jsonl').open() as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    validate_tasks(tasks, protocol)
    rows, configs = {}, {}
    for label in run_test.LABELS:
        values, config = run_test.check_model_output(folder / label, label, tasks, protocol)
        for name in ('model', 'model_revision', 'chat_template_sha256'):
            if config[name] != protocol['expected_training'][name]:
                raise ValueError('TEST model/tokenizer identity changed: ' + label + '/' + name)
        by_task = {row['task_id']: row for row in values}
        rows[label] = [by_task[task['task_id']] for task in tasks]
        configs[label] = config
    run_test.check_comparison(rows, configs)
    # TEST changes the held-out tasks, keeping the completed validation settings.
    original = formal.DEFAULT / 'returned/outputs/eval_one_shot_v4/full_validation/base/generation_config.json'
    baseline = json.loads(original.read_text())
    for key in run_test.COMMON_SETTINGS:
        if key not in ('prompts_sha256', 'prompt_count') and configs['base'][key] != baseline[key]:
            raise ValueError('TEST generation settings differ from validation: ' + key)
    verification = {'status': 'PASS', 'split': 'test',
                    'checked_utc': datetime.now(timezone.utc).isoformat(),
                    'archive_sha256': archive_sha, 'signals_per_model': len(tasks),
                    'total_saved_responses': sum(len(values) for values in rows.values()),
                    'signals_by_design': dict(Counter(t['case_id'] for t in tasks)),
                    'training_and_validation_designs_excluded': True,
                    'same_effective_prompts_seeds_token_lengths_and_settings': True,
                    'generation_settings_match_completed_validation': True,
                    'no_generated_assertion_repair_or_outcome_selection': True,
                    'models': {label: {'generation_config_sha256': formal.sha(
                        (folder / label / 'generation_config.json').read_bytes()),
                        'adapter_files': configs[label]['adapter_files'],
                        'selected_checkpoint': protocol['models'][label].get('selected_checkpoint')}
                               for label in run_test.LABELS}}
    write_json(out / 'input_verification.json', verification)
    return tasks, rows, protocol, archive_sha


def prepare(out):
    if (out / 'manifest.json').exists():
        raise ValueError('TEST formal snapshot exists; use deploy, sync or watch to resume')
    loaded = load_returned(out)
    original = json.loads((formal.DEFAULT / 'manifest.json').read_text())
    worker = formal.DEFAULT / 'formal_worker.frozen.py'
    if formal.sha(worker.read_bytes()) != original['worker_sha256']:
        raise ValueError('Original native worker changed since validation')
    manifest = formal.prepare(out, original['timeout_seconds'], loaded=loaded,
                              labels=run_test.LABELS, remote=REMOTE, worker_source=worker,
                              reference_map=BASE / 'test/reference_map.jsonl')
    manifest.update(test_preparer_sha256=formal.sha(Path(__file__).read_bytes()),
                    input_verification_sha256=formal.sha((out / 'input_verification.json').read_bytes()),
                    reference_map_sha256=formal.sha((BASE / 'test/reference_map.jsonl').read_bytes()),
                    formal_context_split='test', external_cached_checks=0)
    write_json(out / 'manifest.json', manifest)
    (out / 'formal_test.frozen.py').write_bytes(Path(__file__).read_bytes())
    (out / 'formal_preparer.frozen.py').write_bytes(Path(formal.__file__).read_bytes())
    write_comparison(out, formal.summarize(out))
    return manifest


def write_comparison(out, summary):
    manifest = json.loads((out / 'manifest.json').read_text())
    result = {'split': 'test', 'signals_per_model': manifest['protocol']['signal_count'],
              'updated_utc': datetime.now(timezone.utc).isoformat(),
              'same_generation_and_formal_protocol_verified': True,
              'all_models_formally_evaluated': all(m['metrics_final'] for m in summary['models'].values()),
              'counts_include_duplicate_assertion_occurrences': True,
              'models': summary['models'], 'per_design': summary['per_design'],
              'source': str(out / 'summary.json'),
              'manifest_sha256': formal.sha((out / 'manifest.json').read_bytes())}
    write_json(BASE / 'comparison_test_v4.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'deploy', 'sync', 'summary', 'watch'))
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--interval', type=int, default=300)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.out)
    elif args.action == 'deploy':
        formal.deploy(args.out)
        return
    elif args.action == 'watch':
        formal.watch(args.out, args.interval, on_summary=lambda summary: write_comparison(args.out, summary))
        return
    else:
        result = formal.sync(args.out) if args.action == 'sync' else formal.summarize(args.out)
        write_comparison(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
