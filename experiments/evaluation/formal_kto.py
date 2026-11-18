"""Validate the returned KTO run and check it under the frozen v4 formal protocol."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tarfile

from experiments.evaluation import formal
from experiments.evaluation.generate import inference_messages, sample_seed, write_json
from experiments.evaluation.run_full import validate_outputs


BASE = formal.BASE
DEFAULT = BASE / 'formal_kto_validation_v1'
ARCHIVE = BASE / 'eval_one_shot_v4_kto_validation_outputs.tar.gz'
REMOTE = '/storage/ckarfa/narmada/experiments/evaluation_one_shot_v4/formal_kto_validation_v1'
TRAINING = Path('outputs/qwen3_4b_kto_v1_train')
GENERATIONS = Path('outputs/eval_one_shot_v4/full_validation/kto')
RECORDS = Path('output/fixme_svagen/training_results/qwen3_4b_kto_v1')
ALLOWED_ROOTS = (str(TRAINING), str(GENERATIONS), 'scripts/kto_v1',
                 'logs/kto_v1_train.log', 'logs/kto_v1_profile/summary.json',
                 'logs/eval_one_shot_v4/full_kto.log')


def extract_returned(archive, destination):
    archive_sha = formal.sha(archive.read_bytes())
    marker = destination / 'archive_sha256.json'
    if marker.exists():
        if json.loads(marker.read_text())['sha256'] != archive_sha:
            raise ValueError('Return archive changed after extraction')
        return archive_sha
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        if len(members) > 10000 or sum(item.size for item in members) > 512 * 1024 * 1024:
            raise ValueError('Unexpected archive size')
        names = set()
        for item in members:
            path = Path(item.name)
            if (path.is_absolute() or '..' in path.parts or item.name in names
                    or not (item.isfile() or item.isdir()) or item.size > 25 * 1024 * 1024
                    or not any(item.name == root or item.name.startswith(root + '/')
                               for root in ALLOWED_ROOTS)):
                raise ValueError('Unexpected return archive member: ' + item.name)
            names.add(item.name)
        handle.extractall(destination, filter='data')
    write_json(marker, {'sha256': archive_sha})
    return archive_sha


def compare_generation(config, baseline, rows, baseline_rows, tasks, folder, protocol):
    differences_allowed = {'label', 'adapter', 'adapter_files', 'fingerprint'}
    if ({k: v for k, v in config.items() if k not in differences_allowed}
            != {k: v for k, v in baseline.items() if k not in differences_allowed}):
        raise ValueError('KTO generation settings differ from the frozen base/SFT run')
    if formal.sha(json.dumps({k: v for k, v in config.items() if k != 'fingerprint'},
                             sort_keys=True)) != config['fingerprint']:
        raise ValueError('Invalid KTO generation configuration fingerprint')
    by_task = {row['task_id']: row for row in rows}
    baseline_by_task = {row['task_id']: row for row in baseline_rows}
    for task in tasks:
        ident = task['task_id']
        row = by_task[ident]
        original = baseline_by_task[ident]
        if row['seed'] != sample_seed(ident, 0, protocol['generation']['seed']):
            raise ValueError('KTO sampling seed differs for ' + ident)
        expected = formal.sha(json.dumps(inference_messages(task['messages'], 12), sort_keys=True))
        if (row['effective_messages_sha256'] != expected
                or row['effective_messages_sha256'] != original['effective_messages_sha256']
                or row['prompt_tokens'] != original['prompt_tokens']):
            raise ValueError('KTO input differs for ' + ident)
        attempt = folder / 'attempts' / (ident + '__sample_0.json')
        if json.loads(attempt.read_text()) != row:
            raise ValueError('KTO aggregate generation differs from its saved attempt')
    return [by_task[task['task_id']] for task in tasks]


def verify_training(returned, generation_config, sft_config):
    run = returned / TRAINING
    result = json.loads((run / 'result.json').read_text())
    config = json.loads((run / 'run_config.json').read_text())
    identity = config['identity']
    if (result['status'] != 'PASS' or result['mode'] != 'train' or result['global_step'] != 710
            or not result['policy_changed'] or not result['frozen_reference_unchanged']):
        raise ValueError('KTO training did not complete the planned run')
    if formal.sha(json.dumps({k: v for k, v in config.items() if k != 'run_fingerprint'},
                             sort_keys=True)) != config['run_fingerprint']:
        raise ValueError('KTO run configuration fingerprint is invalid')
    for key in ('model', 'model_revision', 'chat_template_sha256'):
        if identity[key] != generation_config[key]:
            raise ValueError('Generation model differs from KTO training: ' + key)
    if (identity['sft_run_config_sha256'] != sft_config['training_config_sha256']
            or identity['sft_adapter_files'] != sft_config['adapter_files']):
        raise ValueError('KTO was initialized from a different SFT model')
    if formal.sha((returned / 'scripts/kto_v1/train_kto.py').read_bytes()) != identity['script_sha256']:
        raise ValueError('Archived original training runner changed')
    if formal.sha((run / 'final_adapter/adapter_config.json').read_bytes()) != generation_config['adapter_files']['adapter_config.json']:
        raise ValueError('Generation adapter configuration differs from the final adapter')
    if identity['packages'] != {**identity['packages'], **generation_config['packages']}:
        raise ValueError('Training and generation packages differ')
    data = Path('output/fixme_svagen/kto_before_reduction_v2')
    if formal.sha((data / 'manifest.json').read_bytes()) != identity['manifest_sha256']:
        raise ValueError('KTO data manifest differs from the prepared dataset')
    for split, expected in identity['data_sha256'].items():
        if formal.sha((data / (split + '.jsonl')).read_bytes()) != expected:
            raise ValueError('KTO data differs: ' + split)
    best = Path(result['best_checkpoint'])
    if best.parent != TRAINING or best.name != 'checkpoint-710':
        raise ValueError('Unexpected selected checkpoint path')
    if generation_config['adapter'] != str(TRAINING / 'final_adapter'):
        raise ValueError('Generation did not use the selected KTO adapter path')
    state = json.loads((returned / best / 'trainer_state.json').read_text())
    marker = json.loads((returned / best / 'checkpoint_complete.json').read_text())
    if (state['global_step'] != 710 or state['best_model_checkpoint'] != str(best)
            or state['best_metric'] != result['best_validation_loss']
            or marker['global_step'] != 710 or marker['run_fingerprint'] != config['run_fingerprint']):
        raise ValueError('Selected checkpoint records disagree with the final result')
    audits = sorted(run.glob('resume_recovery_*.json'))
    if not audits:
        raise ValueError('The resumed run is missing its recovery provenance')
    for audit in audits:
        record = json.loads(audit.read_text())
        if (record['run_fingerprint'] != config['run_fingerprint']
                or record['original_runner_sha256'] != identity['script_sha256']
                or record['frozen_reference_sha256'] != marker['frozen_reference_sha256']
                or record['recovery_script_sha256'] != formal.sha(
                    (returned / 'scripts/kto_v1/resume_kto_verified.py').read_bytes())):
            raise ValueError('Recovery audit differs from the completed run')
    records = [run / 'result.json', run / 'run_config.json', returned / best / 'trainer_state.json',
               returned / best / 'checkpoint_complete.json', run / 'final_adapter/adapter_config.json', *audits]
    RECORDS.mkdir(parents=True, exist_ok=True)
    archived = {}
    for path in records:
        name = path.name if path.name != 'adapter_config.json' else 'final_adapter_config.json'
        destination = RECORDS / name
        if destination.exists() and destination.read_bytes() != path.read_bytes():
            raise ValueError('Preserve existing different training record: ' + str(destination))
        shutil.copyfile(path, destination)
        archived[name] = formal.sha(path.read_bytes())
    provenance = {'source_archive': str(ARCHIVE), 'source_archive_sha256': formal.sha(ARCHIVE.read_bytes()),
                  'files': archived, 'generation_adapter_files': generation_config['adapter_files'],
                  'adapter_weight_bytes_in_return_archive': False,
                  'adapter_identity_evidence': 'Weight file hash recorded by the generation runner; checkpoint tensor marker retained separately.',
                  'validation_history': [row for row in state['log_history'] if 'eval_loss' in row]}
    write_json(RECORDS / 'provenance.json', provenance)
    return provenance


def load_returned(out):
    archive_sha = extract_returned(ARCHIVE, out / 'returned')
    tasks, baseline_rows, protocol, _ = formal.load_returned(formal.DEFAULT)
    folder = out / 'returned' / GENERATIONS
    rows, config = validate_outputs(folder, 'kto', tasks, protocol)
    original_folder = formal.DEFAULT / 'returned/outputs/eval_one_shot_v4/full_validation'
    original_configs = {label: json.loads((original_folder / label / 'generation_config.json').read_text())
                        for label in ('base', 'sft')}
    for label, original in original_configs.items():
        ordered = compare_generation(config, original, rows, baseline_rows[label], tasks, folder, protocol)
    provenance = verify_training(out / 'returned', config, original_configs['sft'])
    verification = {'status': 'PASS', 'checked_utc': datetime.now(timezone.utc).isoformat(),
                    'archive_sha256': archive_sha, 'signals': len(tasks),
                    'signals_by_design': dict(Counter(task['case_id'] for task in tasks)),
                    'same_effective_prompts_seeds_token_lengths_and_generation_settings': True,
                    'training_checkpoint_records_verified': True,
                    'generation_adapter_files': config['adapter_files'],
                    'validation_history': provenance['validation_history'],
                    'test_designs_used': False}
    write_json(out / 'input_verification.json', verification)
    return tasks, {'kto': ordered}, protocol, archive_sha


def reuse_completed(out, source=formal.DEFAULT):
    manifest = json.loads((out / 'manifest.json').read_text())
    original = json.loads((source / 'manifest.json').read_text())
    if (manifest['worker_sha256'] != original['worker_sha256']
            or manifest['timeout_seconds'] != original['timeout_seconds']):
        raise ValueError('Native worker or budget differs; cached results cannot be reused')
    originals = {job['job_id']: job for line in (source / 'jobs.jsonl').read_text().splitlines()
                 if (job := json.loads(line))}
    reused = []
    for line in (out / 'jobs.jsonl').read_text().splitlines():
        job = json.loads(line)
        prior = originals.get(job['job_id'])
        if prior is None:
            continue
        old = source / 'jobs' / job['job_id']
        target = out / 'jobs' / job['job_id']
        result_path = old / 'result.json'
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        if (prior['fingerprint'] != job['fingerprint'] or prior['files'] != job['files']
                or prior['implication'] is not job['implication']
                or result['job_fingerprint'] != job['fingerprint']
                or formal.sha((old / 'jg.log').read_bytes()) != result['jg_log_sha256']):
            raise ValueError('Exact cached check differs from the new formal job')
        for name, expected in job['files'].items():
            if (formal.sha((old / name).read_bytes()) != expected
                    or formal.sha((target / name).read_bytes()) != expected):
                raise ValueError('Instrumented RTL or Tcl changed in a cached job')
        for name in ('result.json', 'jg.log', 'status.tsv'):
            if (old / name).exists():
                shutil.copyfile(old / name, target / name)
        reused.append({'job_id': job['job_id'], 'source': str(old),
                       'result_sha256': formal.sha(result_path.read_bytes())})
    report = {'source': str(source), 'reused_jobs': len(reused), 'jobs': reused,
              'policy': 'Exact complete native input/worker/budget fingerprints only; every KTO occurrence remains in the dataset.'}
    write_json(out / 'cached_checks.json', report)
    manifest['cached_jobs'] = len(reused)
    manifest['new_native_jobs'] = manifest['unique_jobs'] - len(reused)
    manifest['kto_preparer_sha256'] = formal.sha(Path(__file__).read_bytes())
    manifest['input_verification_sha256'] = formal.sha((out / 'input_verification.json').read_bytes())
    manifest['cached_checks_sha256'] = formal.sha((out / 'cached_checks.json').read_bytes())
    write_json(out / 'manifest.json', manifest)
    (out / 'formal_kto.frozen.py').write_bytes(Path(__file__).read_bytes())
    return manifest


def prepare(out):
    if (out / 'manifest.json').exists():
        raise ValueError('KTO formal snapshot exists; use deploy, sync or watch to resume it')
    loaded = load_returned(out)
    original = json.loads((formal.DEFAULT / 'manifest.json').read_text())
    # Keep proof inputs, native worker and budget identical to the completed comparison.
    formal.prepare(out, original['timeout_seconds'], loaded=loaded, labels=('kto',), remote=REMOTE,
                   worker_source=formal.DEFAULT / 'formal_worker.frozen.py')
    return reuse_completed(out)


def write_comparison(out, result):
    baseline = json.loads((formal.DEFAULT / 'summary.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    models = {label: baseline['models'][label] for label in ('base', 'sft')}
    models.update(result['models'])
    per_design = json.loads(json.dumps(baseline['per_design']))
    for case_id, data in result['per_design'].items():
        per_design.setdefault(case_id, {}).update(data)
    comparison = {'split': 'validation', 'signals_per_model': manifest['protocol']['signal_count'],
                  'updated_utc': datetime.now(timezone.utc).isoformat(),
                  'same_generation_and_formal_protocol_verified': True,
                  'all_models_formally_evaluated': all(model['metrics_final'] for model in models.values()),
                  'counts_include_duplicate_assertion_occurrences': True,
                  'models': models, 'per_design': per_design,
                  'sources': {'base_sft': str(formal.DEFAULT / 'summary.json'),
                              'kto': str(out / 'summary.json'),
                              'kto_manifest_sha256': formal.sha((out / 'manifest.json').read_bytes())}}
    write_json(BASE / 'comparison_validation_v4.json', comparison)
    return comparison


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
    elif args.action == 'sync':
        result = formal.sync(args.out)
    elif args.action == 'watch':
        formal.watch(args.out, args.interval, on_summary=lambda result: write_comparison(args.out, result))
        return
    else:
        result = formal.summarize(args.out)
    if args.action != 'prepare':
        write_comparison(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
