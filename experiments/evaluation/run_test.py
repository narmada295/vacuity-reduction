"""Run base, SFT and KTO on the frozen five-design v4 TEST split, with recovery."""

import argparse
from collections import Counter
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import signal
import subprocess
import sys

if __package__:
    from .generate import inference_messages, sample_seed, write_json
    from .run_full import file_hash, validate_inputs, validate_outputs
else:
    from generate import inference_messages, sample_seed, write_json
    from run_full import file_hash, validate_inputs, validate_outputs


LABELS = ('base', 'sft', 'kto')
COMMON_SETTINGS = ('model', 'model_revision', 'chat_template_sha256', 'training_config_sha256',
                   'prompts_sha256', 'prompt_count', 'assertion_cap', 'prompt_version',
                   'one_shot_system_prompt_sha256', 'samples_per_prompt', 'max_new_tokens',
                   'temperature', 'top_p', 'top_k', 'repetition_penalty', 'seed',
                   'quantization', 'script_sha256', 'truncation', 'packages')


def preflight(root, protocol):
    if protocol['split'] != 'test' or tuple(protocol['models']) != LABELS:
        raise ValueError('This launcher requires the frozen base/SFT/KTO TEST protocol')
    tasks = validate_inputs(root, protocol)
    observed = {task['case_id'] for task in tasks}
    if observed != set(protocol['test_designs']) or observed & set(protocol['excluded_designs']):
        raise ValueError('Test design membership overlaps training/validation or differs from the saved split')
    if file_hash(root / protocol['training_config']) != protocol['training_config_sha256']:
        raise ValueError('Original SFT model/tokenizer configuration changed')
    for package, expected in protocol['expected_packages'].items():
        if version(package) != expected:
            raise ValueError('Generation package changed: ' + package)
    for label in ('sft', 'kto'):
        model = protocol['models'][label]
        adapter = root / model['adapter']
        for name, expected in model['adapter_files'].items():
            if file_hash(adapter / name) != expected:
                raise ValueError(label + ' adapter changed: ' + name)
        result = json.loads((root / model['training_result']).read_text())
        if result['status'] != 'PASS':
            raise ValueError(label + ' source training did not complete')
        if label == 'kto' and result['best_checkpoint'] != model['selected_checkpoint']:
            raise ValueError('Selected KTO checkpoint changed after the protocol was frozen')
    for task in tasks:
        # Verify the full original input can receive the unchanged one-shot prompt.
        inference_messages(task['messages'], protocol['generation']['assertion_cap'])
    return tasks


def check_model_output(directory, label, tasks, protocol):
    rows, config = validate_outputs(directory, label, tasks, protocol)
    fingerprint = hashlib.sha256(json.dumps({key: value for key, value in config.items()
                                           if key != 'fingerprint'}, sort_keys=True).encode()).hexdigest()
    if fingerprint != config['fingerprint']:
        raise ValueError('Generation configuration fingerprint is invalid')
    if config['script_sha256'] != protocol['files'][protocol['generator']]:
        raise ValueError('Generation script differs from the frozen test protocol')
    if config['training_config_sha256'] != protocol['training_config_sha256']:
        raise ValueError('Generation used a different base-model configuration')
    if config['packages'] != protocol['expected_packages']:
        raise ValueError('Generation packages differ from the frozen test protocol')
    if config['adapter_files'] != protocol['models'][label]['adapter_files']:
        raise ValueError('Generation used a different ' + label + ' adapter')
    assigned = {task['task_id']: task for task in tasks}
    for row in rows:
        task = assigned[row['task_id']]
        if row['seed'] != sample_seed(row['task_id'], 0, protocol['generation']['seed']):
            raise ValueError('Generation seed differs from the assigned task')
        expected = hashlib.sha256(json.dumps(inference_messages(task['messages'], 12), sort_keys=True).encode()).hexdigest()
        if row['effective_messages_sha256'] != expected:
            raise ValueError('Effective inference input changed')
        attempt = directory / 'attempts' / (row['task_id'] + '__sample_0.json')
        if json.loads(attempt.read_text()) != row:
            raise ValueError('Saved response differs from the aggregate output')
    return rows, config


def check_comparison(rows, configs):
    for label in ('sft', 'kto'):
        for key in COMMON_SETTINGS:
            if configs[label][key] != configs['base'][key]:
                raise ValueError('Base/' + label + ' setting differs: ' + key)
    by_task = {label: {row['task_id']: row for row in values} for label, values in rows.items()}
    for label in ('sft', 'kto'):
        if set(by_task[label]) != set(by_task['base']):
            raise ValueError('Models received different test tasks')
        for ident, original in by_task['base'].items():
            for key in ('seed', 'prompt_tokens', 'effective_messages_sha256'):
                if by_task[label][ident][key] != original[key]:
                    raise ValueError('Effective base/' + label + ' input differs: ' + ident + '/' + key)


def status_from_summary(directory, total, complete=False):
    path = directory / 'summary.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    saved = data.get('saved_attempts', 0)
    return {'state': 'complete' if complete else 'running', 'saved_attempts': saved,
            'remaining_attempts': total - saved, 'outcomes': data.get('outcomes', {})}


def run(root, protocol):
    tasks = preflight(root, protocol)
    output = root / protocol['output_root']
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.full_test.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = output / 'frozen_protocol.json'
    if snapshot.exists() and json.loads(snapshot.read_text()) != protocol:
        raise ValueError('Existing TEST run uses a different frozen protocol')
    write_json(snapshot, protocol)
    state = {'split': 'test', 'state': 'running', 'signals_per_model': len(tasks),
             'total_expected_responses': len(tasks) * 3,
             'models': {label: {'state': 'queued', 'saved_attempts': 0,
                                'remaining_attempts': len(tasks)} for label in LABELS}}
    state_path = output / 'run_status.json'
    logs = root / 'logs/eval_one_shot_v4/test'
    logs.mkdir(parents=True, exist_ok=True)
    rows, configs = {}, {}

    def save_status():
        state['remaining_responses'] = sum(item['remaining_attempts'] for item in state['models'].values())
        write_json(state_path, state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Allocation/session interrupted; saved responses will be reused')

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        for label in LABELS:
            destination = output / label
            summary_path = destination / 'summary.json'
            complete = summary_path.exists() and json.loads(summary_path.read_text()).get('status') == 'complete'
            if not complete:
                state['current'] = label
                state['models'][label] = status_from_summary(destination, len(tasks))
                save_status()
                command = [sys.executable, '-u', str(root / protocol['generator']),
                           '--prompts', str(root / protocol['prompts']), '--label', label,
                           '--training-config', str(root / protocol['training_config']),
                           '--assertion-cap', str(protocol['generation']['assertion_cap']),
                           '--samples-per-prompt', '1',
                           '--max-new-tokens', str(protocol['generation']['max_new_tokens']),
                           '--temperature', str(protocol['generation']['temperature']),
                           '--top-p', str(protocol['generation']['top_p']),
                           '--seed', str(protocol['generation']['seed']), '--out', str(destination)]
                if label != 'base':
                    command.extend(['--adapter', str(root / protocol['models'][label]['adapter'])])
                print('START TEST', label, len(tasks), 'signals', flush=True)
                with (logs / (label + '.log')).open('a', buffering=1) as log:
                    process = subprocess.Popen(command, cwd=str(root), stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, bufsize=1)
                    try:
                        for line in process.stdout:
                            print(line, end='', flush=True)
                            log.write(line)
                            if line.startswith(label + ' '):
                                state['models'][label] = status_from_summary(destination, len(tasks))
                                save_status()
                        code = process.wait()
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
                if code:
                    raise RuntimeError(label + ' generation exited ' + str(code) + '; saved responses preserved')
            rows[label], configs[label] = check_model_output(destination, label, tasks, protocol)
            state['models'][label] = status_from_summary(destination, len(tasks), complete=True)
            save_status()
            print('DONE TEST', label, len(rows[label]), '/', len(tasks), flush=True)
        check_comparison(rows, configs)
        state.update(state='generation_complete', comparison_input_check='PASS',
                     formal_evaluation='pending JasperGold checks; generation does not establish correctness')
        state.pop('current', None)
        save_status()
        print('All three TEST generation runs complete; inputs/settings match.', flush=True)
    except BaseException as error:
        current = state.get('current')
        if current:
            state['models'][current] = status_from_summary(output / current, len(tasks))
            state['models'][current]['state'] = 'interrupted'
        state.update(state='failed_or_interrupted', error=type(error).__name__ + ': ' + str(error))
        save_status()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--protocol', type=Path, default=Path('data/eval_one_shot_v4/full_test_protocol.json'))
    parser.add_argument('--check-only', action='store_true', help='Validate split, files and adapters without loading a GPU model')
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    protocol = json.loads(protocol_path.read_text())
    if args.check_only:
        tasks = preflight(root, protocol)
        print(json.dumps({'status': 'PASS', 'split': 'test', 'signals_per_model': len(tasks),
                          'signals_by_design': dict(Counter(task['case_id'] for task in tasks)),
                          'models': list(LABELS), 'total_generation_requests': len(tasks) * 3,
                          'gpu_generation': 'not started'}, indent=2))
    else:
        run(root, protocol)


if __name__ == '__main__':
    main()
