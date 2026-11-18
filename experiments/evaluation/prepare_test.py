"""Package the reserved test inputs and a frozen three-model v4 launcher."""

from collections import Counter
import json
from pathlib import Path
import tarfile

from experiments.evaluation.generate import file_hash, inference_messages, write_json
from experiments.evaluation.prepare_inputs import prepare


BASE = Path('output/fixme_svagen/evaluation_one_shot_v4')
SOURCE = Path('output/fixme_svagen/sft_before_reduction_v2_complete')


def main():
    source = json.loads((SOURCE / 'manifest.json').read_text())
    test_dir = BASE / 'test'
    if not test_dir.exists():
        prepare(SOURCE, 'test', test_dir, per_signal=True)
    inputs = json.loads((test_dir / 'manifest.json').read_text())
    if (inputs['split'] != 'test' or not inputs['one_prompt_per_signal']
            or inputs['source_manifest_sha256'] != file_hash(SOURCE / 'manifest.json')
            or inputs['source_sha256'] != file_hash(SOURCE / 'test.jsonl')):
        raise ValueError('Test input preparation differs from the frozen source')
    for name, expected in inputs['files'].items():
        if file_hash(test_dir / name) != expected:
            raise ValueError('Prepared test inputs changed: ' + name)
    tasks = [json.loads(line) for line in (test_dir / 'prompts.jsonl').read_text().splitlines()]
    test_designs = source['split_plan']['test']
    excluded = source['split_plan']['train'] + source['split_plan']['validation']
    if ({task['case_id'] for task in tasks} != set(test_designs)
            or set(test_designs) & set(excluded)):
        raise ValueError('Test designs differ from the agreed split')
    for task in tasks:
        if set(task) != {'task_id', 'case_id', 'target_signal', 'messages'}:
            raise ValueError('Inference task contains unexpected metadata')
        if [message['role'] for message in task['messages']] != ['system', 'user']:
            raise ValueError('Inference task contains a reference answer')
        inference_messages(task['messages'], 12)
    validation = json.loads((BASE / 'full_validation_protocol.json').read_text())
    original = BASE / 'formal_validation_v1/returned/outputs/eval_one_shot_v4/full_validation'
    sft = json.loads((original / 'sft/generation_config.json').read_text())
    kto_run = BASE / 'formal_kto_validation_v1/returned'
    kto = json.loads((kto_run / 'outputs/eval_one_shot_v4/full_validation/kto/generation_config.json').read_text())
    training = json.loads((kto_run / 'outputs/qwen3_4b_kto_v1_train/result.json').read_text())
    if training['status'] != 'PASS' or training['best_checkpoint'] != 'outputs/qwen3_4b_kto_v1_train/checkpoint-710':
        raise ValueError('Selected KTO model is not the completed checkpoint 710')
    if sft['packages'] != kto['packages']:
        raise ValueError('Validation models used different generation packages')
    script_root = Path('experiments/evaluation')
    files = {}
    for name in ('generate.py', 'prompts.py', 'run_full.py'):
        relative = 'scripts/eval_one_shot_v4/' + name
        actual = file_hash(script_root / name)
        if actual != validation['files'][relative]:
            raise ValueError('The original v4 inference code changed: ' + name)
        files[relative] = actual
    files['scripts/eval_one_shot_v4/run_test.py'] = file_hash(script_root / 'run_test.py')
    for name in ('prompts.jsonl', 'manifest.json'):
        files['data/eval_one_shot_v4/test/' + name] = file_hash(test_dir / name)
    protocol = {
        'protocol': 'frozen_full_test_v4', 'split': 'test',
        'prompt_version': validation['prompt_version'], 'signal_count': len(tasks),
        'signals_by_design': dict(Counter(task['case_id'] for task in tasks)),
        'test_designs': test_designs, 'excluded_designs': excluded,
        'generator': validation['generator'], 'prompts': 'data/eval_one_shot_v4/test/prompts.jsonl',
        'training_config': validation['training_config'],
        'training_config_sha256': sft['training_config_sha256'],
        'output_root': 'outputs/eval_one_shot_v4/full_test',
        'generation': validation['generation'], 'system_prompt_sha256': validation['system_prompt_sha256'],
        'expected_training': validation['expected_training'], 'expected_packages': sft['packages'],
        'models': {
            'base': {'adapter': None, 'adapter_files': {}},
            'sft': {'adapter': 'outputs/qwen3_4b_sft_v2_train/final_adapter',
                    'adapter_files': sft['adapter_files'],
                    'training_result': 'outputs/qwen3_4b_sft_v2_train/result.json',
                    'selected_checkpoint': 'outputs/qwen3_4b_sft_v2_train/checkpoint-621'},
            'kto': {'adapter': 'outputs/qwen3_4b_kto_v1_train/final_adapter',
                    'adapter_files': kto['adapter_files'],
                    'training_result': 'outputs/qwen3_4b_kto_v1_train/result.json',
                    'selected_checkpoint': training['best_checkpoint']}},
        'files': files, 'source_manifest_sha256': file_hash(SOURCE / 'manifest.json'),
        'selection': validation['selection'].replace('validation', 'test'),
        'reference_assertions_or_labels_in_inference_inputs': False,
        'refinement': False, 'requests_per_signal': 1,
        'test_results_used_to_choose_prompt_or_models': False}
    path = BASE / 'full_test_protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('An existing different test protocol must be preserved')
    write_json(path, protocol)
    bundle = BASE / 'remote_eval_one_shot_v4_test.tar.gz'
    with tarfile.open(bundle, 'w:gz') as handle:
        handle.add(script_root / 'run_test.py', arcname='scripts/eval_one_shot_v4/run_test.py')
        handle.add(path, arcname='data/eval_one_shot_v4/full_test_protocol.json')
        for name in ('prompts.jsonl', 'manifest.json'):
            handle.add(test_dir / name, arcname='data/eval_one_shot_v4/test/' + name)
        handle.add('experiments/evaluation/README.md', arcname='FULL_TEST_V4_README.md')
    digest = file_hash(bundle)
    bundle.with_name(bundle.name + '.sha256').write_text(digest + '  ' + bundle.name + '\n')
    report = {'status': 'PASS', 'scope': 'test', 'signals_per_model': len(tasks),
              'signals_by_design': protocol['signals_by_design'], 'models': list(protocol['models']),
              'total_generation_requests': len(tasks) * 3,
              'prompt_unchanged_from_validation': True,
              'generator_unchanged_from_validation': True,
              'test_designs_disjoint_from_train_and_validation': True,
              'reference_assertions_and_labels_excluded': True,
              'model_adapter_hashes_frozen': True,
              'protocol_sha256': file_hash(path), 'bundle_sha256': digest,
              'gpu_generation': 'not started; run check-only on black6000 first',
              'formal_evaluation': 'pending after test generation'}
    write_json(BASE / 'full_test_preparation_verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
