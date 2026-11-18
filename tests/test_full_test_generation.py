import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.evaluation import run_test
from experiments.evaluation.run_full import file_hash


class FullTestGenerationTests(unittest.TestCase):
    def fixture(self, root):
        task = {'task_id': 'test-task', 'case_id': 'L1/heldout', 'target_signal': 'req',
                'messages': [{'role': 'system', 'content': 'system'},
                             {'role': 'user', 'content': 'Spec and RTL. Generate one assertion for this signal.'}]}
        prompts = root / 'prompts.jsonl'
        prompts.write_text(json.dumps(task) + '\n')
        training = root / 'sft_config.json'
        training.write_text(json.dumps({'model': 'pinned'}))
        protocol = {'split': 'test', 'signal_count': 1, 'signals_by_design': {'L1/heldout': 1},
                    'prompts': 'prompts.jsonl', 'training_config': 'sft_config.json',
                    'training_config_sha256': file_hash(training), 'expected_training': {'model': 'pinned'},
                    'expected_packages': {'torch': 'pinned'}, 'generation': {'assertion_cap': 12},
                    'test_designs': ['L1/heldout'], 'excluded_designs': ['L1/train', 'L1/validation'],
                    'files': {'prompts.jsonl': file_hash(prompts)}, 'models': {'base': {'adapter_files': {}}}}
        for label in ('sft', 'kto'):
            folder = root / label
            folder.mkdir()
            (folder / 'adapter_model.safetensors').write_bytes((label + '-weights').encode())
            (folder / 'adapter_config.json').write_text('{}')
            (folder / 'result.json').write_text(json.dumps({'status': 'PASS', 'best_checkpoint': 'checkpoint-710'}))
            protocol['models'][label] = {'adapter': label,
                'adapter_files': {name: file_hash(folder / name) for name in ('adapter_model.safetensors', 'adapter_config.json')},
                'training_result': label + '/result.json', 'selected_checkpoint': 'checkpoint-710'}
        return task, protocol

    def test_preflight_rejects_validation_split_and_train_overlap(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run_test, 'version', return_value='pinned'):
            root = Path(directory)
            task, protocol = self.fixture(root)
            self.assertEqual(run_test.preflight(root, protocol), [task])
            with self.assertRaisesRegex(ValueError, 'TEST protocol'):
                run_test.preflight(root, dict(protocol, split='validation'))
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                run_test.preflight(root, dict(protocol, excluded_designs=['L1/heldout']))

    def test_changed_checkpoint_input_or_package_cannot_silently_start(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run_test, 'version', return_value='pinned'):
            root = Path(directory)
            _, protocol = self.fixture(root)
            weight = root / 'kto/adapter_model.safetensors'
            original = weight.read_bytes()
            weight.write_bytes(b'different checkpoint')
            with self.assertRaisesRegex(ValueError, 'kto adapter changed'):
                run_test.preflight(root, protocol)
            weight.write_bytes(original)
            with patch.object(run_test, 'version', return_value='changed'):
                with self.assertRaisesRegex(ValueError, 'package changed'):
                    run_test.preflight(root, protocol)
            (root / 'prompts.jsonl').write_text('changed input')
            with self.assertRaisesRegex(ValueError, 'Frozen run file changed'):
                run_test.preflight(root, protocol)

    def comparison_fixture(self):
        config = {key: 'shared' for key in run_test.COMMON_SETTINGS}
        configs = {label: copy.deepcopy(config) for label in run_test.LABELS}
        row = {'task_id': 'test-task', 'seed': 42, 'prompt_tokens': 100, 'effective_messages_sha256': 'same'}
        rows = {label: [dict(row)] for label in run_test.LABELS}
        return rows, configs

    def test_all_three_models_must_receive_same_effective_inputs_and_budget(self):
        rows, configs = self.comparison_fixture()
        run_test.check_comparison(rows, configs)
        rows['kto'][0]['prompt_tokens'] += 1
        with self.assertRaisesRegex(ValueError, 'input differs'):
            run_test.check_comparison(rows, configs)
        rows, configs = self.comparison_fixture()
        configs['kto']['max_new_tokens'] = 'different'
        with self.assertRaisesRegex(ValueError, 'setting differs'):
            run_test.check_comparison(rows, configs)

    def test_relaunch_skips_complete_models_without_loading_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, configs = self.comparison_fixture()
            protocol = {'output_root': 'outputs/full_test'}
            for label in run_test.LABELS:
                folder = root / protocol['output_root'] / label
                folder.mkdir(parents=True)
                (folder / 'summary.json').write_text(json.dumps({'status': 'complete', 'saved_attempts': 1,
                                                               'outcomes': {'generated': 1}}))
            with patch.object(run_test, 'preflight', return_value=[{'task_id': 'test-task'}]), \
                 patch.object(run_test, 'check_model_output', side_effect=lambda directory, label, *args: (rows[label], configs[label])), \
                 patch.object(run_test.subprocess, 'Popen') as process:
                run_test.run(root, protocol)
                process.assert_not_called()
            state = json.loads((root / protocol['output_root'] / 'run_status.json').read_text())
            self.assertEqual(state['split'], 'test')
            self.assertEqual(state['state'], 'generation_complete')
            self.assertEqual(state['remaining_responses'], 0)
            self.assertEqual(set(state['models']), set(run_test.LABELS))


if __name__ == '__main__':
    unittest.main()
