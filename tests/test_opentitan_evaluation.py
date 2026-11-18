import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from experiments.evaluation import run_opentitan
from experiments.evaluation.generate import file_hash, inference_messages
from experiments.evaluation.opentitan_sources import CASES, ROOT, UPSTREAM, REVISION, code_only
from tests.test_full_test_generation import FullTestGenerationTests


class OpenTitanInputsTests(unittest.TestCase):
    def test_original_specifications_and_actual_clocks_are_preserved(self):
        path = ROOT / 'data/designs.jsonl'
        if not path.exists():
            self.skipTest('Prepared OpenTitan snapshot is not present')
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([(r['case_id'].split('/')[-1], r['top']) for r in records], list(CASES))
        self.assertEqual(sum(len(r['signals']) for r in records), 79)
        for record in records:
            ip = record['case_id'].split('/')[-1]
            with self.subTest(ip=ip):
                self.assertIn((UPSTREAM / 'hw/ip' / ip / 'doc/_index.md').read_text(), record['spec'])
                self.assertIn((UPSTREAM / 'hw/ip' / ip / 'data' / (ip + '.hjson')).read_text(), record['spec'])
                self.assertEqual(record['source']['revision'], REVISION)
                self.assertEqual(record['source']['compile_status'], 'PASS')
                self.assertFalse(record['source']['functional_rtl_edits'])
                clock = record['formal_context']['clock']
                names = {s['name'] for s in record['signals']}
                self.assertIn(clock, names)
                self.assertIn(record['formal_context']['reset_signal'], names)
                self.assertIn('posedge ' + clock, record['rtl'])
                self.assertEqual(record['formal_context']['formal_reset_policy'], 'unconstrained')
                self.assertNotRegex(code_only(record['rtl']), r'\b(?:assert|assume|cover)\s*(?:property|\()')

    def test_bundle_hashes_roles_and_frozen_v4_inputs(self):
        path = ROOT / 'remote_opentitan_six_v4.tar.gz'
        if not path.exists():
            self.skipTest('Packaged OpenTitan snapshot is not present')
        protocol = json.loads((ROOT / 'data/protocol.json').read_text())
        with tarfile.open(path) as archive:
            names = archive.getnames()
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names))
            for relative, digest in protocol['files'].items():
                import hashlib
                self.assertEqual(hashlib.sha256(archive.extractfile(relative).read()).hexdigest(), digest)
        tasks = [json.loads(line) for line in (ROOT / 'data/prompts.jsonl').read_text().splitlines()]
        self.assertEqual(len(tasks), protocol['signal_count'])
        self.assertEqual(len({t['task_id'] for t in tasks}), len(tasks))
        for task in tasks:
            self.assertEqual(set(task), {'task_id', 'case_id', 'target_signal', 'messages'})
            self.assertEqual([m['role'] for m in task['messages']], ['system', 'user'])
            transformed = inference_messages(task['messages'], 12)
            self.assertIn('Generate up to 12', transformed[0]['content'])
        self.assertEqual(file_hash(Path('experiments/evaluation/generate.py')),
                         protocol['files'][protocol['generator']])


class OpenTitanRunnerTests(unittest.TestCase):
    def test_preflight_rejects_wrong_cohort_and_changed_adapter(self):
        fixture = FullTestGenerationTests()
        with tempfile.TemporaryDirectory() as directory, patch.object(run_opentitan, 'version', return_value='pinned'):
            root = Path(directory)
            task, protocol = fixture.fixture(root)
            with self.assertRaisesRegex(ValueError, 'TEST protocol'):
                run_opentitan.preflight(root, protocol)
            protocol['cohort'] = 'opentitan_six_v4'
            self.assertEqual(run_opentitan.preflight(root, protocol), [task])
            (root / 'kto/adapter_model.safetensors').write_bytes(b'wrong model')
            with self.assertRaisesRegex(ValueError, 'kto adapter changed'):
                run_opentitan.preflight(root, protocol)

    def test_resume_reuses_all_committed_models_without_loading_gpu(self):
        rows, configs = FullTestGenerationTests().comparison_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {'output_root': 'outputs/opentitan', 'log_root': 'logs/opentitan'}
            for label in run_opentitan.LABELS:
                folder = root / protocol['output_root'] / label
                folder.mkdir(parents=True)
                (folder / 'summary.json').write_text(json.dumps({'status': 'complete', 'saved_attempts': 1,
                                                               'outcomes': {'generated': 1}}))
            with patch.object(run_opentitan, 'preflight', return_value=[{'task_id': 'test-task'}]), \
                 patch.object(run_opentitan, 'profile_inputs', return_value={'status': 'PASS'}), \
                 patch.object(run_opentitan, 'check_model_output', side_effect=lambda path, label, *args: (rows[label], configs[label])), \
                 patch.object(run_opentitan.subprocess, 'Popen') as process:
                run_opentitan.run(root, protocol)
                process.assert_not_called()
            state = json.loads((root / protocol['output_root'] / 'run_status.json').read_text())
            self.assertEqual(state['state'], 'generation_complete')
            self.assertEqual(state['remaining_responses'], 0)
            self.assertEqual(state['comparison_input_check'], 'PASS')


if __name__ == '__main__':
    unittest.main()
