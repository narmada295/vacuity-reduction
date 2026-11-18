import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from experiments.evaluation import formal, formal_test


class FormalTestTests(unittest.TestCase):
    def test_archive_rejects_validation_files_traversal_and_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, kind in (('../escape', tarfile.REGTYPE),
                               ('outputs/eval_one_shot_v4/full_validation/base/result.json', tarfile.REGTYPE),
                               ('outputs/eval_one_shot_v4/full_test/link', tarfile.SYMTYPE)):
                with self.subTest(name=name):
                    archive = root / 'bad.tar.gz'
                    with tarfile.open(archive, 'w:gz') as handle:
                        item = tarfile.TarInfo(name)
                        item.type = kind
                        item.size = 1 if kind == tarfile.REGTYPE else 0
                        handle.addfile(item, io.BytesIO(b'x') if item.size else None)
                    with self.assertRaises(ValueError):
                        formal_test.extract_returned(archive, root / 'returned')
            self.assertFalse((root / 'escape').exists())

    def test_test_population_excludes_train_and_validation_and_answers(self):
        task = {'task_id': 'a', 'case_id': 'L1/test', 'target_signal': 'req',
                'messages': [{'role': 'system'}, {'role': 'user'}]}
        protocol = {'split': 'test', 'models': dict.fromkeys(('base', 'sft', 'kto')),
                    'signal_count': 1, 'signals_by_design': {'L1/test': 1},
                    'test_designs': ['L1/test'], 'excluded_designs': ['L1/train', 'L1/validation']}
        formal_test.validate_tasks([task], protocol)
        with self.assertRaisesRegex(ValueError, 'TEST protocol'):
            formal_test.validate_tasks([task], dict(protocol, split='validation'))
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            formal_test.validate_tasks([task], dict(protocol, excluded_designs=['L1/test']))
        with self.assertRaisesRegex(ValueError, 'unexpected roles'):
            formal_test.validate_tasks([dict(task, messages=task['messages'] + [{'role': 'assistant'}])], protocol)

    def test_context_lookup_uses_test_map_and_rejects_wrong_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'test RTL and spec'}]
            task = {'task_id': 'test-task', 'case_id': 'L1/test', 'messages': messages}
            ref = root / 'test_reference.jsonl'
            ref.write_text(json.dumps({'task_id': 'test-task', 'selected': True,
                                       'source_example_ids': ['test-example']}) + '\n')
            source = root / 'examples.jsonl'
            source.write_text(json.dumps({'id': 'test-example', 'rtl': 'module test(input clk); endmodule',
                'spec': 'test spec', 'formal_context': {'checker_clocks': ['clk'],
                                                     'formal_reset_policy': 'unconstrained'}}) + '\n')
            with patch.object(formal, 'Path', return_value=source), \
                 patch.object(formal, 'chat_record', return_value={'messages': messages}):
                contexts = formal.load_contexts([task], ref)
                self.assertEqual(contexts['test-task']['clock_tcl'], 'clock clk')
                self.assertEqual(contexts['test-task']['top'], 'test')
                with self.assertRaisesRegex(ValueError, 'differs from the inference prompt'):
                    formal.load_contexts([dict(task, messages=[{'role': 'user', 'content': 'wrong design'}])], ref)

    def test_formal_jobs_preserve_occurrences_and_share_only_identical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / 'worker.py'
            worker.write_text('# frozen native runner\n')
            task = {'task_id': 'task', 'case_id': 'L1/test', 'target_signal': 'req'}
            response = dict(task, response='assert property (@(posedge clk) req |-> ack);')
            context = {'rtl': 'module test(input clk,req,ack); endmodule', 'top': 'test',
                       'clock_tcl': 'clock clk', 'formal_context': {'checker_clocks': ['clk']}}
            rows = {label: [dict(response, label=label)] for label in ('base', 'sft', 'kto')}
            with patch.object(formal, 'load_contexts', return_value={'task': context}):
                out = root / 'evaluation'
                manifest = formal.prepare(out, 180, loaded=([task], rows, {'signal_count': 1}, 'archive'),
                                          labels=tuple(rows), worker_source=worker,
                                          reference_map=root / 'test_reference.jsonl')
                self.assertEqual(manifest['unique_jobs'], 1)
                occurrences = [json.loads(line) for line in (out / 'occurrences.jsonl').read_text().splitlines()]
                self.assertEqual(len(occurrences), 3)
                job = json.loads((out / 'jobs.jsonl').read_text())
                tcl = (out / 'jobs' / job['job_id'] / 'run.tcl').read_text()
                self.assertIn('clock clk\nreset -none\nprove -all', tcl)
                self.assertIn(response['response'], (out / 'jobs' / job['job_id'] / 'dut.sv').read_text())
                rows['base'][0]['task_id'] = 'wrong-task'
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    formal.prepare(root / 'mismatch', 180,
                                   loaded=([task], rows, {'signal_count': 1}, 'archive'),
                                   labels=tuple(rows), worker_source=worker)


if __name__ == '__main__':
    unittest.main()
