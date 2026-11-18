import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from experiments.evaluation.formal import sha, summarize
from experiments.evaluation.formal_kto import compare_generation, extract_returned, reuse_completed
from experiments.evaluation.generate import inference_messages, sample_seed


class FormalKtoTests(unittest.TestCase):
    def test_archive_rejects_traversal_symlinks_and_unexpected_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, kind in (('../escape', tarfile.REGTYPE),
                               ('outputs/qwen3_4b_kto_v1_train/link', tarfile.SYMTYPE),
                               ('outputs/eval_one_shot_v4/full_validation/base/result.json', tarfile.REGTYPE)):
                with self.subTest(name=name):
                    archive = root / 'bad.tar.gz'
                    with tarfile.open(archive, 'w:gz') as handle:
                        item = tarfile.TarInfo(name)
                        item.type = kind
                        item.size = 1 if kind == tarfile.REGTYPE else 0
                        handle.addfile(item, io.BytesIO(b'x') if item.size else None)
                    with self.assertRaises(ValueError):
                        extract_returned(archive, root / 'returned')
            self.assertFalse((root / 'escape').exists())

    def test_prompt_seed_token_length_and_sampling_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'attempts').mkdir()
            task = {'task_id': 'task', 'messages': [{'role': 'system', 'content': 'system'},
                    {'role': 'user', 'content': 'Specification. Generate one assertion for this signal.'}]}
            config = {'label': 'kto', 'adapter': 'kto', 'temperature': .7}
            config['fingerprint'] = sha(json.dumps(config, sort_keys=True))
            baseline_config = {'label': 'sft', 'adapter': 'sft', 'temperature': .7}
            row = {'task_id': 'task', 'seed': sample_seed('task', 0, 42), 'prompt_tokens': 50,
                   'effective_messages_sha256': sha(json.dumps(inference_messages(task['messages'], 12), sort_keys=True))}
            attempt = folder / 'attempts/task__sample_0.json'
            attempt.write_text(json.dumps(row))
            protocol = {'generation': {'seed': 42}}
            self.assertEqual(compare_generation(config, baseline_config, [row], [row], [task], folder, protocol), [row])
            for key, value in (('seed', 1), ('prompt_tokens', 51), ('effective_messages_sha256', 'changed')):
                wrong = dict(row, **{key: value})
                attempt.write_text(json.dumps(wrong))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    compare_generation(config, baseline_config, [wrong], [row], [task], folder, protocol)
            with self.assertRaises(ValueError):
                compare_generation(config, dict(baseline_config, temperature=.8), [row], [row], [task], folder, protocol)

    def test_kto_summary_uses_manifest_labels_and_actual_signal_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({'models': {'kto': {}}, 'protocol': {'signal_count': 3}}))
            items = [{'job_id': None, 'label': 'kto', 'task_id': 'a', 'case_id': 'L1/a',
                      'offline_outcome': 'syntax_extraction_error', 'offline_reason': 'bad_syntax'},
                     {'job_id': 'pending', 'label': 'kto', 'task_id': 'b', 'case_id': 'L1/a'}]
            (root / 'occurrences.jsonl').write_text(''.join(json.dumps(item) + '\n' for item in items))
            result = summarize(root)
            self.assertEqual(set(result['models']), {'kto'})
            self.assertEqual(result['models']['kto']['signals_remaining'], 1)
            self.assertEqual(result['models']['kto']['signals_evaluated'], 2)
            self.assertIsNone(result['models']['kto']['proof_success_rate'])

    def test_cache_reuse_requires_exact_input_fingerprints_and_log_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'source', root / 'target'
            source.mkdir()
            target.mkdir()
            shared = {'worker_sha256': 'worker', 'timeout_seconds': 180, 'unique_jobs': 1}
            for path in (source, target):
                (path / 'manifest.json').write_text(json.dumps(shared))
                folder = path / 'jobs/job'
                folder.mkdir(parents=True)
                (folder / 'dut.sv').write_text('original RTL and assertion')
                (folder / 'run.tcl').write_text('original clock/reset/proof context')
            old = source / 'jobs/job'
            (old / 'jg.log').write_text('native proof log')
            job = {'job_id': 'job', 'fingerprint': 'input', 'implication': True,
                   'files': {name: sha((old / name).read_bytes()) for name in ('dut.sv', 'run.tcl')}}
            for path in (source, target):
                (path / 'jobs.jsonl').write_text(json.dumps(job) + '\n')
            result = {'job_fingerprint': 'input', 'jg_log_sha256': sha((old / 'jg.log').read_bytes())}
            (old / 'result.json').write_text(json.dumps(result))
            (target / 'input_verification.json').write_text('{}')
            manifest = reuse_completed(target, source)
            self.assertEqual(manifest['cached_jobs'], 1)
            self.assertEqual((target / 'jobs/job/result.json').read_bytes(), (old / 'result.json').read_bytes())
            (old / 'jg.log').write_text('changed log')
            with self.assertRaisesRegex(ValueError, 'cached check differs'):
                reuse_completed(target, source)
            (old / 'jg.log').write_text('native proof log')
            (target / 'jobs/job/dut.sv').write_text('different RTL')
            with self.assertRaisesRegex(ValueError, 'RTL or Tcl changed'):
                reuse_completed(target, source)


if __name__ == '__main__':
    unittest.main()
