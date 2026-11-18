import copy
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from experiments.evaluation import run_mixed
from experiments.evaluation.prepare_mixed import COHORT, DATA, ROOT, task_for_case
from experiments.evaluation.generate import inference_messages
from fixme_svagen.signal_mapping import validate_mapping


class DocumentedSignalTests(unittest.TestCase):
    def test_rejects_invented_identifiers_quotes_and_port_exclusions(self):
        candidates = [{'name': 'clk', 'kind': 'port'}, {'name': 'counter', 'kind': 'internal'}]
        spec = 'The counter increments on each enabled clock cycle.'
        rows = [{'name': 'clk', 'selected': True, 'information': 'Clock input.', 'spec_quote': '', 'reason': 'Port'},
                {'name': 'counter', 'selected': True, 'information': 'Implements the documented counter.',
                 'spec_quote': spec, 'reason': 'Documented counter'}]
        self.assertEqual(len(validate_mapping({'signals': rows}, candidates, spec)), 2)
        for field, value in [('name', 'invented'), ('spec_quote', 'A requirement that the source never states.')]:
            changed = copy.deepcopy(rows)
            changed[1][field] = value
            with self.assertRaises(ValueError):
                validate_mapping({'signals': changed}, candidates, spec)
        changed = copy.deepcopy(rows)
        changed[0]['selected'] = False
        with self.assertRaisesRegex(ValueError, 'port was excluded'):
            validate_mapping({'signals': changed}, candidates, spec)
        changed = copy.deepcopy(rows)
        changed[1]['spec_quote'] = 'This sentence is invented.'
        filtered = validate_mapping({'signals': changed}, candidates, spec, filter_unsupported=True)
        self.assertFalse(filtered[1]['selected'])
        self.assertTrue(filtered[1]['proposed_selected'])
        self.assertEqual(filtered[1]['proposed_spec_quote'], 'This sentence is invented.')

    def test_two_clock_context_reaches_the_actual_inference_prompt(self):
        case = {'case_id': 'L3/9', 'spec': 'Ethernet status.', 'rtl': 'module eth_macstatus; endmodule',
                'formal_context': {'clock': 'MRxClk', 'checker_clocks': ['MRxClk', 'MTxClk']}}
        signal = {'name': 'Collision', 'information': 'Collision indicator.'}
        task = task_for_case(case, signal)
        content = inference_messages(task['messages'], 12)[1]['content']
        self.assertIn('clock MRxClk\nclock MTxClk', content)
        self.assertEqual(case['formal_context']['clock'], 'MRxClk')


class MixedArtifactsTests(unittest.TestCase):
    def test_inputs_include_internal_targets_without_previous_split_overlap(self):
        path = DATA / 'manifest.json'
        if not path.exists():
            self.skipTest('Prepared mixed snapshot is not present')
        manifest = json.loads(path.read_text())
        records = [json.loads(line) for line in (DATA / 'designs.jsonl').read_text().splitlines()]
        expected = {'OpenTitan/gpio', 'OpenTitan/pwm', 'OpenTitan/uart', 'OpenTitan/i2c', 'L3/8', 'L3/9'}
        self.assertEqual({r['case_id'] for r in records}, expected)
        self.assertFalse(expected & set(manifest['previous_35_designs_excluded']))
        self.assertEqual(sum(len(r['signals']) for r in records), manifest['signals_per_model'])
        self.assertTrue(all(any(s['kind'] == 'internal' for s in r['signals']) for r in records))
        self.assertTrue(all(r['source']['compile_status'] == 'PASS' for r in records))
        for record in records:
            for signal in record['signals']:
                if signal['kind'] == 'internal':
                    self.assertTrue(signal['spec_quote'])
                    self.assertIn(' '.join(signal['spec_quote'].split()), ' '.join(record['spec'].split()))

    def test_bundle_matches_protocol_and_keeps_the_established_v4_generation(self):
        path = ROOT / 'remote_external_mixed_six_v4.tar.gz'
        if not path.exists():
            self.skipTest('Packaged mixed snapshot is not present')
        protocol = json.loads((DATA / 'protocol.json').read_text())
        previous = json.loads(Path('output/fixme_svagen/evaluation_one_shot_v4/full_test_protocol.json').read_text())
        with tarfile.open(path) as archive:
            names = archive.getnames()
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names))
            self.assertFalse(any(Path(n).name == '.env' for n in names))
            for name, digest in protocol['files'].items():
                self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(), digest)
        for name in ('generate.py', 'prompts.py', 'run_full.py'):
            key = 'scripts/eval_one_shot_v4/' + name
            self.assertEqual(protocol['files'][key], previous['files'][key])
        for key in ('generation', 'models', 'system_prompt_sha256'):
            self.assertEqual(protocol[key], previous[key])
        tasks = [json.loads(line) for line in (DATA / 'prompts.jsonl').read_text().splitlines()]
        for task in tasks:
            self.assertEqual(set(task), {'task_id', 'case_id', 'target_signal', 'messages'})
            self.assertEqual([m['role'] for m in task['messages']], ['system', 'user'])

    def test_runner_rejects_a_different_cohort_before_loading_models(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'TEST protocol'):
                run_mixed.preflight(Path(temp), {'split': 'test', 'models': dict.fromkeys(run_mixed.LABELS),
                                                 'cohort': 'opentitan_six_v4'})


if __name__ == '__main__':
    unittest.main()
