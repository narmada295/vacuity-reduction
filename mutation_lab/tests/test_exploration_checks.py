"""Regression cases for failures observed in real exploration runs; no API calls."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from mutation_lab import exploration_prompts as prompts
from mutation_lab.core import Config
from mutation_lab.exploration import run_exploration
from mutation_lab.exploration_checks import predicate_conflicts, semantic_review
from mutation_lab.exploration_probes import ProbeExecutor, summarize_evidence, validate_probe
from mutation_lab.tests.test_exploration import (PIEZO, TINY, TOOLS, arguments,
                                               piezo_fixture, predicate)


class SemanticChecksTests(unittest.TestCase):
    @unittest.skipUnless(TOOLS, 'Verilator/Icarus required')
    def test_existing_bug_review_frame_is_retried_not_used_to_reject_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / 'spec.txt', PIEZO / 'rtl/Piezo_Driver.v', Path(temp) / 'run', 'reset',
                             '--response-retries', '1', '--max-attempts', '1')
            model = piezo_fixture(False)
            original = model._request
            def reply(system, prompt, **kwargs):
                row = original(system, prompt, **kwargs)
                if system == prompts.PLAN_CHECK:
                    payload = model.calls[-1][1]
                    self.assertEqual(payload['review_subject'], 'proposed_mutation')
                    self.assertIn('not yet implemented', payload['design_roles']['proposed'])
                    if sum(s == system for s, _, _ in model.calls) == 1:
                        # Reproduce the observed live response: searches only golden
                        # and rejects a future mutation because it is not present yet.
                        row.pop('review_subject'); row.pop('comparison')
                        row.update(verdict='mismatch', reason='The proposed fault is not present in golden RTL')
                return row
            model._request = reply
            summary = run_exploration(args, client=model)
            self.assertEqual(summary['accepted'], 1)
            self.assertEqual(sum(s == prompts.PLAN_CHECK for s, _, _ in model.calls), 2)
            failed = json.loads((args.out / 'llm_calls/c00004/result.json').read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertIn('proposed_mutation', failed['reason'])

    def test_proposed_comparison_cannot_swap_baseline_or_claim_preservation_false(self):
        config = Config.load(TINY / 'config.json')
        plan = {'golden_behavior': 'q_next = rst ? 0 : d', 'faulty_behavior': 'q_next = d',
                'controls': [{'id': 'C1'}]}
        required = ['golden_behavior', 'actual_change', 'trigger', 'controls', 'consequence']
        raw = {'review_subject': 'proposed_mutation', 'verdict': 'supported', 'reason': 'Proposed reset omission',
               'checks': {name: {'assessment': 'supported', 'reason': 'Evaluated A and hypothetical B'} for name in required},
               'controls': [{'id': 'C1', 'assessment': 'preserved', 'reason': 'rst=0 gives d in both'}],
               'source_evidence': [{'source_file': 'tiny_counter.sv', 'start_line': 1, 'end_line': 1}],
               'comparison': {'golden_rule': plan['golden_behavior'], 'proposed_rule': plan['faulty_behavior'],
                              'failing_case': {'condition': 'rst=1,d=1', 'golden_result': 'q=0', 'proposed_result': 'q=1', 'different': True},
                              'preserved_cases': [{'control_id': 'C1', 'condition': 'rst=0', 'golden_result': 'q=d', 'proposed_result': 'q=d', 'equal': True}]}}
        result = semantic_review(raw, required, config, config.design_root, plan)
        self.assertEqual(result['verdict'], 'supported')
        self.assertIn('not executed', result['comparison_evidence'])
        swapped = copy.deepcopy(raw)
        swapped['comparison']['golden_rule'] = plan['faulty_behavior']
        with self.assertRaisesRegex(ValueError, 'do not swap'):
            semantic_review(swapped, required, config, config.design_root, plan)
        contradictory = copy.deepcopy(raw)
        contradictory['comparison']['preserved_cases'][0]['equal'] = False
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            semantic_review(contradictory, required, config, config.design_root, plan)

    def test_contradictions_and_input_bit_packing(self):
        widths = {'data': 25, 'enabled': 1}
        self.assertTrue(predicate_conflicts([predicate('data', 0), predicate('data[24]', 1)], widths, {'data'}))
        self.assertTrue(predicate_conflicts([predicate('data', 1 << 25)], widths, {'data'}))
        self.assertTrue(predicate_conflicts([predicate('enabled', 1), predicate('enabled', 1, '!=')], widths, {'data'}))
        # Different state samples can legitimately differ. Inputs have one value per cycle.
        pre, post = predicate('enabled', 0), {**predicate('enabled', 1), 'sample': 'post'}
        self.assertFalse(predicate_conflicts([pre, post], widths, {'data'}))
        self.assertTrue(predicate_conflicts([predicate('data', 0), {**predicate('data', 1), 'sample': 'post'}], widths, {'data'}))

    def test_semantic_review_cannot_accept_contradictory_control_or_fabricate_source(self):
        config = Config.load(TINY / 'config.json')
        p = {'controls': [{'id': 'C1'}]}
        raw = {'verdict': 'supported', 'reason': 'A checked case',
               'checks': {'mechanism': {'assessment': 'supported', 'reason': 'A transition'}},
               'controls': [{'id': 'C1', 'assessment': 'contradicted', 'reason': 'This is the trigger'}],
               'source_evidence': [{'source_file': 'tiny_counter.sv', 'start_line': 1, 'end_line': 1}]}
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            semantic_review(raw, ['mechanism'], config, config.design_root, p)
        raw['controls'][0]['assessment'] = 'preserved'
        raw['source_evidence'][0]['excerpt'] = 'Fabricated code that is not in this design'
        result = semantic_review(raw, ['mechanism'], config, config.design_root, p)
        self.assertNotIn('Fabricated', result['source_evidence'][0]['excerpt'])
        self.assertFalse(result['formal_proof'])
        raw['source_evidence'][0]['source_file'] = '../other.sv'
        with self.assertRaisesRegex(ValueError, 'unconfigured'):
            semantic_review(raw, ['mechanism'], config, config.design_root, p)

    @unittest.skipUnless(TOOLS, 'Verilator/Icarus required')
    def test_bad_plan_stops_before_editing(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / 'spec.txt', PIEZO / 'rtl/Piezo_Driver.v', Path(temp) / 'run', 'reset',
                             '--max-attempts', '1')
            model = piezo_fixture(False)
            original = model._request
            def reply(system, prompt, **kwargs):
                row = original(system, prompt, **kwargs)
                if system == prompts.PLAN_CHECK:
                    row['verdict'] = 'mismatch'
                    row['checks']['golden_behavior'] = {'assessment': 'contradicted', 'reason': 'This is already golden behavior'}
                return row
            model._request = reply
            summary = run_exploration(args, client=model)
            self.assertEqual(summary['counts'], {'plan_rejected': 1})
            self.assertFalse(any(s == prompts.EDIT for s, _, _ in model.calls))

    @unittest.skipUnless(TOOLS, 'Verilator/Icarus required')
    def test_edit_mismatch_is_repaired_before_probes_with_same_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / 'spec.txt', PIEZO / 'rtl/Piezo_Driver.v', Path(temp) / 'run', 'reset',
                             '--max-attempts', '1', '--max-repairs', '1')
            model = piezo_fixture(False)
            original = model._request
            def reply(system, prompt, **kwargs):
                row = original(system, prompt, **kwargs)
                if system == prompts.EDIT_CHECK and sum(s == system for s, _, _ in model.calls) == 1:
                    row['verdict'] = 'mismatch'
                    row['checks']['mechanism'] = {'assessment': 'contradicted', 'reason': 'Wrong implementation for the checked plan'}
                return row
            model._request = reply
            summary = run_exploration(args, client=model)
            self.assertEqual(summary['accepted'], 1)
            record = json.loads((args.out / 'mutants/m0001/record.json').read_text())
            self.assertEqual(record['revisions'][0]['status'], 'plan_mismatch')
            plans = [payload['plan'] for system, payload, _ in model.calls if system == prompts.EDIT]
            self.assertEqual(len(plans), 2)
            self.assertEqual(plans[0], plans[1])
            self.assertTrue((args.out / 'mutants/m0001/plan_validation.json').exists())
            self.assertFalse((args.out / 'mutants/m0001/attempts/a1/probes').exists())

    @unittest.skipUnless(TOOLS, 'Verilator/Icarus required')
    def test_prediction_errors_do_not_cause_rtl_repairs(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / 'spec.txt', PIEZO / 'rtl/Piezo_Driver.v', Path(temp) / 'run', 'reset')
            model = piezo_fixture(False)
            model.plan['latency'] = {'min': 20, 'max': 20}
            model.batches[0][0]['observations'][0].update(golden=0, mutant=1)
            summary = run_exploration(args, client=model)
            self.assertEqual(summary['accepted'], 1)
            self.assertEqual(sum(s == prompts.EDIT for s, _, _ in model.calls), 1)
            record = json.loads((args.out / 'mutants/m0001/record.json').read_text())
            timing = record['validation']['timing']
            self.assertEqual(timing['prediction_mismatch_count'], 1)
            measured = timing['measured']['consecutive_trigger']
            self.assertTrue(any(x['kind'] == 'output' and not x['estimate_matches_any'] for x in measured))

    @unittest.skipUnless(TOOLS, 'Verilator/Icarus required')
    def test_unresolved_source_is_not_regenerated_as_a_new_mutant(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / 'spec.txt', PIEZO / 'rtl/Piezo_Driver.v', Path(temp) / 'run', 'reset',
                             '--count', '0', '--max-attempts', '2', '--probe-refinements', '0')
            model = piezo_fixture(True, proposed=2)
            run_exploration(args, client=model)
            second = json.loads((args.out / 'mutants/m0002/record.json').read_text())
            self.assertIn('duplicates m0001', second['reason'])
            self.assertFalse((args.out / 'mutants/m0002/attempts/a1/probes').exists())

    @unittest.skipUnless(TOOLS, 'Verilator/Icarus required')
    def test_old_review_is_not_attached_to_a_failed_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / 'spec.txt', PIEZO / 'rtl/Piezo_Driver.v', Path(temp) / 'run', 'reset',
                             '--max-attempts', '1', '--max-repairs', '1')
            model = piezo_fixture(False)
            broken = copy.deepcopy(model.batches[0]); broken[0]['steps'][0]['inputs']['data'] = 1 << 25
            model.batches.append(broken)
            original = model._request
            def reply(system, prompt, **kwargs):
                row = original(system, prompt, **kwargs)
                if system == prompts.REVIEW:
                    row.update(verdict='revise', plan_match='mismatch', reason='Repair same mechanism')
                return row
            model._request = reply
            run_exploration(args, client=model)
            record = json.loads((args.out / 'mutants/m0001/record.json').read_text())
            self.assertEqual(record['latest_attempt'], 'a2')
            self.assertNotIn('review', record)
            self.assertEqual(record['revisions'][0]['review']['verdict'], 'revise')
            self.assertEqual(record['revisions'][1]['status'], 'unresolved_probes')


class MeasuredProbeTests(unittest.TestCase):
    def fixture(self):
        executor = object.__new__(ProbeExecutor)
        executor.outputs, executor.state = {'Ack'}, {'enabled': 1}
        plan = {'target_signals': ['Ack'], 'trigger_predicates': [predicate('Write', 1), predicate('enabled', 0)],
                'latency': {'min': 7, 'max': 7}, 'exposure': ['state_dependent'],
                'controls': [{'id': 'C1', 'predicates': [predicate('Write', 1), predicate('enabled', 1)]}]}
        cycles = [{'Write': 0}, {'Write': 1}, {'Write': 0}, {'Write': 1}, {'Write': 0}]
        item = {'id': 'enable_then_disable', 'purpose': 'control', 'cycles': cycles,
                'observations': [], 'control_checks': [{'control_id': 'C1', 'cycle': 3}],
                'legality': {'status': 'supported'}}
        golden = {(c, 'Ack'): str(v) for c, v in enumerate([0, 1, 0, 1, 0])}
        golden.update({(c, 'enabled'): str(v) for c, v in enumerate([0, 1, 1, 0, 0])})
        mutant = {**golden, (1, 'Ack'): '0'}
        return executor, plan, item, golden, mutant

    def test_setup_fault_does_not_fail_later_preserved_control(self):
        e, p, item, g, m = self.fixture()
        result = e.characterize(item, g, m, p, {'input_constraints': []})
        control = result['control_checks'][0]
        self.assertTrue(control['activated'])
        self.assertTrue(control['preserved'])
        self.assertEqual(control['start_cycle'], 3)
        self.assertEqual(control['outside_window_differences'][0]['cycle'], 1)
        # If setup is intentionally included in the claim, the mismatch is retained.
        item['control_checks'][0]['start_cycle'] = 1
        self.assertFalse(e.characterize(item, g, m, p, {'input_constraints': []})['control_checks'][0]['preserved'])

    def test_later_difference_inside_default_control_window_still_fails(self):
        e, p, item, g, m = self.fixture()
        m[(4, 'Ack')] = '1'
        result = e.characterize(item, g, m, p, {'input_constraints': []})
        self.assertFalse(result['control_checks'][0]['preserved'])
        self.assertEqual(result['control_checks'][0]['window_differences'][0]['cycle'], 4)

    def test_unreached_trigger_and_preexisting_divergence_do_not_qualify(self):
        e, p, item, g, m = self.fixture()
        item['purpose'] = 'trigger'
        p['controls'] = []
        item['control_checks'] = []
        p['trigger_predicates'] = [predicate('Write', 2)]
        r = e.characterize(item, g, m, p, {'input_constraints': []})
        self.assertFalse(r['probe_verification']['trigger_reached'])
        self.assertFalse(r['intended_observations'])
        p['trigger_predicates'] = [predicate('Write', 1), predicate('enabled', 1)]
        r = e.characterize(item, g, m, p, {'input_constraints': []})
        self.assertEqual(r['trigger_cycles'], [3])
        self.assertFalse(r['first_difference_trigger_aligned'])
        self.assertFalse(r['intended_observations'])

    def test_value_change_requires_target_evidence(self):
        e, p, item, g, m = self.fixture()
        p['target_signals'] = ['enabled']
        item['purpose'] = 'trigger'
        r = e.characterize(item, g, m, p, {'input_constraints': []})
        self.assertTrue(r['output_difference'])
        self.assertFalse(r['probe_verification']['target_changed'])
        self.assertFalse(r['intended_observations'])
        p['target_effect'] = 'update_order'
        self.assertTrue(e.characterize(item, g, m, p, {'input_constraints': []})['intended_observations'])

    def test_empty_predictions_allowed_and_control_window_must_contain_edge(self):
        from dataclasses import replace
        config = replace(Config.load(TINY / 'config.json'), clock='clock', reset='reset')
        model = piezo_fixture(False)
        ports = [{'name': n, 'width': w, 'direction': 'input'} for n, w in [('clock', 1), ('reset', 1), ('Write', 1), ('data', 25)]]
        ports += [{'name': 'Ack', 'width': 1, 'direction': 'output'}]
        raw = copy.deepcopy(model.batches[0][0]); raw['observations'] = []
        self.assertFalse(validate_probe(raw, config, ports, {}, model.plan, 512, 1)['observations'])
        raw = copy.deepcopy(model.batches[0][1]); raw['control_checks'][0]['end_cycle'] = 0
        with self.assertRaisesRegex(ValueError, 'control end_cycle'):
            validate_probe(raw, config, ports, {}, model.plan, 512, 1)


if __name__ == '__main__':
    unittest.main()
