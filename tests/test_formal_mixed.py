import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.evaluation import formal, formal_mixed
from experiments.evaluation.prepare_mixed import task_for_case


class MixedFormalTests(unittest.TestCase):
    def test_scope_ignores_inline_comments_without_rewriting_assertions(self):
        for statement in (
            'assert property (@(posedge clk) req |-> ack) /* a comment */;',
            'assert property (@(posedge clk) req |=> ack) // a comment\n;',
        ):
            original = {'assertion': statement, 'selected_support': ''}
            self.assertIs(formal_mixed.implication_scope(original), True)
            self.assertEqual(original['assertion'], statement)
        self.assertIs(formal_mixed.implication_scope({
            'assertion': 'assert property (@(posedge clk) ack /* not an implication |-> */);'}), False)
        self.assertIs(formal_mixed.implication_scope({
            'assertion': 'assert property (p) /* named property */;',
            'selected_support': 'property p; @(posedge clk) req |-> ack; endproperty'}), True)

    def test_external_context_is_bound_to_the_complete_input_and_both_clocks(self):
        case = {'case_id': 'L3/9', 'top': 'ethernet', 'spec': 'Reports receive and transmit errors.',
                'rtl': 'module ethernet(input MRxClk, MTxClk, Reset, error); endmodule',
                'signals': [{'name': 'error', 'information': 'Error status.'}],
                'formal_context': {'clock': 'MRxClk', 'checker_clocks': ['MRxClk', 'MTxClk'],
                                   'formal_reset_policy': 'unconstrained', 'formal_clocking': None}}
        task = task_for_case(case, case['signals'][0])
        contexts = formal_mixed.load_contexts([task], [case])
        self.assertEqual(contexts[task['task_id']]['clock_tcl'], 'clock MRxClk\nclock MTxClk')
        self.assertEqual(contexts[task['task_id']]['rtl'], case['rtl'])
        changed = dict(case, rtl=case['rtl'] + '\n// changed source')
        with self.assertRaisesRegex(ValueError, 'differs from the original inference prompt'):
            formal_mixed.load_contexts([task], [changed])

    def test_supplied_contexts_never_fall_back_to_teacher_reference_assertions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            worker = root / 'worker.py'
            worker.write_text('# test worker\n')
            task = {'task_id': 'external', 'case_id': 'OpenTitan/test', 'target_signal': 'req'}
            row = dict(task, response='assert property (@(posedge clk) req |-> ack);')
            context = {'rtl': 'module external(input clk,req,ack); endmodule', 'top': 'external',
                       'clock_tcl': 'clock clk', 'formal_context': {'checker_clocks': ['clk']}}
            with patch.object(formal, 'load_contexts', side_effect=AssertionError('Wrong context lookup')):
                result = formal.prepare(root / 'evaluation', 180,
                                        loaded=([task], {'base': [row]}, {'signal_count': 1}, 'archive'),
                                        labels=('base',), worker_source=worker, contexts={'external': context})
                self.assertEqual(result['unique_jobs'], 1)
                with self.assertRaisesRegex(ValueError, 'context population differs'):
                    formal.prepare(root / 'bad', 180,
                                   loaded=([task], {'base': [row]}, {'signal_count': 1}, 'archive'),
                                   labels=('base',), worker_source=worker, contexts={})

    def test_comparison_withholds_pending_rates_and_keeps_undefined_vacuity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / 'formal'
            out.mkdir()
            (out / 'manifest.json').write_text(json.dumps({'protocol': {'signal_count': 1, 'signals_by_design': {'design': 1}}}))
            responses = [{'label': label, 'case_id': 'design', 'no_assertions': False, 'cap_violation': False}
                         for label in ('base', 'sft', 'kto')]
            results = [{'label': label, 'case_id': 'design', 'task_id': 'signal',
                        'assertion_status': 'pending' if label == 'base' else 'proven',
                        'reason': 'formal_pending' if label == 'base' else 'outside_scope_non_implication'}
                       for label in ('base', 'sft', 'kto')]
            for name, rows in [('responses.jsonl', responses), ('results.jsonl', results)]:
                (out / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
            counts = {label: {'metrics_final': label != 'base', 'assertion_occurrences': 1,
                              'proved': int(label != 'base'), 'proved_non_implication': int(label != 'base')}
                      for label in ('base', 'sft', 'kto')}
            summary = {'models': counts, 'per_design': {'design': counts}}
            with patch.object(formal_mixed, 'ROOT', root):
                result = formal_mixed.write_comparison(out, summary)
            per = result['per_design']['design']
            self.assertIsNone(per['base']['proof_success_rate'])
            self.assertEqual(per['base']['signals_remaining'], 1)
            self.assertIsNone(per['sft']['conditional_vacuity_rate'])
            self.assertEqual(per['sft']['proof_success_rate'], 1.0)
            self.assertFalse(result['all_models_formally_evaluated'])
            self.assertTrue((root / 'comparison_external_mixed_v4.csv').exists())


if __name__ == '__main__':
    unittest.main()
