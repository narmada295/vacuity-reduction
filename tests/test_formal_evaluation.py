import unittest

from experiments.evaluation.formal import extract_occurrences
from experiments.evaluation.formal_worker import assess, license_unavailable, statuses


class FormalEvaluationTests(unittest.TestCase):
    def test_embedded_aliases_refer_to_one_goal_and_do_not_hide_proofs(self):
        states = {'top.eval_assert': 'proven', '<embedded>::top.eval_assert': 'proven',
                  'top.eval_trace': 'covered', '<embedded>::top.eval_trace': 'covered',
                  'top.eval_assert:precondition1': 'covered',
                  '<embedded>::top.eval_assert:precondition1': 'covered',
                  'top.eval_assert:contrapositive1': 'covered',
                  '<embedded>::top.eval_assert:contrapositive1': 'covered'}
        result = assess(states, True)
        self.assertEqual(result['assertion_status'], 'proven')
        self.assertEqual(result['trace_status'], 'covered')
        self.assertIs(result['vacuous'], False)
        self.assertEqual(assess(states, False)['reason'], 'outside_scope_non_implication')
        states['<embedded>::top.eval_assert'] = 'cex'
        self.assertEqual(assess(states, True)['assertion_status'], 'conflicting_status')
    def test_successful_license_fallback_does_not_hide_a_syntax_error(self):
        self.assertFalse(license_unavailable('FLEXnet ERROR(-5, 0, 0)\ncheckout succeeded\n'
                                            "[ERROR (VERI-1137)] syntax error near '=>'", False))
        self.assertFalse(license_unavailable('FLEXnet ERROR(-5, 0, 0)\ncheckout succeeded', False))
        self.assertTrue(license_unavailable('FLEXnet ERROR(-15, 0, 0)', False))
        self.assertTrue(license_unavailable('Waiting for license ...', True))
    def test_duplicates_remain_occurrences_and_comment_strings_are_not_assertions(self):
        source = '''```systemverilog
// assert property (fake);
property p; @(posedge clk) req |-> ack; endproperty
a: assert property (p);
b: assert property (p);
```'''
        items, support, error = extract_occurrences(source)
        self.assertEqual(len(items), 2)
        self.assertIsNone(error)
        self.assertEqual([r['assertion'] for r in items], ['assert property (p);'] * 2)
        self.assertNotIn('a:', support)
        self.assertNotIn('b:', support)
        self.assertIn('property p;', support)

    def test_malformed_statements_are_counted_and_not_silently_dropped(self):
        items, _, _ = extract_occurrences('assert property (@(posedge clk) req |-> ack));\nassert property (ok);')
        self.assertEqual(len(items), 2)
        self.assertIsNotNone(items[0]['extraction_error'])
        self.assertIsNone(items[1]['extraction_error'])
        items, _, _ = extract_occurrences('assert property missing_parenthesis;')
        self.assertEqual(len(items), 1)
        self.assertIsNotNone(items[0]['extraction_error'])

    def test_assumptions_and_drivers_are_not_inserted_into_the_golden_rtl(self):
        for statement in ('assume property (req);', 'assign ack = 1;', 'always @(posedge clk) x <= req;'):
            items, _, error = extract_occurrences(statement + '\nassert property (req |-> ack);')
            self.assertEqual(len(items), 1)
            self.assertEqual(error, 'forbidden_generated_context')
        _, _, error = extract_occurrences('// assume things are valid\nassert property (req |-> ack);')
        self.assertIsNone(error)

    def test_nonvacuity_needs_every_related_witness_and_a_nonempty_context(self):
        states = {'top.eval_assert': 'proven', 'top.eval_trace': 'covered',
                  'top.eval_assert:precondition1': 'covered', 'top.eval_assert:contrapositive1': 'covered'}
        self.assertIs(assess(states, True)['vacuous'], False)
        states['top.eval_assert:precondition2'] = 'undetermined'
        self.assertIsNone(assess(states, True)['vacuous'])
        states['top.eval_assert:precondition2'] = 'unreachable'
        result = assess(states, True)
        self.assertIs(result['vacuous'], True)
        self.assertEqual(result['reason'], 'antecedent_never_true')
        states['top.eval_assert:contrapositive1'] = 'unreachable'
        self.assertEqual(assess(states, True)['reason'], 'antecedent_never_true; consequent_always_true')
        states['top.eval_trace'] = 'unreachable'
        self.assertIsNone(assess(states, True)['vacuous'])

    def test_failed_and_non_implication_properties_are_not_nonvacuity_positives(self):
        states = {'top.eval_assert': 'cex'}
        self.assertIsNone(assess(states, True)['vacuous'])
        self.assertEqual(assess(states, True)['reason'], 'assertion_falsified')
        self.assertEqual(assess({'top.eval_assert': 'proven'}, False)['reason'], 'outside_scope_non_implication')
        self.assertIsNone(assess({'top.eval_assert': 'proven'}, False)['vacuous'])


if __name__ == '__main__':
    unittest.main()
