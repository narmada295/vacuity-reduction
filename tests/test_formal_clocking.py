import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.assertion_extraction.formal_clocking import clock_commands
from experiments.assertion_extraction.check_premise_consistency import prepare as consistency_prepare
from experiments.assertion_extraction.formal_redundancy import cmd_init, job_fingerprint
from experiments.assertion_extraction.validate_reduced_set import prepare as validation_prepare
from experiments.checker_coverage.run_batch import tcl_text


class FormalClockContextTests(unittest.TestCase):
    context = {'version': 1, 'declarations': [{'signal': 'clk', 'edges': 'both'}]}

    def test_legacy_defaults_and_invalid_contexts(self):
        self.assertEqual(clock_commands([]), 'clock -none')
        self.assertEqual(clock_commands(['clk']), 'clock clk')
        for context in [
            {'version': 2, 'declarations': []},
            {'version': 1, 'declarations': []},
            {'version': 1, 'declarations': [{'signal': 'clk; exit', 'edges': 'both'}]},
            {'version': 1, 'declarations': [{'signal': 'clk', 'edges': 'invalid'}]},
            {'version': 1, 'declarations': [{'signal': 'clk', 'edges': 'both'}] * 2},
        ]:
            with self.assertRaises(ValueError):
                clock_commands(['clk'], context)

    def test_all_proof_stages_share_context_and_reset_stays_unconstrained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rtl = 'module top(input clk, rst_b, a, b); endmodule'
            rows = []
            for index, expression in enumerate([
                '@(posedge clk) a |-> b', '@(negedge rst_b) b == 0'
            ], 1):
                rows.append({'row_number':index, 'case_id':'L3/sample', 'design':'top',
                    'rtl_sha256':hashlib.sha256(rtl.encode()).hexdigest(),
                    'formal_reset_policy':'unconstrained', 'clock':'clk',
                    'property_expression':expression, 'assertion':'assert property ('+expression+');',
                    'extraction_status':'ok', 'selected':True, 'rtl':rtl})
            source = root/'selected.jsonl'
            source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            case = root/'case.json'
            case.write_text(json.dumps({'rows':rows, 'formal_clocking':self.context}))
            consistency = root/'consistency'
            consistency_prepare(SimpleNamespace(input=str(source), case=str(case), out=str(consistency)))
            run = root/'pruning'
            cmd_init(SimpleNamespace(input=str(source), case=str(case), rtl=None, decls=None, out=str(run)))
            state_path = run/'state.json'
            state = json.loads(state_path.read_text())
            self.assertEqual(state['formal_clocking'], self.context)
            pruning_tcl = next((run/'jobs').glob('*/run.tcl')).read_text()
            before = job_fingerprint('1', ['2'], state)
            legacy_state = dict(state, formal_clocking=None)
            self.assertNotEqual(before, job_fingerprint('1', ['2'], legacy_state))
            state['remaining'] = ['1']
            state['removed'] = [{'target_id':'2', 'result':'PROVEN', 'removed':True,
                                 'statuses':{'TARGET_CHECK':'proven'}}]
            state_path.write_text(json.dumps(state))
            validation = root/'validation'
            validation_prepare(SimpleNamespace(run=str(run), out=str(validation)))
            scripts = [(consistency/'run.tcl').read_text(), pruning_tcl,
                       (validation/'run.tcl').read_text(),
                       tcl_text('top', ['clk','rst_b'], False, self.context),
                       tcl_text('top', ['clk','rst_b'], True, self.context)]
            for script in scripts:
                self.assertIn('clock -both_edges clk\n', script)
                self.assertNotIn('clock rst_b', script)
                self.assertIn('reset -none', script)


if __name__ == '__main__':
    unittest.main()
