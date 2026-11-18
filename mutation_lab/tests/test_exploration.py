"""Controller tests use scripted model replies and real Verilator/Icarus execution."""

from __future__ import annotations

import argparse
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mutation_lab import exploration_prompts as prompts
from mutation_lab.exploration import run_exploration, _history_pairs
from mutation_lab.exploration_cli import register_exploration
from mutation_lab.exploration_probes import (ProbeExecutor, discover_interface,
                                            predicate_holds, validate_probe)
from mutation_lab.exploration_schema import (exploration_slots, normalize_numeric_json,
                                             predicates, quote, validate_analysis)
from mutation_lab.core import Config, Edit, apply_edits, write_json


ROOT = Path(__file__).resolve().parents[1]
PIEZO = ROOT / "examples/piezo_driver"
TINY = ROOT / "examples/tiny_counter"
TOOLS = all(shutil.which(x) for x in ("verilator", "iverilog", "vvp"))


def arguments(spec, rtl, out, reset=None, *extra):
    parser = argparse.ArgumentParser()
    register_exploration(parser.add_subparsers(dest="action", required=True))
    values = ["explore-campaign", "--spec", str(spec), "--rtl", str(rtl), "--out", str(out),
              "--count", "1", "--max-attempts", "2", "--max-llm-calls", "40",
              "--response-retries", "0", "--max-repairs", "0", "--no-cross-region"]
    if reset:
        values += ["--reset", reset]
    return parser.parse_args(values + list(extra))


def predicate(left, right, op="=="):
    return {"left": left, "op": op, "right": right, "sample": "pre"}


def analysis(quote, signals, family):
    return {"summary": "Small test circuit with a specified output behavior.",
        "behaviors": [{"id": "B1", "description": quote, "spec_quote": quote,
            "signals": signals, "trigger": "An input operation occurs", "expected": quote,
            "timing": "At the specified edge or settled sample", "confidence": "explicit"}],
        "regions": [{"id": "F1", "name": "Output behavior", "description": quote,
            "signals": signals, "behavior_ids": ["B1"], "related_regions": [],
            "families": [{"family": family, "reason": "A plausible mistake in this output logic",
                          "confidence": "supported"}]}], "input_constraints": [], "ambiguities": []}


def make_plan(family, target, trigger, mechanism):
    return {"family": family, "secondary_families": [], "region_ids": ["F1"],
        "behavior_ids": ["B1"], "mechanism": mechanism, "target_signals": [target],
        "golden_behavior": "Every valid operation has the specified response",
        "faulty_behavior": "The planned input sequence produces an incorrect response",
        "trigger": "The selected input operation and state coincide",
        "required_history": [], "exposure": ["sequence_dependent"],
        "observable_consequence": "The output differs at the specified observation",
        "realism": "An incorrect feedback dependency in the update logic",
        "trigger_predicates": trigger, "latency": {"min": 0, "max": 0},
        "controls": [{"id": "C1", "description": "An ordinary isolated operation remains correct",
                      "predicates": [predicate("Write", 1), predicate("Ack", 0)]}]}


def probe(identifier, purpose, steps, quote, observations=None, control=False):
    return {"id": identifier, "purpose": purpose, "description": identifier,
            "steps": [{"inputs": x} for x in steps], "observations": observations or [],
            "control_checks": [{"control_id": "C1", "cycle": 1}] if control else [],
            "legality": {"status": "supported", "reason": "Specified operation with initialized inputs",
                         "spec_quote": quote}}


class ScriptedModel:
    def __init__(self, design_analysis, plan, edit, batches, proposed=1):
        self.analysis, self.plan, self.edit = design_analysis, plan, edit
        self.batches = batches
        self.proposed = proposed
        self.calls = []
        self.probe_round = 0

    def _request(self, system, prompt, **_kwargs):
        payload = json.JSONDecoder().raw_decode(prompt.split("STAGE INPUT:\n", 1)[1])[0]
        self.calls.append((system, payload, prompt))
        assert "FULL SPECIFICATION:" in prompt and "FULL GOLDEN RTL:" in prompt
        if system == prompts.ANALYZE:
            return copy.deepcopy(self.analysis)
        if system == prompts.PLAN:
            plans = [{**copy.deepcopy(self.plan), "mechanism": self.plan["mechanism"] + f" variant {i}"}
                     for i in range(self.proposed)]
            return {"plans": plans, "exploration_note": "Scripted test plans"}
        if system == prompts.SIMILARITY:
            return {"assessments": [{"plan_id": p["id"], "relationship": "novel", "similar_to": [],
                                     "reason": "Fixture mechanism"} for p in payload["proposed"]]}
        if system == prompts.EDIT:
            return {"edits": [self.edit]}
        if system in (prompts.PLAN_CHECK, prompts.EDIT_CHECK):
            names = (["golden_behavior", "actual_change", "trigger", "controls", "consequence"]
                     if system == prompts.PLAN_CHECK else ["mechanism", "trigger", "controls", "single_defect"])
            # Source-owned references are copied by the controller, never fabricated excerpts.
            citation = {"source_file": self.edit['source_file'], "start_line": 1, "end_line": 1}
            result = {"verdict": "supported", "reason": "Scripted semantic assessment for controller tests",
                      "checks": {name: {"assessment": "supported", "reason": "Fixture contract"} for name in names},
                      "controls": [{"id": c['id'], "assessment": "preserved", "reason": "Fixture preserved operation"}
                                   for c in payload['plan']['controls']], "source_evidence": [citation]}
            if system == prompts.EDIT_CHECK:
                result['golden_source_evidence'] = [citation]
            else:
                result['review_subject'] = 'proposed_mutation'
                result['comparison'] = {
                    'golden_rule': payload['plan']['golden_behavior'],
                    'proposed_rule': payload['plan']['faulty_behavior'],
                    'failing_case': {'condition': 'Fixture trigger and setup',
                                     'golden_result': 'Specified response',
                                     'proposed_result': 'Incorrect response', 'different': True},
                    'preserved_cases': [{'control_id': c['id'], 'condition': 'Fixture control',
                                         'golden_result': 'Correct control response',
                                         'proposed_result': 'Correct control response', 'equal': True}
                                        for c in payload['plan']['controls']]}
            return result
        if system == prompts.PROBES:
            result = self.batches[min(self.probe_round, len(self.batches) - 1)]
            self.probe_round += 1
            return {"probes": copy.deepcopy(result)}
        if system == prompts.REVIEW:
            return {"verdict": "accept", "reason": "Actual fixture consequence confirmed by RTL simulation",
                    "issues": [], "realism": {"assessment": "plausible", "reason": "Feedback guard error"},
                    "specification": {"assessment": "supported", "spec_quote": self.analysis["behaviors"][0]["spec_quote"],
                                      "reason": "The measured output contradicts the stated response"},
                    "plan_match": "supported", "implemented_behavior": "The edited update changes the response",
                    "observed_behavior": "Known output difference, exercised and preserved control",
                    "unexpected_effects": [], "unverified_claims": ["Not an exhaustive specification proof"],
                    "measured_fact_check": payload["evidence"]["fact_check"]}
        raise AssertionError("Unexpected stage")


def piezo_fixture(initial_miss=True, proposed=1):
    quote = "The module acknowledges write operations by asserting the `Ack` output in the cycle following a write"
    model_analysis = analysis(quote, ["Write", "Ack", "reset", "data"], "interface_protocol")
    plan = make_plan("interface_protocol", "Ack", [predicate("Write", 1), predicate("Ack", 1),
                                                 predicate("reset", 0)], "Incorrect acknowledgement feedback guard")
    edit = {"source_file": "Piezo_Driver.v", "old": "Ack     <= (reset) ? 0 : Write;",
            "new": "Ack     <= (reset) ? 0 : (Write & ~Ack);"}
    first = {"reset": 1, "Write": 0, "data": 0}
    isolated = [first, {"reset": 0, "Write": 1}, {"Write": 0}, {}, {}]
    sustained = [first, {"reset": 0, "Write": 1}, {"Write": 1}, {"Write": 0}, {}]
    control = probe("control", "control", isolated, quote, control=True)
    adversarial = probe("adversarial", "adversarial", [first,
        {"reset": 1, "Write": 1}, {"reset": 0, "Write": 0}, {}, {}], quote)
    observation = [{"cycle": 2, "signal": "Ack", "golden": 1, "mutant": 0}]
    initial = [probe("missed_trigger", "trigger", isolated, quote, observation), control, adversarial]
    refined = [probe("consecutive_trigger", "trigger", sustained, quote, observation)]
    batches = [initial, refined] if initial_miss else [[refined[0], control, adversarial]]
    return ScriptedModel(model_analysis, plan, edit, batches, proposed)


class ExplorationTests(unittest.TestCase):
    def test_spec_evidence_copies_source_without_accepting_paraphrases(self):
        source = "A `counter` increments on every **enabled** clock.\nReset clears the counter.\n"
        recovered = quote("a counter increments on every enabled clock.", source)
        self.assertIn(recovered, source)
        self.assertIn("**enabled**", recovered)
        self.assertEqual(quote({"start_line": 2, "end_line": 2}, source), "Reset clears the counter.")
        with self.assertRaises(ValueError):
            quote("The counter decrements on each enabled clock.", source)

    def test_simple_predicate_normalization_and_constant_part_select(self):
        parsed = predicates(["data[7:4] == 10"], {"data"}, {"data": 8}, 1)
        self.assertTrue(predicate_holds(parsed, 0, [{"data": 0xa5}], {}))
        with self.assertRaises(ValueError):
            predicates(["data == 1 || data == 2"], {"data"}, {"data": 8})
        with self.assertRaises(ValueError):
            predicates(["data[8:4] == 10"], {"data"}, {"data": 8})

    def test_numeric_json_recovery_preserves_strings_and_rejects_expressions(self):
        raw = '''{"data":25'h1000001,"hex":0xff,"text":"25'h1000001", "quoted":"a\\\"b"}'''
        recovered = json.loads(normalize_numeric_json(raw))
        self.assertEqual(recovered["data"], 16777217)
        self.assertEqual(recovered["hex"], 255)
        self.assertEqual(recovered["text"], "25'h1000001")
        with self.assertRaises(ValueError):
            normalize_numeric_json('''{"data":4'hff}''')
        with self.assertRaises(json.JSONDecodeError):
            json.loads(normalize_numeric_json('{"data":0xff + 1}'))

    def test_environment_constraints_cannot_mislabel_dut_responses(self):
        config = Config.load(TINY / "config.json")
        spec = (TINY / "spec.txt").read_text()
        q = "A request is accepted when start is high and the counter is idle."
        a = analysis(q, ["start", "active", "done"], "control")
        a["input_constraints"] = [{"id": "E1", "description": "An output response is not an input restriction",
            "status": "established", "spec_quote": q, "when": [predicate("start", 1)],
            "require": [predicate("done", 0)]}]
        widths = {"clk": 1, "rst_n": 1, "start": 1, "active": 1, "done": 1}
        with self.assertRaisesRegex(ValueError, "restrict external inputs"):
            validate_analysis(a, config, spec, set(widths), widths, {"clk", "rst_n", "start"})

    def test_array_response_and_raw_failed_response_recording(self):
        from dataclasses import replace
        from mutation_lab.planner import LLMPlanner
        with tempfile.TemporaryDirectory() as temp:
            raw_file = Path(temp) / "raw.txt"
            client = object.__new__(LLMPlanner)
            client.config = replace(Config.load(TINY / "config.json"),
                                    planner={"raw_response_file": str(raw_file)})
            client.model, client.endpoint, client.key = "test", "https://example.invalid", "test-secret"
            payload = {"choices": [{"finish_reason": "stop", "message": {"content": '[{"id":"p1"}]'}}]}
            with patch("mutation_lab.planner.urlopen", return_value=io.StringIO(json.dumps(payload))):
                self.assertEqual(client._request("system", "prompt", array_key="probes"), {"probes": [{"id": "p1"}]})
            self.assertEqual(raw_file.read_text(), '[{"id":"p1"}]')
            payload["choices"][0]["finish_reason"] = "length"
            payload["choices"][0]["message"]["content"] = '{"truncated":'
            with patch("mutation_lab.planner.urlopen", return_value=io.StringIO(json.dumps(payload))):
                with self.assertRaisesRegex(ValueError, "truncated"):
                    client._request("system", "prompt")
            self.assertEqual(raw_file.read_text(), '{"truncated":')

    def test_delayed_output_does_not_require_delayed_first_state_change(self):
        executor = object.__new__(ProbeExecutor)
        executor.outputs, executor.state = {"done"}, {"count": 2}
        plan = {"target_signals": ["count"], "trigger_predicates": [predicate("start", 1)],
                "latency": {"min": 2, "max": 2}, "controls": []}
        item = {"id": "p", "purpose": "trigger", "cycles": [{"start": 0}, {"start": 1},
                {"start": 0}, {"start": 0}], "observations": [
                    {"cycle": 3, "signal": "done", "golden": 0, "mutant": 1}],
                "control_checks": [], "legality": {"status": "supported"}}
        golden = {(c, s): "0" for c in range(4) for s in ("count", "done")}
        mutant = {**golden, (1, "count"): "1", (3, "done"): "1"}
        result = executor.characterize(item, golden, mutant, plan, {"input_constraints": []})
        self.assertTrue(result["first_difference_trigger_aligned"])
        self.assertTrue(any(d['signal'] == 'done' for d in result['intended_observations']))

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_adaptive_consecutive_write_fault_and_score_compatibility(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "campaign"
            args = arguments(PIEZO / "spec.txt", PIEZO / "rtl/Piezo_Driver.v", out, "reset")
            client = piezo_fixture()
            golden_before = (PIEZO / "rtl/Piezo_Driver.v").read_bytes()
            # New mode must not invoke the old fixed stimulus generator.
            with patch("mutation_lab.quality_validation.make_probes", side_effect=AssertionError("legacy probes used")):
                summary = run_exploration(args, client=client)
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["stop_reason"], "requested_count")
            self.assertEqual(client.probe_round, 2)
            record = json.loads((out / "mutants/m0001/record.json").read_text())
            self.assertEqual(record["validation"]["fact_check"]["first_intended_difference"]["cycle"], 2)
            self.assertEqual(record["validation"]["fact_check"]["controls_preserved"], ["C1"])
            self.assertEqual(record["status"], "generated_unqualified")
            self.assertTrue((out / "mutants/m0001/rtl/Piezo_Driver.v").is_file())
            saved = Config.load(out / "config.direct.json")
            self.assertEqual(record["edit"]["source_snapshot"], saved.source_snapshot())
            self.assertEqual(golden_before, (PIEZO / "rtl/Piezo_Driver.v").read_bytes())
            requests = list((out / "llm_calls").glob("*/request.json"))
            self.assertTrue(requests)
            self.assertTrue(all("FULL SPECIFICATION:" in p.read_text() for p in requests))
            from mutation_lab.exploration_compare import compare_probes
            comparison = compare_probes(out)
            self.assertFalse(comparison["mutants"][0]["legacy_output_difference"])
            self.assertTrue(comparison["mutants"][0]["targeted_output_difference"])

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_more_than_five_plans_and_pending_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / "spec.txt", PIEZO / "rtl/Piezo_Driver.v", Path(temp) / "run", "reset",
                             "--plans-per-batch", "20")
            summary = run_exploration(args, client=piezo_fixture(False, proposed=12))
            self.assertEqual(summary["plans_proposed"], 12)
            self.assertEqual(summary["accepted"], 1)
            archive = json.loads((args.out / "mechanism_archive.json").read_text())
            self.assertEqual(len(archive), 12)
            self.assertEqual(sum("outcome" not in p for p in archive), 11)

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_unobserved_not_equivalent_or_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / "spec.txt", PIEZO / "rtl/Piezo_Driver.v", Path(temp) / "run", "reset",
                             "--probe-refinements", "0", "--max-attempts", "1")
            summary = run_exploration(args, client=piezo_fixture())
            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["counts"], {"unresolved": 1})
            record = json.loads((args.out / "mutants/m0001/record.json").read_text())
            self.assertEqual(record["validation"]["evidence_level"], "unobserved")
            self.assertFalse((args.out / "mutants/m0001/rtl").exists())

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_call_budget_preserves_partial_record(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / "spec.txt", PIEZO / "rtl/Piezo_Driver.v", Path(temp) / "run", "reset",
                             "--max-llm-calls", "4")
            summary = run_exploration(args, client=piezo_fixture())
            self.assertEqual(summary["stop_reason"], "max_llm_calls")
            self.assertEqual(summary["counts"], {"unresolved_budget": 1})
            self.assertTrue((args.out / "mutants/m0001/record.json").exists())

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_second_sequential_design_active_low_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            quote = "After an accepted request, done must remain low for three clock cycles"
            a = analysis(quote, ["start", "done", "active", "count", "rst_n"], "timing")
            p = make_plan("timing", "count", [predicate("start", 1), predicate("active", 0),
                                             predicate("rst_n", 1)], "Load a counter with one fewer remaining cycles")
            p["latency"] = {"min": 0, "max": 3}
            p["controls"][0]["predicates"] = [predicate("start", 0), predicate("active", 0)]
            e = {"source_file": "tiny_counter.sv", "old": "count <= 2'd2;", "new": "count <= 2'd1;"}
            init = {"rst_n": 0, "start": 0}
            trigger = probe("request", "trigger", [init, {"rst_n": 1, "start": 1}, {"start": 0}, {}, {}, {}],
                            quote, [{"cycle": 3, "signal": "done", "golden": 0, "mutant": 1}])
            control = probe("idle", "control", [init, {"rst_n": 1}, {}, {}], quote, control=True)
            adversarial = probe("cancel", "adversarial", [init, {"rst_n": 1, "start": 1},
                                {"rst_n": 0, "start": 0}, {"rst_n": 1}, {}, {}], quote)
            args = arguments(TINY / "spec.txt", TINY / "rtl/tiny_counter.sv", Path(temp) / "run", "rst_n",
                             *(["--synthesis-check"] if shutil.which("yosys") else []))
            self.assertEqual(run_exploration(args, client=ScriptedModel(a, p, e, [[trigger, control, adversarial]]))["accepted"], 1)
            from mutation_lab.campaign import retry_scoring
            with patch("mutation_lab.campaign.run_check", side_effect=[
                    {"status": "completed", "properties": {"EVAL_NO_EARLY_DONE": "proven"}},
                    {"status": "completed", "properties": {"EVAL_NO_EARLY_DONE": "cex"}}]):
                retry_scoring(argparse.Namespace(out=args.out, assertions=TINY / "eval.sv",
                                                max_assertions=0, jg_timeout=None))
            scored = json.loads((args.out / "evaluation/results.json").read_text())
            self.assertEqual(scored["summary"]["killed_mutants"], 1)

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_combinational_design_without_clock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rtl, spec = root / "parity.sv", root / "spec.txt"
            rtl.write_text("module parity(input [1:0] word, output wire y); assign y = ^word; endmodule\n")
            quote = "Parity output equals the XOR of both input bits."
            spec.write_text(quote)
            a = analysis(quote, ["word", "y"], "data")
            p = make_plan("data", "y", [predicate("word[1]", 1)], "Ignore the upper operand bit")
            p["exposure"] = ["immediate"]
            p["controls"][0]["predicates"] = [predicate("word", 0)]
            e = {"source_file": "parity.sv", "old": "assign y = ^word;", "new": "assign y = word[0];"}
            trigger = probe("upper_bit", "trigger", [{"word": 0}, {"word": 2}], quote,
                            [{"cycle": 1, "signal": "y", "golden": 1, "mutant": 0}])
            control = probe("zero", "control", [{"word": 0}, {}], quote, control=True)
            adversarial = probe("both_bits", "adversarial", [{"word": 0}, {"word": 3}], quote)
            args = arguments(spec, rtl, root / "run")
            summary = run_exploration(args, client=ScriptedModel(a, p, e, [[trigger, control, adversarial]]))
            self.assertEqual(summary["accepted"], 1)
            self.assertIsNone(Config.load(args.out / "config.direct.json").clock)

    def test_functional_regions_overlap_and_cross_only_declared_relationships(self):
        a = analysis("A sufficiently long specification quote.", ["a", "b"], "control")
        region = copy.deepcopy(a["regions"][0])
        region["id"] = "F2"
        a["regions"][0]["related_regions"] = ["F2"]
        a["regions"].append(region)
        slots = exploration_slots(a, True, False)
        self.assertEqual(len(slots), 3)
        self.assertEqual(slots[-1]["region_ids"], ["F1", "F2"])

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_missing_controls_override_generous_review(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / "spec.txt", PIEZO / "rtl/Piezo_Driver.v", Path(temp) / "run", "reset",
                             "--probe-refinements", "0", "--max-attempts", "1")
            model = piezo_fixture(False)
            model.plan["controls"][0]["predicates"] = [predicate("data", 1)]
            summary = run_exploration(args, client=model)
            self.assertEqual(summary["accepted"], 0)
            record = json.loads((args.out / "mutants/m0001/record.json").read_text())
            self.assertEqual(record["review"]["verdict"], "accept")
            self.assertEqual(record["status"], "unresolved")
            self.assertEqual(record["validation"]["fact_check"]["controls_missing"], ["C1"])

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_relabeling_a_control_does_not_create_adversarial_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / "spec.txt", PIEZO / "rtl/Piezo_Driver.v", Path(temp) / "run", "reset",
                             "--probe-refinements", "0", "--max-attempts", "1")
            model = piezo_fixture(False)
            model.batches[0][2]["steps"] = copy.deepcopy(model.batches[0][1]["steps"])
            summary = run_exploration(args, client=model)
            self.assertEqual(summary["accepted"], 0)
            record = json.loads((args.out / "mutants/m0001/record.json").read_text())
            self.assertEqual(record["validation"]["fact_check"]["adversarial_probes_exercised"], 0)

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_control_compares_whole_trajectory(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / "spec.txt", PIEZO / "rtl/Piezo_Driver.v", Path(temp) / "run", "reset",
                             "--probe-refinements", "0", "--max-attempts", "1")
            model = piezo_fixture(False)
            # The claimed control initially agrees, but differs on the following write.
            model.batches[0][1]["steps"] = copy.deepcopy(model.batches[0][0]["steps"])
            summary = run_exploration(args, client=model)
            self.assertEqual(summary["accepted"], 0)
            record = json.loads((args.out / "mutants/m0001/record.json").read_text())
            mismatch = record["validation"]["control_mismatches"][0]
            self.assertTrue(mismatch["activated"])
            self.assertFalse(mismatch["preserved"])

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_probe_format_retry_does_not_consume_edit_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / "spec.txt", PIEZO / "rtl/Piezo_Driver.v", Path(temp) / "run", "reset",
                             "--response-retries", "1", "--probe-refinements", "0")
            model = piezo_fixture(False)
            broken = copy.deepcopy(model.batches[0])
            broken[0]["steps"][0]["inputs"]["invented_signal"] = 1
            model.batches.insert(0, broken)
            summary = run_exploration(args, client=model)
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(model.probe_round, 2)
            self.assertEqual(sum(s == prompts.EDIT for s, _, _ in model.calls), 1)

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_failed_probe_response_is_unresolved_without_rewriting_rtl(self):
        with tempfile.TemporaryDirectory() as temp:
            args = arguments(PIEZO / "spec.txt", PIEZO / "rtl/Piezo_Driver.v", Path(temp) / "run", "reset",
                             "--max-attempts", "1", "--max-repairs", "1")
            model = piezo_fixture(False)
            model.batches[0][0]["steps"][0]["inputs"]["data"] = 1 << 25
            summary = run_exploration(args, client=model)
            self.assertEqual(summary["counts"], {"unresolved": 1})
            self.assertEqual(sum(s == prompts.EDIT for s, _, _ in model.calls), 1)
            record = json.loads((args.out / "mutants/m0001/record.json").read_text())
            self.assertEqual(record["revisions"][0]["status"], "unresolved_probes")

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_analysis_cache_and_repeated_planning_rounds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = arguments(PIEZO / "spec.txt", PIEZO / "rtl/Piezo_Driver.v", root / "first", "reset",
                             "--analysis-cache", str(root / "cached.json"))
            run_exploration(args, client=piezo_fixture(False))
            second = arguments(PIEZO / "spec.txt", PIEZO / "rtl/Piezo_Driver.v", root / "second", "reset",
                               "--analysis-cache", str(root / "cached.json"), "--count", "0",
                               "--plans-per-batch", "1", "--max-attempts", "2", "--max-planning-calls", "2")
            model = piezo_fixture(False)
            original_request = model._request
            def respond(system, prompt, **kwargs):
                reply = original_request(system, prompt, **kwargs)
                if system == prompts.PLAN:
                    # A second distinct plan description reaches independent RTL editing;
                    # identical actual RTL is correctly rejected later.
                    n = sum(s == prompts.PLAN for s, _, _ in model.calls)
                    reply["plans"][0]["mechanism"] += " exploration round " + str(n)
                return reply
            model._request = respond
            summary = run_exploration(second, client=model)
            self.assertFalse(any(s == prompts.ANALYZE for s, _, _ in model.calls))
            self.assertEqual(summary["planning_calls"], 2)
            self.assertEqual(summary["attempts"], 2)
            self.assertEqual(summary["accepted"], 1)
            second_record = json.loads((second.out / "mutants/m0002/record.json").read_text())
            self.assertIn("duplicates m0001", second_record["reason"])

    def test_probe_schema_rejects_driving_clock_width_overflow_and_missing_initialization(self):
        from dataclasses import replace
        config = replace(Config.load(TINY / "config.json"), clock="clock", reset="reset")
        ports = [{"name": name, "direction": "input", "width": width} for name, width in
                 (("clock", 1), ("reset", 1), ("Write", 1), ("data", 25))]
        ports += [{"name": "Ack", "direction": "output", "width": 1}]
        model = piezo_fixture(False)
        raw = model.batches[0][0]
        for change in ({"clock": 1}, {"data": 1 << 25}, {"Write": True}):
            broken = copy.deepcopy(raw)
            broken["steps"][0]["inputs"].update(change)
            with self.assertRaises(ValueError):
                validate_probe(broken, config, ports, {}, model.plan, 512, 1)
        broken = copy.deepcopy(raw)
        del broken["steps"][0]["inputs"]["data"]
        with self.assertRaisesRegex(ValueError, "initialize every"):
            validate_probe(broken, config, ports, {}, model.plan, 512, 1)
        with self.assertRaisesRegex(ValueError, "exceeds --max-probe-cycles"):
            validate_probe(raw, config, ports, {}, model.plan, 2, 1)

    @unittest.skipUnless(TOOLS, "Verilator/Icarus required")
    def test_additional_disable_effect_is_measured(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = Config.load(TINY / "config.json")
            from dataclasses import replace
            config = replace(config, design_root=PIEZO / "rtl", sources=(Path("Piezo_Driver.v"),),
                             include_dirs=(Path("."),), top="Piezo_Driver", clock="clock", reset="reset")
            ports, state = discover_interface(config, root / "interface")
            old = "enabled <= (reset) ? 0 : ((Write) ? data[24] : enabled);"
            apply_edits(config, (Edit("Piezo_Driver.v", old, old.replace("<=", "=", 1)),), root / "mutant")
            p = make_plan("timing", "enabled", [predicate("Write", 1), predicate("enabled", 0),
                                                predicate("data[24]", 1)], "Blocking enable capture")
            p['target_effect'] = 'update_order'
            q = piezo_fixture().analysis["behaviors"][0]["spec_quote"]
            steps = [{"reset": 1, "Write": 0, "data": 0}, {"reset": 0, "Write": 1, "data": 3},
                     {"data": (1 << 24) | 3}, {"Write": 0}, {}, {}, {"Write": 1, "data": 3}, {}]
            raw = probe("disable", "adversarial", steps, q)
            item = validate_probe(raw, config, ports, state, p, 512, 1)
            executor = ProbeExecutor(config, ports, state, root)
            golden, mutant = executor.execute(root / "mutant/rtl", item, root / "simulation")
            measured = executor.characterize(item, golden, mutant, p, {"input_constraints": []})
            self.assertEqual(measured["first_difference"]["cycle"], 6)
            self.assertEqual(measured["first_difference"]["signal"], "Piezo")
            self.assertTrue(measured["first_difference_trigger_aligned"])
            self.assertFalse(measured['probe_verification']['target_changed'])
            self.assertTrue(measured["additional_differences"])


if __name__ == "__main__":
    unittest.main()
