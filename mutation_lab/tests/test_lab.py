from __future__ import annotations

import json
import io
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mutation_lab.cli import mutation_schedule, run_evaluation, run_generation
from mutation_lab.campaign import direct_config, retry_scoring, run_campaign
from mutation_lab.assertions import load_assertions
from mutation_lab.core import (Config, Edit, Proposal, ScenarioMutation, apply_edits, apply_proposal,
                               apply_scenario, rtl_tokens, stage_sources, write_json)
from mutation_lab.formal import named_status, parse_statuses, prepare_job
from mutation_lab.planner import LLMPlanner, _json_object
from mutation_lab.scenario_validation import validate_scenario
from mutation_lab.trace_guided import run_trace_campaign
from mutation_lab.differential_formal import build_miter
from mutation_lab.scenario_validation import _ports
from mutation_lab.quality_campaign import (Behavior, FaultPlan, _concept_key,
                                           _request_critique, _request_edits, _request_plan,
                                           run_quality_campaign)
from mutation_lab.quality_validation import BehaviorAuditor, make_probes, obvious_constant_edit


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tiny_counter"
PIEZO = Path(__file__).resolve().parents[1] / "examples" / "piezo_driver"


def _model_with_measured_critic(items: list, critic: dict):
    responses = iter(items)

    def answer(_system: str, prompt: str, **_kwargs) -> dict:
        try:
            response = next(responses)
        except StopIteration:
            audit_text = prompt.split("MEASURED BEHAVIOR AUDIT:\n", 1)[1].split(
                "\n\nPRIOR ACCEPTED FAULT PLANS:", 1)[0]
            audit = json.loads(audit_text)
            return {**critic, "measured_first_difference": audit["witness_check"],
                    "observed_behavior": "Measured target and output divergence"}
        if isinstance(response, Exception):
            raise response
        return response

    return answer


class MutationLabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config.load(EXAMPLE / "config.json")

    def test_provider_options_preserve_model_identity_and_replace_thinking_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            options = Path(temp) / "options.json"
            options.write_text(json.dumps({"temperature": 1, "reasoning_effort": "high"}))
            args = Namespace(spec=PIEZO / "spec.txt", rtl=[PIEZO / "rtl" / "Piezo_Driver.v"],
                             out=Path(temp) / "run", design_root=None, include_dir=[],
                             focus_rtl=[], top=None, clock=None, reset="reset", jg_mode="ssh",
                             jg_timeout=180, jg_executable="jg", env_file=Path(".env"),
                             endpoint="https://api.tokenfactory.nebius.com/v1/chat/completions",
                             model="moonshotai/Kimi-K3", key_env="NEBIUS_API_KEY1",
                             max_context_chars=100000, planner_max_tokens=16384, lint_arg=[],
                             planner_extra_body=options, planner_timeout=300)
            planner = direct_config(args)["planner"]
            self.assertEqual(planner["extra_body"], {"temperature": 1, "reasoning_effort": "high"})
            self.assertEqual(planner["timeout_seconds"], 300)
            args.planner_extra_body = None
            self.assertFalse(direct_config(args)["planner"]["extra_body"]["chat_template_kwargs"]["thinking"])
            args.planner_extra_body = options
            options.write_text(json.dumps({"model": "unexpected-model"}))
            with self.assertRaisesRegex(ValueError, "cannot override"):
                direct_config(args)

    def test_model_request_logs_usage_and_rejects_truncated_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "requests.jsonl"
            client = object.__new__(LLMPlanner)
            client.config = replace(self.config, planner={"max_tokens": 16384,
                "timeout_seconds": 300, "request_log": str(log_path),
                "extra_body": {"reasoning_effort": "high", "temperature": 1}})
            client.model = "moonshotai/Kimi-K3"
            client.endpoint = "https://example.invalid/v1/chat/completions"
            client.key = "secret-test-key"
            payload = {"model": client.model, "usage": {"completion_tokens": 17},
                       "choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}]}
            with patch("mutation_lab.planner.urlopen", return_value=io.StringIO(json.dumps(payload))) as send:
                self.assertEqual(client._request("system", "private prompt"), {"ok": True})
                body = json.loads(send.call_args.args[0].data)
                self.assertEqual(body["reasoning_effort"], "high")
                self.assertEqual(body["temperature"], 1)
                self.assertEqual(send.call_args.kwargs["timeout"], 300)
            payload["choices"][0]["finish_reason"] = "length"
            with patch("mutation_lab.planner.urlopen", return_value=io.StringIO(json.dumps(payload))):
                with self.assertRaisesRegex(ValueError, "truncated"):
                    client._request("system", "private prompt")
            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual([row["finish_reason"] for row in rows], ["stop", "length"])
            self.assertEqual(rows[0]["usage"]["completion_tokens"], 17)
            self.assertNotIn("secret-test-key", log_path.read_text())
            self.assertNotIn("private prompt", log_path.read_text())

    def test_fault_predicates_normalize_json_integers_without_accepting_booleans(self) -> None:
        client = Mock()
        behavior = Behavior("R001", "S001", "Writes update the enable flag.",
                            "Writes update the enable flag.", ("enabled", "Write"))
        proposal = {"requirement_id": "R001", "category": "simple",
                    "trigger": "An enabling write occurs while disabled",
                    "golden_behavior": "The write sets enabled to one",
                    "faulty_behavior": "The enabling write is ignored",
                    "root_cause": "The enable update guard is inverted",
                    "target_signal": "enabled", "why_realistic": "An inverted update guard",
                    "trigger_predicates": [{"left": "Write", "op": "==", "right": 1}],
                    "control_predicates": [[{"left": "Write", "op": "==", "right": 0}]]}
        client._request.return_value = proposal
        plan = _request_plan(client, behavior, "simple", "RTL", {"enabled", "Write"}, [], 100000)
        self.assertEqual(plan.trigger_predicates[0]["right"], "1")
        self.assertEqual(plan.control_predicates[0][0]["right"], "0")
        proposal["trigger_predicates"][0]["right"] = True
        with self.assertRaisesRegex(ValueError, "operands/op"):
            _request_plan(client, behavior, "simple", "RTL", {"enabled", "Write"}, [], 100000)

    def test_behavior_probes_cover_data_transitions_and_distinct_buses(self) -> None:
        ports = [{"name": "clk", "direction": "input", "width": 1},
                 {"name": "rst_n", "direction": "input", "width": 1},
                 {"name": "request", "direction": "input", "width": 1},
                 {"name": "cancel", "direction": "input", "width": 1},
                 {"name": "data_a", "direction": "input", "width": 4},
                 {"name": "data_b", "direction": "input", "width": 4},
                 {"name": "done", "direction": "output", "width": 1}]
        probes = make_probes(self.config, ports, 48)
        self.assertTrue(any(any("data_a" in cycle for cycle in probe.cycles)
                            for probe in probes))
        self.assertTrue(any(any("data_b" in cycle for cycle in probe.cycles)
                            for probe in probes))
        self.assertTrue(any(probe.kind == "reset_overlap" for probe in probes))
        self.assertTrue(any(probe.event == ("request", "cancel") and
                            any(cycle.get("request") == cycle.get("cancel") == 1
                                for cycle in probe.cycles) for probe in probes))
        self.assertTrue(any(probe.kind == "repeat" and
                            any(cycle.get("data_a", 0) & 8 for cycle in probe.cycles) and
                            any("data_a" in cycle and not cycle["data_a"] & 8
                                for cycle in probe.cycles[2:]) for probe in probes))

    def test_behavior_audit_checks_state_sequence_and_trigger(self) -> None:
        config = replace(self.config, design_root=PIEZO / "rtl",
                         sources=(Path("Piezo_Driver.v"),), top="Piezo_Driver",
                         clock="clock", reset="reset")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            enable_old = "enabled <= (reset) ? 0 : ((Write) ? data[24] : enabled);"
            enable_edit = Edit("Piezo_Driver.v", enable_old,
                               "enabled <= (reset) ? 0 : (enabled ? enabled : (Write ? data[24] : enabled));")
            apply_edits(config, (enable_edit,), root / "sequence")
            count_old = "count   <= (reset | (count == compare)) ? 24'h000000 : count + 1;"
            count_edit = Edit("Piezo_Driver.v", count_old,
                              count_old.replace("(count == compare)", "(count >= compare)"))
            apply_edits(config, (count_edit,), root / "mismatch")
            stuck_edit = Edit("Piezo_Driver.v", enable_old,
                              "enabled <= (reset) ? 0 : 0;")
            apply_edits(config, (stuck_edit,), root / "stuck")
            with BehaviorAuditor(config) as auditor:
                sequence = auditor.assess(root / "sequence" / "rtl",
                    SimpleNamespace(category="state_sequence", target_signal="enabled",
                                    trigger="Write=1 while enabled=1",
                                    root_cause="The update is blocked after enabling"),
                    (enable_edit,))
                mismatch = auditor.assess(root / "mismatch" / "rtl",
                    SimpleNamespace(category="simple", target_signal="count",
                                    trigger="count == compare",
                                    root_cause="The equality guard changed"),
                    (count_edit,))
                stuck = auditor.assess(root / "stuck" / "rtl",
                    SimpleNamespace(category="simple", target_signal="enabled",
                                    trigger="Write=1", root_cause="Enable is stuck low"),
                    (stuck_edit,))
            self.assertEqual(sequence["status"], "passed")
            self.assertTrue(sequence["output_witness"])
            self.assertEqual(mismatch["status"], "trigger_mismatch")
            self.assertEqual(stuck["status"], "stuck_constant")
            self.assertTrue(obvious_constant_edit((Edit("Piezo_Driver.v", enable_old,
                                                        "enabled <= 0;"),)))

    def test_reset_unknown_and_intentional_first_interaction_are_kept(self) -> None:
        config = replace(self.config, design_root=(PIEZO / "rtl").resolve(),
                         sources=(Path("Piezo_Driver.v"),), top="Piezo_Driver",
                         clock="clock", reset="reset")
        old = "Piezo   <= (reset | ~enabled) ? 0 : ((count == compare) ? ~Piezo : Piezo);"
        reset_edit = Edit("Piezo_Driver.v", old,
                          "Piezo   <= (reset) ? Piezo : ((~enabled) ? 0 : ((count == compare) ? ~Piezo : Piezo));")
        first_edit = Edit("Piezo_Driver.v", old, old +
                          "\n        if (Write && !reset && !enabled && (count == compare)) Piezo <= data[24];")
        overbroad_edit = Edit("Piezo_Driver.v", old, old +
                              "\n        if (Write && !reset && !enabled && (count == compare)) Piezo <= 1;")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            apply_edits(config, (reset_edit,), root / "reset")
            apply_edits(config, (first_edit,), root / "first")
            apply_edits(config, (overbroad_edit,), root / "overbroad")
            with BehaviorAuditor(config) as auditor:
                reset = auditor.assess(root / "reset" / "rtl",
                    SimpleNamespace(category="state_sequence", target_signal="Piezo",
                                    trigger="reset=1 after activity",
                                    preserve_first_operation=True,
                                    root_cause="Piezo holds its value on reset"),
                    (reset_edit,))
                first = auditor.assess(root / "first" / "rtl",
                    SimpleNamespace(category="interaction", target_signal="Piezo",
                                    trigger="Write=1 and count==compare and data[24]=1",
                                    preserve_first_operation=False,
                                    root_cause="New enable affects output too soon"),
                    (first_edit,))
                overbroad = auditor.assess(root / "overbroad" / "rtl",
                    SimpleNamespace(category="interaction", target_signal="Piezo",
                                    trigger="Write=1 and count==compare",
                                    control_predicates=((
                                        {"left": "Write", "op": "==", "right": "1"},
                                        {"left": "data[24]", "op": "==", "right": "0"}),),
                                    preserve_first_operation=False,
                                    root_cause="The output ignores the data enable bit"),
                    (overbroad_edit,))
            self.assertEqual(reset["status"], "passed")
            self.assertEqual(reset["unknown_result"]["mutant"], "x")
            self.assertEqual(first["status"], "passed")
            self.assertIsNotNone(first["ordinary_difference"])
            self.assertEqual(overbroad["status"], "control_mismatch")
            self.assertGreater(overbroad["control_probe_hits"][0], 0)
            self.assertFalse(obvious_constant_edit((Edit("Piezo_Driver.v", old,
                old + "\n        if (Write) Piezo <= 1;"),)))

    def test_plan_identity_ignores_overlapping_requirement_and_category(self) -> None:
        common = dict(trigger="request=1", golden_behavior="Respond",
                      faulty_behavior="Response is missed", root_cause="Request update is gated by busy",
                      target_signal="done", why_realistic="An incorrect guard")
        self.assertEqual(_concept_key(FaultPlan("R001", "simple", **common)),
                         _concept_key(FaultPlan("R009", "interaction", **common)))

    def test_format_retry_keeps_edit_repair_budget_for_rtl_faults(self) -> None:
        client = Mock()
        client._request.side_effect = [ValueError("Model response must be a JSON object"),
            {"edits": [{"source_file": "tiny_counter.sv", "old": "count <= 2'd2;",
                        "new": "count <= 2'd1;"}]}]
        edits = _request_edits(client, "FAULT PLAN", 10000, self.config)
        self.assertEqual(len(edits), 1)
        self.assertEqual(client._request.call_count, 2)
        self.assertIn("FORMAT ERROR", client._request.call_args.args[1])

    def test_state_sequence_plan_retries_missing_control_contract(self) -> None:
        client = Mock()
        base = {"requirement_id": "R001", "category": "state_sequence",
                "trigger": "start=1 on a second request",
                "golden_behavior": "The first request completes",
                "faulty_behavior": "A later request does not complete",
                "root_cause": "The busy state blocks the second request",
                "target_signal": "count", "why_realistic": "Incorrect state gating"}
        client._request.side_effect = [base, {**base, "preserve_first_operation": True,
                                                   "trigger_predicates": [
                                                       {"left": "start", "op": "==", "right": "1"}],
                                                   "control_predicates": [[
                                                       {"left": "count", "op": "==", "right": "0"}]]}]
        behavior = Behavior("R001", "S001", "A request eventually completes.",
                            "A request eventually completes", ("start", "count"))
        plan = _request_plan(client, behavior, "state_sequence", "RTL", {"start", "count"},
                             [], 10000)
        self.assertTrue(plan.preserve_first_operation)
        self.assertEqual(client._request.call_count, 2)

    def test_interaction_needs_two_activating_conditions(self) -> None:
        client = Mock()
        base = {"requirement_id": "R001", "category": "interaction",
                "trigger": "request=1 with busy=1",
                "golden_behavior": "Request completes",
                "faulty_behavior": "Request is missed",
                "root_cause": "The busy guard blocks the request",
                "target_signal": "count", "why_realistic": "Incorrect guard"}
        request = {"left": "request", "op": "==", "right": "1"}
        client._request.side_effect = [
            {**base, "trigger_predicates": [request,
                {"left": "reset", "op": "==", "right": "0"}]},
            {**base, "trigger_predicates": [request,
                {"left": "busy", "op": "==", "right": "1"}],
                "control_predicates": [
                    [request, {"left": "busy", "op": "==", "right": "0"}],
                    [{"left": "request", "op": "==", "right": "0"},
                     {"left": "busy", "op": "==", "right": "1"}]]}]
        behavior = Behavior("R001", "S001", "A request completes when accepted.",
                            "A request completes when accepted", ("request", "count"))
        plan = _request_plan(client, behavior, "interaction", "RTL",
                             {"request", "busy", "reset", "count"}, [], 10000,
                             reset="reset")
        self.assertEqual(len(plan.trigger_predicates), 2)
        self.assertEqual(client._request.call_count, 2)

    def test_critic_must_echo_measured_difference(self) -> None:
        client = Mock()
        witness = {"probe": "p003", "cycle": 7, "signal": "done",
                   "golden": "0", "mutant": "1"}
        response = {"verdict": "accept", "scores": {
            "spec_relevance": 5, "realism": 4, "selectivity": 4,
            "plan_edit_match": 5, "diversity": 4}, "issues": [],
            "reason": "The trace matches", "observed_behavior": "Done rises early"}
        client._request.side_effect = [
            {**response, "measured_first_difference": {**witness, "cycle": 3}},
            {**response, "measured_first_difference": witness}]
        behavior = Behavior("R001", "S001", "Done rises after a request.",
                            "Done rises after a request", ("done",))
        plan = FaultPlan("R001", "timing", "start=1", "Done rises later",
                         "Done rises early", "Timer preload is too small", "done", "Plausible")
        result = _request_critique(client, behavior, plan, (), "RTL", [], 10000,
                                  {"witness_check": witness, "trace_excerpt": {"rows": []}})
        self.assertEqual(result["measured_first_difference"], witness)
        self.assertEqual(client._request.call_count, 2)
        self.assertIn("trace fact check", client._request.call_args.args[1])

    def test_requirement_mapping_accepts_json_array_and_retries_bad_shape(self) -> None:
        item = {"spec_quote": "Reset cancels a request.",
                "intended_behavior": "Reset cancels every active request",
                "related_signals": ["rst_n", "active"]}
        self.assertEqual(_json_object(json.dumps([item]), array_key="requirements"),
                         {"requirements": [item]})
        self.assertEqual(_json_object("Requirements:\n```json\n" + json.dumps([item]) + "\n```",
                                      array_key="requirements"), {"requirements": [item]})
        self.assertEqual(_json_object('[{"source_file":"dut.sv","old":"a","new":"b"}]',
                                      array_key="edits")["edits"][0]["source_file"], "dut.sv")
        from mutation_lab.quality_campaign import _map_behaviors, _spec_passages
        client = Mock()
        client._request.side_effect = [ValueError("Model response must be a JSON object"),
                                       {"requirements": [item]}]
        behaviors = _map_behaviors(client, self.config, (EXAMPLE / "spec.txt").read_text(),
                                   (EXAMPLE / "rtl" / "tiny_counter.sv").read_text(),
                                   {"rst_n", "active"}, 2, 100000)
        self.assertEqual(len(behaviors), 1)
        self.assertEqual(client._request.call_count, 2)
        passages = _spec_passages((EXAMPLE / "spec.txt").read_text())
        spec_id = next(key for key, text in passages.items() if "Reset cancels a request" in text)
        client._request.reset_mock()
        client._request.side_effect = None
        client._request.return_value = {"requirements": [{
            "spec_id": spec_id, "spec_quote": "A made-up quotation that is not in the spec",
            "intended_behavior": "Reset cancels every active request",
            "related_signals": ["rst_n", "active"]}]}
        grounded = _map_behaviors(client, self.config, (EXAMPLE / "spec.txt").read_text(),
                                  (EXAMPLE / "rtl" / "tiny_counter.sv").read_text(),
                                  {"rst_n", "active"}, 2, 100000)
        self.assertEqual(grounded[0].spec_id, spec_id)
        self.assertEqual(grounded[0].spec_quote, passages[spec_id])

    def test_spec_passages_keep_markdown_requirements_whole(self) -> None:
        from mutation_lab.quality_campaign import _spec_passages
        passages = _spec_passages((PIEZO / "spec.txt").read_text())
        self.assertTrue(any(item.startswith("- **Enable Flag (`enabled`):**") and
                            "updated on a write operation" in item
                            for item in passages.values()))
        self.assertFalse(any(item.startswith("(`enabled`)") for item in passages.values()))

    def test_quality_campaign_refines_broad_edit_without_assertions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rtl = PIEZO / "rtl" / "Piezo_Driver.v"
            golden = rtl.read_bytes()
            args = Namespace(spec=PIEZO / "spec.txt", rtl=[rtl], out=root / "quality",
                             count=1, max_attempts=1, max_repairs=1, max_requirements=2,
                             categories="interaction", quality_threshold=19,
                             design_root=None, include_dir=[], focus_rtl=[], top=None,
                             clock=None, reset="reset", jg_mode="local", jg_timeout=60,
                             jg_executable="jg", env_file=Path(".env"), endpoint="unused",
                             model="unused", key_env="unused", max_context_chars=100000,
                             planner_max_tokens=4096, lint_arg=[])
            quote = ("At each positive edge of the `clock`, the `count` register "
                     "increments unless it matches the `compare` value, in which case it resets to 0.")
            old_ack = "Ack     <= (reset) ? 0 : Write;"
            old_count = "count   <= (reset | (count == compare)) ? 24'h000000 : count + 1;"
            responses = [
                {"requirements": [{"spec_quote": quote,
                                   "intended_behavior": "Count increments until it equals compare",
                                   "related_signals": ["count", "compare", "Write"]}]},
                {"requirement_id": "R001", "category": "interaction",
                 "trigger": "Write while count is running",
                 "trigger_predicates": [{"left": "Write", "op": "==", "right": "1"},
                                        {"left": "count", "op": "!=", "right": "compare"}],
                 "control_predicates": [
                     [{"left": "Write", "op": "==", "right": "1"},
                      {"left": "count", "op": "==", "right": "compare"}],
                     [{"left": "Write", "op": "==", "right": "0"},
                      {"left": "count", "op": "!=", "right": "compare"}]],
                 "preserve_first_operation": True,
                 "golden_behavior": "Count keeps its current phase on Write",
                 "faulty_behavior": "Write restarts the running count",
                 "root_cause": "Write is included in the counter reset guard",
                 "target_signal": "count", "why_realistic": "A reconfiguration path restarts a timer"},
                {"edits": [{"source_file": "Piezo_Driver.v", "old": old_ack,
                            "new": "Ack     <= (reset) ? 0 : ~Write;"}]},
                {"edits": [{"source_file": "Piezo_Driver.v", "old": old_count,
                            "new": old_count.replace("(count == compare)",
                                                     "(count == compare) | Write")}]},
            ]
            critic = {"verdict": "accept", "scores": {"spec_relevance": 4, "realism": 4,
                                                      "selectivity": 4, "plan_edit_match": 5,
                                                      "diversity": 4},
                      "issues": [], "reason": "Edit matches the planned fault"}
            args.out.mkdir()
            write_json(args.out / "config.direct.json", direct_config(args))
            write_json(args.out / "setup_error.json", {"stage": "requirement_mapping"})
            with patch("mutation_lab.quality_campaign.LLMPlanner") as planner:
                planner.return_value._request.side_effect = _model_with_measured_critic(responses, critic)
                run_quality_campaign(args)
                prompts = [call.args[1] for call in planner.return_value._request.call_args_list]
            self.assertEqual(len(prompts), 5)
            self.assertIn("control_operation_changed", prompts[3])
            self.assertIn("trace_excerpt", prompts[4])
            self.assertTrue(all("compare_write_update" not in prompt for prompt in prompts))
            record = json.loads((args.out / "mutants" / "m0001" / "record.json").read_text())
            self.assertEqual(record["status"], "generated_unqualified")
            self.assertEqual([item["status"] for item in record["revisions"]],
                             ["behavior_rejected", "accepted"])
            self.assertEqual(record["quality"]["total"], 21)
            self.assertEqual(record["behavior_audit"]["status"], "passed")
            summary = json.loads((args.out / "summary.json").read_text())
            self.assertEqual(summary["evidence_profile"]["with_output_witness"], 1)
            self.assertEqual(summary["evidence_profile"]["with_first_operation_difference"], 0)
            self.assertFalse((args.out / "setup_error.json").exists())
            self.assertEqual(record["edit"]["source_snapshot"],
                             Config.load(args.out / "config.direct.json").source_snapshot())
            self.assertIn("| Write", (args.out / "mutants" / "m0001" / "rtl" / "Piezo_Driver.v").read_text())
            self.assertEqual(rtl.read_bytes(), golden)
            with patch("mutation_lab.campaign.run_check", side_effect=[
                    {"status": "completed", "properties": {"ML_ASSERT_0001": "proven"}},
                    {"status": "completed", "properties": {"ML_ASSERT_0001": "cex"}}]):
                retry_scoring(Namespace(out=args.out,
                                        assertions=PIEZO / "piezo_driver_final_assertions.txt",
                                        max_assertions=1, jg_timeout=None))
            scored = json.loads((args.out / "evaluation" / "results.json").read_text())
            self.assertEqual(scored["summary"]["candidate_mutants"], 1)
            self.assertEqual(scored["summary"]["killed_mutants"], 1)

    def test_quality_campaign_uses_same_flow_on_another_design(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = Namespace(spec=EXAMPLE / "spec.txt",
                             rtl=[EXAMPLE / "rtl" / "tiny_counter.sv"],
                             out=Path(temp) / "quality", count=1, max_attempts=1,
                             max_repairs=0, max_requirements=1, categories="timing",
                             quality_threshold=19, design_root=None, include_dir=[],
                             focus_rtl=[], top=None, clock=None, reset="rst_n",
                             jg_mode="local", jg_timeout=60, jg_executable="jg",
                             env_file=Path(".env"), endpoint="unused", model="unused",
                             key_env="unused", max_context_chars=100000,
                             planner_max_tokens=4096, lint_arg=[])
            responses = [
                ValueError("model did not return mapped requirements"),
                ValueError("model did not return mapped requirements"),
                ValueError("model did not return mapped requirements"),
                {"requirement_id": "R001", "category": "timing",
                 "trigger": "An accepted start request",
                 "golden_behavior": "Done rises on the fourth cycle",
                 "faulty_behavior": "Done rises one cycle early",
                 "root_cause": "The initial countdown value is too small",
                 "target_signal": "count", "why_realistic": "An off-by-one timer preload"},
                {"edits": [{"source_file": "tiny_counter.sv", "old": "count <= 2'd2;",
                            "new": "count <= 2'd1;"}]},
            ]
            critic = {"verdict": "accept", "scores": {"spec_relevance": 5, "realism": 5,
                                                      "selectivity": 4, "plan_edit_match": 5,
                                                      "diversity": 5},
                      "issues": [], "reason": "Plausible off-by-one preload"}
            with patch("mutation_lab.quality_campaign.LLMPlanner") as planner:
                planner.return_value._request.side_effect = _model_with_measured_critic(responses, critic)
                run_quality_campaign(args)
            record = json.loads((args.out / "mutants" / "m0001" / "record.json").read_text())
            self.assertEqual(record["status"], "generated_unqualified")
            self.assertEqual(record["category"], "timing")
            self.assertEqual(json.loads((args.out / "summary.json").read_text())
                             ["requirement_mapping"], "deterministic_fallback")
            self.assertIn("count <= 2'd1;", (args.out / "mutants" / "m0001" / "rtl" / "tiny_counter.sv").read_text())

    def test_trace_campaign_measures_second_write_faults(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "trace"
            args = Namespace(spec=PIEZO / "spec.txt", rtl=[PIEZO / "rtl" / "Piezo_Driver.v"],
                             out=output, reset="reset", count=5, max_seeds=18,
                             planner="offline", formal_compare=False,
                             design_root=None, include_dir=[], focus_rtl=[], top=None,
                             clock=None, jg_mode="local", jg_timeout=60,
                             jg_executable="jg", env_file=Path(".env"),
                             endpoint="unused", model="unused", key_env="unused",
                             max_context_chars=100000, planner_max_tokens=4096,
                             lint_arg=[])
            golden = args.rtl[0].read_bytes()
            run_trace_campaign(args)
            records = [json.loads(path.read_text()) for path in
                       sorted((output / "mutants").glob("*/record.json"))]
            self.assertEqual(len(records), 5)
            self.assertTrue(all(item["status"] == "generated_unqualified" for item in records))
            self.assertEqual(len({item["behavior_cluster"] for item in records}), 5)
            for item in records:
                trace = item["measured_trace"]
                self.assertGreaterEqual(trace["first_target_difference_cycle"],
                                        trace["second_event_cycle"])
                self.assertGreaterEqual(trace["first_output_difference_cycle"],
                                        trace["first_target_difference_cycle"])
                self.assertEqual(trace["behavior_seeds"], 14)
            self.assertEqual(args.rtl[0].read_bytes(), golden)

            config = Config.load(output / "config.direct.json")
            ports, _ = _ports(config, config.design_root, Path(temp))
            miter = build_miter(config, config.design_root,
                                output / "mutants" / "m0001" / "rtl", ports,
                                Path(temp) / "miter")
            self.assertIn("MUTLAB_OUTPUT_EQUIV: assert property",
                          (miter.design_root / "miter.sv").read_text())

            source = args.rtl[0].read_text()
            old = "count   <= (reset | (count == compare)) ? 24'h000000 : count + 1;"
            self.assertIn(old, source)
            response = {"ranked_ids": ["n1"], "reasons": {"n1": "A write restarts the timer"},
                        "novel_ideas": [{"id": "n1", "category": "interaction",
                                         "event": "Write", "target": "count",
                                         "source_file": "Piezo_Driver.v",
                                         "description": "A second write restarts the running count",
                                         "edits": [{"source_file": "Piezo_Driver.v", "old": old,
                                                    "new": old.replace("(count == compare)",
                                                                       "(count == compare) | Write")}] }]}
            args.out = Path(temp) / "llm_trace"
            args.planner = "llm"
            with patch("mutation_lab.trace_guided.LLMPlanner") as planner:
                planner.return_value._request.return_value = response
                run_trace_campaign(args)
                planner.return_value._request.assert_called_once()
            selected = json.loads((args.out / "mutants" / "m0001" / "record.json").read_text())
            self.assertEqual(selected["idea"]["template"], "llm_constrained")
            self.assertEqual(selected["status"], "generated_unqualified")

    def test_one_exact_edit_is_isolated(self) -> None:
        proposal = Proposal.parse(json.loads((EXAMPLE / "proposals.jsonl").read_text()), self.config)
        golden = (self.config.design_root / "tiny_counter.sv").read_text()
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "mutant"
            edit = apply_proposal(self.config, proposal, destination)
            mutant = (destination / "rtl" / "tiny_counter.sv").read_text()
            self.assertEqual(mutant.count("count <= 2'd1;"), 1)
            self.assertEqual((self.config.design_root / "tiny_counter.sv").read_text(), golden)
            self.assertGreater(edit["line"], 1)

    def test_reject_unknown_source_and_ambiguous_edit(self) -> None:
        raw = json.loads((EXAMPLE / "proposals.jsonl").read_text())
        raw["source_file"] = "../outside.sv"
        with self.assertRaises(ValueError):
            Proposal.parse(raw, self.config)
        raw["source_file"] = "tiny_counter.sv"
        raw["old"] = "end"
        proposal = Proposal.parse(raw, self.config)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "exactly once"):
                apply_proposal(self.config, proposal, Path(temp) / "mutant")

    def test_token_filter_and_fill_quotas(self) -> None:
        self.assertEqual(rtl_tokens("assign x = y; // comment\n"),
                         rtl_tokens("assign x=y;\n"))
        self.assertNotEqual(rtl_tokens("assign x = (a == b);"),
                            rtl_tokens("assign x = (a != b);"))
        self.assertNotEqual(rtl_tokens('initial $display("//a");'),
                            rtl_tokens('initial $display("//b");'))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proposals = root / "proposals.jsonl"
            original = json.loads((EXAMPLE / "proposals.jsonl").read_text())
            comment = {**original, "old": "count <= 2'd2;",
                       "new": "count <= 2'd2; // misleading fault"}
            first = {**original, "old": "count <= 2'd2;", "new": "count <= 2'd1;"}
            duplicate_result = {**original,
                "old": "active <= 1'b1;\n                count <= 2'd2;",
                "new": "active <= 1'b1;\n                count <= 2'd1;"}
            second = {**original, "old": "count <= 2'd2;", "new": "count <= 2'd0;"}
            proposals.write_text("\n".join(json.dumps(item) for item in
                                            (comment, first, duplicate_result, second)) + "\n")
            output = root / "run"
            run_generation(Namespace(config=EXAMPLE / "config.json", out=output,
                                     count=2, mix=None, proposals=proposals,
                                     generate_only=True, fill_quotas=True,
                                     max_attempts=4))
            records = [json.loads(path.read_text()) for path in
                       sorted((output / "mutants").glob("*/record.json"))]
            self.assertEqual([item["status"] for item in records],
                             ["no_token_change", "generated_unqualified",
                              "duplicate_rtl", "generated_unqualified"])
            self.assertEqual(records[2]["duplicate_of"], "m0002")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["requested_mix"], {"simple": 2})
            self.assertEqual(summary["accepted_mix"], {"simple": 2})
            self.assertEqual(summary["attempts"], 4)

    def test_mixed_schedule_and_related_scenario_edits(self) -> None:
        self.assertEqual(mutation_schedule(99, "simple=2,timing=1,state_sequence=1"),
                         ["simple", "timing", "state_sequence", "simple"])
        lines = [json.loads(line) for line in (PIEZO / "mixed_proposals.jsonl").read_text().splitlines()]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "design_root": str(PIEZO / "rtl"), "sources": ["Piezo_Driver.v"],
                "top": "Piezo_Driver", "clock": "clock",
                "spec_file": str(PIEZO / "spec.txt"),
                "requirements": [{"id": "SPEC", "text": "Choose from the spec",
                                  "oracle": "ORACLE_UNUSED", "reach": "REACH_UNUSED"}],
            }))
            config = Config.load(config_path)
            mutation = ScenarioMutation.parse(lines[2], config, "state_sequence")
            original = (PIEZO / "rtl" / "Piezo_Driver.v").read_text()
            result = apply_scenario(config, mutation, root / "mutant")
            edited = (root / "mutant" / "rtl" / "Piezo_Driver.v").read_text()
            self.assertIn("reg delayed_write;", edited)
            self.assertIn("Ack     <= (reset) ? 0 : delayed_write;", edited)
            self.assertEqual(len(result["changes"]), 2)
            self.assertEqual((PIEZO / "rtl" / "Piezo_Driver.v").read_text(), original)
            lines[2]["edits"].append({"source_file": "Piezo_Driver.v",
                                      "old": "enabled;", "new": "enabled /*fault*/;"})
            overlapping = ScenarioMutation.parse(lines[2], config, "state_sequence")
            with self.assertRaisesRegex(ValueError, "overlap"):
                apply_scenario(config, overlapping, root / "overlap")
            self.assertEqual((root / "overlap" / "rtl" / "Piezo_Driver.v").read_text(),
                             original)
            lines[1]["scenario"]["spec_excerpt"] = "A fabricated requirement"
            with self.assertRaisesRegex(ValueError, "quote"):
                ScenarioMutation.parse(lines[1], config, "timing")

    def test_llm_plans_scenario_before_edit(self) -> None:
        lines = [json.loads(line) for line in (PIEZO / "mixed_proposals.jsonl").read_text().splitlines()]
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "config.json"
            config_path.write_text(json.dumps({
                "design_root": str(PIEZO / "rtl"), "sources": ["Piezo_Driver.v"],
                "top": "Piezo_Driver", "spec_file": str(PIEZO / "spec.txt"),
                "requirements": [{"id": "SPEC", "text": "Choose from the spec",
                                  "oracle": "ORACLE_UNUSED", "reach": "REACH_UNUSED"}],
            }))
            planner = object.__new__(LLMPlanner)
            planner.config = Config.load(config_path)
            planner.scenario_mode = "two-call"
            planner._request = Mock(side_effect=[lines[1]["scenario"],
                                                 {"edits": lines[1]["edits"]}])
            mutation = planner.propose([], "timing")
            self.assertEqual(mutation.scenario.category, "timing")
            self.assertEqual(planner._request.call_count, 2)
            self.assertIn("PROPOSED FAULT SCENARIO", planner._request.call_args.args[1])

            planner.scenario_mode = "one-call"
            planner._request = Mock(return_value=lines[1])
            combined = planner.propose([], "timing")
            self.assertEqual(combined, mutation)
            planner._request.assert_called_once()
            focus_prompt = planner._context([], "state_sequence")
            self.assertIn("UNDEREXPLORED RTL UPDATE: count", focus_prompt)
            self.assertIn("UNDEREXPLORED RTL UPDATE: compare", planner._context(
                [{"category": "state_sequence", "focus_target": "count"}],
                "state_sequence"))
            self.assertIn("first transaction succeeds", focus_prompt)
            self.assertIn("state retained or cleared", planner._context(
                [{"category": "state_sequence", "focus_target": "count"}],
                "state_sequence"))
            self.assertIn("REQUESTED CATEGORY: timing", planner._request.call_args.args[1])
            planner.scenario_mode = "two-call"
            planner.require_witness = True
            planner._request = Mock(return_value=lines[1]["scenario"])
            with self.assertRaisesRegex(ValueError, "concrete witness"):
                planner.propose([], "timing")
            planner._request.assert_called_once()
            planned = {**lines[1]["scenario"], "witness": {
                "cycles": [{"reset": 1}, {"reset": 0}],
                "observations": [{"cycle": 0, "signal": "Ack", "golden": 0, "mutant": 0},
                                 {"cycle": 1, "signal": "Ack", "golden": 1, "mutant": 0}],
            }}
            planner._request = Mock(return_value=planned)
            with self.assertRaisesRegex(ValueError, "normal-operation control"):
                planner.propose([], "timing")
            planner._request.assert_called_once()

    def test_selective_fault_controls_reject_broad_failure(self) -> None:
        lines = [json.loads(line) for line in (PIEZO / "mixed_proposals.jsonl").read_text().splitlines()]
        control = {
            "cycles": [{"reset": 1, "Write": 0, "data": 0},
                       {"reset": 0, "Write": 1, "data": "0x1000003"},
                       {"Write": 0}],
            "observations": [
                {"cycle": 0, "signal": "Ack", "golden": 0, "mutant": 0},
                {"cycle": 1, "signal": "Ack", "golden": 1, "mutant": 1},
                {"cycle": 2, "signal": "Ack", "golden": 0, "mutant": 0},
            ],
        }
        witness = {
            "cycles": [{"reset": 1, "Write": 0, "data": 0},
                       {"reset": 0, "Write": 1, "data": "0x1000004"},
                       {"Write": 0}, {"Write": 1}],
            "observations": [
                {"cycle": 0, "signal": "Ack", "golden": 0, "mutant": 0},
                {"cycle": 3, "signal": "Ack", "golden": 1, "mutant": 0},
            ],
            "controls": [control],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "design_root": str(PIEZO / "rtl"), "sources": ["Piezo_Driver.v"],
                "top": "Piezo_Driver", "clock": "clock", "reset": "reset",
                "spec_file": str(PIEZO / "spec.txt"),
                "requirements": [{"id": "SPEC", "text": "Choose from the spec",
                                  "oracle": "ORACLE_UNUSED", "reach": "REACH_UNUSED"}],
            }))
            config = Config.load(config_path)
            selective = ScenarioMutation.parse({
                "scenario": {**lines[2]["scenario"], "witness": witness},
                "edits": [
                    {"source_file": "Piezo_Driver.v", "old": "reg enabled;",
                     "new": "reg enabled;\n    reg seen_write;\n    always @(posedge clock) begin\n"
                            "      if (reset) seen_write <= 1'b0;\n"
                            "      else if (Write) seen_write <= 1'b1;\n    end"},
                    {"source_file": "Piezo_Driver.v",
                     "old": "Ack     <= (reset) ? 0 : Write;",
                     "new": "Ack     <= (reset) ? 0 : (Write && !seen_write);"},
                ],
            }, config, "state_sequence")
            selective_dir = root / "selective"
            apply_scenario(config, selective, selective_dir)
            result = validate_scenario(config, selective_dir, witness,
                                       category="state_sequence", require_controls=True)
            self.assertEqual(result["status"], "passed", result)
            self.assertEqual(result["controls"], 1)
            broad = ScenarioMutation.parse({
                "scenario": {**lines[2]["scenario"], "witness": witness},
                "edits": [
                    selective.edits[0].__dict__,
                    {"source_file": "Piezo_Driver.v",
                     "old": "Ack     <= (reset) ? 0 : Write;",
                     "new": "Ack     <= (reset) ? 0 : (Write && !seen_write && data[23:0] != 24'd3);"},
                ],
            }, config, "state_sequence")
            broad_dir = root / "broad"
            apply_scenario(config, broad, broad_dir)
            self.assertEqual(validate_scenario(config, broad_dir, witness,
                                               category="state_sequence",
                                               require_controls=True)["status"],
                             "control_divergence")

    def test_concrete_witness_checks_golden_and_mutant(self) -> None:
        lines = [json.loads(line) for line in (PIEZO / "mixed_proposals.jsonl").read_text().splitlines()]
        witness = {
            "cycles": [{"reset": 1, "Write": 0, "data": 0},
                       {"reset": 0, "Write": 1, "data": "0x1000003"},
                       {"Write": 0}],
            "observations": [
                {"cycle": 1, "signal": "Ack", "golden": 1, "mutant": 0},
                {"cycle": 2, "signal": "Ack", "golden": 0, "mutant": 1},
            ],
        }
        lines[2]["scenario"]["witness"] = witness
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "design_root": str(PIEZO / "rtl"), "sources": ["Piezo_Driver.v"],
                "top": "Piezo_Driver", "clock": "clock", "reset": "reset",
                "spec_file": str(PIEZO / "spec.txt"),
                "requirements": [{"id": "SPEC", "text": "Choose from the spec",
                                  "oracle": "ORACLE_UNUSED", "reach": "REACH_UNUSED"}],
            }))
            config = Config.load(config_path)
            mutation = ScenarioMutation.parse(lines[2], config, "state_sequence")
            destination = root / "mutant"
            apply_scenario(config, mutation, destination)
            self.assertEqual(validate_scenario(config, destination, witness)["status"], "passed")
            wrong = json.loads(json.dumps(witness))
            wrong["observations"][0]["mutant"] = 1
            self.assertEqual(validate_scenario(config, destination, wrong)["status"],
                             "scenario_mismatch")
            wrong = json.loads(json.dumps(witness))
            wrong["observations"][0]["golden"] = 0
            self.assertEqual(validate_scenario(config, destination, wrong)["status"],
                             "bad_witness")
            early_witness = {
                "cycles": [{"reset": 1, "Write": 0, "data": 0},
                           {"reset": 0}, {}, {}],
                "observations": [
                    {"cycle": 2, "signal": "Ack", "golden": 0, "mutant": 1},
                    {"cycle": 3, "signal": "Ack", "golden": 0, "mutant": 1},
                ],
            }
            early_mutation = ScenarioMutation.parse({
                "scenario": {**lines[2]["scenario"], "witness": early_witness},
                "edits": [{"source_file": "Piezo_Driver.v",
                           "old": "Ack     <= (reset) ? 0 : Write;",
                           "new": "Ack     <= (reset) ? 0 : 1'b1;"}],
            }, config, "state_sequence")
            early_dir = root / "early"
            apply_scenario(config, early_mutation, early_dir)
            self.assertEqual(validate_scenario(config, early_dir, early_witness)["status"],
                             "early_divergence")
            proposals = root / "proposals.jsonl"
            proposals.write_text(json.dumps(lines[2]) + "\n")
            run_generation(Namespace(config=config_path, out=root / "campaign",
                                     count=1, mix="state_sequence=1", proposals=proposals,
                                     generate_only=True))
            record = json.loads((root / "campaign" / "mutants" / "m0001" /
                                 "record.json").read_text())
            self.assertEqual(record["scenario_validation"]["status"], "passed")
            lines[2]["scenario"]["witness"]["observations"][0]["mutant"] = 1
            proposals.write_text(json.dumps(lines[2]) + "\n")
            run_generation(Namespace(config=config_path, out=root / "bad_campaign",
                                     count=1, mix="state_sequence=1", proposals=proposals,
                                     generate_only=True))
            record = json.loads((root / "bad_campaign" / "mutants" / "m0001" /
                                 "record.json").read_text())
            self.assertEqual(record["status"], "scenario_mismatch")

    def test_named_statuses(self) -> None:
        statuses = parse_statuses("[0] top.checker.ORACLE_LATENCY proven\n"
                                  "[1] top.checker.REACH_REQUEST covered\n")
        self.assertEqual(named_status(statuses, "ORACLE_LATENCY"), "proven")
        self.assertEqual(named_status(statuses, "MISSING"), "missing_or_ambiguous")

    def test_sangam_text_assertion_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "assertions.txt"
            path.write_text("Design assertion set\n[001] Signals: done\n"
                            "property done_late;\n  @(posedge clk) start |=> !done;\n"
                            "endproperty\nassert property (done_late);\n"
                            "[002] Signals: done\n"
                            "assert property (@(posedge clk) !done);\n")
            assertions = load_assertions(path)
            self.assertEqual(len(assertions.labels), 2)
            self.assertIn("property done_late;", assertions.inline_code)
            self.assertIn("ML_ASSERT_0002: assert property", assertions.inline_code)

    def test_custom_tcl_keeps_design_specific_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            template = root / "custom.tcl"
            template.write_text("{{ANALYZE_COMMAND}}\nelaborate -top {{TOP}} -param WIDTH=8\n"
                                "clock clk\nreset {~rst_n}\nprove -all\n"
                                "report -summary -force -result\nexit -force\n")
            config = replace(self.config, jasper={"tcl_template": str(template)})
            directory = root / "golden"
            stage_sources(config, directory)
            prepare_job(config, directory, self.config.oracle_file)
            script = (directory / "run.tcl").read_text()
            self.assertIn("-param WIDTH=8", script)
            self.assertIn("checker.sv", script)
            self.assertNotIn("{{", script)

    def test_generate_qualify_then_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_jg = root / "fake_jg"
            fake_jg.write_text("""#!/usr/bin/env python3
from pathlib import Path
rtl = Path('rtl/tiny_counter.sv').read_text()
checker = Path('checker.sv').read_text()
mutant = "count <= 2'd1;" in rtl
if 'ORACLE_LATENCY' in checker:
    print('[0] tiny_counter.oracle_inst.ORACLE_LATENCY ' + ('cex' if mutant else 'proven'))
    print('[1] tiny_counter.oracle_inst.REACH_REQUEST covered')
else:
    print('[0] tiny_counter.evaluation_inst.EVAL_NO_EARLY_DONE ' + ('cex' if mutant else 'proven'))
""")
            fake_jg.chmod(0o755)
            config_path = root / "config.json"
            raw = json.loads((EXAMPLE / "config.json").read_text())
            raw["design_root"] = str(EXAMPLE / "rtl")
            raw["spec_file"] = str(EXAMPLE / "spec.txt")
            raw["oracle_file"] = str(EXAMPLE / "oracle.sv")
            raw["evaluation_file"] = str(EXAMPLE / "eval.sv")
            raw["jasper"] = {"mode": "local", "executable": str(fake_jg)}
            config_path.write_text(json.dumps(raw))
            output = root / "run"
            run_generation(Namespace(config=config_path, out=output, count=1,
                                     proposals=EXAMPLE / "proposals.jsonl", generate_only=False))
            record = json.loads((output / "mutants" / "m0001" / "record.json").read_text())
            self.assertEqual(record["status"], "qualified")
            self.assertIn("prove -all", (output / "golden" / "run.tcl").read_text())
            run_evaluation(Namespace(config=config_path, out=output))
            results = json.loads((output / "evaluation" / "results.json").read_text())
            self.assertEqual(results["mutants"][0]["labels"]["EVAL_NO_EARLY_DONE"], "killed")

    def test_direct_campaign_and_score_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_jg = root / "fake_jg"
            fake_jg.write_text("""#!/usr/bin/env python3
from pathlib import Path
rtl = Path('rtl/tiny_counter.sv').read_text()
mutant = "count <= 2'd1;" in rtl
print('[0] tiny_counter.ML_ASSERT_0001 ' + ('cex' if mutant else 'proven'))
""")
            fake_jg.chmod(0o755)
            proposals = root / "proposals.jsonl"
            proposal = json.loads((EXAMPLE / "proposals.jsonl").read_text())
            proposal["requirement_id"] = "SPEC"
            proposals.write_text(json.dumps(proposal) + "\n")
            assertions = root / "assertions.jsonl"
            assertions.write_text(json.dumps({
                "assertion": "assert property (@(posedge clk) start |=> !done);",
                "row_number": 9,
            }) + "\n")
            output = root / "campaign"
            run_campaign(Namespace(
                spec=EXAMPLE / "spec.txt", rtl=[EXAMPLE / "rtl" / "tiny_counter.sv"],
                assertions=assertions, out=output, count=1, proposals=proposals,
                generate_only=False, max_assertions=0, design_root=None, include_dir=[],
                focus_rtl=[], top=None, clock=None, reset=None, jg_mode="local",
                jg_timeout=30, jg_executable=str(fake_jg), env_file=root / ".env",
                endpoint="https://example.invalid/v1/chat/completions", model="fake",
                key_env="FAKE_KEY", max_context_chars=100000, lint_arg=[],
            ))
            result = json.loads((output / "evaluation" / "results.json").read_text())
            self.assertEqual(result["summary"]["killed_mutants"], 1)
            self.assertEqual(result["summary"]["assertions_killing_any_mutant"], 1)
            self.assertFalse(result["summary"]["spec_confirmed_bugs"])
            retry_scoring(Namespace(out=output, assertions=assertions,
                                    max_assertions=0, jg_timeout=None))
            self.assertTrue((output / "evaluation_2" / "kill_matrix.csv").exists())

    def test_mixed_campaign_reports_category_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_jg = root / "fake_jg"
            fake_jg.write_text("""#!/usr/bin/env python3
from pathlib import Path
rtl = Path('rtl/Piezo_Driver.v').read_text()
mutant = ('~Write;' in rtl or "compare + 24'd1" in rtl or 'delayed_write' in rtl)
print('[0] Piezo_Driver.ML_ASSERT_0001 ' + ('cex' if mutant else 'proven'))
""")
            fake_jg.chmod(0o755)
            assertions = root / "assertions.jsonl"
            assertions.write_text(json.dumps({"assertion":
                "assert property (@(posedge clock) reset |=> !Ack);"}) + "\n")
            output = root / "campaign"
            run_campaign(Namespace(
                spec=PIEZO / "spec.txt", rtl=[PIEZO / "rtl" / "Piezo_Driver.v"],
                assertions=assertions, out=output, count=1,
                mix="simple=1,timing=1,state_sequence=1",
                proposals=PIEZO / "mixed_proposals.jsonl", generate_only=False,
                max_assertions=0, design_root=None, include_dir=[], focus_rtl=[],
                top=None, clock=None, reset=None, jg_mode="local", jg_timeout=30,
                jg_executable=str(fake_jg), env_file=root / ".env",
                endpoint="https://example.invalid/v1/chat/completions", model="fake",
                key_env="FAKE_KEY", max_context_chars=100000, lint_arg=[],
            ))
            result = json.loads((output / "evaluation" / "results.json").read_text())
            self.assertEqual(result["summary"]["candidate_mutants"], 3)
            self.assertEqual(result["summary"]["by_category"]["timing"]["killed"], 1)
            self.assertEqual(result["summary"]["by_category"]["state_sequence"]["killed"], 1)
            self.assertIn("category", (output / "evaluation" / "kill_matrix.csv").read_text())


if __name__ == "__main__":
    unittest.main()
