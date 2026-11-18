import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

from fixme_svagen.jasper import license_connection_error, parse_log
from fixme_svagen.cli import Case, process_case, rebuild_jsonl
from fixme_svagen.generate import GENERATION_CONFIG, SangamGenerator
from fixme_svagen.resume import is_transient
from fixme_svagen.sva import clock_reset, extract_assertions, insert_in_module
from fixme_svagen.sva import Assertion


class SvaExtractionTests(unittest.TestCase):
    def test_inline_and_named_property_are_individual_rows(self):
        response = """```systemverilog
localparam int LIMIT = 2;
assert property (@(posedge clk) a |-> (b && (count < LIMIT)));
property delayed; @(posedge clk) a |=> b; endproperty
assert property (delayed);
```"""
        assertions = extract_assertions(response)
        self.assertEqual(len(assertions), 2)
        self.assertTrue(all(a.implication for a in assertions))
        self.assertIn("property delayed", assertions[1].support)
        self.assertIn("localparam int LIMIT", assertions[0].support)

    def test_insertion_targets_selected_module(self):
        rtl = "module first; endmodule\nmodule second; endmodule\n"
        result = insert_in_module(rtl, "second", "sangam_assert: assert property (1);")
        self.assertEqual(result.count("sangam_assert"), 1)
        self.assertLess(result.index("module second"), result.index("sangam_assert"))

    def test_reset_after_clock_in_port_declaration(self):
        rtl = "module m(clk, rst_n); input wire clk, rst_n; always @(posedge clk) begin end endmodule"
        self.assertEqual(clock_reset(rtl), ("clk", "rst_n"))

    def test_reset_detection_skips_data_bus_containing_rst(self):
        rtl = """module m(input clk, input [1:0] arburst, input cpurst_b);
always @(posedge clk or negedge cpurst_b) begin end
endmodule"""
        self.assertEqual(clock_reset(rtl), ("clk", "cpurst_b"))

    def test_reset_detection_accepts_single_bit_vector(self):
        rtl = """module m(input clk, input [0:0] rst_n);
always @(posedge clk) begin end
endmodule"""
        self.assertEqual(clock_reset(rtl), ("clk", "rst_n"))

    def test_named_property_uses_its_own_body_for_implication_scope(self):
        response = """```systemverilog
property p_plain; @(posedge clk) a; endproperty
property p_imp; @(posedge clk) a |-> b; endproperty
assert property (p_plain);
assert property (p_imp);
```"""
        self.assertEqual([a.implication for a in extract_assertions(response)], [False, True])


class JasperLabelTests(unittest.TestCase):
    def test_license_feature_fallback_is_not_connection_failure(self):
        log = "checkout failed (error: -5) (FLEXnet ERROR(-5): No such feature exists.)\ncheckout succeeded"
        self.assertFalse(license_connection_error(log))
        self.assertTrue(license_connection_error("FLEXnet ERROR(-15): Cannot connect to license server system"))

    def test_related_cover_rows_determine_reason(self):
        template = """[1] top.sangam_assert proven PRE Infinite 0.0 s
[2] top.sangam_assert:precondition1 {pre} PRE 1 0.0 s
[3] top.sangam_assert:contrapositive1 {contra} PRE 1 0.0 s
"""
        a = parse_log(template.format(pre="unreachable", contra="covered"), 0)
        b = parse_log(template.format(pre="covered", contra="unreachable"), 0)
        both = parse_log(template.format(pre="unreachable", contra="unreachable"), 0)
        normal = parse_log(template.format(pre="covered", contra="covered"), 0)
        self.assertEqual(a.label(True), (True, "antecedent_never_true"))
        self.assertEqual(b.label(True), (True, "consequent_always_true"))
        self.assertEqual(both.label(True), (True, "antecedent_never_true; consequent_always_true"))
        self.assertEqual(normal.label(True), (False, "neither_targeted_vacuity_reason"))
        self.assertEqual(normal.label(False), (None, "outside_scope_non_implication"))
        self.assertEqual(parse_log("[1] top.sangam_assert cex PRE 1 0.0 s", 0).label(False),
                         (None, "outside_scope_non_implication"))

    def test_unfinished_proof_is_not_negative_label(self):
        result = parse_log("[1] top.sangam_assert undetermined PRE 2 3.0 s", 0)
        self.assertEqual(result.label(True), (None, "formal_inconclusive_or_error"))


class DatasetWriterTests(unittest.TestCase):
    def test_retry_classifier_accepts_nebius_certificate_case(self):
        self.assertTrue(is_transient("certificate verify failed: Hostname mismatch"))
        self.assertTrue(is_transient("JasperGold preflight failed: assertion=license_unavailable"))
        self.assertTrue(is_transient("TimeoutError: timed out"))
        self.assertTrue(is_transient("Nebius request failed: Temporary failure in name resolution"))
        self.assertTrue(is_transient("Unable to connect to port 22 on 172.16.82.69"))
        self.assertFalse(is_transient("Assertion parser rejected malformed output"))

    def test_candidate_formal_result_survives_signal_restart(self):
        class LLM:
            model = "deepseek-ai/DeepSeek-V4-Flash-0731"

        class Jasper:
            timeout = 120

            def __init__(self):
                self.calls = 0

            def check(self, *_):
                self.calls += 1
                return parse_log("[1] top.sangam_assert proven PRE Infinite 0.0 s", 0)

        assertion = Assertion("assert property (@(posedge clk) a |-> b);", "", True)
        with tempfile.TemporaryDirectory() as directory:
            jasper = Jasper()
            for _ in range(2):
                generator = SangamGenerator(LLM(), jasper,
                                            "module top(input clk,a,b); endmodule",
                                            "top", "clk", None, Path(directory), 4)
                self.assertEqual(generator.check(assertion).assertion_status, "proven")
            self.assertEqual(jasper.calls, 1)

    def test_corrupt_candidate_checkpoint_is_rechecked_and_replaced(self):
        class LLM:
            model = "deepseek-ai/DeepSeek-V4-Flash-0731"

        class Jasper:
            timeout = 120

            def __init__(self):
                self.calls = 0

            def check(self, *_):
                self.calls += 1
                return parse_log("[1] top.sangam_assert proven PRE Infinite 0.0 s", 0)

        assertion = Assertion("assert property (@(posedge clk) a |-> b);", "", True)
        for corrupt in ("", "{", "null", "[]"):
            with self.subTest(checkpoint=corrupt), tempfile.TemporaryDirectory() as directory:
                workdir = Path(directory)
                jasper = Jasper()

                def restart():
                    return SangamGenerator(LLM(), jasper, "module top(input clk,a,b); endmodule",
                                           "top", "clk", None, workdir, 4)

                restart().check(assertion)
                checkpoint = next((workdir / "candidate_results").glob("*.json"))
                checkpoint.write_text(corrupt)
                self.assertEqual(restart().check(assertion).assertion_status, "proven")
                self.assertEqual(jasper.calls, 2)
                self.assertEqual(json.loads(checkpoint.read_text())["assertion_status"], "proven")
                restart().check(assertion)
                self.assertEqual(jasper.calls, 2)

    def test_generation_combination_and_final_dedup(self):
        sva = "assert property (@(posedge clk) a |-> b);"

        class LLM:
            def chat(self, system, user):
                if "Critique every issue" in user:
                    return "[Analysis] Looks sound. [Score] 50"
                return f"```systemverilog\n{sva}\n```"

        class Jasper:
            def check(self, *_):
                return parse_log("""[1] top.sangam_assert proven PRE Infinite 0.0 s
[2] top.sangam_assert:precondition1 covered PRE 1 0.0 s
[3] top.sangam_assert:contrapositive1 covered PRE 1 0.0 s
""", 0)

        with tempfile.TemporaryDirectory() as directory:
            generator = SangamGenerator(LLM(), Jasper(), "module top(input clk,a,b); endmodule",
                                         "top", "clk", None, Path(directory), 0)
            final, trace = generator.run_signal({"name": "a", "information": "a implies b"}, "a implies b")
            self.assertEqual([assertion.text for assertion in final], [sva])
            self.assertEqual(trace["valid_candidate_count"], 1)

    def test_final_prompt_shares_declarations_without_dropping_candidates(self):
        response = """```systemverilog
localparam int LIMIT = 4;
property p_a; @(posedge clk) a |-> b; endproperty
property p_b; @(posedge clk) b |-> a; endproperty
assert property (p_a);
assert property (p_b);
assert property (@(posedge clk) count < LIMIT);
```"""

        class LLM:
            final_prompt = None

            def chat(self, system, user):
                if "Critique every issue" in user:
                    return "[Score] 50"
                if "Return every unique valid assertion" in user:
                    self.final_prompt = user
                return response

        class Jasper:
            def check(self, *_):
                return parse_log("[1] top.sangam_assert proven PRE Infinite 0.0 s", 0)

        with tempfile.TemporaryDirectory() as directory:
            llm = LLM()
            generator = SangamGenerator(llm, Jasper(), "module top; endmodule",
                                        "top", "clk", None, Path(directory), 0)
            final, _ = generator.run_signal({"name": "a", "information": "handshake"}, "handshake")
            expected = extract_assertions(response)
            self.assertEqual(final, expected)
            self.assertEqual(llm.final_prompt.count("localparam int LIMIT = 4;"), 1)
            self.assertEqual(llm.final_prompt.count("property p_a;"), 1)
            self.assertEqual(llm.final_prompt.count("property p_b;"), 1)
            for assertion in expected:
                self.assertEqual(llm.final_prompt.count(assertion.text), 1)

    def test_clockless_final_retries_if_model_invents_clock(self):
        sva = "assert property (a |-> b);"

        class LLM:
            def chat(self, system, user):
                if "Critique every issue" in user:
                    return "[Score] 50"
                if "Your previous answer changed" in user:
                    return f"```systemverilog\n{sva}\n```"
                if "Return every unique valid assertion" in user:
                    return "```systemverilog\nassert property (@(posedge clk) a |-> b);\n```"
                return f"```systemverilog\n{sva}\n```"

        class Jasper:
            def check(self, *_):
                return parse_log("[1] top.sangam_assert proven PRE Infinite 0.0 s", 0)

        with tempfile.TemporaryDirectory() as directory:
            generator = SangamGenerator(LLM(), Jasper(), "module top(input a,b); endmodule",
                                        "top", None, None, Path(directory), 0)
            final, trace = generator.run_signal({"name": "a", "information": "a implies b"}, "a implies b")
            self.assertEqual([assertion.text for assertion in final], [sva])
            self.assertNotIn("posedge", trace["final_response"])

    def test_final_assertion_and_evidence_become_one_jsonl_row(self):
        class LLM:
            model = "deepseek-ai/DeepSeek-V4-Flash-0731"

        class Jasper:
            timeout = 120

            def __init__(self):
                self.calls = 0
                self.resets = []

            def check(self, *args):
                self.calls += 1
                self.resets.append(args[-1])
                return parse_log("""[1] top.sangam_assert proven PRE Infinite 0.0 s
[2] top.sangam_assert:precondition1 unreachable PRE Infinite 0.0 s
[3] top.sangam_assert:contrapositive1 covered PRE 1 0.0 s
""", 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.txt"
            rtl_path = root / "rtl.v"
            spec_path.write_text("Signal a must imply b.")
            rtl_path.write_text("module top(input logic clk, a, b); always @(posedge clk) begin end endmodule")
            output = root / "out"
            workdir = output / "cases" / "L1" / "sample"
            signal_dir = workdir / "signals" / "000_a" / f"iterations_4_{GENERATION_CONFIG}"
            signal_dir.mkdir(parents=True)
            (workdir / "signals.json").write_text(json.dumps([{"name": "a", "information": "a implies b"}]))
            (signal_dir / "final.json").write_text(json.dumps([
                {"text": "assert property (@(posedge clk) a |-> b);", "support": "", "implication": True}
            ]))
            case = Case("L1/sample", "L1", spec_path, rtl_path)
            jasper = Jasper()
            record = process_case(case, output, LLM(), jasper, "revision", 4, 0)
            self.assertEqual(record["status"], "done")
            self.assertEqual(jasper.calls, 1)
            self.assertEqual(jasper.resets, [None])
            self.assertEqual(record["formal_reset_policy"], "unconstrained")
            (workdir / "case.json").unlink()
            resumed = process_case(case, output, LLM(), jasper, "revision", 4, 0)
            self.assertEqual(resumed["rows"], record["rows"])
            self.assertEqual(jasper.calls, 1)
            self.assertEqual(rebuild_jsonl(output, "revision", LLM.model, 4), 1)
            row = json.loads((output / "dataset.jsonl").read_text())
            self.assertEqual(row["spec"], spec_path.read_text())
            self.assertEqual(row["rtl"], rtl_path.read_text())
            self.assertTrue(row["vacuous"])
            self.assertEqual(row["reason"], "antecedent_never_true")
            self.assertEqual(row["refinement_iterations"], 4)
            self.assertEqual(row["generation_config"], GENERATION_CONFIG)
            self.assertEqual(rebuild_jsonl(output, "revision", LLM.model, 0), 0)

            # A provider model replacement must keep completed cases and rows,
            # then resume a partial case without relabelling legacy signals.
            class Replacement:
                model = "deepseek-ai/DeepSeek-V4.1-Flash-1017"

            replacement = Replacement()
            reuse = {LLM.model}
            self.assertEqual(process_case(case, output, replacement, jasper, "revision", 4, 0,
                                          reuse_models=reuse), resumed)
            self.assertEqual(rebuild_jsonl(output, "revision", replacement.model, 4, reuse), 1)
            (workdir / "case.json").unlink()
            (workdir / "signals.json").write_text(json.dumps([
                {"name": "a", "information": "a implies b"},
                {"name": "b", "information": "b implies a"}]))
            new_assertion = Assertion("assert property (@(posedge clk) b |-> a);", "", True)
            with patch.object(SangamGenerator, "run_signal", return_value=([new_assertion], {})) as run:
                mixed = process_case(case, output, replacement, jasper, "revision", 4, 0,
                                     reuse_models=reuse)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(jasper.calls, 2)  # Original final proof was reused.
            self.assertEqual([r["model"] for r in mixed["rows"]], [LLM.model, replacement.model])
            self.assertEqual(set(mixed["generation_models"]), {LLM.model, replacement.model})
            self.assertEqual(json.loads((signal_dir / "final.json").read_text())[0]["text"],
                             record["rows"][0]["assertion"])
            self.assertEqual(rebuild_jsonl(output, "revision", replacement.model, 4, reuse), 2)
            self.assertEqual(rebuild_jsonl(output, "revision", replacement.model, 4), 0)

    def test_zero_final_assertions_are_recorded_as_failure(self):
        class LLM:
            model = "deepseek-ai/DeepSeek-V4-Flash-0731"

        class Jasper:
            timeout = 120

            def check(self, *_):
                self.fail("No final assertions should trigger a JasperGold check")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path, rtl_path = root / "spec.txt", root / "rtl.v"
            spec_path.write_text("a")
            rtl_path.write_text("module top(input clk,a); always @(posedge clk) begin end endmodule")
            output = root / "out"
            workdir = output / "cases" / "L1" / "sample"
            signal_dir = workdir / "signals" / "000_a" / f"iterations_4_{GENERATION_CONFIG}"
            (signal_dir / "candidate_logs").mkdir(parents=True)
            (signal_dir / "candidate_logs" / "failure.log").write_text(
                "multiple event control statements is not supported for synthesis")
            (workdir / "signals.json").write_text(json.dumps([{"name": "a", "information": "a"}]))
            (signal_dir / "final.json").write_text("[]")
            case = Case("L1/sample", "L1", spec_path, rtl_path)
            record = process_case(case, output, LLM(), Jasper(), "revision", 4, 0)
            self.assertEqual(record["status"], "no_final_assertions")
            self.assertIn("cannot elaborate", record["failure_reason"])
            self.assertEqual(rebuild_jsonl(output, "revision", LLM.model, 4), 0)
            self.assertEqual(process_case(case, output, LLM(), Jasper(), "revision", 4, 0), record)


if __name__ == "__main__":
    unittest.main()
