import json
from pathlib import Path
import tempfile
import unittest

from experiments.training.prepare_kto import collect_case, covered_label, feedback_label, kto_record
from experiments.training.prepare_sft import SYSTEM_PROMPT, sha256


def source(name="p", **changes):
    row = {"case_id": "L1/example", "spec": "Requests require an acknowledgement.",
           "rtl": "module example(input clk, req, ack); endmodule",
           "assertion": "assert property (" + name + ");",
           "support_code": "property " + name + "; @(posedge clk) req |-> ack; endproperty",
           "assertion_status": "proven", "vacuous": False,
           "reason": "neither_targeted_vacuity_reason", "signals": ["req"],
           "clock": "clk", "reset": None, "formal_reset_policy": "unconstrained",
           "jasper_log": "original.log"}
    row.update(changes)
    row.update(spec_sha256=sha256(row["spec"]), rtl_sha256=sha256(row["rtl"]))
    return row


SIGNALS = [{"name": "req", "information": "Request input."}]


class KtoPreparationTests(unittest.TestCase):
    def test_feedback_accepts_proven_non_implication_but_excludes_failed_or_unknown(self):
        self.assertIs(feedback_label(source()), True)
        self.assertIs(feedback_label(source(vacuous=True, reason="consequent_always_true")), False)
        self.assertIs(feedback_label(source(vacuous=None, reason="outside_scope_non_implication")), True)
        for changes in ({"assertion_status": "cex"}, {"assertion_status": "bounded_proven"},
                        {"vacuous": None, "reason": "related_cover_inconclusive_or_unsupported"},
                        {"assertion_status": "cex", "vacuous": None, "reason": "outside_scope_non_implication"},
                        {"vacuous": True, "reason": "outside_scope_non_implication"}):
            self.assertIsNone(feedback_label(source(**changes)))
        self.assertIsNone(covered_label({"precondition": {"a": "covered"}, "contrapositive": {}}))
        self.assertIs(covered_label({"precondition": {"a": "unreachable"}, "contrapositive": {}}), False)
        self.assertIsNone(covered_label({"precondition": {"a": "bounded_unreachable"}, "contrapositive": {}}))

    def test_non_implication_needs_proof_but_no_related_covers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "original.log").write_text(
                "[<embedded>] % clock clk\n[1] example.sangam_assert proven\n")
            row = source(vacuous=None, reason="outside_scope_non_implication",
                         support_code="property p; @(posedge clk) req == ack; endproperty")
            case = {"case_id": "L1/example", "rows": [row]}
            examples, stats, _ = collect_case(case, SIGNALS, "case.json", None, root, {})
            self.assertEqual(len(examples), 1)
            example = examples[0]
            self.assertIs(example["label"], True)
            self.assertIsNone(example["vacuous"])
            self.assertEqual(example["category"], "proven_non_implication")
            self.assertEqual(example["reason"], "outside_scope_non_implication")
            self.assertEqual(stats["categories"], {"proven_non_implication": 1})
            self.assertIs(kto_record(example)["label"], True)
            (root / "original.log").write_text(
                "[<embedded>] % clock clk\n[1] example.sangam_assert cex\n")
            examples, stats, _ = collect_case(case, SIGNALS, "case.json", None, root, {})
            self.assertEqual(examples, [])
            self.assertEqual(stats["excluded_rows"]["original_named_proof_not_confirmed"], 1)

    def test_non_implication_label_cannot_hide_an_implication(self):
        case = {"case_id": "L1/example", "rows": [source(vacuous=None, reason="outside_scope_non_implication")]}
        examples, stats, _ = collect_case(case, SIGNALS, "case.json", None, Path("."), {})
        self.assertEqual(examples, [])
        self.assertEqual(stats["excluded_rows"]["property_scope_conflicts_with_saved_category"], 1)

    def test_named_cover_evidence_dedup_and_no_label_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "original.log").write_text(
                "[<embedded>] % clock clk\n"
                "[1] example.sangam_assert proven\n"
                "[2] example.sangam_assert:precondition1 covered\n"
                "[3] example.sangam_assert:contrapositive1 covered\n")
            case = {"case_id": "L1/example", "rows": [source(), source(name="renamed")]}
            examples, stats, conflicts = collect_case(case, SIGNALS, "case.json", None, root, {})
            self.assertEqual(stats["unique_examples"], 1)
            self.assertEqual(stats["selected_before_deduplication"], 2)
            self.assertEqual(conflicts, [])
            record = kto_record(examples[0])
            self.assertEqual(set(record), {"prompt", "completion", "label"})
            self.assertEqual(record["prompt"][0]["content"], SYSTEM_PROMPT)
            self.assertEqual([m["role"] for m in record["prompt"]], ["system", "user"])
            self.assertNotIn("neither_targeted_vacuity_reason", json.dumps(record["prompt"]))
            self.assertIn("property p;", record["completion"][0]["content"])

    def test_corrected_context_needs_matching_row_audit_and_missing_covers_are_not_negatives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "original.log").write_text("[<embedded>] % clock clk\n[1] example.sangam_assert proven\n")
            case = {"case_id": "L1/example", "rows": [source(vacuous=True, reason="consequent_always_true")]}
            examples, stats, _ = collect_case(case, SIGNALS, "case.json", None, root, {})
            self.assertEqual(examples, [])
            self.assertEqual(stats["excluded_rows"]["related_cover_evidence_incomplete_or_deferred_diagnostic"], 1)
            case["formal_clocking"] = {"version": 1, "declarations": [{"signal": "clk", "edges": "both"}]}
            examples, stats, _ = collect_case(case, SIGNALS, "case.json", None, root, {})
            self.assertEqual(examples, [])
            self.assertEqual(stats["excluded_rows"]["explicit_clock_context_without_row_recheck"], 1)

    def test_conflicting_duplicate_groups_are_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, state in (("original.log", "covered"), ("negative.log", "unreachable")):
                (root / filename).write_text(
                    "[<embedded>] % clock clk\n"
                    "[1] example.sangam_assert proven\n"
                    "[2] example.sangam_assert:precondition1 covered\n"
                    "[3] example.sangam_assert:contrapositive1 " + state + "\n")
            case = {"case_id": "L1/example", "rows": [source(), source(name="q", vacuous=True,
                    reason="consequent_always_true", jasper_log="negative.log")]}
            examples, stats, conflicts = collect_case(case, SIGNALS, "case.json", None, root, {})
            self.assertEqual(examples, [])
            self.assertEqual(stats["quarantined_conflicting_groups"], 1)
            self.assertEqual(len(conflicts[0]["provenance"]["sources"]), 2)


if __name__ == "__main__":
    unittest.main()
