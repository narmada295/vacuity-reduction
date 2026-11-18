import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.training.prepare_sft import (
    category, chat_record, examples_for_case, export, load_split, minimal_support, sha256,
)
from experiments.training.verify_sft import verify


def row(case_id="L1/example", name="p", body="@(posedge clk) req |-> ack", **changes):
    spec = "Specification of " + case_id
    rtl = "module " + case_id.replace("/", "_") + "; endmodule"
    result = {
        "case_id": case_id, "spec": spec, "rtl": rtl,
        "spec_sha256": sha256(spec), "rtl_sha256": sha256(rtl),
        "assertion": "assert property (" + name + ");",
        "support_code": "property " + name + "; " + body + "; endproperty",
        "assertion_status": "proven", "vacuous": False,
        "reason": "neither_targeted_vacuity_reason", "signals": ["req"],
        "clock": "clk", "reset": "rst_n", "formal_reset_policy": "unconstrained",
        "jasper_log": "fixture_clock.log",
    }
    result.update(changes)
    return result


SIGNALS = [{"name": "req", "information": "Request input."},
           {"name": "ack", "information": "Acknowledgment output."}]


class SftPreparationTests(unittest.TestCase):
    def test_design_quotas_reject_a_missing_training_design(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            plan = {
                "train": ["L1/a", "L2/b", "L3/c"],
                "validation": ["L1/v"], "test": ["L3/t"],
                "design_quotas": {
                    "train": {"L1": 1, "L2": 1, "L3": 2},
                    "validation": {"L1": 1, "L2": 0, "L3": 0},
                    "test": {"L1": 0, "L2": 0, "L3": 1},
                },
            }
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "Design quota mismatch for train"):
                load_split(path)
            plan["train"].append("L3/d")
            path.write_text(json.dumps(plan))
            self.assertEqual(len(load_split(path)[1]), 6)
            plan["pending_allowed"] = ["L3/d"]
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "only test designs may be pending"):
                load_split(path)
            plan["pending_allowed"] = ["L3/t"]
            path.write_text(json.dumps(plan))
            self.assertEqual(load_split(path)[0]["pending_allowed"], ["L3/t"])

    def test_only_proven_conclusive_labels_enter_sft(self):
        self.assertEqual(category(row()), "proven_non_vacuous_implication")
        self.assertEqual(category(row(vacuous=None, reason="outside_scope_non_implication")),
                         "proven_non_implication")
        for changes in [
            {"assertion_status": "cex", "vacuous": False},
            {"assertion_status": "cex", "vacuous": None, "reason": "outside_scope_non_implication"},
            {"assertion_status": "undetermined"},
            {"vacuous": True, "reason": "consequent_always_true"},
            {"vacuous": None, "reason": "related_cover_inconclusive_or_unsupported"},
        ]:
            self.assertIsNone(category(row(**changes)))

    def test_dedup_ignores_property_names_and_preserves_origin_evidence(self):
        first = row(name="first")
        second = row(name="second", body="@( posedge clk ) req  |-> ack", signals=["ack"])
        case = {"case_id": "L1/example", "rows": [first, second]}
        examples, stats = examples_for_case(case, SIGNALS, "case.json")
        self.assertEqual(len(examples), 1)
        self.assertEqual(stats["exact_duplicates_removed"], 1)
        self.assertEqual(examples[0]["target_signal"], "req")
        self.assertEqual(examples[0]["signal_info"], SIGNALS[0])
        self.assertEqual([s["signals"] for s in examples[0]["provenance"]["sources"]], [["req"], ["ack"]])
        self.assertIn("property first;", examples[0]["assertion"])
        self.assertNotIn("property second;", examples[0]["assertion"])

    def test_dedup_respects_formal_context_and_expands_constant_meaning(self):
        first = row(body="@(posedge clk) req |-> count < WIDTH")
        first["support_code"] = "localparam WIDTH = 8;\n" + first["support_code"]
        second = row(name="alias", body="@(posedge clk) req |-> count < (8)")
        third = row(name="other_context", body="@(posedge clk) req |-> count < (8)", clock="other_clk")
        examples, _ = examples_for_case({"case_id": "L1/example", "rows": [first, second, third]}, SIGNALS, "case.json")
        self.assertEqual(len(examples), 2)
        self.assertIn("localparam WIDTH = 8;", examples[0]["assertion"])

    def test_support_keeps_dependencies_without_unrelated_property_targets(self):
        source = row()
        source["support_code"] = (
            "localparam WIDTH = 8;\n"
            "property helper; @(posedge clk) count < WIDTH; endproperty\n"
            "property p; req |-> helper; endproperty\n"
            "property unrelated; @(posedge clk) secret_signal; endproperty"
        )
        support = minimal_support(source)
        self.assertIn("localparam WIDTH = 8;", support)
        self.assertIn("property helper;", support)
        self.assertIn("property p;", support)
        self.assertNotIn("unrelated", support)
        self.assertNotIn("secret_signal", support)
        source["support_code"] += "\nalways @(posedge clk) req <= 1;"
        with self.assertRaisesRegex(ValueError, "Unsupported support"):
            minimal_support(source)

    def test_original_support_clock_context_survives_support_cleanup(self):
        first = row()
        second = row(name="q")
        second["support_code"] += "\nproperty unused; @(posedge aux_clk) req; endproperty"
        examples, stats = examples_for_case({"case_id": "L1/example", "rows": [first, second]}, SIGNALS, "case.json")
        self.assertEqual(stats["exact_duplicates_removed"], 0)
        self.assertEqual(len(examples), 2)
        self.assertNotIn("property unused", examples[1]["assertion"])
        self.assertEqual(examples[1]["formal_context"]["checker_clocks"], ["clk", "aux_clk"])
        self.assertIn("clock aux_clk", chat_record(examples[1])["messages"][1]["content"])

    def test_explicit_clock_override_requires_audited_sft_labels_and_renders_both_edges(self):
        context = {"version": 1, "declarations": [{"signal": "main_clk", "edges": "both"}]}
        source = row()
        case = {"case_id": "L1/example", "rows": [source], "formal_clocking": context}
        with self.assertRaisesRegex(ValueError, "matching SFT proof/vacuity evidence"):
            examples_for_case(case, SIGNALS, "case.json")
        source["sft_formal_recheck"] = {"label": "SFT_CHECK_0001", "assessment_sha256": "fixture"}
        examples, _ = examples_for_case(case, SIGNALS, "case.json")
        self.assertEqual(examples[0]["formal_context"]["checker_clocks"], ["main_clk"])
        self.assertIn("clock -both_edges main_clk", chat_record(examples[0])["messages"][1]["content"])
        self.assertEqual(examples[0]["provenance"]["sources"][0]["sft_formal_recheck"],
                         source["sft_formal_recheck"])

    def test_inconsistent_labels_hashes_and_missing_signal_info_fail(self):
        for source in [
            row(vacuous=None, reason="outside_scope_non_implication"),
            row(spec_sha256="wrong"),
            row(signals=["missing_signal"]),
            row(assertion="assert property (undefined);")
        ]:
            with self.assertRaises(ValueError):
                examples_for_case({"case_id": "L1/example", "rows": [source]}, SIGNALS, "case.json")

    def test_chat_contains_inputs_and_code_without_labels_or_other_assertions(self):
        examples, _ = examples_for_case({"case_id": "L1/example", "rows": [row()]}, SIGNALS, "case.json")
        chat = chat_record(examples[0])
        prompt = chat["messages"][1]["content"]
        self.assertIn(examples[0]["spec"], prompt)
        self.assertIn(examples[0]["rtl"], prompt)
        self.assertIn("Request input.", prompt)
        self.assertIn("reset -none", prompt)
        self.assertNotIn("neither_targeted_vacuity_reason", prompt)
        self.assertNotIn("assert property (p)", prompt)
        self.assertEqual(chat["messages"][2]["content"].count("assert property"), 1)

    def test_export_holds_out_designs_allows_pending_test_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases"
            (root / "fixture_clock.log").write_text("[<embedded>] % clock clk\n")
            plan = {"train": ["L1/train"], "validation": ["L2/val"],
                    "test": ["L3/5", "L3/6"], "pending_allowed": ["L3/6"],
                    "excluded": {"L3/3": "excluded by user"}}
            originals = {}
            for case_id in ["L1/train", "L2/val", "L3/5", "L3/3"]:
                folder = cases / case_id
                folder.mkdir(parents=True)
                for name, value in [("case.json", {"case_id": case_id, "status": "done", "rows": [row(case_id)]}),
                                    ("signals.json", SIGNALS)]:
                    path = folder / name
                    path.write_text(json.dumps(value))
                    originals[path] = path.read_bytes()
            split_path = root / "split.json"
            split_path.write_text(json.dumps(plan))
            output = root / "out"
            manifest = export(cases, split_path, output)
            self.assertEqual(manifest["totals"]["unique_examples"], 3)
            self.assertEqual(manifest["pending"][0]["case_id"], "L3/6")
            records = [json.loads(line) for line in (output / "examples.jsonl").read_text().splitlines()]
            self.assertEqual({r["case_id"] for r in records}, {"L1/train", "L2/val", "L3/5"})
            for split, expected in [("train", "L1/train"), ("validation", "L2/val"), ("test", "L3/5")]:
                exported = json.loads((output / (split + ".jsonl")).read_text())
                self.assertEqual(exported["case_id"], expected)
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)
            self.assertEqual(verify(output)["status"], "PASS")
            train_path = output / "train.jsonl"
            original_chat = train_path.read_text()
            altered_chat = json.loads(original_chat)
            altered_chat["messages"][2]["content"] = "assert property (1);"
            train_path.write_text(json.dumps(altered_chat) + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify(output)
            train_path.write_text(original_chat)
            with self.assertRaisesRegex(ValueError, "already exists"):
                export(cases, split_path, output)
            broken = copy.deepcopy(plan)
            broken["train"].append("L3/5")
            split_path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                load_split(split_path)


if __name__ == "__main__":
    unittest.main()
