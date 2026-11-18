import json
from pathlib import Path
import tempfile
import unittest

from experiments.training.train_kto import (
    cached_batch, class_weights, complete_checkpoint, digest, measure,
    resolve_resume, smoke_selection, validate_record, write_json,
)


class KtoTrainingTests(unittest.TestCase):
    def test_class_weights_balance_feedback_without_duplicating_data(self):
        weights = class_weights({"desirable": 4948, "undesirable": 726})
        self.assertEqual(weights["desirable"], 1)
        self.assertAlmostEqual(4948 * weights["desirable"], 726 * weights["undesirable"])
        with self.assertRaises(ValueError):
            class_weights({"desirable": 1, "undesirable": 0})

    def test_scope_and_split_gates(self):
        plan = {"train": ["L1/a"], "validation": ["L2/b"], "test": ["L3/c"]}
        example = {"split": "train", "case_id": "L1/a", "label": True,
                   "category": "proven_non_implication", "assertion_status": "proven",
                   "vacuous": None, "assertion": "assert property ($bits(req) == 1);"}
        record = {"prompt": [{"role": "system", "content": "Generate one assertion."},
                             {"role": "user", "content": "Specification and RTL."}],
                  "completion": [{"role": "assistant", "content": "```systemverilog\n" + example["assertion"] + "\n```"}],
                  "label": True}
        validate_record(record, example, "train", plan)
        with self.assertRaises(ValueError):
            validate_record(record, dict(example, case_id="L3/c"), "train", plan)
        with self.assertRaises(ValueError):
            validate_record(record, dict(example, vacuous=False), "train", plan)
        with self.assertRaises(ValueError):
            validate_record(dict(record, label=False), example, "train", plan)

    def test_resume_ignores_partial_checkpoint_and_rejects_changed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(resolve_resume(root, "auto", "run"))
            good = root / "checkpoint-4"
            good.mkdir()
            for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (good / name).write_text("fixture")
            write_json(good / "trainer_state.json", {"global_step": 4})
            write_json(good / "checkpoint_complete.json", {"global_step": 4, "run_fingerprint": "run",
                                                          "frozen_reference_sha256": "sft", "policy_sha256": "policy"})
            self.assertTrue(complete_checkpoint(good, "run"))
            self.assertFalse(complete_checkpoint(good, "different"))
            partial = root / "checkpoint-5"
            partial.mkdir()
            write_json(partial / "trainer_state.json", {"global_step": 5})
            self.assertEqual(resolve_resume(root, "auto", "run"), str(good))
            with self.assertRaises(ValueError):
                resolve_resume(root, str(partial), "run")
            (good / "optimizer.pt").unlink()
            self.assertFalse(complete_checkpoint(good, "run"))

    def test_reference_cache_survives_partial_write_and_rejects_mismatched_or_nonfinite_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0.json"
            path.with_suffix(".json.tmp").write_text('{"partial":')
            self.assertIsNone(cached_batch(path, "key", 0, 2))
            value = {"key": "key", "start": 0, "count": 2,
                     "ref_logps": [-1.0, -2.0], "ref_KL_logps": [-3.0, -4.0]}
            write_json(path, value)
            self.assertEqual(cached_batch(path, "key", 0, 2), value)
            for key, start, count in (("wrong", 0, 2), ("key", 2, 2), ("key", 0, 1)):
                with self.assertRaises(ValueError):
                    cached_batch(path, key, start, count)
            write_json(path, dict(value, ref_logps=[float("nan"), -2.0]))
            with self.assertRaises(ValueError):
                cached_batch(path, "key", 0, 2)

    def test_smoke_covers_long_inputs_outputs_negatives_and_non_implications(self):
        lengths = []
        for split, count in (("train", 12), ("validation", 6)):
            for n in range(count):
                kind = ("proven_non_implication", "proven_non_vacuous_implication", "proven_vacuous_implication")[n % 3]
                lengths.append({"index": n, "split": split, "category": kind,
                                "label": kind != "proven_vacuous_implication",
                                "total_tokens": n + 10, "completion_tokens": 30 - n})
        selection = smoke_selection(lengths)
        self.assertEqual(len(selection["train"]), 8)
        self.assertEqual(len(selection["validation"]), 4)
        self.assertIn(11, selection["train"])
        self.assertIn(0, selection["train"])
        selected = [r for r in lengths if r["split"] == "train" and r["index"] in selection["train"]]
        self.assertIn("proven_non_implication", {r["category"] for r in selected})
        self.assertIn(False, {r["label"] for r in selected})

    def test_prompt_mask_boundary_is_not_silently_changed(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2] if len(messages) == 2 else [1, 2, 3, 4]
        record = {"prompt": [{}, {}], "completion": [{}]}
        self.assertEqual(measure(Tokenizer(), record), {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4})
        class WrongTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2] if len(messages) == 2 else [1, 3, 4]
        with self.assertRaises(ValueError):
            measure(WrongTokenizer(), record)


if __name__ == "__main__":
    unittest.main()
