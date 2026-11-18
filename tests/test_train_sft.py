import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.training.train_sft import (
    checkpoint_complete, encode_record, make_checkpoint_callback,
    resolve_resume, smoke_ids, write_json,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, truncation, return_dict):
        if truncation is not False:
            raise AssertionError("Full inputs must not be truncated")
        ids = [1]
        for item in messages:
            if item["role"] == "assistant":
                ids += [2] + [3] * len(item["content"])
            else:
                ids += [4] * len(item["content"])
        if add_generation_prompt:
            ids += [2]
        return ids


def checkpoint(folder, step, fingerprint="run"):
    folder.mkdir()
    for name in ("adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (folder / name).write_text("saved")
    write_json(folder / "trainer_state.json", {"global_step": step})
    write_json(folder / "checkpoint_complete.json", {"global_step": step, "run_fingerprint": fingerprint})


class SftTrainingPreflightTests(unittest.TestCase):
    def test_loss_mask_preserves_full_prompt_and_all_output_tokens(self):
        record = {"id": "sample", "messages": [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "x" * 17500},
            {"role": "assistant", "content": "assert property (req |-> ack);"},
        ]}
        tokenizer = FakeTokenizer()
        ids = tokenizer.apply_chat_template(record["messages"], tokenize=True,
            add_generation_prompt=False, truncation=False, return_dict=False)
        prompt = tokenizer.apply_chat_template(record["messages"][:-1], tokenize=True,
            add_generation_prompt=True, truncation=False, return_dict=False)
        expected = {"total_tokens": len(ids), "prompt_tokens": len(prompt),
                    "assistant_tokens": len(ids) - len(prompt)}
        encoded = encode_record(tokenizer, record, expected)
        self.assertGreater(len(encoded["input_ids"]), 16384)
        self.assertTrue(all(label == -100 for label in encoded["labels"][:len(prompt)]))
        self.assertEqual(encoded["labels"][len(prompt):], ids[len(prompt):])
        self.assertEqual(sum(label != -100 for label in encoded["labels"]), expected["assistant_tokens"])
        with self.assertRaisesRegex(ValueError, "lengths changed"):
            encode_record(tokenizer, record, dict(expected, total_tokens=1))

    def test_smoke_never_updates_validation_and_selects_two_memory_extremes(self):
        lengths = {
            "long_train": {"id": "long_train", "split": "train", "total_tokens": 10000, "assistant_tokens": 10},
            "long_output": {"id": "long_output", "split": "train", "total_tokens": 5000, "assistant_tokens": 2000},
            "val": {"id": "val", "split": "validation", "total_tokens": 18000, "assistant_tokens": 100},
        }
        chosen = smoke_ids(lengths)
        self.assertEqual(chosen["train"], {"long_train", "long_output"})
        self.assertEqual(chosen["validation"], {"val"})
        lengths["long_train"]["assistant_tokens"] = 3000
        self.assertEqual(len(smoke_ids(lengths)["train"]), 2)

    def test_resume_skips_partial_or_incompatible_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint(root / "checkpoint-1", 1)
            checkpoint(root / "checkpoint-2", 2)
            (root / "checkpoint-2" / "optimizer.pt").unlink()
            checkpoint(root / "checkpoint-3", 3, "different_run")
            self.assertEqual(Path(resolve_resume(root, "auto", "run")).name, "checkpoint-1")
            self.assertFalse(checkpoint_complete(root / "checkpoint-2", "run"))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                resolve_resume(root, str(root / "checkpoint-2"), "run")

    def test_stop_requests_save_at_optimizer_boundary_and_marker_follows_complete_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stop = {"stop_requested": True}
            callback = make_checkpoint_callback(object, "run", 600, 25, stop)
            args = SimpleNamespace(output_dir=str(root))
            state = SimpleNamespace(global_step=1)
            control = SimpleNamespace(should_save=False, should_training_stop=False, should_evaluate=True)
            callback.on_step_end(args=args, state=state, control=control)
            self.assertTrue(control.should_save)
            self.assertTrue(control.should_training_stop)
            self.assertFalse(control.should_evaluate)
            control.should_evaluate = True
            callback.on_epoch_end(args=args, state=state, control=control)
            self.assertFalse(control.should_evaluate)
            checkpoint(root / "checkpoint-1", 1)
            (root / "checkpoint-1" / "checkpoint_complete.json").unlink()
            callback.on_save(args=args, state=state, control=control)
            self.assertTrue(checkpoint_complete(root / "checkpoint-1", "run"))
            (root / "checkpoint-1" / "scheduler.pt").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing"):
                callback.on_save(args=args, state=state, control=control)
            self.assertFalse((root / "checkpoint-1" / "checkpoint_complete.json").exists())


if __name__ == "__main__":
    unittest.main()
