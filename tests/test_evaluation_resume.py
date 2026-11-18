"""CPU checks for recovery when an allocation ends between disk writes."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

from experiments.evaluation.generate import saved_attempts, save_progress, write_json


class EvaluationResumeTests(unittest.TestCase):
    def test_hard_kill_recovers_committed_attempt_not_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            attempts = out / "attempts"
            attempts.mkdir()
            tasks = [{"task_id": str(n), "case_id": "L1/example", "target_signal": "s" + str(n)}
                     for n in range(3)]
            expected = [(task, 0, attempts / (task["task_id"] + ".json")) for task in tasks]

            def record(task, status="generated"):
                return dict(task, sample=0, label="base", generation_config_fingerprint="frozen",
                            status=status, response="assert property (1);")

            write_json(expected[0][2], record(tasks[0]))
            save_progress(out, expected, "frozen", "base", "running")
            # Commit a second response, then kill before updating aggregate views.
            child_code = (
                "import json, os, signal\n"
                "from pathlib import Path\n"
                "from experiments.evaluation.generate import write_json\n"
                "write_json(Path(os.environ['RESUME_ATTEMPT']), json.loads(os.environ['RESUME_RECORD']))\n"
                "os.kill(os.getpid(), signal.SIGKILL)\n"
            )
            child = subprocess.run([sys.executable, "-c", child_code], env=dict(
                os.environ, RESUME_ATTEMPT=str(expected[1][2]),
                RESUME_RECORD=json.dumps(record(tasks[1], "generation_error"))))
            self.assertEqual(child.returncode, -signal.SIGKILL)
            self.assertEqual(json.loads((out / "summary.json").read_text())["saved_attempts"], 1)
            expected[2][2].with_suffix(".json.tmp").write_text('{"unfinished":')
            recovered = save_progress(out, expected, "frozen", "base", "loading_model")
            self.assertEqual([r["task_id"] for r in recovered], ["0", "1"])
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["saved_attempts"], 2)
            self.assertEqual(summary["remaining_attempts"], 1)
            self.assertEqual(summary["outcomes"], {"generated": 1, "generation_error": 1})
            self.assertEqual(len((out / "generations.jsonl").read_text().splitlines()), 2)
            missing = [task["task_id"] for task, _, path in expected if not path.exists()]
            self.assertEqual(missing, ["2"])

    def test_partial_resume_rejects_mixed_configuration_and_wrong_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.json"
            task = {"task_id": "t", "case_id": "L1/example", "target_signal": "s"}
            row = dict(task, sample=0, label="sft", status="generated",
                       generation_config_fingerprint="frozen")
            for change in ({"generation_config_fingerprint": "other"}, {"target_signal": "wrong"}):
                write_json(path, dict(row, **change))
                with self.assertRaises(ValueError):
                    saved_attempts([(task, 0, path)], "frozen", "sft")


if __name__ == "__main__":
    unittest.main()
