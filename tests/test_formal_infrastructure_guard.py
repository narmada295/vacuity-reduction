import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from experiments.evaluation.formal_infrastructure_guard import recover


class InfrastructureGuardTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory)
        root = base / 'evaluation'
        proc = base / 'proc'
        folder = root / 'jobs' / ('a' * 24)
        project = folder / 'jgproject'
        project.mkdir(parents=True)
        (project / 'host.999.lock').write_text('dead owner')
        (project / 'jg_console.log').write_text('old license failure')
        (folder / 'jg.log').write_text('original launch evidence')
        (folder / 'dut.sv').write_text('original RTL and assertion')
        (folder / 'run.tcl').write_text('original formal context')
        worker = proc / '100'
        worker.mkdir(parents=True)
        (worker / 'cmdline').write_bytes(b'python3\0' + str(root / 'formal_worker.py').encode()
                                        + b'\0--root\0' + str(root).encode() + b'\0')
        (worker / 'status').write_text('State:\tT (stopped)\n')
        state = {'state': 'waiting_for_license', 'pid': 100, 'current_job': 'a' * 24}
        (root / 'status.json').write_text(json.dumps(state))
        return root, proc, folder, state

    def test_dead_project_is_archived_with_evidence_and_worker_resumes(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('experiments.evaluation.formal_infrastructure_guard.os.kill') as kill:
            root, proc, folder, _ = self.fixture(directory)
            result = recover(root, proc)
            self.assertEqual(result['dead_lock_owners'], [999])
            archive = Path(result['archive'])
            self.assertEqual((archive / 'jgproject/jg_console.log').read_text(), 'old license failure')
            self.assertEqual((archive / 'jg_launch.log').read_text(), 'original launch evidence')
            self.assertEqual((folder / 'dut.sv').read_text(), 'original RTL and assertion')
            self.assertEqual((folder / 'run.tcl').read_text(), 'original formal context')
            self.assertFalse((folder / 'jgproject').exists())
            self.assertEqual([call.args for call in kill.call_args_list],
                             [(100, signal.SIGSTOP), (100, signal.SIGCONT)])

    def test_live_lock_owner_or_live_cwd_prevents_recovery(self):
        for reason in ('owner', 'cwd', 'completed'):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory, \
                 patch('experiments.evaluation.formal_infrastructure_guard.os.kill') as kill:
                root, proc, folder, _ = self.fixture(directory)
                if reason == 'owner':
                    (proc / '999').mkdir()
                elif reason == 'cwd':
                    (proc / '101').mkdir()
                    (proc / '101/cwd').symlink_to(folder / 'jgproject', target_is_directory=True)
                else:
                    (folder / 'result.json').write_text('{}')
                self.assertIsNone(recover(root, proc))
                self.assertTrue((folder / 'jgproject').exists())
                if reason != 'completed':
                    self.assertEqual(kill.call_args.args, (100, signal.SIGCONT))
                else:
                    kill.assert_not_called()

    def test_running_job_or_wrong_worker_is_never_interrupted(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('experiments.evaluation.formal_infrastructure_guard.os.kill') as kill:
            root, proc, folder, state = self.fixture(directory)
            (root / 'status.json').write_text(json.dumps(dict(state, state='running')))
            self.assertIsNone(recover(root, proc))
            (root / 'status.json').write_text(json.dumps(state))
            (proc / '100/cmdline').write_bytes(b'unrelated-process\0')
            self.assertIsNone(recover(root, proc))
            kill.assert_not_called()
            self.assertTrue((folder / 'jgproject').exists())


if __name__ == '__main__':
    unittest.main()
