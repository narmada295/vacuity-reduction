import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class SftLauncherTests(unittest.TestCase):
    def run_launcher(self, root, smoke_status):
        scripts = root / "scripts/sft_v2"
        scripts.mkdir(parents=True)
        shutil.copyfile(Path(__file__).resolve().parents[1] /
                        "experiments/training/start_sft_v2.sh", scripts / "start.sh")
        data = root / "data/sft_v2"
        data.mkdir(parents=True)
        (data / "manifest.json").write_text(json.dumps({
            "split_plan": {"design_quotas": {"train": {"L1": 9, "L2": 9, "L3": 7}}},
            "splits": {"train": {"designs": 25, "L1_designs": 9, "L2_designs": 9, "L3_designs": 7}},
        }))
        binaries = root / "bin"
        binaries.mkdir()
        fake = binaries / "python"
        fake.write_text("#!" + sys.executable + "\n" + '''import json,os,sys
from pathlib import Path
args=sys.argv[1:]
if args==['-']:
 os.execv(sys.executable,[sys.executable,'-'])
if '--resume' in args:
 raise SystemExit('A fresh run must not resume old optimizer state')
if '--mode' not in args:
 out=Path(args[args.index('--out')+1]);out.mkdir(parents=True)
elif args[args.index('--mode')+1]=='smoke':
 out=Path(args[args.index('--out')+1]);out.mkdir(parents=True)
 (out/'result.json').write_text(json.dumps({'status':os.environ['SFT_LAUNCH_TEST_SMOKE_STATUS'],
  'global_step':2,'complete_checkpoints':['checkpoint-2']}))
else:
 Path('full_run_started').write_text(json.dumps(args))
''')
        fake.chmod(0o755)
        env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                   CUDA_VISIBLE_DEVICES="test-only-no-gpu-use",
                   SFT_LAUNCH_TEST_SMOKE_STATUS=smoke_status)
        return subprocess.run(["bash", str(scripts / "start.sh")], env=env,
                              text=True, capture_output=True)

    def test_successful_smoke_starts_a_fresh_full_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_launcher(root, "PASS")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            args = json.loads((root / "full_run_started").read_text())
            self.assertEqual(args[args.index('--data-dir') + 1], 'data/sft_v2')
            self.assertEqual(args[args.index('--out') + 1], 'outputs/qwen3_4b_sft_v2_train')

    def test_interrupted_smoke_does_not_start_full_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_launcher(root, "stopped_with_checkpoint")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "full_run_started").exists())


if __name__ == "__main__":
    unittest.main()
