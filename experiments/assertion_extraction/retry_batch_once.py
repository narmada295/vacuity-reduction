#!/usr/bin/env python3
"""Restart a remote reduction batch once after it exits or loses its process."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def running(pid):
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--jg", default="jg")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retry-delay", type=int, default=180)
    parser.add_argument("--poll", type=int, default=60)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf8"))
    root = Path(plan["remote_base"]) / "results"
    status_path = root / plan.get("status_name", "reduction_25_status.json")

    while True:
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf8"))
            if status.get("state") == "complete":
                print("Primary batch completed; no retry needed.", flush=True)
                return 0
            if status.get("state") == "finished_with_failures":
                break
            if status.get("state") == "running" and not running(status.get("pid")):
                print("Primary batch process ended before completion; resuming.", flush=True)
                break
        time.sleep(args.poll)

    command = [sys.executable, str(root / "run_batch.py"),
               "--plan", args.plan, "--jg", args.jg,
               "--timeout", str(args.timeout),
               "--retry-delay", str(args.retry_delay)]
    print("Starting one checkpoint-based follow-up pass.", flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
