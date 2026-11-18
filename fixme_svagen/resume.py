"""Run the checkpointed dataset CLI through temporary service interruptions."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


TRANSIENT_ERRORS = (
    "hostname mismatch",
    "nebius http 429",
    "nebius http 500",
    "nebius http 502",
    "nebius http 503",
    "nebius http 504",
    "nebius http 529",
    "system is overloaded",
    "jaspergold license unavailable",
    "jaspergold license checkout failed",
    "license_unavailable",
    "jaspergold ssh connection lost",
    "jaspergold timed out",
    "jaspergold hard timeout",
    "timeouterror: timed out",
    "temporary failure in name resolution",
    "unable to connect to port 22",
    "connection reset by peer",
    "remote end closed connection",
)


def is_transient(output: str) -> bool:
    return any(error in output.lower() for error in TRANSIENT_ERRORS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("output/fixme_svagen/run.log"))
    parser.add_argument("--max-restarts", type=int, default=48)
    parser.add_argument("--retry-delay", type=int, default=120)
    args, cli_args = parser.parse_known_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "fixme_svagen.cli", *cli_args]
    with args.log.open("a", buffering=1) as log:
        for attempt in range(args.max_restarts + 1):
            message = f"Dataset run attempt {attempt + 1}: {' '.join(command)}\n"
            print(message, end="", flush=True)
            log.write(message)
            recent: deque[str] = deque(maxlen=100)
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       bufsize=1, env=os.environ.copy())
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    recent.append(line)
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                return 130
            if code == 0:
                return 0
            tail = "".join(recent)
            if attempt >= args.max_restarts or not is_transient(tail):
                message = f"Dataset run stopped with exit code {code}; inspect {args.log}.\n"
                print(message, end="", flush=True)
                log.write(message)
                return code
            message = (f"Temporary service failure; resuming saved checkpoints "
                       f"in {args.retry_delay} seconds.\n")
            print(message, end="", flush=True)
            log.write(message)
            try:
                time.sleep(args.retry_delay)
            except KeyboardInterrupt:
                return 130
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
