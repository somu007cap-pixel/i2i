"""Run a command while streaming combined stdout/stderr to console and a log."""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


def main() -> int:
    """Run a child command while mirroring output to console and a log file."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 3:
        print(
            "Usage: run_with_log.py [--append] LOG_FILE COMMAND [ARGS...]",
            file=sys.stderr,
        )
        return 2
    args = sys.argv[1:]
    append = False
    if args and args[0] == "--append":
        append = True
        args = args[1:]
    if len(args) < 2:
        print(
            "Usage: run_with_log.py [--append] LOG_FILE COMMAND [ARGS...]",
            file=sys.stderr,
        )
        return 2
    log_path = Path(args[0])
    command = args[1:]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8", errors="replace") as log_handle:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
