"""Download a public seizure benchmark dataset into a separate folder.

This script keeps the public benchmark separate from the Empatica/Mayo
clinical pipeline. The default source is OpenNeuro SeizeIT2, which is a
public wearable epilepsy dataset with a dedicated accession.

Usage:
    python seizure_detection/download_public_benchmark.py
    python seizure_detection/download_public_benchmark.py --dest C:/I2I/public_benchmark
    python seizure_detection/download_public_benchmark.py --source openneuro --dataset ds005873

Requirements:
    - openneuro CLI installed and available on PATH
    - internet access

The script downloads into a standalone folder and writes a small manifest so
the benchmark lane can be referenced separately from the main clinical data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET = "ds005873"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "public_benchmark_data"
DEFAULT_SNAPSHOT = "1.1.0"


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def download_openneuro(dataset: str, destination: Path, snapshot: str | None) -> None:
    executable = shutil.which("openneuro")
    if executable is None:
        raise SystemExit(
            "openneuro CLI not found on PATH. Install it with:\n"
            "  deno install -A --global jsr:@openneuro/cli -n openneuro"
        )

    destination.mkdir(parents=True, exist_ok=True)
    command = [executable, "download"]
    if snapshot:
        command.extend(["--snapshot", snapshot])
    command.extend([dataset, str(destination)])
    run_command(command)


def write_manifest(destination: Path, source: str, dataset: str, snapshot: str | None) -> None:
    manifest = {
        "source": source,
        "dataset": dataset,
        "snapshot": snapshot,
        "destination": str(destination),
        "retrieval_policy": "download metadata, then datalad get -r . for annexed content",
    }
    (destination / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="openneuro", choices=["openneuro"])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Audit an existing checkout without downloading or modifying it",
    )
    parser.add_argument(
        "--get-all",
        action="store_true",
        help="After download, retrieve all annexed files with DataLad",
    )
    args = parser.parse_args()

    destination = Path(args.dest).expanduser().resolve()
    if args.verify_only:
        audit_script = Path(__file__).resolve().parent / "public_data_audit.py"
        result = subprocess.run(
            [sys.executable, str(audit_script), "--root", str(destination)],
            check=False,
        )
        return result.returncode
    if args.source == "openneuro":
        download_openneuro(args.dataset, destination, args.snapshot)
        if args.get_all:
            datalad = shutil.which("datalad")
            if datalad is None:
                raise SystemExit("DataLad is required for --get-all but was not found on PATH")
            run_command([datalad, "get", "-r", "."], cwd=destination)
        write_manifest(destination, args.source, args.dataset, args.snapshot)
        print(f"Downloaded benchmark dataset to: {destination}")
        print("Point any auxiliary benchmark scripts at this folder separately.")
        return 0

    raise SystemExit(f"Unsupported source: {args.source}")


if __name__ == "__main__":
    raise SystemExit(main())
