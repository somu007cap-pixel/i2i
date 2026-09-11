"""Restore SeizeIT2 event labels without downloading EEG signal payloads."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_REPOSITORY = "https://github.com/OpenNeuroDatasets/ds005873.git"


def run(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, check=False, text=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def restore(destination: Path, repository: str) -> int:
    # Make repeated campaign launches network-free once the local checkout is
    # complete. This also prevents a transient GitHub outage from invalidating
    # an already verified dataset.
    existing_events = list(destination.rglob("*_events.tsv"))
    if existing_events and len(existing_events) >= 2850:
        print(
            f"Existing public annotation set detected ({len(existing_events)} files); "
            "skipping network restoration"
        )
        return len(existing_events)
    git = shutil.which("git")
    if not git:
        raise SystemExit("git is required to restore public event annotations")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="seizeit2_metadata_") as temp:
        checkout = Path(temp) / "dataset"
        run([
            git,
            "clone",
            "--branch",
            "1.1.0",
            "--filter=blob:none",
            "--no-checkout",
            repository,
            str(checkout),
        ])
        tree = subprocess.run(
            [git, "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        )
        event_paths = [
            Path(line.strip())
            for line in tree.stdout.splitlines()
            if line.strip().endswith("_events.tsv")
        ]
        if not event_paths:
            raise SystemExit("No *_events.tsv files were found in the source repository")
        for relative in event_paths:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                result = subprocess.run(
                    [git, "show", f"HEAD:{relative.as_posix()}"],
                    cwd=checkout,
                    stdout=handle,
                    check=False,
                )
            if result.returncode:
                raise SystemExit(result.returncode)
    print(f"Restored {len(event_paths)} event TSV files into {destination}")
    return len(event_paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    args = parser.parse_args()
    restore(Path(args.dest).expanduser().resolve(), args.repository)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
