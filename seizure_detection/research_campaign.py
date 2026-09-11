"""Run a resumable, evidence-first multi-week detection research campaign.

This controller deliberately does not promote a model or alter the final test
split. It schedules the existing experiment runner in documented phases and
keeps a durable manifest so an interrupted campaign can continue safely.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve().parent / "run_tsmixer_experiments.py"
CAMPAIGN_ROOT = Path(__file__).resolve().parent / "outputs" / "research_campaign"
MANIFEST = CAMPAIGN_ROOT / "campaign_manifest.json"
HEARTBEAT = CAMPAIGN_ROOT / "heartbeat.json"


PHASES: list[dict[str, Any]] = [
    {
        "name": "public_data_gate",
        "description": "Audit public signals, metadata, labels, and modality compatibility before training.",
        "kind": "public_audit",
        "env": {},
    },
    {
        "name": "public_external_evaluation",
        "description": "Evaluate retained non-EEG public signals against restored per-recording seizure events.",
        "kind": "public_evaluation",
        "requires_metrics": True,
        "env": {},
    },
    {
        "name": "baseline_reproduction",
        "description": "Reproduce the current focus matrix with a clean, auditable run.",
        "args": ["--focus-final"],
        "kind": "detection",
        "epochs": 24,
        "env": {"DETECTION_RANDOM_SEED": "42", "SEIZURE_DURATION_SECONDS": "300"},
    },
    {
        "name": "seed_robustness",
        "description": "Repeat the pre-specified finalist configurations across seeds.",
        "args": ["--robust-final"],
        "kind": "detection",
        "epochs": 36,
        "env": {},
    },
    {
        "name": "label_sensitivity",
        "description": "Quantify sensitivity to the onset-duration labeling assumption.",
        # These names may have been run by the seed phase, but the label policy
        # is a different scientific condition and must be rerun deliberately.
        "args": [
            "--only",
            ",".join(
                [
                    "robust_label120s_booster_tuned_patch8_base_eda_boost_ppg_temp",
                    "robust_label300s_booster_tuned_patch8_base_eda_boost_ppg_temp",
                    "robust_label600s_booster_tuned_patch8_base_eda_boost_ppg_temp",
                    "robust_label120s_patch8_base_temp_boost_ppg_eda",
                    "robust_label300s_patch8_base_temp_boost_ppg_eda",
                    "robust_label600s_patch8_base_temp_boost_ppg_eda",
                ]
            ),
        ],
        "kind": "detection",
        "epochs": 36,
        "env": {
            "SEIZURE_DURATION_SECONDS": "120",
            "PHASE1_CACHE": "0",
        },
    },
    {
        "name": "training_stability",
        "description": "Longer finalist training runs with a fixed evaluation protocol.",
        # Do not resume by experiment name: the 72-epoch condition must not be
        # silently satisfied by an earlier 24-epoch archive.
        "args": ["--focus-final"],
        "kind": "detection",
        "epochs": 72,
        "env": {
            "DETECTION_EARLY_STOPPING_PATIENCE": "14",
            "DETECTION_RANDOM_SEED": "123",
        },
    },
    {
        "name": "quality_audit",
        "description": "Audit whether completed evidence is scientifically reportable; never promote weak results.",
        "kind": "quality_audit",
        "env": {},
    },
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise SystemExit(f"Cannot parse existing manifest: {MANIFEST}")
    return {
        "campaign_version": "1",
        "created_at": now(),
        "updated_at": now(),
        "status": "created",
        "holdout_policy": (
            "The test split is never used for phase selection, threshold tuning, "
            "or promotion. Only validation may select configurations."
        ),
        "phases": {
            phase["name"]: {"status": "pending", "attempts": 0}
            for phase in PHASES
        },
    }


def save_manifest(manifest: dict[str, Any]) -> None:
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = now()
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_heartbeat(manifest: dict[str, Any], phase: str, message: str) -> None:
    payload = {
        "timestamp": now(),
        "pid": os.getpid(),
        "status": manifest.get("status"),
        "phase": phase,
        "message": message,
        "manifest": str(MANIFEST),
    }
    HEARTBEAT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def phase_command(phase: dict[str, Any], args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(SCRIPT), *phase["args"]]
    command.extend(["--epochs", str(phase["epochs"])])
    command.extend(["--max-sessions", str(args.max_sessions)])
    command.extend(["--max-train-windows", str(args.max_train_windows)])
    command.extend(["--batch-size", str(args.batch_size)])
    command.extend(["--patience", str(args.patience)])
    return command


def run_phase(
    manifest: dict[str, Any], phase: dict[str, Any], args: argparse.Namespace
) -> bool:
    name = phase["name"]
    record = manifest["phases"][name]
    record["attempts"] = int(record.get("attempts", 0)) + 1
    record["started_at"] = now()
    record["status"] = "running"
    manifest["status"] = "running"
    save_manifest(manifest)
    write_heartbeat(manifest, name, "phase started")

    env = os.environ.copy()
    env.update({key: str(value) for key, value in phase["env"].items()})
    env.update(
        {
            "PYTHONUTF8": "1",
            "TSMIXER_STOP_ON_FAILURE": "0",
            "RESEARCH_CAMPAIGN_MANIFEST": str(MANIFEST),
        }
    )
    log_path = CAMPAIGN_ROOT / f"{name}.log"
    if phase.get("kind") == "public_audit":
        public_root = os.environ.get("PUBLIC_BENCHMARK_ROOT", str(ROOT / "public_benchmark_data"))
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "public_data_audit.py"),
            "--root",
            public_root,
            "--output",
            str(ROOT / "public_benchmark_outputs"),
        ]
    elif phase.get("kind") == "public_evaluation":
        public_root = os.environ.get("PUBLIC_BENCHMARK_ROOT", str(ROOT / "public_benchmark_data"))
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "public_non_eeg_evaluator.py"),
            "--root", public_root,
            "--output", str(ROOT / "public_benchmark_outputs"),
            "--train-metrics",
            "--max-windows", str(args.max_public_windows),
        ]
    elif phase.get("kind") == "quality_audit":
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "campaign_quality_audit.py"),
            "--manifest", str(MANIFEST),
            "--output", str(CAMPAIGN_ROOT / "quality_audit.json"),
        ]
    elif phase.get("kind") == "modern_methods":
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "modern_method_benchmark.py"),
            "--output", str(CAMPAIGN_ROOT / "modern_methods"),
        ]
    else:
        command = phase_command(phase, args)
    record["command"] = command
    record["environment"] = phase["env"]
    record["log"] = str(log_path)
    save_manifest(manifest)

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] START {name}\n")
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            for line in process.stdout:
                output_queue.put(line)
            output_queue.put(None)

        threading.Thread(target=read_output, daemon=True).start()
        last_heartbeat = 0.0
        stream_closed = False
        while process.poll() is None or not stream_closed:
            try:
                line = output_queue.get(timeout=1.0)
            except queue.Empty:
                line = ""
            if line is None:
                stream_closed = True
            elif line:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            current_time = time.monotonic()
            if current_time - last_heartbeat >= args.heartbeat_minutes * 60:
                write_heartbeat(manifest, name, "phase running")
                last_heartbeat = current_time
        return_code = process.returncode
        log.write(f"[{now()}] END {name} exit_code={return_code}\n")

    record["finished_at"] = now()
    record["exit_code"] = return_code
    record["status"] = "completed" if return_code == 0 else "failed"
    save_manifest(manifest)
    write_heartbeat(manifest, name, f"phase finished exit_code={return_code}")
    return return_code == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List scheduled phases")
    parser.add_argument("--phase", action="append", help="Run only this phase; repeatable")
    parser.add_argument("--reset", action="store_true", help="Create a new manifest")
    parser.add_argument("--max-sessions", type=int, default=146)
    parser.add_argument("--max-train-windows", type=int, default=600000)
    parser.add_argument("--max-public-windows", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument(
        "--heartbeat-minutes", type=float, default=5.0,
        help="Heartbeat interval while a phase is running",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for phase in PHASES:
            print(f"{phase['name']}: {phase['description']}")
        return 0

    if args.reset and MANIFEST.exists():
        backup = MANIFEST.with_name(f"campaign_manifest_{int(time.time())}.json")
        MANIFEST.replace(backup)
        print(f"Previous manifest moved to {backup}")

    manifest = load_manifest()
    selected_names = set(args.phase or [phase["name"] for phase in PHASES])
    unknown = selected_names - {phase["name"] for phase in PHASES}
    if unknown:
        print("Unknown phase(s): " + ", ".join(sorted(unknown)))
        return 2

    for phase in PHASES:
        if phase["name"] not in selected_names:
            continue
        # Zero-skip policy: a completed record is historical evidence only.
        # Every selected phase executes again so no stale artifact can satisfy
        # the current campaign.
        if not run_phase(manifest, phase, args):
            manifest["status"] = "paused_after_failure"
            save_manifest(manifest)
            print(f"Phase failed; rerun this command to resume: {phase['name']}")
            return 1

    selected_records = [manifest["phases"][name] for name in selected_names]
    all_complete = all(record.get("status") == "completed" for record in selected_records)
    manifest["status"] = "completed" if all_complete else "incomplete"
    manifest["finished_at"] = now()
    save_manifest(manifest)
    write_heartbeat(manifest, "none", "campaign completed")
    print(f"Campaign manifest: {MANIFEST}")
    return 0 if all_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
