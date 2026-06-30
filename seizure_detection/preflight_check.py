"""Preflight checks for the clean marathon experiment run."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "seizure_detection" / "outputs"
REPORT_DIR = OUTPUT_DIR / "marathon"
REPORT_PATH = REPORT_DIR / "preflight_report.json"

REQUIRED_IMPORTS = [
    "tensorflow",
    "keras",
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "matplotlib",
]

REQUIRED_DATA_FILES = ["ACC.csv", "BVP.csv", "EDA.csv", "TEMP.csv", "HR.csv"]
GENERATED_PATTERNS = [
    "*.keras",
    "*.tflite",
    "results.json",
    "edge_feasibility_report.json",
    "EDGE_FEASIBILITY_SUMMARY.md",
    "TSMIXER_EXPERIMENT_SUMMARY.md",
    "BASELINE_COMPARISON_SUMMARY.md",
]


def check(condition: bool, name: str, detail: str, failures: list[dict]) -> None:
    status = "ok" if condition else "fail"
    print(f"[{status.upper()}] {name}: {detail}")
    if not condition:
        failures.append({"name": name, "detail": detail})


def command_version(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            shell=executable.lower().endswith((".cmd", ".bat")),
        )
    except Exception as exc:
        return f"ERROR: {exc}"
    text = (completed.stdout or completed.stderr).strip().splitlines()
    return text[0] if text else f"exit={completed.returncode}"


def find_generated_artifacts() -> list[str]:
    found: list[str] = []
    roots = [OUTPUT_DIR, ROOT / "showcase_outputs", ROOT / "prediction_outputs_local"]
    for base in roots:
        if not base.exists():
            continue
        for pattern in GENERATED_PATTERNS:
            for path in base.rglob(pattern):
                if OUTPUT_DIR / "marathon" in path.parents:
                    continue
                if ROOT / "showcase_outputs" / "logs" in path.parents:
                    continue
                found.append(str(path.relative_to(ROOT)))
    return sorted(found)


def data_inventory() -> dict:
    patient_dirs = sorted(ROOT.glob("Mayo_*"))
    sessions = 0
    complete_sessions = 0
    missing_examples = []
    patient_label_files = []
    missing_patient_labels = []
    for patient_dir in patient_dirs:
        label_file = find_patient_label_file(patient_dir)
        if label_file:
            patient_label_files.append(str(label_file.relative_to(ROOT)))
        else:
            missing_patient_labels.append(str(patient_dir.relative_to(ROOT)))
        chronic_dir = patient_dir / "Empatica_Chronic"
        if not chronic_dir.exists():
            continue
        for session_dir in chronic_dir.iterdir():
            if not session_dir.is_dir():
                continue
            sessions += 1
            missing = [
                name for name in REQUIRED_DATA_FILES if not (session_dir / name).exists()
            ]
            if missing:
                if len(missing_examples) < 10:
                    missing_examples.append(
                        {
                            "session": str(session_dir.relative_to(ROOT)),
                            "missing": missing,
                        }
                    )
            else:
                complete_sessions += 1
    additional_label_files = sorted((ROOT / "SeizureTimesOnly").glob("*.txt"))
    return {
        "patients": len(patient_dirs),
        "sessions": sessions,
        "complete_sessions": complete_sessions,
        "patient_label_files": len(patient_label_files),
        "additional_label_files": len(additional_label_files),
        "missing_patient_labels": missing_patient_labels,
        "label_examples": patient_label_files[:10],
        "missing_examples": missing_examples,
    }


def find_patient_label_file(patient_dir: Path) -> Path | None:
    for folder in (patient_dir, patient_dir / "Empatica_Chronic"):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.txt")):
            if path.name.lower() not in {"info.txt", "readme.txt"}:
                return path
    return None


def main() -> int:
    failures: list[dict] = []
    report: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version,
        "checks": {},
    }

    print("=" * 70)
    print("MARATHON PREFLIGHT CHECK")
    print("=" * 70)

    check(sys.version_info[:2] in {(3, 11), (3, 12)}, "python_version", sys.version, failures)

    imports = {}
    for module_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            imports[module_name] = getattr(module, "__version__", "imported")
            check(True, f"import_{module_name}", imports[module_name], failures)
        except Exception as exc:
            imports[module_name] = f"ERROR: {exc}"
            check(False, f"import_{module_name}", imports[module_name], failures)
    report["checks"]["imports"] = imports

    inventory = data_inventory()
    report["checks"]["data_inventory"] = inventory
    check(inventory["patients"] >= 5, "patient_dirs", str(inventory["patients"]), failures)
    check(inventory["sessions"] > 0, "session_dirs", str(inventory["sessions"]), failures)
    check(
        inventory["complete_sessions"] > 0,
        "complete_sensor_sessions",
        str(inventory["complete_sessions"]),
        failures,
    )
    check(
        inventory["patient_label_files"] == inventory["patients"],
        "patient_label_files",
        (
            f"{inventory['patient_label_files']}/{inventory['patients']} "
            f"patient labels, missing={inventory['missing_patient_labels']}"
        ),
        failures,
    )
    check(
        not inventory["missing_examples"],
        "required_sensor_files",
        "all scanned sessions contain ACC/BVP/EDA/TEMP/HR"
        if not inventory["missing_examples"]
        else json.dumps(inventory["missing_examples"][:3]),
        failures,
    )

    strict_env = {
        "EDGE_BENCHMARK": os.environ.get("EDGE_BENCHMARK"),
        "EDGE_FAIL_ON_BENCHMARK_FAILURE": os.environ.get("EDGE_FAIL_ON_BENCHMARK_FAILURE"),
        "EDGE_REQUIRE_INT8": os.environ.get("EDGE_REQUIRE_INT8"),
        "TSMIXER_STOP_ON_FAILURE": os.environ.get("TSMIXER_STOP_ON_FAILURE"),
    }
    report["checks"]["strict_env"] = strict_env
    check(strict_env["EDGE_BENCHMARK"] == "1", "edge_benchmark_enabled", str(strict_env), failures)
    check(
        strict_env["EDGE_FAIL_ON_BENCHMARK_FAILURE"] == "0",
        "edge_benchmark_nonfatal",
        str(strict_env),
        failures,
    )
    check(
        strict_env["EDGE_REQUIRE_INT8"] == "0",
        "edge_int8_report_only",
        str(strict_env),
        failures,
    )
    check(
        strict_env["TSMIXER_STOP_ON_FAILURE"] == "1",
        "experiment_stop_on_failure",
        str(strict_env),
        failures,
    )

    resume_mode = os.environ.get("MARATHON_RESUME", "0") == "1"
    generated = find_generated_artifacts()
    report["checks"]["generated_artifacts"] = generated
    if resume_mode:
        print(
            "[INFO] clean_generated_artifacts: resume mode enabled; "
            f"preserving {len(generated)} existing artifact(s)"
        )
    else:
        check(
            not generated,
            "clean_generated_artifacts",
            "none found" if not generated else json.dumps(generated[:10]),
            failures,
        )

    cli_versions = {
        "node": command_version(["node", "--version"]),
        "npm": command_version(["npm", "--version"]),
        "platformio": command_version(["platformio", "--version"]),
        "edge_impulse": command_version(["edge-impulse-daemon", "--version"]),
    }
    report["checks"]["optional_cli_versions"] = cli_versions
    for name, value in cli_versions.items():
        print(f"[INFO] optional_{name}: {value or 'not found'}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report["status"] = "failed" if failures else "passed"
    report["failures"] = failures
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Preflight report: {REPORT_PATH}")

    if failures:
        print("Preflight failed. Fix the listed issue before starting the marathon run.")
        return 1
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
