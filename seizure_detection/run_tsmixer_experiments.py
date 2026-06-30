"""
Run staged Patched Dual-Stream TSMixer experiments.

Each experiment forces a clean retrain, archives results under
seizure_detection/outputs/tsmixer_experiments, and keeps a timestamped log.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "seizure_detection"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "tsmixer_experiments"
EXPECTED_PIPELINE_VERSION = "edge_safe_patched_dual_stream_tsmixer_v15_dsp_session_norm"
RESUMABLE_STATUSES = {"completed"}


EXPERIMENTS: List[Dict] = [
    {
        "name": "tiny_patch8_base_eda_boost_ppg_temp",
        "description": "TinyML-budget patched TSMixer under 100k parameters; ACC+EDA baseline, BVP+HR+TEMP booster.",
        "env": {
            "DETECTION_PATCH_SIZE": "8",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "2",
            "DETECTION_TSMIXER_HIDDEN_DIM": "32",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "EDA",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR,TEMP",
            "DETECTION_LOSS": "focal",
            "DETECTION_NEGATIVE_RATIO": "5",
        },
    },
    {
        "name": "tiny_patch16_base_temp_boost_ppg_eda",
        "description": "Smallest recommendation-style patched TSMixer; ACC+TEMP baseline, BVP+HR+EDA booster.",
        "env": {
            "DETECTION_PATCH_SIZE": "16",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "2",
            "DETECTION_TSMIXER_HIDDEN_DIM": "32",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "TEMP",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR,EDA",
            "DETECTION_LOSS": "focal",
            "DETECTION_NEGATIVE_RATIO": "5",
        },
    },
    {
        "name": "patch8_base_eda_temp_boost_ppg",
        "description": "Default patched TSMixer: ACC+EDA+TEMP baseline stream, BVP+HR booster stream.",
        "env": {
            "DETECTION_PATCH_SIZE": "8",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "3",
            "DETECTION_TSMIXER_HIDDEN_DIM": "48",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "EDA,TEMP",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR",
            "DETECTION_LOSS": "focal",
            "DETECTION_NEGATIVE_RATIO": "5",
        },
    },
    {
        "name": "patch8_base_eda_boost_ppg_temp",
        "description": "Primary stream emphasizes ACC+EDA; booster stream adds BVP+HR+TEMP.",
        "env": {
            "DETECTION_PATCH_SIZE": "8",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "3",
            "DETECTION_TSMIXER_HIDDEN_DIM": "48",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "EDA",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR,TEMP",
            "DETECTION_LOSS": "focal",
            "DETECTION_NEGATIVE_RATIO": "5",
        },
    },
    {
        "name": "patch8_base_temp_boost_ppg_eda",
        "description": "Recommendation-style split: ACC+TEMP baseline stream, BVP+HR+EDA booster stream.",
        "env": {
            "DETECTION_PATCH_SIZE": "8",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "3",
            "DETECTION_TSMIXER_HIDDEN_DIM": "48",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "TEMP",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR,EDA",
            "DETECTION_LOSS": "focal",
            "DETECTION_NEGATIVE_RATIO": "5",
        },
    },
    {
        "name": "patch4_base_eda_boost_ppg_temp",
        "description": "Smaller patches for higher temporal resolution with ACC+EDA baseline and BVP+HR+TEMP booster.",
        "env": {
            "DETECTION_PATCH_SIZE": "4",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "3",
            "DETECTION_TSMIXER_HIDDEN_DIM": "48",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "EDA",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR,TEMP",
            "DETECTION_LOSS": "focal",
            "DETECTION_NEGATIVE_RATIO": "5",
        },
    },
    {
        "name": "patch4_sensitive_base_eda_boost_ppg_temp",
        "description": "Event-sensitivity run: same ACC+EDA baseline and BVP+HR+TEMP booster with stronger positive sampling.",
        "env": {
            "DETECTION_PATCH_SIZE": "4",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "3",
            "DETECTION_TSMIXER_HIDDEN_DIM": "48",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "EDA",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR,TEMP",
            "DETECTION_LOSS": "focal",
            "DETECTION_FOCAL_GAMMA": "3.0",
            "DETECTION_NEGATIVE_RATIO": "2",
            "DETECTION_BATCH_SIZE": "32",
            "DETECTION_DEFAULT_ALARM_BUDGET_PER_HOUR": "25",
            "DETECTION_ALARM_BUDGETS_PER_HOUR": "1,5,10,25,50,100",
        },
    },
    {
        "name": "tiny_sensitive_patch16_base_temp_boost_ppg_eda",
        "description": "TinyML sensitivity run under 100k parameters with stronger positive sampling.",
        "env": {
            "DETECTION_PATCH_SIZE": "16",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "2",
            "DETECTION_TSMIXER_HIDDEN_DIM": "32",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "TEMP",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR,EDA",
            "DETECTION_LOSS": "focal",
            "DETECTION_FOCAL_GAMMA": "3.0",
            "DETECTION_NEGATIVE_RATIO": "2",
            "DETECTION_DEFAULT_ALARM_BUDGET_PER_HOUR": "25",
            "DETECTION_ALARM_BUDGETS_PER_HOUR": "1,5,10,25,50,100",
        },
    },
    {
        "name": "sswce_patch8_base_eda_boost_ppg_temp",
        "description": "Sensitivity-specificity weighted loss with ACC+EDA baseline and BVP+HR+TEMP add-on.",
        "env": {
            "DETECTION_PATCH_SIZE": "8",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "3",
            "DETECTION_TSMIXER_HIDDEN_DIM": "48",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "EDA",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR,TEMP",
            "DETECTION_LOSS": "sswce",
            "DETECTION_SENSITIVITY_WEIGHT": "0.90",
            "DETECTION_SPECIFICITY_WEIGHT": "0.10",
            "DETECTION_NEGATIVE_RATIO": "3",
            "DETECTION_DEFAULT_ALARM_BUDGET_PER_HOUR": "25",
            "DETECTION_ALARM_BUDGETS_PER_HOUR": "1,5,10,25,50,100",
        },
    },
    {
        "name": "sswce_patch8_base_temp_boost_ppg_eda",
        "description": "Sensitivity-specificity weighted loss with ACC+TEMP baseline and BVP+HR+EDA add-on.",
        "env": {
            "DETECTION_PATCH_SIZE": "8",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "3",
            "DETECTION_TSMIXER_HIDDEN_DIM": "48",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "TEMP",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR,EDA",
            "DETECTION_LOSS": "sswce",
            "DETECTION_SENSITIVITY_WEIGHT": "0.90",
            "DETECTION_SPECIFICITY_WEIGHT": "0.10",
            "DETECTION_NEGATIVE_RATIO": "3",
            "DETECTION_DEFAULT_ALARM_BUDGET_PER_HOUR": "25",
            "DETECTION_ALARM_BUDGETS_PER_HOUR": "1,5,10,25,50,100",
        },
    },
    {
        "name": "sswce_tiny_patch16_base_temp_boost_ppg_eda",
        "description": "TinyML-budget sensitivity-specificity weighted run for edge-feasible seizure detection.",
        "env": {
            "DETECTION_PATCH_SIZE": "16",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "2",
            "DETECTION_TSMIXER_HIDDEN_DIM": "32",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "TEMP",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR,EDA",
            "DETECTION_LOSS": "sswce",
            "DETECTION_SENSITIVITY_WEIGHT": "0.90",
            "DETECTION_SPECIFICITY_WEIGHT": "0.10",
            "DETECTION_NEGATIVE_RATIO": "3",
            "DETECTION_DEFAULT_ALARM_BUDGET_PER_HOUR": "25",
            "DETECTION_ALARM_BUDGETS_PER_HOUR": "1,5,10,25,50,100",
        },
    },
    {
        "name": "booster_tuned_patch8_base_eda_boost_ppg_temp",
        "description": "Focused booster run: ACC+EDA baseline, BVP+HR+TEMP add-on, smaller validation-selected Pro fusion weights.",
        "env": {
            "DETECTION_PATCH_SIZE": "8",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "3",
            "DETECTION_TSMIXER_HIDDEN_DIM": "48",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "EDA",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR,TEMP",
            "DETECTION_LOSS": "focal",
            "DETECTION_NEGATIVE_RATIO": "4",
            "DETECTION_PRO_FUSION_WEIGHTS": "0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.35,0.50",
            "DETECTION_ALARM_BUDGETS_PER_HOUR": "1,5,10,15,25,50,100",
        },
    },
    {
        "name": "booster_tuned_patch8_base_temp_boost_ppg_eda",
        "description": "Focused booster run: ACC+TEMP baseline, BVP+HR+EDA add-on, smaller validation-selected Pro fusion weights.",
        "env": {
            "DETECTION_PATCH_SIZE": "8",
            "DETECTION_PATCH_EMBED_DIM": "8",
            "DETECTION_TSMIXER_BLOCKS": "3",
            "DETECTION_TSMIXER_HIDDEN_DIM": "48",
            "DETECTION_BASELINE_SECONDARY_FEATURES": "TEMP",
            "DETECTION_BOOSTER_SECONDARY_FEATURES": "BVP,HR,EDA",
            "DETECTION_LOSS": "focal",
            "DETECTION_NEGATIVE_RATIO": "4",
            "DETECTION_PRO_FUSION_WEIGHTS": "0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.35,0.50",
            "DETECTION_ALARM_BUDGETS_PER_HOUR": "1,5,10,15,25,50,100",
        },
    },
]


def run_command(command: List[str], env: Dict[str, str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
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
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def archive_file(source: Path, target_dir: Path, name: str) -> None:
    if source.exists():
        shutil.copy2(source, target_dir / name)


def archive_outputs(experiment: Dict, target_dir: Path) -> None:
    archive_file(OUTPUT_DIR / "results.json", target_dir, "results.json")
    results = load_json(OUTPUT_DIR / "results.json") or {}
    pipeline_version = results.get("pipeline_version", EXPECTED_PIPELINE_VERSION)
    standard_variant = results.get("standard_mode", {}).get("variant_name")
    pro_variant = results.get("pro_mode", {}).get("variant_name")
    if standard_variant:
        archive_file(
            OUTPUT_DIR / f"{standard_variant}.keras",
            target_dir,
            f"{standard_variant}.keras",
        )
    if pro_variant:
        archive_file(
            OUTPUT_DIR / f"{pro_variant}.keras",
            target_dir,
            f"{pro_variant}.keras",
        )
    archive_file(
        OUTPUT_DIR / "seizure_model_standard.tflite",
        target_dir,
        "seizure_model_standard.tflite",
    )
    archive_file(
        OUTPUT_DIR / "seizure_model.tflite",
        target_dir,
        "seizure_model.tflite",
    )
    archive_file(
        OUTPUT_DIR / "event_level_diagnostics.json",
        target_dir,
        "event_level_diagnostics.json",
    )
    archive_file(
        OUTPUT_DIR / "edge_feasibility" / "edge_feasibility_report.json",
        target_dir,
        "edge_feasibility_report.json",
    )
    archive_file(
        OUTPUT_DIR / "edge_feasibility" / "EDGE_FEASIBILITY_SUMMARY.md",
        target_dir,
        "EDGE_FEASIBILITY_SUMMARY.md",
    )
    metadata_path = target_dir / "experiment_metadata.json"
    existing = load_json(metadata_path) or {}
    metadata = {
        **existing,
        "name": experiment["name"],
        "description": experiment["description"],
        "env": experiment["env"],
        "pipeline_version": pipeline_version,
        "archived_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def latest_experiment_dir(name: str) -> Path | None:
    """Return the newest archived directory for an experiment name, if any."""
    if not EXPERIMENT_DIR.exists():
        return None
    matches = [
        path
        for path in EXPERIMENT_DIR.iterdir()
        if path.is_dir() and path.name.endswith(f"_{name}")
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def experiment_status(name: str) -> str | None:
    """Read the latest archived status for an experiment name."""
    target_dir = latest_experiment_dir(name)
    if target_dir is None:
        return None
    metadata = load_json(target_dir / "experiment_metadata.json") or {}
    results = load_json(target_dir / "results.json") or {}
    archived_version = metadata.get("pipeline_version") or results.get("pipeline_version")
    if archived_version != EXPECTED_PIPELINE_VERSION:
        return "pipeline_version_mismatch"
    status_value = metadata.get("status")
    if status_value == "detection_completed" and metadata.get("edge_status") == "completed":
        return "completed"
    if status_value:
        return str(status_value)
    status = metadata.get("detection_exit_code")
    if status == 0 and metadata.get("edge_status") == "completed":
        return "completed"
    if status == 0:
        return "detection_completed"
    return "detection_failed"


def update_metadata(target_dir: Path, **updates) -> None:
    path = target_dir / "experiment_metadata.json"
    payload = load_json(path) or {}
    payload.setdefault("pipeline_version", EXPECTED_PIPELINE_VERSION)
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_env(experiment: Dict, args) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "TF_ENABLE_ONEDNN_OPTS": "0",
            "FORCE_RERUN": "1",
            "DETECTION_FORCE_RETRAIN_VARIANTS": "1",
            "DETECTION_EDGE_SAFE_MODEL": "1",
            "DETECTION_MAX_SESSIONS": str(args.max_sessions),
            "DETECTION_EPOCHS": str(args.epochs),
            "DETECTION_BATCH_SIZE": str(args.batch_size),
            "DETECTION_MAX_TRAIN_WINDOWS": str(args.max_train_windows),
            "DETECTION_EARLY_STOPPING_PATIENCE": str(args.patience),
            "EDGE_BENCHMARK": os.environ.get("EDGE_BENCHMARK", "1"),
            "EDGE_FAIL_ON_BENCHMARK_FAILURE": os.environ.get(
                "EDGE_FAIL_ON_BENCHMARK_FAILURE", "1"
            ),
            "EDGE_REQUIRE_INT8": os.environ.get("EDGE_REQUIRE_INT8", "1"),
        }
    )
    env.update(experiment["env"])
    return env


def run_experiment(experiment: Dict, args) -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    target_dir = EXPERIMENT_DIR / f"{stamp}_{experiment['name']}"
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / "run.log"
    env = build_env(experiment, args)
    header = [
        f"Experiment: {experiment['name']}",
        f"Description: {experiment['description']}",
        f"Started: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"Max sessions: {args.max_sessions}",
        f"Epochs: {args.epochs}",
        f"Batch size: {env.get('DETECTION_BATCH_SIZE', args.batch_size)}",
        "",
    ]
    log_path.write_text("\n".join(header), encoding="utf-8")
    print("\n" + "=" * 78)
    print(header[0])
    print(header[1])
    print("=" * 78)
    code = run_command(
        [sys.executable, str(SCRIPT_DIR / "phase4_validation.py")],
        env,
        log_path,
    )
    if code != 0:
        print(f"Detection run failed for {experiment['name']} with exit code {code}")
        update_metadata(
            target_dir,
            name=experiment["name"],
            description=experiment["description"],
            env=experiment["env"],
            detection_exit_code=code,
            status="detection_failed",
        )
        return code
    update_metadata(target_dir, detection_exit_code=0, status="detection_completed")
    code = run_command(
        [sys.executable, str(SCRIPT_DIR / "edge_feasibility.py")],
        env,
        log_path,
    )
    if code != 0:
        print(f"Edge profiling failed for {experiment['name']} with exit code {code}")
        update_metadata(
            target_dir,
            edge_exit_code=code,
            edge_status="failed",
            status="edge_failed_detection_archived",
        )
        archive_outputs(experiment, target_dir)
        return code
    else:
        update_metadata(
            target_dir,
            edge_exit_code=0,
            edge_status="completed",
            status="completed",
        )
    archive_outputs(experiment, target_dir)
    run_command(
        [sys.executable, str(SCRIPT_DIR / "summarize_tsmixer_experiments.py")],
        env,
        log_path,
    )
    print(f"Archived experiment outputs: {target_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List experiments and exit.")
    parser.add_argument("--only", default="", help="Comma-separated experiment names to run.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments that already finished successfully and continue from the first unfinished one.",
    )
    parser.add_argument("--max-sessions", type=int, default=146)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-windows", type=int, default=600000)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    if args.list:
        for item in EXPERIMENTS:
            print(f"{item['name']}: {item['description']}")
        return 0

    selected = EXPERIMENTS
    if args.only:
        names = {name.strip() for name in args.only.split(",") if name.strip()}
        selected = [item for item in EXPERIMENTS if item["name"] in names]
        missing = names - {item["name"] for item in selected}
        if missing:
            print("Unknown experiment(s): " + ", ".join(sorted(missing)))
            return 2

    if args.resume:
        resumed: List[Dict] = []
        skipped: List[str] = []
        rerun_reasons: List[str] = []
        for item in selected:
            status = experiment_status(item["name"])
            if status in RESUMABLE_STATUSES:
                skipped.append(item["name"])
                continue
            if status:
                rerun_reasons.append(f"{item['name']} ({status})")
            resumed.append(item)
        selected = resumed
        if skipped:
            print("Resume mode skipped completed experiments: " + ", ".join(skipped))
        if rerun_reasons:
            print("Resume mode will rerun: " + ", ".join(rerun_reasons))

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    for experiment in selected:
        code = run_experiment(experiment, args)
        if code != 0:
            if os.environ.get("TSMIXER_STOP_ON_FAILURE", "1") == "1":
                return code
            print(
                f"Continuing after {experiment['name']} failure because "
                "TSMIXER_STOP_ON_FAILURE is not set."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
