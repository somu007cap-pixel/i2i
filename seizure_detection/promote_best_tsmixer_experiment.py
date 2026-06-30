"""Promote the best archived TSMixer experiment to active output artifacts."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "seizure_detection" / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "tsmixer_experiments"
MARATHON_DIR = OUTPUT_DIR / "marathon"
PROMOTION_REPORT = OUTPUT_DIR / "promoted_experiment.json"
EXPECTED_PIPELINE_VERSION = "edge_safe_patched_dual_stream_tsmixer_v15_dsp_session_norm"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def event_counts(metrics: dict[str, Any]) -> tuple[int, int]:
    event = metrics.get("event_level") or {}
    return int(event.get("detected_events", 0)), int(event.get("event_count", 0))


def false_alarm_delta(standard: dict[str, Any], pro: dict[str, Any]) -> float:
    return float(standard.get("false_alarms_per_hour", 0.0)) - float(
        pro.get("false_alarms_per_hour", 0.0)
    )


def product_score(results: dict[str, Any], edge: dict[str, Any] | None) -> tuple:
    standard = results.get("standard_mode", {})
    pro = results.get("pro_mode", {})
    std_events, _ = event_counts(standard)
    pro_events, _ = event_counts(pro)
    event_gain = pro_events - std_events
    recall_gain = float(pro.get("recall", 0.0)) - float(standard.get("recall", 0.0))
    auc_gain = float(pro.get("auc", 0.0)) - float(standard.get("auc", 0.0))
    fa_delta = false_alarm_delta(standard, pro)
    params = 10**9
    if edge:
        params = int(
            edge.get("models", {})
            .get("pro", {})
            .get("keras", {})
            .get("parameters", params)
        )
    return (
        1 if event_gain > 0 else 0,
        1 if recall_gain > 0 else 0,
        event_gain,
        recall_gain,
        fa_delta,
        pro_events,
        float(pro.get("recall", 0.0)),
        float(pro.get("auc", 0.0)),
        auc_gain,
        -params,
    )


def is_promotable(folder: Path, results: dict[str, Any]) -> bool:
    metadata = load_json(folder / "experiment_metadata.json") or {}
    edge = load_json(folder / "edge_feasibility_report.json")
    pipeline_version = metadata.get("pipeline_version") or results.get("pipeline_version")
    return (
        pipeline_version == EXPECTED_PIPELINE_VERSION
        and metadata.get("status") in {None, "detection_completed", "completed"}
        and metadata.get("edge_status") == "completed"
        and edge is not None
    )


def latest_marathon_window() -> tuple[datetime | None, datetime | None]:
    status_files = sorted(
        MARATHON_DIR.glob("marathon_*_status.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in status_files:
        start = None
        end = None
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("START TSMIXER "):
                start = datetime.fromisoformat(line.split("START TSMIXER ", 1)[1])
            elif line.startswith("END TSMIXER "):
                end = datetime.fromisoformat(line.rsplit(" ", 1)[-1])
        if start:
            return start, end
    return None, None


def folder_timestamp(folder: Path) -> datetime | None:
    parts = folder.name.split("_", 2)
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime("_".join(parts[:2]), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def candidate_folders() -> list[Path]:
    folders = [path for path in EXPERIMENT_DIR.iterdir() if path.is_dir()]
    start, end = latest_marathon_window()
    if start:
        selected = []
        for folder in folders:
            stamp = folder_timestamp(folder)
            if stamp and stamp >= start and (end is None or stamp <= end):
                selected.append(folder)
        if selected:
            return selected
    return folders


def copy_if_exists(source: Path, target: Path, required: bool = True) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    elif required:
        raise FileNotFoundError(f"Required promotion artifact missing: {source}")


def promote(folder: Path) -> dict[str, Any]:
    results = load_json(folder / "results.json")
    if not results:
        raise FileNotFoundError(f"Missing results.json in {folder}")
    edge = load_json(folder / "edge_feasibility_report.json") or {}
    standard_variant = results.get("standard_mode", {}).get("variant_name")
    pro_variant = results.get("pro_mode", {}).get("variant_name")

    copy_if_exists(folder / "results.json", OUTPUT_DIR / "results.json")
    copy_if_exists(
        folder / "edge_feasibility_report.json",
        OUTPUT_DIR / "edge_feasibility" / "edge_feasibility_report.json",
        required=False,
    )
    copy_if_exists(
        folder / "EDGE_FEASIBILITY_SUMMARY.md",
        OUTPUT_DIR / "edge_feasibility" / "EDGE_FEASIBILITY_SUMMARY.md",
        required=False,
    )
    copy_if_exists(
        folder / "seizure_model_standard.tflite",
        OUTPUT_DIR / "seizure_model_standard.tflite",
    )
    copy_if_exists(folder / "seizure_model.tflite", OUTPUT_DIR / "seizure_model.tflite")
    if standard_variant:
        copy_if_exists(
            folder / f"{standard_variant}.keras",
            OUTPUT_DIR / f"{standard_variant}.keras",
        )
    if pro_variant:
        copy_if_exists(folder / f"{pro_variant}.keras", OUTPUT_DIR / f"{pro_variant}.keras")

    standard = results.get("standard_mode", {})
    pro = results.get("pro_mode", {})
    std_events, total_events = event_counts(standard)
    pro_events, _ = event_counts(pro)
    report = {
        "timestamp": datetime.now().isoformat(),
        "promoted_folder": str(folder),
        "standard_variant": standard_variant,
        "pro_variant": pro_variant,
        "selection_policy": (
            "Prefer Pro event gain, Pro recall gain, fewer false alarms, Pro event "
            "count, Pro recall, and Pro AUC. Test data is used only after the "
            "experiment sweep for final artifact promotion."
        ),
        "standard": {
            "auc": standard.get("auc"),
            "recall": standard.get("recall"),
            "events": f"{std_events}/{total_events}",
            "false_alarms_per_hour": standard.get("false_alarms_per_hour"),
        },
        "pro": {
            "auc": pro.get("auc"),
            "recall": pro.get("recall"),
            "events": f"{pro_events}/{total_events}",
            "false_alarms_per_hour": pro.get("false_alarms_per_hour"),
        },
        "score": list(product_score(results, edge)),
    }
    PROMOTION_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    candidates = []
    for folder in candidate_folders():
        results = load_json(folder / "results.json")
        if not results:
            continue
        if not is_promotable(folder, results):
            continue
        edge = load_json(folder / "edge_feasibility_report.json")
        candidates.append((product_score(results, edge), folder))
    if not candidates:
        raise RuntimeError("No completed TSMixer experiments found for promotion.")
    candidates.sort(reverse=True, key=lambda item: item[0])
    report = promote(candidates[0][1])
    print("Promoted TSMixer experiment:")
    print(f"  {report['promoted_folder']}")
    print(
        f"  Standard {report['standard']['events']}, recall={report['standard']['recall']:.4f}, "
        f"AUC={report['standard']['auc']:.4f}, FA/hr={report['standard']['false_alarms_per_hour']:.2f}"
    )
    print(
        f"  Pro      {report['pro']['events']}, recall={report['pro']['recall']:.4f}, "
        f"AUC={report['pro']['auc']:.4f}, FA/hr={report['pro']['false_alarms_per_hour']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
