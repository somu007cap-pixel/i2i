"""Summarize archived TSMixer experiment results."""

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "seizure_detection" / "outputs" / "tsmixer_experiments"
SUMMARY_PATH = EXPERIMENT_DIR / "TSMIXER_EXPERIMENT_SUMMARY.md"
MARATHON_DIR = ROOT / "seizure_detection" / "outputs" / "marathon"
EXPECTED_PIPELINE_VERSION = "edge_safe_patched_dual_stream_tsmixer_v15_dsp_session_norm"


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def selected_model_edge(edge):
    if not edge:
        return {}
    models = edge.get("models", {})
    pro = models.get("pro", {})
    keras = pro.get("keras", {})
    tflite = pro.get("tflite", {}).get("calibrated_existing", {})
    return {
        "params": keras.get("parameters"),
        "tflite_kb": tflite.get("size_kb"),
        "tensor_kb": tflite.get("estimated_tensor_arena_proxy_kb"),
    }


def row_for_experiment(folder: Path):
    results = load_json(folder / "results.json")
    metadata = load_json(folder / "experiment_metadata.json") or {}
    edge = load_json(folder / "edge_feasibility_report.json")
    if not results:
        return None
    pipeline_version = metadata.get("pipeline_version") or results.get("pipeline_version")
    if pipeline_version != EXPECTED_PIPELINE_VERSION:
        return None
    if metadata.get("edge_status") != "completed" or edge is None:
        return None
    standard = results.get("standard_mode", {})
    pro = results.get("pro_mode", {})
    pro_event = pro.get("event_level", {}) or {}
    std_event = standard.get("event_level", {}) or {}
    profile = selected_model_edge(edge)
    return {
        "folder": folder.name,
        "name": metadata.get("name", folder.name),
        "description": metadata.get("description", ""),
        "standard_variant": standard.get("variant_name"),
        "pro_variant": pro.get("variant_name"),
        "standard_auc": standard.get("auc", 0.0),
        "pro_auc": pro.get("auc", 0.0),
        "pro_pr_auc": pro.get("pr_auc", 0.0),
        "standard_recall": standard.get("recall", 0.0),
        "pro_recall": pro.get("recall", 0.0),
        "standard_events": f"{std_event.get('detected_events', 0)}/{std_event.get('event_count', 0)}",
        "pro_events": f"{pro_event.get('detected_events', 0)}/{pro_event.get('event_count', 0)}",
        "standard_false_alarms": standard.get("false_alarms_per_hour", 0.0),
        "pro_false_alarms": pro.get("false_alarms_per_hour", 0.0),
        "params": profile.get("params"),
        "tflite_kb": profile.get("tflite_kb"),
        "tensor_kb": profile.get("tensor_kb"),
    }


def score(row):
    return (
        float(row["pro_auc"] or 0.0),
        float(row["pro_pr_auc"] or 0.0),
        float(row["pro_recall"] or 0.0),
        -float(row["pro_false_alarms"] or 0.0),
    )


def folder_run_date(folder_name: str) -> str:
    parts = folder_name.split("_", 2)
    if not parts:
        return ""
    return parts[0]


def folder_timestamp(folder_name: str):
    parts = folder_name.split("_", 2)
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime("_".join(parts[:2]), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def latest_marathon_tsmixer_window():
    status_files = sorted(
        MARATHON_DIR.glob("marathon_*_status.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for status_file in status_files:
        start = None
        end = None
        for line in status_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("START TSMIXER "):
                try:
                    start = datetime.fromisoformat(line.split("START TSMIXER ", 1)[1])
                except ValueError:
                    start = None
            elif line.startswith("END TSMIXER "):
                try:
                    end = datetime.fromisoformat(line.rsplit(" ", 1)[-1])
                except ValueError:
                    end = None
        if start:
            return start, end
    return None, None


def product_score(row):
    event_gain = parse_event_count(row["pro_events"]) - parse_event_count(
        row["standard_events"]
    )
    recall_gain = float(row["pro_recall"] or 0.0) - float(row["standard_recall"] or 0.0)
    auc_gain = float(row["pro_auc"] or 0.0) - float(row["standard_auc"] or 0.0)
    false_alarm_delta = float(row["pro_false_alarms"] or 0.0)
    return (
        event_gain,
        recall_gain,
        auc_gain,
        -false_alarm_delta,
        float(row["pro_auc"] or 0.0),
    )


def parse_event_count(value: str) -> int:
    try:
        return int(str(value).split("/", 1)[0])
    except Exception:
        return 0


def latest_rows(rows):
    start, end = latest_marathon_tsmixer_window()
    if start:
        selected = []
        for row in rows:
            timestamp = folder_timestamp(row["folder"])
            if timestamp is None:
                continue
            if timestamp >= start and (end is None or timestamp <= end):
                selected.append(row)
        if selected:
            return selected
    dates = [folder_run_date(row["folder"]) for row in rows]
    dates = [date for date in dates if date]
    if not dates:
        return rows
    latest_date = max(dates)
    return [row for row in rows if folder_run_date(row["folder"]) == latest_date]


def append_rows(lines, rows):
    lines += [
        "| Experiment | Standard AUC | Standard recall | Standard events | Standard FA/hr | Pro AUC | Pro recall | Pro events | Pro FA/hr | AUC gain | Recall gain | Event gain | Params | TFLite KB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        params = row["params"]
        tflite = row["tflite_kb"]
        auc_gain = float(row["pro_auc"] or 0.0) - float(row["standard_auc"] or 0.0)
        recall_gain = float(row["pro_recall"] or 0.0) - float(
            row["standard_recall"] or 0.0
        )
        event_gain = parse_event_count(row["pro_events"]) - parse_event_count(
            row["standard_events"]
        )
        lines.append(
            "| {name} | {std_auc:.4f} | {std_recall:.4f} | {std_events} | {std_fa:.2f} | "
            "{pro_auc:.4f} | {pro_recall:.4f} | {pro_events} | {pro_fa:.2f} | "
            "{auc_gain:.4f} | {recall_gain:.4f} | {event_gain:+d} | {params} | {tflite} |".format(
                name=row["name"],
                std_auc=float(row["standard_auc"] or 0.0),
                std_recall=float(row["standard_recall"] or 0.0),
                std_events=row["standard_events"],
                std_fa=float(row.get("standard_false_alarms") or 0.0),
                pro_auc=float(row["pro_auc"] or 0.0),
                pro_recall=float(row["pro_recall"] or 0.0),
                pro_events=row["pro_events"],
                pro_fa=float(row["pro_false_alarms"] or 0.0),
                auc_gain=auc_gain,
                recall_gain=recall_gain,
                event_gain=event_gain,
                params=f"{int(params):,}" if params is not None else "-",
                tflite=f"{float(tflite):.1f}" if tflite is not None else "-",
            )
        )


def main():
    rows = []
    if EXPERIMENT_DIR.exists():
        for folder in sorted(EXPERIMENT_DIR.iterdir()):
            if folder.is_dir():
                row = row_for_experiment(folder)
                if row:
                    rows.append(row)
    latest = latest_rows(rows)
    latest.sort(key=product_score, reverse=True)
    rows.sort(key=score, reverse=True)
    lines = [
        "# TSMixer Experiment Summary",
        "",
        "## Latest Marathon Run",
        "",
        "Ranking in this section favors the product claim: event gain, recall gain, AUC gain, and lower Pro false alarms.",
        "",
    ]
    append_rows(lines, latest)
    lines += [
        "",
        "## Full Archive",
        "",
        "Archive ranking uses Pro AUC first, then PR-AUC, recall, and lower false alarms.",
        "",
    ]
    append_rows(lines, rows)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
