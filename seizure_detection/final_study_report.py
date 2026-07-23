"""Build a compact final-study report from detection experiment artifacts."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "seizure_detection" / "outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "tsmixer_experiments"
BASELINE_REPORT = OUTPUT_DIR / "baselines" / "baseline_comparison_report.json"
PROMOTED_REPORT = OUTPUT_DIR / "promoted_experiment.json"
REPORT_PATH = OUTPUT_DIR / "FINAL_STUDY_REPORT.md"
ROBUSTNESS_JSON = OUTPUT_DIR / "final_study_report.json"
LABEL_AUDIT = OUTPUT_DIR / "label_metadata_audit.json"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def event_text(metrics: dict[str, Any]) -> str:
    event = metrics.get("event_level") or {}
    return f"{event.get('detected_events', 0)}/{event.get('event_count', 0)}"


def alert_text(metrics: dict[str, Any]) -> str:
    event = metrics.get("event_alert_level") or {}
    if not event:
        return "n/a"
    return f"{event.get('detected_events', 0)}/{event.get('event_count', 0)}"


def alert_false_alarm_text(metrics: dict[str, Any]) -> str:
    event = metrics.get("event_alert_level") or {}
    if not event:
        return "n/a"
    return fmt(event.get("false_alerts_per_hour"), 2)


def event_sensitivity_ci(metrics: dict[str, Any]) -> str:
    event = metrics.get("event_level") or {}
    low = event.get("event_sensitivity_ci95_low")
    high = event.get("event_sensitivity_ci95_high")
    if low is None or high is None:
        return "n/a"
    return f"{fmt(event.get('event_sensitivity'))} [{fmt(low)}, {fmt(high)}]"


def row_from_results(folder: Path) -> dict[str, Any] | None:
    results = load_json(folder / "results.json")
    metadata = load_json(folder / "experiment_metadata.json") or {}
    if not results:
        return None
    standard = results.get("standard_mode", {})
    pro = results.get("pro_mode", {})
    return {
        "folder": folder.name,
        "name": metadata.get("name", folder.name),
        "description": metadata.get("description", ""),
        "env": metadata.get("env", {}),
        "standard": standard,
        "pro": pro,
        "split_strategy": results.get("split_strategy"),
        "claim_scope": results.get("claim_scope"),
        "label_policy": results.get("label_policy", {}),
        "matched_false_alarm_operating_point": results.get(
            "matched_false_alarm_operating_point", {}
        ),
    }


def collect_experiments() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not EXPERIMENT_DIR.exists():
        return rows
    for folder in sorted(EXPERIMENT_DIR.iterdir()):
        if not folder.is_dir():
            continue
        row = row_from_results(folder)
        if row is not None:
            rows.append(row)
    return rows


def metric(row: dict[str, Any], tier: str, key: str) -> float:
    try:
        return float(row[tier].get(key, float("nan")))
    except Exception:
        return float("nan")


def detected_events(row: dict[str, Any], tier: str) -> int:
    return int((row[tier].get("event_level") or {}).get("detected_events", 0))


def event_sensitivity(row: dict[str, Any], tier: str) -> float:
    try:
        return float((row[tier].get("event_level") or {}).get("event_sensitivity", float("nan")))
    except Exception:
        return float("nan")


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(tier: str, key: str) -> list[float]:
        result = [metric(row, tier, key) for row in rows]
        return [value for value in result if not math.isnan(value)]

    def mean_std(items: list[float]) -> dict[str, float | None]:
        if not items:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": float(statistics.mean(items)),
            "std": float(statistics.stdev(items)) if len(items) > 1 else 0.0,
            "n": len(items),
        }

    return {
        "count": len(rows),
        "standard_auc": mean_std(values("standard", "auc")),
        "pro_auc": mean_std(values("pro", "auc")),
        "standard_recall": mean_std(values("standard", "recall")),
        "pro_recall": mean_std(values("pro", "recall")),
        "standard_false_alarms_per_hour": mean_std(
            values("standard", "false_alarms_per_hour")
        ),
        "pro_false_alarms_per_hour": mean_std(
            values("pro", "false_alarms_per_hour")
        ),
        "standard_events_mean": mean_std(
            [float(detected_events(row, "standard")) for row in rows]
        ),
        "pro_events_mean": mean_std(
            [float(detected_events(row, "pro")) for row in rows]
        ),
        "standard_event_sensitivity": mean_std(
            [
                value
                for value in (event_sensitivity(row, "standard") for row in rows)
                if not math.isnan(value)
            ]
        ),
        "pro_event_sensitivity": mean_std(
            [
                value
                for value in (event_sensitivity(row, "pro") for row in rows)
                if not math.isnan(value)
            ]
        ),
        "pro_auc_gain": mean_std(
            [
                metric(row, "pro", "auc") - metric(row, "standard", "auc")
                for row in rows
                if not math.isnan(metric(row, "pro", "auc"))
                and not math.isnan(metric(row, "standard", "auc"))
            ]
        ),
        "pro_recall_gain": mean_std(
            [
                metric(row, "pro", "recall") - metric(row, "standard", "recall")
                for row in rows
                if not math.isnan(metric(row, "pro", "recall"))
                and not math.isnan(metric(row, "standard", "recall"))
            ]
        ),
        "pro_event_sensitivity_gain": mean_std(
            [
                event_sensitivity(row, "pro") - event_sensitivity(row, "standard")
                for row in rows
                if not math.isnan(event_sensitivity(row, "pro"))
                and not math.isnan(event_sensitivity(row, "standard"))
            ]
        ),
    }


def robustness_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        name = str(row.get("name", ""))
        if name.startswith("robust_seed"):
            base = name.split("_", 2)[2] if name.count("_") >= 2 else name
            groups[f"seed_variability::{base}"].append(row)
        elif name.startswith("robust_label"):
            base = name.split("_", 2)[2] if name.count("_") >= 2 else name
            groups[f"label_policy::{base}"].append(row)
    return dict(groups)


def best_baseline() -> dict[str, Any] | None:
    report = load_json(BASELINE_REPORT)
    if not report:
        return None
    candidates = []
    for name, item in report.get("baselines", {}).items():
        metrics = item.get("metrics", {})
        event = metrics.get("event_level") or {}
        candidates.append(
            (
                int(event.get("detected_events", 0)),
                float(metrics.get("recall", 0.0)),
                -float(metrics.get("false_alarms_per_hour", 0.0)),
                float(metrics.get("auc", 0.0)),
                name,
                item,
            )
        )
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, _, _, _, name, item = candidates[0]
    return {"name": name, **item}


def baseline_operating_point_rows() -> list[str]:
    report = load_json(BASELINE_REPORT)
    if not report:
        return []
    rows = [
        "| Model | Budget FA/hr | Recall | Events | Alert false alerts/hour |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in sorted(report.get("baselines", {}).items()):
        grid = (item.get("metrics") or {}).get("fixed_validation_alarm_budgets", {})
        for budget in ("1.0", "5.0", "10.0", "25.0", "50.0"):
            metrics = grid.get(budget) or grid.get(str(float(budget)))
            if not metrics:
                continue
            event = metrics.get("event_level") or {}
            alert = metrics.get("event_alert_level") or {}
            rows.append(
                "| {name} | {budget} | {recall} | {events} | {alert_fa} |".format(
                    name=name,
                    budget=budget,
                    recall=fmt(metrics.get("recall")),
                    events=f"{event.get('detected_events', 0)}/{event.get('event_count', 0)}",
                    alert_fa=fmt(alert.get("false_alerts_per_hour"), 2),
                )
            )
    return rows if len(rows) > 2 else []


def failure_notes(row: dict[str, Any]) -> list[str]:
    notes = []
    for tier in ("standard", "pro"):
        event = (row.get(tier, {}).get("event_level") or {})
        missed = [
            item
            for item in event.get("events", [])
            if item.get("detected") is False
        ]
        if missed:
            sessions = sorted({str(item.get("session_id")) for item in missed})
            notes.append(
                f"{tier.title()} missed {len(missed)} event(s), across sessions: "
                + ", ".join(sessions)
            )
    return notes


def write_report() -> None:
    rows = collect_experiments()
    promoted = load_json(PROMOTED_REPORT) or {}
    active = load_json(OUTPUT_DIR / "results.json") or {}
    active_row = {
        "name": "active_promoted_result",
        "standard": active.get("standard_mode", {}),
        "pro": active.get("pro_mode", {}),
    }
    groups = robustness_groups(rows)
    group_summary = {name: summarize_group(items) for name, items in groups.items()}
    baseline = best_baseline()

    payload = {
        "active_timestamp": active.get("timestamp"),
        "claim_scope": active.get("claim_scope"),
        "promoted": promoted,
        "active_standard": active_row["standard"],
        "active_pro": active_row["pro"],
        "robustness_groups": group_summary,
        "best_baseline": baseline,
        "failure_notes": failure_notes(active_row),
    }
    ROBUSTNESS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Final Study Report",
        "",
        "This report summarizes the study as a patient-held-out feasibility analysis, "
        "not a clinically validated detector.",
        "",
        "## Submission Verdict",
        "",
        "- The current results are suitable for a rigorous prototype submission when "
        "framed as weak-label wearable seizure-event detection.",
        "- The strongest defensible claim is not that every Pro run dominates the "
        "Standard tier, but that add-on physiological sensing can improve recall, "
        "ranking, or event detection under some operating points while preserving "
        "edge-feasible model size.",
        "- The remaining limitation is scientific rather than cosmetic: the source "
        "labels provide onset timestamps without exact seizure offsets.",
        "",
        "## Active Promoted TSMixer Result",
        "",
        "| Tier | Variant | AUC | PR-AUC | Recall | Event sensitivity | Alert events | Alert false alerts/hour | Window false alarms/hour |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for tier in ("standard", "pro"):
        metrics = active_row[tier]
        lines.append(
            "| {tier} | `{variant}` | {auc} | {pr_auc} | {recall} | {event_ci} | {alert_events} | {alert_fa} | {fa} |".format(
                tier=tier.title(),
                variant=metrics.get("variant_name", "n/a"),
                auc=fmt(metrics.get("auc")),
                pr_auc=fmt(metrics.get("pr_auc")),
                recall=fmt(metrics.get("recall")),
                event_ci=event_sensitivity_ci(metrics),
                alert_events=alert_text(metrics),
                alert_fa=alert_false_alarm_text(metrics),
                fa=fmt(metrics.get("false_alarms_per_hour"), 2),
            )
        )
    standard_active = active_row["standard"]
    pro_active = active_row["pro"]
    auc_gain = float(pro_active.get("auc", 0.0)) - float(standard_active.get("auc", 0.0))
    recall_gain = float(pro_active.get("recall", 0.0)) - float(
        standard_active.get("recall", 0.0)
    )
    event_gain = (
        float((pro_active.get("event_level") or {}).get("event_sensitivity", 0.0))
        - float((standard_active.get("event_level") or {}).get("event_sensitivity", 0.0))
    )
    alarm_delta = float(pro_active.get("false_alarms_per_hour", 0.0)) - float(
        standard_active.get("false_alarms_per_hour", 0.0)
    )
    lines += [
        "",
        "## Active Result Trade-Off",
        "",
        (
            f"- Pro AUC change: {fmt(auc_gain)}; recall change: {fmt(recall_gain)}; "
            f"event-sensitivity change: {fmt(event_gain)}; false-alarms/hour change: "
            f"{fmt(alarm_delta, 2)}."
        ),
        (
            "- Interpretation: the add-on configuration should be presented as an "
            "event-sensitivity and ranking trade-off, not as uniformly better across "
            "all metrics."
        ),
        "",
        "Raw accuracy is intentionally not used as a primary success metric because "
        "the all-normal baseline is very high under rare seizure windows.",
        "",
        "## Label Metadata",
        "",
    ]
    label_audit = load_json(LABEL_AUDIT) or {}
    label_totals = label_audit.get("totals", {})
    if label_totals:
        lines += [
            f"- Label rows scanned: {label_totals.get('rows', 0)}.",
            f"- Rows with usable onset-offset fields: {label_totals.get('usable_offset_rows', 0)}.",
        ]
        if int(label_totals.get("usable_offset_rows", 0)) == 0:
            lines.append(
                "- Source labels are treated as onset-derived weak labels; exact "
                "seizure offsets are not invented."
            )
        else:
            lines.append(
                "- Usable offset fields were found for some rows and should be used "
                "where available."
            )
    else:
        lines.append("- Label metadata audit has not been generated yet.")
    lines += [
        "",
        "## Event-Level Post-Processing",
        "",
        "- Window scores are converted into alert episodes using a validation-selected "
        "refractory interval, so repeated high-score windows in the same episode "
        "are not counted as separate user-facing alarms.",
        "- Alert-level false alarms/hour is reported separately from raw window false "
        "alarms/hour.",
        "",
        "## Robustness Groups",
        "",
    ]
    if group_summary:
        lines += [
            "| Group | Runs | Pro AUC | Pro recall | Pro event sensitivity | Pro FA/hr | Pro AUC gain | Pro recall gain | Pro event-sensitivity gain |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, summary in sorted(group_summary.items()):
            lines.append(
                "| {name} | {n} | {auc} | {recall} | {event_sens} | {fa} | {auc_gain} | {recall_gain} | {event_gain} |".format(
                    name=name,
                    n=summary["count"],
                    auc=mean_std_text(summary["pro_auc"]),
                    recall=mean_std_text(summary["pro_recall"]),
                    event_sens=mean_std_text(summary["pro_event_sensitivity"]),
                    fa=mean_std_text(summary["pro_false_alarms_per_hour"], 2),
                    auc_gain=mean_std_text(summary["pro_auc_gain"]),
                    recall_gain=mean_std_text(summary["pro_recall_gain"]),
                    event_gain=mean_std_text(summary["pro_event_sensitivity_gain"]),
                )
            )
        lines += [
            "",
            "Note: label-policy groups can change the number and duration of positive "
            "events, so event sensitivity is more comparable than raw detected-event "
            "counts across those groups.",
        ]
    else:
        lines.append(
            "No robustness experiments were found yet. Run the robustness marathon to "
            "populate seed and label-policy sensitivity results."
        )

    lines += ["", "## Strongest Baseline Reference", ""]
    if baseline:
        metrics = baseline.get("metrics", {})
        lines.append(
            "- `{name}`: AUC {auc}, recall {recall}, events {events}, false alarms/hour {fa}.".format(
                name=baseline.get("name"),
                auc=fmt(metrics.get("auc")),
                recall=fmt(metrics.get("recall")),
                events=event_text(metrics),
                fa=fmt(metrics.get("false_alarms_per_hour"), 2),
            )
        )
    else:
        lines.append("- Baseline report not found.")

    operating_rows = baseline_operating_point_rows()
    lines += ["", "## Same Operating-Point Baselines", ""]
    if operating_rows:
        lines += operating_rows
    else:
        lines.append(
            "- Same-budget baseline rows are not available yet; rerun the marathon "
            "after the baseline evaluator update."
        )

    notes = failure_notes(active_row)
    lines += ["", "## Failure Notes", ""]
    if notes:
        lines += [f"- {note}" for note in notes]
    else:
        lines.append("- No event-level failure details were available.")

    lines += [
        "",
        "## Interpretation Guardrails",
        "",
        "- Treat this as a feasibility study under weak labels.",
        "- Compare event sensitivity and false alarms/hour before raw accuracy.",
        "- Report TFLite results as software feasibility, not physical MCU deployment.",
        "- Firmware-facing PSoC Edge E84 artifacts are prepared under "
        "`seizure_detection/firmware_c/`; measured board latency should be added "
        "only after a successful hardware run.",
        "",
        "## Firmware Deployment Package",
        "",
        "- INT8 TFLite model header: `seizure_detection/firmware_c/model_data.h`.",
        "- Static-memory TFLite Micro harness: `seizure_detection/firmware_c/main.cpp`.",
        "- Streaming DSP ring buffer and online normalization helpers: "
        "`seizure_detection/firmware_c/sensor_dsp.c` and `.h`.",
        "- Target profile and PSoC Edge E84 porting plan: "
        "`seizure_detection/firmware_c/edge_target_profile.md`.",
        "",
        "## Figure Package",
        "",
        "- Dissertation figures are generated under `seizure_detection/outputs/dissertation_figures/`.",
        "- Use `README_FIGURES.md` in that folder as the placement guide for report chapters.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)


def mean_std_text(summary: dict[str, Any], digits: int = 4) -> str:
    if not summary or summary.get("mean") is None:
        return "n/a"
    return f"{fmt(summary.get('mean'), digits)}+-{fmt(summary.get('std'), digits)}"


if __name__ == "__main__":
    write_report()
