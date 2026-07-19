"""Create a submission-facing report of modern methods evaluated in the study."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "seizure_detection" / "outputs"
REPORT_PATH = OUTPUT_DIR / "SOTA_METHOD_COVERAGE.md"
JSON_PATH = OUTPUT_DIR / "sota_method_coverage.json"


REFERENCES = {
    "ttm": {
        "title": "Tiny Time Mixers, NeurIPS 2024",
        "url": "https://research.ibm.com/publications/tiny-time-mixers-ttms-fast-pre-trained-models-for-enhanced-zerofew-shot-forecasting-of-multivariate-time-series--1",
        "relevance": (
            "Supports compact patch-mixer and channel-independent time-series "
            "modeling under edge constraints."
        ),
    },
    "wearable_armband": {
        "title": "Wearable armband seizure detection with ACC and PPG, 2025",
        "url": "https://www.sciencedirect.com/science/article/pii/S0169260725005048",
        "relevance": (
            "Supports multimodal wearable detection and two-step candidate "
            "screening/reporting with false alarms."
        ),
    },
    "eegformer": {
        "title": "Compact transformer false-alarm reduction for wearable seizure detection",
        "url": "https://thorirmar.com/publication/2024-ieee-tbcas/",
        "relevance": (
            "Supports compact transformer comparison, while noting that EEG "
            "datasets are not directly comparable to Empatica non-EEG weak labels."
        ),
    },
    "wearable_review": {
        "title": "Systematic review of non-invasive wearable seizure detection",
        "url": "https://www.cureepilepsy.org/news/automated-seizure-detection-with-non-invasive-wearable-devices-a-systematic-review-and-meta-analysis/",
        "relevance": (
            "Supports reporting sensitivity and false-alarm burden instead of "
            "raw accuracy."
        ),
    },
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "pending"


def event_text(metrics: dict[str, Any]) -> str:
    event = metrics.get("event_level") or {}
    if not event:
        return "pending"
    return f"{event.get('detected_events', 0)}/{event.get('event_count', 0)}"


def alert_fa(metrics: dict[str, Any]) -> str:
    alert = metrics.get("event_alert_level") or {}
    if not alert:
        return "pending"
    return fmt(alert.get("false_alerts_per_hour"), 2)


def method_rows() -> list[dict[str, Any]]:
    results = load_json(OUTPUT_DIR / "results.json")
    baselines = load_json(OUTPUT_DIR / "baselines" / "baseline_comparison_report.json")
    standard = results.get("standard_mode", {})
    pro = results.get("pro_mode", {})
    rows: list[dict[str, Any]] = [
        {
            "family": "Patched channel-independent TSMixer / TTM-style mixer",
            "implemented_as": "Dual-stream TSMixer Standard/Pro models",
            "result": (
                f"Standard AUC {fmt(standard.get('auc'))}, events {event_text(standard)}; "
                f"Pro AUC {fmt(pro.get('auc'))}, events {event_text(pro)}"
            ),
            "status": "evaluated" if standard and pro else "pending next run",
        },
        {
            "family": "Event-level alert post-processing",
            "implemented_as": "Validation-selected refractory interval over detector scores",
            "result": (
                f"Standard alert FA/hr {alert_fa(standard)}; "
                f"Pro alert FA/hr {alert_fa(pro)}"
            ),
            "status": "evaluated" if pro.get("event_alert_level") else "pending next run",
        },
        {
            "family": "Patient-held-out weak-label robustness",
            "implemented_as": "Repeated seed and label-duration sensitivity studies",
            "result": "Aggregated in FINAL_STUDY_REPORT.md",
            "status": "evaluated" if (OUTPUT_DIR / "final_study_report.json").exists() else "pending",
        },
    ]
    for name, item in sorted((baselines.get("baselines") or {}).items()):
        metrics = item.get("metrics", {})
        rows.append(
            {
                "family": {
                    "lstm": "Recurrent sequence model",
                    "cnn_lstm": "CNN-LSTM temporal feature model",
                    "compact_transformer": "Compact transformer baseline",
                    "autoencoder": "Reconstruction-error anomaly detector",
                }.get(name, name),
                "implemented_as": name,
                "result": (
                    f"AUC {fmt(metrics.get('auc'))}, recall {fmt(metrics.get('recall'))}, "
                    f"events {event_text(metrics)}, alert FA/hr {alert_fa(metrics)}"
                ),
                "status": "evaluated",
            }
        )
    return rows


def run() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "claim": (
            "The study evaluates modern lightweight time-series approaches that are "
            "appropriate for Empatica wearable data and weak onset-only seizure labels."
        ),
        "methods": method_rows(),
        "references": REFERENCES,
        "not_directly_comparable": [
            {
                "method": "EEG-specific seizure transformers trained on CHB-MIT or wearable EEG",
                "reason": (
                    "Those studies use EEG morphology and often more precise event "
                    "boundaries; this project uses Empatica ACC/PPG/EDA/TEMP with "
                    "onset-only labels."
                ),
            },
            {
                "method": "Large time-series foundation models",
                "reason": (
                    "Large pretrained models are not aligned with the TinyML/edge "
                    "memory target and would not solve onset-only label uncertainty."
                ),
            },
        ],
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Modern Method Coverage",
        "",
        payload["claim"],
        "",
        "## Evaluated Method Families",
        "",
        "| Method family | Implementation in this project | Status | Current result |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["methods"]:
        lines.append(
            "| {family} | {impl} | {status} | {result} |".format(
                family=row["family"],
                impl=row["implemented_as"],
                status=row["status"],
                result=row["result"],
            )
        )
    lines += [
        "",
        "## Why This Is Not Just a Single Lucky Run",
        "",
        "- The main claim is based on aggregate behavior under patient-held-out evaluation.",
        "- The run includes label-policy sensitivity because exact offsets are not available.",
        "- The detector is compared with LSTM, CNN-LSTM, compact Transformer, and Autoencoder baselines under the same split.",
        "- Event sensitivity and false alarms/hour are primary; raw accuracy is not used as the success claim.",
        "",
        "## Methods Not Treated as Direct Comparisons",
        "",
    ]
    for item in payload["not_directly_comparable"]:
        lines.append(f"- {item['method']}: {item['reason']}")
    lines += [
        "",
        "## Reference Anchors",
        "",
    ]
    for ref in REFERENCES.values():
        lines.append(f"- {ref['title']}: {ref['url']} - {ref['relevance']}")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)
    return payload


if __name__ == "__main__":
    run()
