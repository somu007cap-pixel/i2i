"""Print a compact console summary after a marathon run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "seizure_detection" / "outputs"
PREDICTION_DIR = ROOT / "prediction_outputs_local"
PUBLIC_BENCHMARK_DIR = ROOT / "public_benchmark_outputs"


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON artifact if it exists and is readable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def fmt(value: Any, digits: int = 4) -> str:
    """Format numeric metrics without failing on missing values."""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def event_text(metrics: dict[str, Any]) -> str:
    """Format event-level detection counts."""
    event = metrics.get("event_level") or {}
    return f"{event.get('detected_events', 'n/a')}/{event.get('event_count', 'n/a')}"


def print_detection() -> None:
    """Print headline detection metrics."""
    results = load_json(OUTPUT_DIR / "results.json")
    print("\n[Detection]")
    if not results or "_error" in results:
        print("  No readable detection results found.")
        return
    standard = results.get("standard_mode") or {}
    pro = results.get("pro_mode") or {}
    print(f"  Split: {results.get('split_strategy', 'n/a')}")
    print(
        "  Standard: "
        f"AUC={fmt(standard.get('auc'))}, recall={fmt(standard.get('recall'))}, "
        f"events={event_text(standard)}, FA/hr={fmt(standard.get('false_alarms_per_hour'), 2)}"
    )
    print(
        "  Pro:      "
        f"AUC={fmt(pro.get('auc'))}, recall={fmt(pro.get('recall'))}, "
        f"events={event_text(pro)}, FA/hr={fmt(pro.get('false_alarms_per_hour'), 2)}"
    )
    print(f"  Results: {OUTPUT_DIR / 'results.json'}")


def print_prediction() -> None:
    """Print headline prediction/risk-model metrics."""
    metrics = load_json(PREDICTION_DIR / "metrics.json")
    print("\n[Prediction / Risk Modelling]")
    if not metrics or "_error" in metrics:
        print("  No readable prediction metrics found.")
        return
    print(f"  Status: {metrics.get('status', 'n/a')}")
    for horizon in ("300", "900", "1800"):
        item = (metrics.get("metrics") or {}).get(horizon) or {}
        if item:
            print(
                f"  {int(horizon) // 60} min: "
                f"AUC={fmt(item.get('auc'))}, PR-AUC={fmt(item.get('pr_auc'))}"
            )
    print(f"  Metrics: {PREDICTION_DIR / 'metrics.json'}")


def print_baselines() -> None:
    """Print the strongest available baseline comparison rows."""
    report = load_json(OUTPUT_DIR / "baselines" / "baseline_comparison_report.json")
    print("\n[Baseline Models]")
    if not report or "_error" in report:
        print("  No readable baseline report found.")
        return
    for name, payload in (report.get("baselines") or {}).items():
        metrics = payload.get("metrics") or {}
        print(
            f"  {name}: AUC={fmt(metrics.get('auc'))}, "
            f"recall={fmt(metrics.get('recall'))}, events={event_text(metrics)}"
        )
    print(f"  Report: {OUTPUT_DIR / 'baselines' / 'baseline_comparison_report.json'}")


def print_matrix() -> None:
    """Print TSMixer experiment-matrix summary location."""
    matrix_dir = OUTPUT_DIR / "tsmixer_experiments"
    summary = matrix_dir / "TSMIXER_EXPERIMENT_SUMMARY.md"
    print("\n[TSMixer Experiment Matrix]")
    if summary.exists():
        print(f"  Summary: {summary}")
    elif matrix_dir.exists():
        count = len([path for path in matrix_dir.iterdir() if path.is_dir()])
        print(f"  Archived experiment folders: {count}")
    else:
        print("  No matrix artifacts found.")


def print_public_benchmark() -> None:
    """Print public EDF benchmark summary."""
    report = load_json(PUBLIC_BENCHMARK_DIR / "public_benchmark_summary.json")
    print("\n[Public EDF Benchmark]")
    if not report or "_error" in report:
        print("  No readable public benchmark summary found.")
        return
    print(
        f"  EDF files={report.get('edf_file_count', 'n/a')}, "
        f"subjects={report.get('unique_subject_count', 'n/a')}, "
        f"duration={report.get('total_duration_hours', 'n/a')} hours"
    )
    print(f"  Summary: {PUBLIC_BENCHMARK_DIR / 'public_benchmark_summary.json'}")


def main() -> int:
    """Print all available run summaries."""
    print("\n" + "=" * 70)
    print("LATEST RUN SUMMARY")
    print("=" * 70)
    print_detection()
    print_prediction()
    print_baselines()
    print_matrix()
    print_public_benchmark()
    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
