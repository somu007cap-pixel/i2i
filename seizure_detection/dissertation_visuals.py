"""Generate dissertation figures from seizure-detection run artifacts.

The figures in this module are intentionally artifact-driven. They use the
current detection, baseline, experiment, and edge-feasibility outputs instead of
inventing generic ML plots that are not supported by saved data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "seizure_detection" / "outputs"
FIGURE_DIR = OUTPUT_DIR / "dissertation_figures"
RESULTS_PATH = OUTPUT_DIR / "results.json"
BASELINE_PATH = OUTPUT_DIR / "baselines" / "baseline_comparison_report.json"
EDGE_PATH = OUTPUT_DIR / "edge_feasibility" / "edge_feasibility_report.json"
EXPERIMENT_SUMMARY_PATH = OUTPUT_DIR / "tsmixer_experiments" / "TSMIXER_EXPERIMENT_SUMMARY.md"


COLORS = {
    "standard": "#2E86AB",
    "pro": "#A23B72",
    "baseline": "#6C757D",
    "success": "#2CA02C",
    "warning": "#F58518",
    "danger": "#D62728",
    "light": "#F7F7F7",
    "dark": "#2B2B2B",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def event_label(metrics: dict[str, Any]) -> str:
    event = metrics.get("event_level") or {}
    return f"{event.get('detected_events', 0)}/{event.get('event_count', 0)}"


def plot_class_distribution(results: dict[str, Any]) -> Path:
    standard = results.get("standard_mode", {})
    tn = int(standard.get("true_negatives", 0))
    fp = int(standard.get("false_positives", 0))
    fn = int(standard.get("false_negatives", 0))
    tp = int(standard.get("true_positives", 0))
    normal = tn + fp
    seizure = tp + fn

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(["Non-seizure windows", "Seizure windows"], [normal, seizure], color=[COLORS["standard"], COLORS["warning"]])
    ax.set_title("Held-Out Test Window Distribution")
    ax.set_ylabel("Window count")
    ax.set_yscale("log")
    ax.text(0.5, 0.92, f"Seizure prevalence: {fmt(seizure / max(normal + seizure, 1) * 100, 3)}%", transform=ax.transAxes, ha="center")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height()):,}", ha="center", va="bottom")
    return save(fig, "01_class_imbalance_log_scale.png")


def plot_standard_pro_metrics(results: dict[str, Any]) -> Path:
    standard = results.get("standard_mode", {})
    pro = results.get("pro_mode", {})
    metrics = [
        ("AUC", "auc"),
        ("PR-AUC", "pr_auc"),
        ("Recall", "recall"),
        ("Balanced accuracy", "balanced_accuracy"),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    std_values = [float(standard.get(key, 0.0)) for _, key in metrics]
    pro_values = [float(pro.get(key, 0.0)) for _, key in metrics]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - width / 2, std_values, width, label="Standard", color=COLORS["standard"])
    ax.bar(x + width / 2, pro_values, width, label="Pro", color=COLORS["pro"])
    ax.set_title("Standard vs Pro Window-Level Metrics")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in metrics])
    ax.set_ylim(0, max(0.75, max(std_values + pro_values) * 1.15))
    ax.legend(frameon=False)
    subtitle = (
        f"Events: Standard {event_label(standard)} -> Pro {event_label(pro)}; "
        f"FA/hr: {fmt(standard.get('false_alarms_per_hour'), 2)} -> {fmt(pro.get('false_alarms_per_hour'), 2)}"
    )
    ax.text(0.5, -0.22, subtitle, transform=ax.transAxes, ha="center")
    return save(fig, "02_standard_vs_pro_metrics.png")


def confusion_matrix_from_metrics(metrics: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [int(metrics.get("true_negatives", 0)), int(metrics.get("false_positives", 0))],
            [int(metrics.get("false_negatives", 0)), int(metrics.get("true_positives", 0))],
        ],
        dtype=int,
    )


def draw_confusion(ax: plt.Axes, matrix: np.ndarray, title: str) -> None:
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Seizure"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Seizure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > threshold else COLORS["dark"]
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center", color=color, fontsize=9)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_confusion_matrices(results: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    draw_confusion(axes[0], confusion_matrix_from_metrics(results.get("standard_mode", {})), "Standard")
    draw_confusion(axes[1], confusion_matrix_from_metrics(results.get("pro_mode", {})), "Pro")
    fig.suptitle("Held-Out Test Confusion Matrices")
    fig.tight_layout()
    return save(fig, "03_confusion_matrices.png")


def plot_alarm_budget_grid(results: dict[str, Any]) -> Path | None:
    grid = (
        results.get("matched_false_alarm_operating_point", {})
        .get("fixed_validation_alarm_budgets", {})
    )
    budgets = grid.get("budgets_per_hour") or []
    standard_grid = grid.get("standard") or {}
    pro_grid = grid.get("pro") or {}
    if not budgets or not standard_grid or not pro_grid:
        return None

    std_recall, pro_recall, std_fa, pro_fa, std_events, pro_events = [], [], [], [], [], []
    for budget in budgets:
        key = str(float(budget))
        if key not in standard_grid:
            key = str(budget)
        if key not in standard_grid or key not in pro_grid:
            continue
        s = standard_grid[key]
        p = pro_grid[key]
        std_recall.append(float(s.get("recall", 0.0)))
        pro_recall.append(float(p.get("recall", 0.0)))
        std_fa.append(float(s.get("false_alarms_per_hour", 0.0)))
        pro_fa.append(float(p.get("false_alarms_per_hour", 0.0)))
        std_events.append(int((s.get("event_level") or {}).get("detected_events", 0)))
        pro_events.append(int((p.get("event_level") or {}).get("detected_events", 0)))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].plot(budgets[: len(std_recall)], std_recall, marker="o", label="Standard", color=COLORS["standard"])
    axes[0].plot(budgets[: len(pro_recall)], pro_recall, marker="o", label="Pro", color=COLORS["pro"])
    axes[0].set_title("Recall Across Validation Alarm Budgets")
    axes[0].set_xlabel("Validation false alarms/hour budget")
    axes[0].set_ylabel("Held-out test recall")
    axes[0].legend(frameon=False)

    x = np.arange(len(std_events))
    width = 0.36
    axes[1].bar(x - width / 2, std_events, width, label="Standard", color=COLORS["standard"])
    axes[1].bar(x + width / 2, pro_events, width, label="Pro", color=COLORS["pro"])
    axes[1].set_title("Detected Events Across Alarm Budgets")
    axes[1].set_xlabel("Validation false alarms/hour budget")
    axes[1].set_ylabel("Detected events")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(int(float(b))) for b in budgets[: len(std_events)]])
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return save(fig, "04_alarm_budget_sweep.png")


def plot_baseline_comparison(results: dict[str, Any], baseline_report: dict[str, Any]) -> Path | None:
    baselines = baseline_report.get("baselines") or {}
    if not baselines:
        return None
    names = ["TSMixer Standard", "TSMixer Pro"]
    auc_values = [float(results.get("standard_mode", {}).get("auc", 0.0)), float(results.get("pro_mode", {}).get("auc", 0.0))]
    recall_values = [float(results.get("standard_mode", {}).get("recall", 0.0)), float(results.get("pro_mode", {}).get("recall", 0.0))]
    event_values = [
        int((results.get("standard_mode", {}).get("event_level") or {}).get("detected_events", 0)),
        int((results.get("pro_mode", {}).get("event_level") or {}).get("detected_events", 0)),
    ]
    for name, item in baselines.items():
        metrics = item.get("metrics", {})
        names.append(name)
        auc_values.append(float(metrics.get("auc", 0.0)))
        recall_values.append(float(metrics.get("recall", 0.0)))
        event_values.append(int((metrics.get("event_level") or {}).get("detected_events", 0)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = [COLORS["standard"], COLORS["pro"]] + [COLORS["baseline"]] * (len(names) - 2)
    for ax, values, title, ylabel in [
        (axes[0], auc_values, "ROC-AUC", "AUC"),
        (axes[1], recall_values, "Recall", "Recall"),
        (axes[2], event_values, "Event Detection", "Detected events"),
    ]:
        ax.barh(names, values, color=colors)
        ax.set_title(title)
        ax.set_xlabel(ylabel)
        ax.invert_yaxis()
    fig.suptitle("TSMixer Compared With Baseline Models")
    fig.tight_layout()
    return save(fig, "05_baseline_comparison.png")


def experiment_rows_from_summary() -> list[dict[str, Any]]:
    if not EXPERIMENT_SUMMARY_PATH.exists():
        return []
    rows = []
    in_latest = False
    for line in EXPERIMENT_SUMMARY_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Latest Marathon Run"):
            in_latest = True
            continue
        if line.startswith("## Full Archive"):
            break
        if not in_latest or not line.startswith("| ") or line.startswith("| ---") or "Experiment" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 13:
            continue
        rows.append(
            {
                "name": parts[0],
                "std_events": parts[3],
                "pro_events": parts[7],
                "auc_gain": float(parts[9]),
                "recall_gain": float(parts[10]),
                "event_gain": int(parts[11].replace("+", "")),
                "event_sensitivity_gain": parse_event_sensitivity(parts[7])
                - parse_event_sensitivity(parts[3]),
            }
        )
    return rows


def parse_event_sensitivity(value: str) -> float:
    try:
        detected, total = str(value).split("/", 1)
        return float(detected) / max(float(total), 1.0)
    except Exception:
        return 0.0


def plot_ablation_summary() -> Path | None:
    rows = experiment_rows_from_summary()
    if not rows:
        return None
    names = [row["name"] for row in rows]
    event_sensitivity_gain = [row["event_sensitivity_gain"] for row in rows]
    recall_gain = [row["recall_gain"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    y = np.arange(len(rows))
    axes[0].barh(y, event_sensitivity_gain, color=COLORS["success"])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    axes[0].set_title("Pro Event-Sensitivity Gain")
    axes[0].set_xlabel("Event-sensitivity gain")
    axes[1].barh(y, recall_gain, color=COLORS["pro"])
    axes[1].set_title("Pro Recall Gain")
    axes[1].set_xlabel("Recall gain")
    fig.suptitle("Sensor/Fusion Ablation Summary")
    fig.tight_layout()
    return save(fig, "06_ablation_summary.png")


def plot_edge_footprint(edge_report: dict[str, Any]) -> Path | None:
    models = edge_report.get("models") or {}
    if not models:
        return None
    tiers, sizes, tensors, latency = [], [], [], []
    for tier in ("standard", "pro"):
        payload = models.get(tier) or {}
        tflite = (payload.get("tflite") or {}).get("calibrated_existing") or {}
        bench = (payload.get("benchmark") or {}).get("calibrated_existing") or {}
        tiers.append(tier.title())
        sizes.append(float(tflite.get("size_kb", 0.0)))
        tensors.append(float(tflite.get("estimated_tensor_arena_proxy_kb", 0.0)))
        latency.append(float(bench.get("median_ms", 0.0)))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    for ax, values, title, ylabel in [
        (axes[0], sizes, "TFLite Model Size", "KB"),
        (axes[1], tensors, "Tensor Arena Proxy", "KB"),
        (axes[2], latency, "PC TFLite Latency Proxy", "ms"),
    ]:
        ax.bar(tiers, values, color=[COLORS["standard"], COLORS["pro"]])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    fig.suptitle("Software Edge-Feasibility Profile")
    fig.tight_layout()
    return save(fig, "07_edge_feasibility.png")


def plot_event_failure_analysis(results: dict[str, Any]) -> Path | None:
    standard_events = (results.get("standard_mode", {}).get("event_level") or {}).get("events") or []
    pro_events = (results.get("pro_mode", {}).get("event_level") or {}).get("events") or []
    if not standard_events or not pro_events:
        return None
    labels = [f"E{i+1}" for i in range(max(len(standard_events), len(pro_events)))]
    std_scores = [float(item.get("max_score", 0.0)) for item in standard_events]
    pro_scores = [float(item.get("max_score", 0.0)) for item in pro_events]
    std_detected = [bool(item.get("detected")) for item in standard_events]
    pro_detected = [bool(item.get("detected")) for item in pro_events]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.bar(x - width / 2, std_scores, width, label="Standard max score", color=[COLORS["standard"] if ok else "#BFD7EA" for ok in std_detected])
    ax.bar(x + width / 2, pro_scores, width, label="Pro max score", color=[COLORS["pro"] if ok else "#E8C5DA" for ok in pro_detected])
    ax.set_title("Event-Level Failure Analysis")
    ax.set_xlabel("Held-out seizure event")
    ax.set_ylabel("Maximum model score in event interval")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.text(0.5, -0.22, "Darker bars indicate detected events; lighter bars indicate missed events.", transform=ax.transAxes, ha="center")
    return save(fig, "08_event_failure_analysis.png")


def write_figure_index(paths: list[Path]) -> Path:
    descriptions = {
        "01_class_imbalance_log_scale.png": "Use in Data Preparation / Dataset Characteristics. Shows severe class imbalance and why raw accuracy is misleading.",
        "02_standard_vs_pro_metrics.png": "Use in Results. Shows Standard vs Pro metric trade-offs at the selected operating point.",
        "03_confusion_matrices.png": "Use in Results or Appendix. Shows actual error counts under the selected operating point.",
        "04_alarm_budget_sweep.png": "Use in Evaluation Methodology. Shows sensitivity to false-alarm operating budget.",
        "05_baseline_comparison.png": "Use in Baseline Comparison. Shows where TSMixer wins or remains competitive.",
        "06_ablation_summary.png": "Use in Ablation Study. Shows which sensor/fusion configurations improved Pro performance.",
        "07_edge_feasibility.png": "Use in Edge Feasibility. Keep wording as software profiling, not hardware deployment.",
        "08_event_failure_analysis.png": "Use in Failure Analysis. Shows which held-out seizure events were missed.",
    }
    lines = [
        "# Dissertation Figure Index",
        "",
        "These figures are generated from saved project artifacts. They are suitable for thesis/report use because they reflect actual run outputs.",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`: {descriptions.get(path.name, 'Project artifact visualization.')}")
    index_path = FIGURE_DIR / "README_FIGURES.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def generate_all() -> list[Path]:
    setup_style()
    results = load_json(RESULTS_PATH)
    baselines = load_json(BASELINE_PATH)
    edge = load_json(EDGE_PATH)
    if not results:
        raise FileNotFoundError(f"Missing detection results: {RESULTS_PATH}")

    paths: list[Path] = [
        plot_class_distribution(results),
        plot_standard_pro_metrics(results),
        plot_confusion_matrices(results),
    ]
    for maybe_path in [
        plot_alarm_budget_grid(results),
        plot_baseline_comparison(results, baselines),
        plot_ablation_summary(),
        plot_edge_footprint(edge),
        plot_event_failure_analysis(results),
    ]:
        if maybe_path is not None:
            paths.append(maybe_path)
    paths.append(write_figure_index(paths))
    return paths


if __name__ == "__main__":
    generated = generate_all()
    print("Generated dissertation figures:")
    for item in generated:
        print(f"  {item}")
