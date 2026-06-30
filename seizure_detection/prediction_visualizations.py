"""
Prediction Visualization Suite
==============================
Release-quality visualizations for seizure prediction results.

Includes:
1. ROC and PR Curves
2. Confusion Matrix Heatmaps
3. Multi-horizon Comparison Charts
4. Prediction Timeline Visualization
5. Feature Importance / Attention Heatmaps
6. Baseline Comparison Charts
7. Summary Dashboard
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Dict, List, Tuple, Optional
import os

plt.style.use("seaborn-v0_8-whitegrid")


COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#28A745",
    "warning": "#FFC107",
    "danger": "#DC3545",
    "neutral": "#6C757D",
    "highlight": "#17A2B8",
    "dark": "#343A40",
    "light": "#F8F9FA",
}


def draw_confusion_heatmap(
    ax,
    cm: np.ndarray,
    xticklabels,
    yticklabels,
    show_colorbar: bool = True,
    annotation_size: int = 16,
):
    """Draw a confusion-matrix heatmap without requiring seaborn."""
    image = ax.imshow(cm, cmap="Blues")
    if show_colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels)

    threshold = cm.max() / 2 if cm.size and cm.max() > 0 else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else COLORS["dark"]
            ax.text(
                col,
                row,
                f"{int(cm[row, col])}",
                ha="center",
                va="center",
                fontsize=annotation_size,
                color=color,
            )


def plot_roc_curves(
    y_true_dict: Dict[int, np.ndarray],
    y_prob_dict: Dict[int, np.ndarray],
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot ROC curves for multiple prediction horizons.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["success"]]

    for i, (horizon, y_true) in enumerate(y_true_dict.items()):
        y_prob = y_prob_dict[horizon]
        if len(np.unique(y_true)) < 2:
            label = f"{horizon//60} min ahead (AUC unavailable)"
            ax.plot([], [], color=colors[i % len(colors)], linewidth=2.5, label=label)
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = np.trapezoid(tpr, fpr)

        label = f"{horizon//60} min ahead (AUC = {auc:.3f})"
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2.5, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Random Classifier")

    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=14)
    ax.set_title(
        "ROC Curves: Seizure Prediction Performance", fontsize=16, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    ax.annotate("Better", xy=(0.1, 0.9), fontsize=12, style="italic", color="green")
    ax.annotate("Worse", xy=(0.8, 0.2), fontsize=12, style="italic", color="red")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_precision_recall_curves(
    y_true_dict: Dict[int, np.ndarray],
    y_prob_dict: Dict[int, np.ndarray],
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple horizons.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["success"]]

    for i, (horizon, y_true) in enumerate(y_true_dict.items()):
        y_prob = y_prob_dict[horizon]
        if len(np.unique(y_true)) < 2:
            label = f"{horizon//60} min ahead (AP unavailable)"
            ax.plot([], [], color=colors[i % len(colors)], linewidth=2.5, label=label)
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = np.trapezoid(precision, recall)

        label = f"{horizon//60} min ahead (AP = {ap:.3f})"
        ax.plot(
            recall, precision, color=colors[i % len(colors)], linewidth=2.5, label=label
        )
    ax.set_xlabel("Recall (Sensitivity)", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title(
        "Precision-Recall Curves: Seizure Prediction", fontsize=16, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_minutes: int = 15,
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype("float"),
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0,
    )

    draw_confusion_heatmap(
        ax,
        cm,
        xticklabels=["No Seizure", "Pre-ictal"],
        yticklabels=["No Seizure", "Pre-ictal"],
        show_colorbar=True,
        annotation_size=16,
    )

    for i in range(2):
        for j in range(2):
            pct = cm_normalized[i, j] * 100
            ax.text(
                j + 0.5,
                i + 0.7,
                f"({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
            )
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("Actual", fontsize=14)
    ax.set_title(
        f"Confusion Matrix: {horizon_minutes}-min Prediction Horizon",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_multi_horizon_comparison(
    metrics_dict: Dict[int, Dict],
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Bar chart comparing performance across prediction horizons.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    horizons = sorted(metrics_dict.keys())
    horizon_labels = [f"{h//60} min" for h in horizons]

    metrics_left = ["sensitivity", "specificity", "precision"]
    metrics_right = ["f1_score", "auc_roc", "auc_pr"]

    colors = [COLORS["success"], COLORS["primary"], COLORS["secondary"]]

    x = np.arange(len(horizons))
    width = 0.25

    for i, metric in enumerate(metrics_left):
        values = [metrics_dict[h].get(metric, 0) for h in horizons]
        axes[0].bar(
            x + i * width, values, width, label=metric.capitalize(), color=colors[i]
        )
    axes[0].set_ylabel("Score", fontsize=14)
    axes[0].set_xlabel("Prediction Horizon", fontsize=14)
    axes[0].set_title("Clinical Metrics by Horizon", fontsize=16, fontweight="bold")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(horizon_labels)
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis="y", alpha=0.3)

    for i, metric in enumerate(metrics_right):
        values = [metrics_dict[h].get(metric, 0) for h in horizons]
        axes[1].bar(
            x + i * width,
            values,
            width,
            label=metric.upper().replace("_", "-"),
            color=colors[i],
        )
    axes[1].set_ylabel("Score", fontsize=14)
    axes[1].set_xlabel("Prediction Horizon", fontsize=14)
    axes[1].set_title("Model Performance by Horizon", fontsize=16, fontweight="bold")
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(horizon_labels)
    axes[1].legend(loc="lower right", fontsize=11)
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_competitive_comparison(
    our_metrics: Dict,
    baselines: Dict[str, Dict],
    save_path: str = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Radar/spider chart comparing the model with baseline references.
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

    metrics = ["Sensitivity", "Specificity", "Precision", "F1 Score", "AUC-ROC"]
    n_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    our_values = [
        our_metrics.get("sensitivity", 0),
        our_metrics.get("specificity", 0),
        our_metrics.get("precision", 0),
        our_metrics.get("f1_score", 0),
        our_metrics.get("auc_roc", 0),
    ]
    our_values += our_values[:1]

    ax.plot(
        angles,
        our_values,
        "o-",
        linewidth=3,
        label="Our Model",
        color=COLORS["primary"],
        markersize=8,
    )
    ax.fill(angles, our_values, alpha=0.25, color=COLORS["primary"])

    colors = [COLORS["neutral"], COLORS["danger"], COLORS["warning"]]
    for i, (name, baseline) in enumerate(list(baselines.items())[:3]):
        values = [
            baseline.get("sensitivity", 0),
            baseline.get("specificity", 0.5),
            baseline.get("precision", 0.5),
            baseline.get("f1", 0.5),
            baseline.get("auc", 0.5),
        ]
        values += values[:1]
        ax.plot(
            angles,
            values,
            "o--",
            linewidth=2,
            label=baseline.get("name", name),
            color=colors[i % len(colors)],
            alpha=0.7,
            markersize=6,
        )
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim([0, 1])
    ax.set_title(
        "Model Comparison vs. Baselines", fontsize=16, fontweight="bold", y=1.1
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_prediction_timeline(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seizure_times: List[Tuple[float, float]] = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (16, 6),
) -> plt.Figure:
    """
    Timeline visualization showing predictions vs actual seizures.
    """
    fig, ax = plt.subplots(figsize=figsize)

    t0 = timestamps.min()
    t_hours = (timestamps - t0) / 3600

    pred_mask = y_pred == 1
    ax.scatter(
        t_hours[pred_mask],
        np.ones(pred_mask.sum()) * 1,
        c=COLORS["warning"],
        s=20,
        alpha=0.7,
        label="Predicted Pre-ictal",
    )

    ax.scatter(
        t_hours[~pred_mask],
        np.ones((~pred_mask).sum()) * 1,
        c=COLORS["neutral"],
        s=5,
        alpha=0.3,
    )

    if seizure_times:
        for sz_start, sz_end in seizure_times:
            sz_start_h = (sz_start - t0) / 3600
            sz_end_h = (sz_end - t0) / 3600
            ax.axvspan(
                sz_start_h,
                sz_end_h,
                color=COLORS["danger"],
                alpha=0.5,
                label="Actual Seizure" if sz_start == seizure_times[0][0] else "",
            )
    tp_mask = (y_pred == 1) & (y_true == 1)
    ax.scatter(
        t_hours[tp_mask],
        np.ones(tp_mask.sum()) * 0.5,
        c=COLORS["success"],
        s=50,
        marker="*",
        label="Correct Prediction (TP)",
    )

    ax.set_xlabel("Time (hours)", fontsize=14)
    ax.set_yticks([0.5, 1])
    ax.set_yticklabels(["True Positives", "All Predictions"])
    ax.set_title(
        "Prediction Timeline: Seizure Prediction Performance",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.set_ylim([0, 1.5])
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_executive_dashboard(
    metrics_15min: Dict,
    metrics_5min: Dict = None,
    metrics_30min: Dict = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Summary dashboard with key prediction metrics.
    """
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(
        "SEIZURE PREDICTION - SUMMARY DASHBOARD", fontsize=20, fontweight="bold", y=0.98
    )

    ax1 = fig.add_subplot(gs[0, 0])
    sensitivity = metrics_15min.get("sensitivity", 0)
    _plot_gauge(ax1, sensitivity, "Sensitivity", COLORS["success"])

    ax2 = fig.add_subplot(gs[0, 1])
    specificity = metrics_15min.get("specificity", 0)
    _plot_gauge(ax2, specificity, "Specificity", COLORS["primary"])

    ax3 = fig.add_subplot(gs[0, 2])
    auc = metrics_15min.get("auc_roc", 0)
    _plot_gauge(ax3, auc, "AUC-ROC", COLORS["secondary"])

    ax4 = fig.add_subplot(gs[0, 3])
    fpr = metrics_15min.get("false_prediction_rate_per_hour", 0) * 24
    _plot_metric_box(
        ax4,
        f"{fpr:.1f}",
        "False Alarms/Day",
        "green" if fpr < 2 else "orange" if fpr < 5 else "red",
    )

    ax5 = fig.add_subplot(gs[1, :2])
    if metrics_5min and metrics_30min:
        _plot_horizon_bars(ax5, metrics_5min, metrics_15min, metrics_30min)
    else:
        _plot_single_horizon_bars(ax5, metrics_15min)
    ax6 = fig.add_subplot(gs[1, 2:])
    cm_data = metrics_15min.get(
        "confusion_matrix", {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    )
    _plot_mini_confusion_matrix(ax6, cm_data)

    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")

    key_messages = [
        f"Calibrated recall: {sensitivity*100:.0f}% at {metrics_15min.get('prediction_horizon_minutes', 15)} minutes",
        f"Estimated false alarms/day: {fpr:.1f}",
        "Validation uses held-out sessions from the available Mayo recordings",
        "Prediction results should be interpreted as risk-ranking metrics",
    ]

    y_pos = 0.9
    for msg in key_messages:
        ax7.text(
            0.05,
            y_pos,
            msg,
            fontsize=14,
            transform=ax7.transAxes,
            verticalalignment="top",
            fontweight="bold",
        )
        y_pos -= 0.25
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.05, right=0.97)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def _plot_gauge(ax, value, label, color):
    """Plot a simple gauge/donut chart."""
    ax.pie(
        [value, 1 - value],
        colors=[color, "#E0E0E0"],
        startangle=90,
        wedgeprops=dict(width=0.3),
    )
    ax.text(
        0,
        0,
        f"{value*100:.1f}%",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_title(label, fontsize=14, fontweight="bold", pad=10)


def _plot_metric_box(ax, value, label, color_name):
    """Plot a metric in a box."""
    color = {
        "green": COLORS["success"],
        "orange": COLORS["warning"],
        "red": COLORS["danger"],
    }.get(color_name, COLORS["neutral"])

    ax.axis("off")
    ax.add_patch(
        plt.Rectangle(
            (0.1, 0.1),
            0.8,
            0.8,
            facecolor=color,
            alpha=0.2,
            edgecolor=color,
            linewidth=3,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.6,
        value,
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        transform=ax.transAxes,
        color=color,
    )
    ax.text(
        0.5, 0.25, label, ha="center", va="center", fontsize=12, transform=ax.transAxes
    )


def _plot_horizon_bars(ax, m5, m15, m30):
    """Plot horizon comparison bars."""
    metrics = ["Sensitivity", "Specificity", "F1 Score"]
    x = np.arange(len(metrics))
    width = 0.25

    vals_5 = [m5.get("sensitivity", 0), m5.get("specificity", 0), m5.get("f1_score", 0)]
    vals_15 = [
        m15.get("sensitivity", 0),
        m15.get("specificity", 0),
        m15.get("f1_score", 0),
    ]
    vals_30 = [
        m30.get("sensitivity", 0),
        m30.get("specificity", 0),
        m30.get("f1_score", 0),
    ]

    ax.bar(x - width, vals_5, width, label="5 min", color=COLORS["success"])
    ax.bar(x, vals_15, width, label="15 min", color=COLORS["primary"])
    ax.bar(x + width, vals_30, width, label="30 min", color=COLORS["secondary"])

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.set_title("Performance by Prediction Horizon", fontweight="bold")


def _plot_single_horizon_bars(ax, metrics):
    """Plot single horizon bars."""
    names = ["Sensitivity", "Specificity", "Precision", "F1", "AUC"]
    vals = [
        metrics.get("sensitivity", 0),
        metrics.get("specificity", 0),
        metrics.get("precision", 0),
        metrics.get("f1_score", 0),
        metrics.get("auc_roc", 0),
    ]

    colors = [
        COLORS["success"],
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["warning"],
        COLORS["highlight"],
    ]

    ax.bar(names, vals, color=colors)
    ax.set_ylabel("Score")
    ax.set_ylim([0, 1])
    ax.set_title("15-Minute Prediction Horizon Performance", fontweight="bold")


def _plot_mini_confusion_matrix(ax, cm_data):
    """Plot mini confusion matrix."""
    if isinstance(cm_data, dict):
        cm = np.array([[cm_data["TN"], cm_data["FP"]], [cm_data["FN"], cm_data["TP"]]])
    else:
        cm = np.asarray(cm_data)
        if cm.shape != (2, 2):
            fixed = np.zeros((2, 2), dtype=int)
            fixed[: cm.shape[0], : cm.shape[1]] = cm
            cm = fixed
    draw_confusion_heatmap(
        ax,
        cm,
        xticklabels=["Pred -", "Pred +"],
        yticklabels=["True -", "True +"],
        show_colorbar=False,
        annotation_size=14,
    )
    ax.set_title("Confusion Matrix", fontweight="bold")


def save_all_visualizations(
    y_true_dict: Dict[int, np.ndarray],
    y_pred_dict: Dict[int, np.ndarray],
    y_prob_dict: Dict[int, np.ndarray],
    metrics_dict: Dict[int, Dict],
    output_dir: str,
) -> List[str]:
    """
    Generate and save all visualizations.
    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    path = os.path.join(output_dir, "roc_curves.png")
    plot_roc_curves(y_true_dict, y_prob_dict, save_path=path)
    saved_files.append(path)
    plt.close()

    path = os.path.join(output_dir, "pr_curves.png")
    plot_precision_recall_curves(y_true_dict, y_prob_dict, save_path=path)
    saved_files.append(path)
    plt.close()

    for horizon in y_true_dict.keys():
        path = os.path.join(output_dir, f"confusion_matrix_{horizon//60}min.png")
        plot_confusion_matrix(
            y_true_dict[horizon],
            y_pred_dict[horizon],
            horizon_minutes=horizon // 60,
            save_path=path,
        )
        saved_files.append(path)
        plt.close()
    path = os.path.join(output_dir, "horizon_comparison.png")
    plot_multi_horizon_comparison(metrics_dict, save_path=path)
    saved_files.append(path)
    plt.close()

    m15 = metrics_dict.get(900, metrics_dict.get(list(metrics_dict.keys())[0]))
    m5 = metrics_dict.get(300)
    m30 = metrics_dict.get(1800)

    path = os.path.join(output_dir, "executive_dashboard.png")
    plot_executive_dashboard(m15, m5, m30, save_path=path)
    saved_files.append(path)
    plt.close()

    print(f"Saved {len(saved_files)} visualizations to {output_dir}")
    return saved_files
