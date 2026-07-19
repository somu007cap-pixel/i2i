"""
Phase 4: Training and Validation
=================================
Train and evaluate the Dual-Branch TSMixer detection models.
"""

import os
import subprocess
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import hashlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Dict, Optional, Union, List
import json
from datetime import datetime
from types import SimpleNamespace
from phase3_tsmixer_model import (
    ChannelIndependentTSMixerBlock,
    ChannelIndependentTimeMixer,
    GatedBoosterFusion,
    GatedTimeMixerBlock,
    ChannelPatchEmbedding,
    MLPBlock,
    ModalityDropout,
    SecondaryFeatureSelector,
    SensitivitySpecificityWeightedBinaryCrossentropy,
    SummaryStats,
    TSMixerBlock,
    build_dual_branch_tsmixer,
    compile_model,
    model_size_profile,
)

tf.get_logger().setLevel("ERROR")

PIPELINE_VERSION = "edge_safe_patched_dual_stream_tsmixer_v15_dsp_session_norm"
TARGET_RATE_HZ = 32
DETECTION_WINDOW_SECONDS = float(os.environ.get("DETECTION_WINDOW_SECONDS", "5"))
DETECTION_WINDOW_SIZE = max(32, int(round(DETECTION_WINDOW_SECONDS * TARGET_RATE_HZ)))
EXPECTED_PRIMARY_SHAPE = (DETECTION_WINDOW_SIZE, 3)
EXPECTED_SECONDARY_SHAPE = (DETECTION_WINDOW_SIZE, 4)
SECONDARY_FEATURES = ["BVP", "HR", "EDA", "TEMP"]
PREDICT_BATCH_SIZE = int(os.environ.get("DETECTION_PREDICT_BATCH_SIZE", "8192"))
PRO_CANDIDATE_FEATURES = {
    "pro_ppg_bvp_hr": ["BVP", "HR"],
    "pro_ppg_bvp_hr_eda": ["BVP", "HR", "EDA"],
    "pro_ppg_bvp_hr_temp": ["BVP", "HR", "TEMP"],
    "pro_full_bvp_hr_eda_temp": ["BVP", "HR", "EDA", "TEMP"],
}
STANDARD_CANDIDATE_FEATURES = {
    "standard_acc_only": [],
    "standard_acc_eda": ["EDA"],
    "standard_acc_temp": ["TEMP"],
    "standard_acc_eda_temp": ["EDA", "TEMP"],
}
PRODUCT_ALLOCATION_CANDIDATES = {
    "base_acc__addon_ppg_eda_temp": {
        "standard_variant": "standard_acc_only",
        "pro_variant": "pro_full_bvp_hr_eda_temp",
        "base_optional": [],
        "addon_optional": ["EDA", "TEMP"],
        "description": "Base has ACC only; add-on has required PPG plus EDA and TEMP.",
    },
    "base_acc_eda__addon_ppg_temp": {
        "standard_variant": "standard_acc_eda",
        "pro_variant": "pro_full_bvp_hr_eda_temp",
        "base_optional": ["EDA"],
        "addon_optional": ["TEMP"],
        "description": (
            "Base has ACC plus EDA; Pro keeps the base EDA signal and adds PPG plus TEMP."
        ),
    },
    "base_acc_temp__addon_ppg_eda": {
        "standard_variant": "standard_acc_temp",
        "pro_variant": "pro_full_bvp_hr_eda_temp",
        "base_optional": ["TEMP"],
        "addon_optional": ["EDA"],
        "description": (
            "Base has ACC plus TEMP; Pro keeps the base TEMP signal and adds PPG plus EDA."
        ),
    },
    "base_acc_eda_temp__addon_ppg": {
        "standard_variant": "standard_acc_eda_temp",
        "pro_variant": "pro_full_bvp_hr_eda_temp",
        "base_optional": ["EDA", "TEMP"],
        "addon_optional": [],
        "description": (
            "Base has ACC plus EDA and TEMP; Pro keeps those base signals and adds PPG."
        ),
    },
}


def get_patient_id(session_path: str) -> str:
    """Infer the Mayo patient id from a session path."""
    parts = os.path.abspath(session_path).split(os.sep)
    for part in reversed(parts):
        if part.startswith("Mayo_"):
            return part
    return "unknown"


def split_patient_paths(
    session_paths: List[str],
    seizure_intervals: Dict[str, List[Tuple[float, float]]],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, List[str]]:
    """Split by patient first, then keep all sessions from each patient together."""
    try:
        from phase2_data_generator_fast import get_actual_session_intervals
    except ImportError:
        from .phase2_data_generator_fast import get_actual_session_intervals
    rng = np.random.default_rng(random_state)
    patient_sessions: Dict[str, List[str]] = {}
    patient_has_seizure: Dict[str, bool] = {}
    for path in session_paths:
        patient_id = get_patient_id(path)
        patient_sessions.setdefault(patient_id, []).append(path)
        patient_has_seizure[patient_id] = (
            patient_has_seizure.get(patient_id, False)
            or len(get_actual_session_intervals(path, seizure_intervals)) > 0
        )
    seizure_patients = [
        patient_id for patient_id, has_sz in patient_has_seizure.items() if has_sz
    ]
    normal_patients = [
        patient_id for patient_id, has_sz in patient_has_seizure.items() if not has_sz
    ]
    rng.shuffle(seizure_patients)
    rng.shuffle(normal_patients)

    def split_group(patient_ids: List[str]):
        if not patient_ids:
            return [], [], []
        n_total = len(patient_ids)
        n_test = max(1, int(round(n_total * test_size))) if n_total >= 3 else 0
        n_val = max(1, int(round(n_total * val_size))) if n_total >= 4 else 0
        if n_test + n_val >= n_total:
            n_test = 1 if n_total >= 3 else 0
            n_val = 1 if n_total >= 4 else 0
        test_ids = patient_ids[:n_test]
        val_ids = patient_ids[n_test : n_test + n_val]
        train_ids = patient_ids[n_test + n_val :]
        test = [session for patient_id in test_ids for session in patient_sessions[patient_id]]
        val = [session for patient_id in val_ids for session in patient_sessions[patient_id]]
        train = [session for patient_id in train_ids for session in patient_sessions[patient_id]]
        return train, val, test

    sz_train, sz_val, sz_test = split_group(seizure_patients)
    normal_train, normal_val, normal_test = split_group(normal_patients)

    splits = {
        "train": sz_train + normal_train,
        "val": sz_val + normal_val,
        "test": sz_test + normal_test,
    }
    for paths in splits.values():
        rng.shuffle(paths)
    return splits


def summarize_session_splits(
    session_splits: Dict[str, List[str]],
    seizure_intervals: Dict[str, List[Tuple[float, float]]],
) -> Dict[str, Dict]:
    """Summarize held-out split composition for auditability."""
    try:
        from phase2_data_generator_fast import get_actual_session_intervals
    except ImportError:
        from .phase2_data_generator_fast import get_actual_session_intervals
    summary = {}
    for split_name, paths in session_splits.items():
        patients = {}
        seizure_sessions = 0
        episode_keys = set()
        for path in paths:
            patient_id = get_patient_id(path)
            patients[patient_id] = patients.get(patient_id, 0) + 1
            episodes = get_actual_session_intervals(path, seizure_intervals)
            if episodes:
                seizure_sessions += 1
                for start, end in episodes:
                    episode_keys.add((round(float(start), 3), round(float(end), 3)))
        summary[split_name] = {
            "sessions": len(paths),
            "seizure_sessions": seizure_sessions,
            "seizure_episodes": len(episode_keys),
            "patients": patients,
        }
    return summary


def print_split_audit(summary: Dict[str, Dict]) -> None:
    """Print patient/session composition for the current split."""
    print("\nSplit audit:")
    for split_name, item in summary.items():
        patient_text = ", ".join(
            f"{patient}:{count}" for patient, count in sorted(item["patients"].items())
        )
        print(
            f"  {split_name.title()}: {item['sessions']} sessions, "
            f"{item['seizure_sessions']} seizure sessions, "
            f"{item['seizure_episodes']} seizure episodes | patients: {patient_text}"
        )


def normalize_session_splits(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Preserve local session-level z-score normalization across splits."""
    normalized = dict(data)
    p_channels = data["primary_train"].shape[-1]
    s_channels = data["secondary_train"].shape[-1]
    normalized["normalization"] = {
        "primary_mean": np.zeros((1, 1, p_channels), dtype=np.float32).tolist(),
        "primary_std": np.ones((1, 1, p_channels), dtype=np.float32).tolist(),
        "secondary_mean": np.zeros((1, 1, s_channels), dtype=np.float32).tolist(),
        "secondary_std": np.ones((1, 1, s_channels), dtype=np.float32).tolist(),
    }
    return normalized


def print_label_counts(data: Dict[str, np.ndarray]) -> None:
    for split in ("train", "val", "test"):
        labels = data[f"labels_{split}"]
        positives = int(np.sum(labels))
        negatives = int(len(labels) - positives)
        pct = positives / len(labels) * 100 if len(labels) else 0
        print(
            f"  {split.title()}: {len(labels)} samples ({positives} seizure, {negatives} normal, {pct:.2f}% seizure)"
        )


def split_events(
    session_paths: List[str],
    seizure_intervals: Dict[str, List[Tuple[float, float]]],
) -> List[Dict[str, Union[str, float, int]]]:
    try:
        from phase2_data_generator_fast import get_actual_session_intervals
    except ImportError:
        from .phase2_data_generator_fast import get_actual_session_intervals

    events = []
    for path in session_paths:
        session_id = os.path.basename(path)
        for event_index, (start, end) in enumerate(
            get_actual_session_intervals(path, seizure_intervals)
        ):
            events.append(
                {
                    "session_id": session_id,
                    "event_index": int(event_index),
                    "start": float(start),
                    "end": float(end),
                }
            )
    return events


def event_level_metrics_from_probabilities(
    data: Dict[str, np.ndarray],
    split: str,
    probabilities: np.ndarray,
    threshold: float,
) -> Optional[Dict[str, Union[int, float]]]:
    session_ids = data.get(f"window_session_ids_{split}")
    window_starts = data.get(f"window_starts_{split}")
    window_ends = data.get(f"window_ends_{split}")
    events = data.get(f"events_{split}")
    labels = data.get(f"labels_{split}")
    if (
        session_ids is None
        or window_starts is None
        or window_ends is None
        or events is None
        or labels is None
    ):
        return None
    predictions = probabilities >= threshold
    detected_events = 0
    event_count = len(events)
    event_rows = []
    for event in events:
        session_mask = session_ids == event["session_id"]
        overlap = (
            session_mask
            & (window_starts <= float(event["end"]))
            & (window_ends >= float(event["start"]))
        )
        detected = bool(np.any(predictions & overlap))
        detected_events += int(detected)
        event_rows.append(
            {
                "session_id": event["session_id"],
                "event_index": int(event["event_index"]),
                "start": float(event["start"]),
                "end": float(event["end"]),
                "overlap_windows": int(np.sum(overlap)),
                "detected": detected,
                "max_score": (
                    float(np.max(probabilities[overlap]))
                    if np.any(overlap)
                    else float("nan")
                ),
            }
        )
    positive_windows = predictions
    seizure_windows = labels > 0.5
    sensitivity = float(detected_events / max(event_count, 1))
    if event_count > 0:
        z = 1.96
        denominator = 1.0 + (z * z / event_count)
        centre = sensitivity + (z * z / (2.0 * event_count))
        margin = z * np.sqrt(
            (sensitivity * (1.0 - sensitivity) / event_count)
            + (z * z / (4.0 * event_count * event_count))
        )
        ci_low = float(max(0.0, (centre - margin) / denominator))
        ci_high = float(min(1.0, (centre + margin) / denominator))
    else:
        ci_low = float("nan")
        ci_high = float("nan")
    return {
        "event_count": int(event_count),
        "detected_events": int(detected_events),
        "event_sensitivity": sensitivity,
        "event_sensitivity_ci95_low": ci_low,
        "event_sensitivity_ci95_high": ci_high,
        "event_sensitivity_ci95_method": "Wilson score interval",
        "positive_windows": int(np.sum(positive_windows)),
        "positive_seizure_windows": int(np.sum(positive_windows & seizure_windows)),
        "false_positive_windows": int(np.sum(positive_windows & ~seizure_windows)),
        "events": event_rows,
    }


def prepare_data(
    primary: np.ndarray,
    secondary: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Split data into train/val/test sets.

    Args:
        primary: Primary features (watch)
        secondary: Secondary features (sensors)
        labels: Binary labels
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed

    Returns:
        Dictionary with train/val/test splits
    """

    (
        primary_trainval,
        primary_test,
        secondary_trainval,
        secondary_test,
        labels_trainval,
        labels_test,
    ) = train_test_split(
        primary,
        secondary,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    val_frac = val_size / (1 - test_size)
    (
        primary_train,
        primary_val,
        secondary_train,
        secondary_val,
        labels_train,
        labels_val,
    ) = train_test_split(
        primary_trainval,
        secondary_trainval,
        labels_trainval,
        test_size=val_frac,
        random_state=random_state,
        stratify=labels_trainval,
    )

    return {
        "primary_train": primary_train,
        "secondary_train": secondary_train,
        "labels_train": labels_train,
        "primary_val": primary_val,
        "secondary_val": secondary_val,
        "labels_val": labels_val,
        "primary_test": primary_test,
        "secondary_test": secondary_test,
        "labels_test": labels_test,
    }


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute class weights for imbalanced data."""
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def build_balanced_training_view(
    data: Dict[str, np.ndarray],
    negative_ratio: int = 5,
    max_train_windows: int = 600_000,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Return a train-balanced copy of data while leaving val/test untouched."""
    labels = data["labels_train"]
    positive_idx = np.flatnonzero(labels > 0.5)
    negative_idx = np.flatnonzero(labels <= 0.5)

    if len(positive_idx) == 0 or len(negative_idx) == 0:
        print("Training balance skipped: train split has only one class.")
        return data
    rng = np.random.default_rng(random_state)
    max_negatives = min(len(negative_idx), len(positive_idx) * int(negative_ratio))
    max_total = int(max_train_windows)
    if len(positive_idx) + max_negatives > max_total:
        max_negatives = max(1, max_total - len(positive_idx))
    sampled_negative_idx = rng.choice(negative_idx, size=max_negatives, replace=False)
    selected_idx = np.concatenate([positive_idx, sampled_negative_idx])
    rng.shuffle(selected_idx)

    balanced = dict(data)
    balanced["primary_train"] = data["primary_train"][selected_idx]
    balanced["secondary_train"] = data["secondary_train"][selected_idx]
    balanced["labels_train"] = data["labels_train"][selected_idx]
    balanced["training_sampling"] = {
        "strategy": "all positives plus sampled negatives",
        "original_train_windows": int(len(labels)),
        "original_train_positives": int(len(positive_idx)),
        "sampled_train_windows": int(len(selected_idx)),
        "sampled_train_positives": int(np.sum(balanced["labels_train"])),
        "negative_ratio": int(negative_ratio),
        "max_train_windows": int(max_train_windows),
    }
    print("\nBalanced training view:")
    print(f"  Original train: {len(labels):,} windows, {len(positive_idx):,} seizure")
    print(
        f"  Sampled train: {len(selected_idx):,} windows, {int(np.sum(balanced['labels_train'])):,} seizure"
    )
    return balanced


class TrainingCallback(keras.callbacks.Callback):
    """Custom callback for training progress."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            f"  Epoch {epoch + 1}: "
            f"loss={logs.get('loss', 0):.4f}, "
            f"pr_auc={logs.get('pr_auc', 0):.4f}, "
            f"val_loss={logs.get('val_loss', 0):.4f}, "
            f"val_pr_auc={logs.get('val_pr_auc', 0):.4f}, "
            f"val_auc={logs.get('val_auc', 0):.4f}"
        )


def train_model(
    model: keras.Model,
    data: Dict[str, np.ndarray],
    epochs: int = 50,
    batch_size: int = 64,
    early_stopping_patience: int = 10,
    save_path: str = None,
) -> keras.callbacks.History:
    """
    Train the model.

    Args:
        model: Compiled Keras model
        data: Dictionary with train/val/test splits
        epochs: Max training epochs
        batch_size: Batch size
        early_stopping_patience: Early stopping patience
        save_path: Path to save best model

    Returns:
        Training history
    """
    negative_ratio = int(os.environ.get("DETECTION_NEGATIVE_RATIO", "5"))
    max_train_windows = int(os.environ.get("DETECTION_MAX_TRAIN_WINDOWS", "600000"))
    train_data = build_balanced_training_view(
        data,
        negative_ratio=negative_ratio,
        max_train_windows=max_train_windows,
    )
    if "training_sampling" in train_data:
        data["training_sampling"] = train_data["training_sampling"]
    class_weights = compute_class_weights(train_data["labels_train"])
    print(f"Class weights: {class_weights}")

    monitor_metric = os.environ.get("DETECTION_MONITOR", "val_pr_auc")
    print(f"Checkpoint/early stopping monitor: {monitor_metric}")

    callbacks = [
        TrainingCallback(),
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    if save_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                save_path,
                monitor=monitor_metric,
                save_best_only=True,
                mode="max",
                verbose=1,
            )
        )
    print(f"\nTraining for up to {epochs} epochs...")
    print("=" * 50)

    history = model.fit(
        x=[train_data["primary_train"], train_data["secondary_train"]],
        y=train_data["labels_train"],
        validation_data=(
            [data["primary_val"], data["secondary_val"]],
            data["labels_val"],
        ),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )

    return history


def evaluate_model(
    model: keras.Model,
    primary: np.ndarray,
    secondary: np.ndarray,
    labels: np.ndarray,
    mode_name: str = "Standard",
    stride_seconds: float = 0.5,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate model on a test set.

    Args:
        model: Trained Keras model
        primary: Primary features
        secondary: Secondary features (can be zeros for standard mode)
        labels: True labels
        mode_name: Name for display

    Returns:
        Dictionary of metrics
    """
    probabilities = model.predict(
        [primary, secondary],
        batch_size=PREDICT_BATCH_SIZE,
        verbose=0,
    ).reshape(-1)
    predictions = (probabilities >= threshold).astype(np.int32)
    cm = confusion_matrix(labels.astype(np.int32), predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    recording_hours = max(len(labels) * stride_seconds / 3600, 1e-9)
    total = max(tp + tn + fp + fn, 1)
    positives = tp + fn
    negatives = tn + fp
    threshold_accuracy = float((tp + tn) / max(tp + tn + fp + fn, 1))
    threshold_precision = float(tp / max(tp + fp, 1))
    threshold_recall = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(negatives, 1))
    balanced_accuracy = float((threshold_recall + specificity) / 2.0)
    positive_rate = float(positives / total)
    all_normal_baseline_accuracy = float(negatives / total)
    threshold_f1 = float(
        2
        * threshold_precision
        * threshold_recall
        / max(threshold_precision + threshold_recall, 1e-12)
    )
    unique_labels = np.unique(labels.astype(np.int32))
    auc = (
        float(roc_auc_score(labels, probabilities))
        if len(unique_labels) > 1
        else float("nan")
    )
    pr_auc = (
        float(average_precision_score(labels, probabilities))
        if len(unique_labels) > 1
        else float("nan")
    )
    eps = 1e-7
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    loss = -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))

    results = {
        "loss": float(loss),
        "accuracy": threshold_accuracy,
        "auc": auc,
        "pr_auc": pr_auc,
        "precision": threshold_precision,
        "recall": threshold_recall,
        "f1": threshold_f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "positive_rate": positive_rate,
        "all_normal_baseline_accuracy": all_normal_baseline_accuracy,
    }
    results.update(
        {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "false_alarms_per_hour": float(fp / recording_hours),
            "decision_threshold": float(threshold),
        }
    )
    print_evaluation_results(mode_name, results)
    return results


def print_evaluation_results(mode_name: str, results: Dict[str, float]) -> None:
    """Print evaluation metrics in the standard report format."""
    print(f"\n{mode_name} Mode Results:")
    print("-" * 30)
    for metric, value in results.items():
        if isinstance(value, dict):
            if metric == "event_level":
                print(
                    "  event_level: "
                    f"{value.get('detected_events', 0)}/{value.get('event_count', 0)} events, "
                    f"sensitivity={value.get('event_sensitivity', 0.0):.4f}"
                )
            else:
                print(f"  {metric}: {value}")
            continue
        display_metric = metric
        suffix = ""
        if metric == "accuracy":
            display_metric = "accuracy_diagnostic"
            suffix = " (not a success metric for this imbalanced split)"
        elif metric == "all_normal_baseline_accuracy":
            display_metric = "all_normal_baseline_accuracy_diagnostic"
            suffix = " (all-negative baseline)"
        if isinstance(value, (int, np.integer)):
            print(f"  {display_metric}: {value}{suffix}")
        elif not isinstance(value, dict):
            print(f"  {display_metric}: {value:.4f}{suffix}")


def predict_probabilities(
    model: keras.Model,
    primary: np.ndarray,
    secondary: np.ndarray,
) -> np.ndarray:
    """Predict probabilities in explicit chunks to avoid large TensorFlow copies."""
    total = len(primary)
    if total != len(secondary):
        raise ValueError(
            f"Primary/secondary length mismatch: {total} vs {len(secondary)}"
        )
    if total == 0:
        return np.array([], dtype=np.float32)
    chunk_size = max(1, int(os.environ.get("DETECTION_PREDICT_CHUNK_SIZE", "4096")))
    outputs = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        batch_prob = model.predict(
            [primary[start:end], secondary[start:end]],
            batch_size=min(PREDICT_BATCH_SIZE, end - start),
            verbose=0,
        ).reshape(-1)
        outputs.append(batch_prob.astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def metrics_from_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    stride_seconds: float,
) -> Dict[str, float]:
    """Compute thresholded and ranking metrics from cached probabilities."""
    predictions = (probabilities >= threshold).astype(np.int32)
    cm = confusion_matrix(labels.astype(np.int32), predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    recording_hours = max(len(labels) * stride_seconds / 3600, 1e-9)
    total = max(tp + tn + fp + fn, 1)
    positives = tp + fn
    negatives = tn + fp
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(negatives, 1))
    balanced_accuracy = float((recall + specificity) / 2.0)
    positive_rate = float(positives / total)
    all_normal_baseline_accuracy = float(negatives / total)
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    unique_labels = np.unique(labels.astype(np.int32))
    eps = 1e-7
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    loss = -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    return {
        "loss": float(loss),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "auc": (
            float(roc_auc_score(labels, probabilities))
            if len(unique_labels) > 1
            else float("nan")
        ),
        "pr_auc": (
            float(average_precision_score(labels, probabilities))
            if len(unique_labels) > 1
            else float("nan")
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "positive_rate": positive_rate,
        "all_normal_baseline_accuracy": all_normal_baseline_accuracy,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "false_alarms_per_hour": float(fp / recording_hours),
        "decision_threshold": float(threshold),
    }


def evaluate_alarm_budget_grid(
    val_labels: np.ndarray,
    val_probabilities: np.ndarray,
    test_labels: np.ndarray,
    test_probabilities: np.ndarray,
    budgets: List[float],
    stride_seconds: float,
    test_data: Optional[Dict[str, np.ndarray]] = None,
    alert_refractory_minutes: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """Evaluate test metrics at thresholds chosen from validation alarm budgets."""
    results = {}
    for budget in budgets:
        threshold = threshold_for_false_alarm_rate(
            val_labels,
            val_probabilities,
            budget,
            stride_seconds,
        )
        metrics = metrics_from_probabilities(
            test_labels,
            test_probabilities,
            threshold,
            stride_seconds,
        )
        metrics["validation_false_alarm_budget_per_hour"] = float(budget)
        if test_data is not None:
            event_metrics = event_level_metrics_from_probabilities(
                test_data, "test", test_probabilities, threshold
            )
            if event_metrics is not None:
                metrics["event_level"] = event_metrics
            alert_metrics = event_alert_metrics_from_probabilities(
                test_data,
                "test",
                test_probabilities,
                threshold,
                stride_seconds,
                alert_refractory_minutes,
            )
            if alert_metrics is not None:
                metrics["event_alert_level"] = alert_metrics
        results[str(budget)] = metrics
    return results


def false_alarm_rate_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    stride_seconds: float,
) -> float:
    """Return validation false alarms/hour for a threshold."""
    normal = labels <= 0.5
    false_positives = int(np.sum(probabilities[normal] >= threshold))
    recording_hours = max(len(labels) * stride_seconds / 3600, 1e-9)
    return float(false_positives / recording_hours)


def alert_mask_with_refractory(
    data: Dict[str, np.ndarray],
    split: str,
    probabilities: np.ndarray,
    threshold: float,
    refractory_windows: int,
) -> np.ndarray:
    """Convert window scores into first-alert windows with per-session refractory."""
    raw = probabilities >= threshold
    refractory_windows = max(int(refractory_windows), 0)
    session_ids = data.get(f"window_session_ids_{split}")
    if session_ids is None or len(session_ids) != len(probabilities):
        if refractory_windows <= 0:
            return raw.copy()
        alerts = np.zeros_like(raw, dtype=bool)
        last_alert = -refractory_windows - 1
        for idx, is_alert in enumerate(raw):
            if is_alert and idx - last_alert > refractory_windows:
                alerts[idx] = True
                last_alert = idx
        return alerts
    alerts = np.zeros_like(raw, dtype=bool)
    for session_id in np.unique(session_ids):
        idx = np.flatnonzero(session_ids == session_id)
        last_alert = -refractory_windows - 1
        for local_pos, global_idx in enumerate(idx):
            if raw[global_idx] and local_pos - last_alert > refractory_windows:
                alerts[global_idx] = True
                last_alert = local_pos
    return alerts


def event_alert_metrics_from_probabilities(
    data: Dict[str, np.ndarray],
    split: str,
    probabilities: np.ndarray,
    threshold: float,
    stride_seconds: float,
    refractory_minutes: float,
) -> Optional[Dict[str, Union[int, float, str]]]:
    """Evaluate event detections using alert episodes rather than every hot window."""
    refractory_windows = int(round(max(refractory_minutes, 0.0) * 60.0 / stride_seconds))
    alerts = alert_mask_with_refractory(
        data, split, probabilities, threshold, refractory_windows
    )
    session_ids = data.get(f"window_session_ids_{split}")
    window_starts = data.get(f"window_starts_{split}")
    window_ends = data.get(f"window_ends_{split}")
    events = data.get(f"events_{split}")
    labels = data.get(f"labels_{split}")
    if (
        session_ids is None
        or window_starts is None
        or window_ends is None
        or events is None
        or labels is None
    ):
        return None
    event_count = len(events)
    detected_events = 0
    alert_in_event = np.zeros_like(alerts, dtype=bool)
    event_rows = []
    for event in events:
        overlap = (
            (session_ids == event["session_id"])
            & (window_starts <= float(event["end"]))
            & (window_ends >= float(event["start"]))
        )
        detected = bool(np.any(alerts & overlap))
        detected_events += int(detected)
        alert_in_event |= alerts & overlap
        event_rows.append(
            {
                "session_id": event["session_id"],
                "event_index": int(event["event_index"]),
                "detected": detected,
                "alert_count": int(np.sum(alerts & overlap)),
                "max_score": (
                    float(np.max(probabilities[overlap]))
                    if np.any(overlap)
                    else float("nan")
                ),
            }
        )
    recording_hours = max(len(labels) * stride_seconds / 3600.0, 1e-9)
    false_alerts = int(np.sum(alerts & ~alert_in_event))
    sensitivity = float(detected_events / max(event_count, 1))
    return {
        "post_processing": "first alert per refractory interval",
        "refractory_minutes": float(refractory_minutes),
        "refractory_windows": int(refractory_windows),
        "event_count": int(event_count),
        "detected_events": int(detected_events),
        "event_sensitivity": sensitivity,
        "alert_count": int(np.sum(alerts)),
        "false_alert_count": false_alerts,
        "false_alerts_per_hour": float(false_alerts / recording_hours),
        "events": event_rows,
    }


def select_event_alert_postprocessing(
    data: Dict[str, np.ndarray],
    split: str,
    probabilities: np.ndarray,
    threshold: float,
    stride_seconds: float,
) -> Dict[str, Union[int, float, str]]:
    """Select refractory post-processing on validation labels only."""
    candidates = [
        float(value)
        for value in os.environ.get(
            "DETECTION_ALERT_REFRACTORY_MINUTES", "0,1,2,5,10,15"
        ).split(",")
        if value.strip()
    ]
    best = None
    best_score = None
    for minutes in candidates or [0.0]:
        metrics = event_alert_metrics_from_probabilities(
            data, split, probabilities, threshold, stride_seconds, minutes
        )
        if metrics is None:
            continue
        score = (
            100.0 * float(metrics.get("event_sensitivity", 0.0))
            + float(metrics.get("detected_events", 0))
            - 0.2 * float(metrics.get("false_alerts_per_hour", 0.0))
        )
        if best_score is None or score > best_score:
            best_score = score
            best = metrics
    if best is None:
        return {
            "post_processing": "not available",
            "refractory_minutes": 0.0,
            "refractory_windows": 0,
        }
    return best


def causal_smooth_probabilities(
    data: Dict[str, np.ndarray],
    split: str,
    probabilities: np.ndarray,
    window_count: int,
) -> np.ndarray:
    """Apply causal rolling-mean score smoothing within each session."""
    window_count = max(int(window_count), 1)
    if window_count <= 1:
        return probabilities.copy()
    session_ids = data.get(f"window_session_ids_{split}")
    if session_ids is None or len(session_ids) != len(probabilities):
        return probabilities.copy()
    smoothed = probabilities.astype(np.float64, copy=True)
    for session_id in np.unique(session_ids):
        idx = np.flatnonzero(session_ids == session_id)
        if len(idx) == 0:
            continue
        values = probabilities[idx].astype(np.float64)
        cumulative = np.cumsum(np.insert(values, 0, 0.0))
        result = np.empty_like(values, dtype=np.float64)
        for pos in range(len(values)):
            start = max(0, pos - window_count + 1)
            result[pos] = (cumulative[pos + 1] - cumulative[start]) / (pos - start + 1)
        smoothed[idx] = result
    return smoothed.astype(probabilities.dtype, copy=False)


def score_validation_operating_point(
    metrics: Dict[str, float],
) -> float:
    """Rank validation post-processing without using held-out test labels."""
    event = metrics.get("event_level") or {}
    return float(
        200.0 * event.get("detected_events", 0)
        + 40.0 * event.get("event_sensitivity", 0.0)
        + 20.0 * metrics.get("recall", 0.0)
        + 5.0 * metrics.get("balanced_accuracy", 0.0)
        + metrics.get("auc", 0.0)
        - 0.05 * metrics.get("false_alarms_per_hour", 0.0)
    )


def select_score_smoothing(
    data: Dict[str, np.ndarray],
    split: str,
    probabilities: np.ndarray,
    target_false_alarms_per_hour: float,
    stride_seconds: float,
    mode_name: str,
) -> Tuple[int, np.ndarray, float]:
    """Select a causal smoothing window on validation data only."""
    candidates = [
        int(value)
        for value in os.environ.get(
            "DETECTION_SMOOTHING_WINDOWS", "1,3,5,9,15,25"
        ).split(",")
        if value.strip()
    ]
    if not candidates:
        candidates = [1]
    labels = data[f"labels_{split}"]
    best_window = 1
    best_prob = probabilities.copy()
    best_threshold = threshold_for_false_alarm_rate(
        labels, best_prob, target_false_alarms_per_hour, stride_seconds
    )
    best_metrics = metrics_from_probabilities(
        labels, best_prob, best_threshold, stride_seconds
    )
    event_metrics = event_level_metrics_from_probabilities(
        data, split, best_prob, best_threshold
    )
    if event_metrics is not None:
        best_metrics["event_level"] = event_metrics
    best_score = score_validation_operating_point(best_metrics)

    for window_count in candidates:
        candidate_prob = causal_smooth_probabilities(
            data, split, probabilities, window_count
        )
        candidate_threshold = threshold_for_false_alarm_rate(
            labels, candidate_prob, target_false_alarms_per_hour, stride_seconds
        )
        candidate_metrics = metrics_from_probabilities(
            labels, candidate_prob, candidate_threshold, stride_seconds
        )
        candidate_event = event_level_metrics_from_probabilities(
            data, split, candidate_prob, candidate_threshold
        )
        if candidate_event is not None:
            candidate_metrics["event_level"] = candidate_event
        candidate_score = score_validation_operating_point(candidate_metrics)
        if candidate_score > best_score:
            best_window = window_count
            best_prob = candidate_prob
            best_threshold = candidate_threshold
            best_score = candidate_score

    print(
        f"Selected {mode_name} causal score smoothing: "
        f"{best_window} window(s), threshold={best_threshold:.4f}"
    )
    return best_window, best_prob, float(best_threshold)


def threshold_for_false_alarm_rate(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_false_alarms_per_hour: float,
    stride_seconds: float,
) -> float:
    """Choose the lowest validation threshold that stays within an alarm budget."""
    normal_scores = np.sort(probabilities[labels <= 0.5])[::-1]
    if len(normal_scores) == 0:
        return 0.5
    recording_hours = max(len(labels) * stride_seconds / 3600, 1e-9)
    allowed_fp = int(np.floor(max(target_false_alarms_per_hour, 0.0) * recording_hours))
    if allowed_fp <= 0:
        return float(np.nextafter(np.max(normal_scores), np.inf))
    if allowed_fp >= len(normal_scores):
        return float(np.min(normal_scores))
    return float(normal_scores[allowed_fp - 1])


def calibrate_threshold(
    model: keras.Model,
    primary: np.ndarray,
    secondary: np.ndarray,
    labels: np.ndarray,
    mode_name: str,
) -> float:
    """Choose a decision threshold on validation data using F1 score."""
    probabilities = predict_probabilities(model, primary, secondary)
    return calibrate_threshold_from_probabilities(probabilities, labels, mode_name)


def calibrate_threshold_from_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    mode_name: str,
) -> float:
    """Choose a decision threshold from already-computed validation scores."""
    positives = int(np.sum(labels))
    if positives == 0 or positives == len(labels):
        print(
            f"Threshold calibration skipped for {mode_name}: validation has one class."
        )
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = int(np.nanargmax(f1))
    threshold = float(thresholds[best_idx])
    print(
        f"Calibrated {mode_name} threshold on validation: {threshold:.4f} "
        f"(precision={precision[best_idx]:.4f}, recall={recall[best_idx]:.4f}, f1={f1[best_idx]:.4f})"
    )
    return threshold


def make_secondary_variant(
    secondary: np.ndarray, enabled_features: List[str]
) -> np.ndarray:
    """Zero out secondary channels not available in a sensor configuration."""
    variant = np.zeros_like(secondary)
    for feature in enabled_features:
        if feature not in SECONDARY_FEATURES:
            raise ValueError(f"Unknown secondary feature: {feature}")
        idx = SECONDARY_FEATURES.index(feature)
        variant[:, :, idx] = secondary[:, :, idx]
    return variant


def mask_secondary_data(
    data: Dict[str, np.ndarray],
    enabled_features: List[str],
    variant_name: str,
) -> Dict[str, np.ndarray]:
    """Return a shallow data copy with secondary channels masked for a product tier."""
    masked = dict(data)
    for split in ("train", "val", "test"):
        masked[f"secondary_{split}"] = make_secondary_variant(
            data[f"secondary_{split}"],
            enabled_features,
        )
    masked["enabled_secondary_features"] = list(enabled_features)
    masked["variant_name"] = variant_name
    return masked


def parse_feature_env(name: str, default: List[str]) -> Tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return tuple(default)
    values = []
    for item in raw.split(","):
        item = item.strip().upper()
        if item:
            values.append(item)
    return tuple(values)


def build_detection_model_for_variant(
    variant_name: str,
    secondary_dropout: float = 0.0,
) -> keras.Model:
    """Build and compile a fresh model for one product-tier sensor variant."""
    model = build_dual_branch_tsmixer(
        primary_shape=EXPECTED_PRIMARY_SHAPE,
        secondary_shape=EXPECTED_SECONDARY_SHAPE,
        n_blocks=int(os.environ.get("DETECTION_TSMIXER_BLOCKS", "3")),
        hidden_dim=int(os.environ.get("DETECTION_TSMIXER_HIDDEN_DIM", "48")),
        modality_dropout_prob=secondary_dropout,
        baseline_secondary_features=parse_feature_env(
            "DETECTION_BASELINE_SECONDARY_FEATURES", ["EDA", "TEMP"]
        ),
        booster_secondary_features=parse_feature_env(
            "DETECTION_BOOSTER_SECONDARY_FEATURES", ["BVP", "HR"]
        ),
        patch_size=int(os.environ.get("DETECTION_PATCH_SIZE", "8")),
        patch_embed_dim=int(os.environ.get("DETECTION_PATCH_EMBED_DIM", "8")),
    )
    model = compile_model(model, learning_rate=1e-3)
    print(f"\nBuilt model for {variant_name}")
    profile = model_size_profile(model)
    print(
        f"Model footprint: {profile['parameters']:,} parameters "
        f"(under 100k: {profile['under_100k_parameters']})"
    )
    return model


def validation_pr_auc(
    model: keras.Model,
    data: Dict[str, np.ndarray],
    return_probabilities: bool = False,
):
    """Score a trained variant on validation probabilities for model selection."""
    labels = data["labels_val"]
    if len(np.unique(labels.astype(np.int32))) < 2:
        score = float("nan")
        probabilities = np.zeros(len(labels), dtype=np.float32)
    else:
        probabilities = predict_probabilities(
            model,
            data["primary_val"],
            data["secondary_val"],
        )
        score = float(average_precision_score(labels, probabilities))
    if return_probabilities:
        return score, probabilities
    return score


def validation_allocation_metrics(
    standard_model: keras.Model,
    standard_data: Dict[str, np.ndarray],
    pro_model: keras.Model,
    pro_data: Dict[str, np.ndarray],
    stride_seconds: float,
    standard_probabilities: Optional[np.ndarray] = None,
    pro_probabilities: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Score one Standard/Pro allocation pair on validation only."""
    default_alarm_budget = float(
        os.environ.get("DETECTION_DEFAULT_ALARM_BUDGET_PER_HOUR", "10")
    )
    standard_prob = standard_probabilities
    if standard_prob is None:
        standard_prob = predict_probabilities(
            standard_model,
            standard_data["primary_val"],
            standard_data["secondary_val"],
        )
    pro_prob = pro_probabilities
    if pro_prob is None:
        pro_prob = predict_probabilities(
            pro_model,
            pro_data["primary_val"],
            pro_data["secondary_val"],
        )
    standard_threshold = threshold_for_false_alarm_rate(
        standard_data["labels_val"],
        standard_prob,
        default_alarm_budget,
        stride_seconds,
    )
    standard_metrics = metrics_from_probabilities(
        standard_data["labels_val"],
        standard_prob,
        standard_threshold,
        stride_seconds,
    )
    standard_event_metrics = event_level_metrics_from_probabilities(
        standard_data,
        "val",
        standard_prob,
        standard_threshold,
    )
    if standard_event_metrics:
        standard_metrics["event_level"] = standard_event_metrics
    fusion_weight, fused_pro_prob, pro_metrics = select_pro_fusion(
        pro_data["labels_val"],
        standard_prob,
        pro_prob,
        standard_metrics,
        default_alarm_budget,
        stride_seconds,
        validation_data=pro_data,
    )
    return {
        "standard": standard_metrics,
        "pro": pro_metrics,
        "pro_score_fusion": {
            "policy": (
                "Pro combines the Standard detector score with the add-on detector "
                "score; the blend weight is selected on validation only."
            ),
            "pro_model_weight": float(fusion_weight),
        },
        "default_alarm_budget_per_hour": float(default_alarm_budget),
        "recall_gain": float(pro_metrics["recall"] - standard_metrics["recall"]),
        "auc_gain": float(pro_metrics["auc"] - standard_metrics["auc"]),
        "pr_auc_gain": float(pro_metrics["pr_auc"] - standard_metrics["pr_auc"]),
    }


def allocation_selection_score(
    metrics: Dict[str, Dict[str, float]],
) -> Tuple[float, ...]:
    standard_recall = float(metrics.get("standard", {}).get("recall", 0.0))
    pro_recall = float(metrics.get("pro", {}).get("recall", 0.0))
    standard_events = metrics.get("standard", {}).get("event_level", {}) or {}
    pro_events = metrics.get("pro", {}).get("event_level", {}) or {}
    standard_event_count = int(standard_events.get("detected_events", 0))
    pro_event_count = int(pro_events.get("detected_events", 0))
    standard_balanced_accuracy = float(
        metrics.get("standard", {}).get("balanced_accuracy", 0.0)
    )
    pro_balanced_accuracy = float(metrics.get("pro", {}).get("balanced_accuracy", 0.0))
    standard_false_alarms = float(
        metrics.get("standard", {}).get("false_alarms_per_hour", float("inf"))
    )
    pro_false_alarms = float(
        metrics.get("pro", {}).get("false_alarms_per_hour", float("inf"))
    )
    balanced_accuracy_gain = pro_balanced_accuracy - standard_balanced_accuracy
    event_gain = pro_event_count - standard_event_count
    recall_gain = float(metrics.get("recall_gain", 0.0))
    auc_gain = float(metrics.get("auc_gain", 0.0))
    false_alarm_reduction = standard_false_alarms - pro_false_alarms
    return (
        1.0 if event_gain >= 0 else 0.0,
        event_gain,
        1.0 if recall_gain >= 0 else 0.0,
        1.0 if balanced_accuracy_gain >= 0 else 0.0,
        pro_recall,
        recall_gain,
        balanced_accuracy_gain,
        auc_gain,
        false_alarm_reduction,
        standard_recall,
    )


def pro_fusion_weights() -> List[float]:
    """Candidate Pro score-fusion weights selected using validation only."""
    raw = os.environ.get(
        "DETECTION_PRO_FUSION_WEIGHTS",
        "0.05,0.10,0.15,0.20,0.25,0.35,0.50,0.75,1.0",
    )
    weights: List[float] = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            weight = float(value)
        except ValueError:
            continue
        if 0.0 <= weight <= 1.0:
            weights.append(weight)
    return weights or [1.0]


def blend_pro_scores(
    standard_probabilities: np.ndarray,
    pro_probabilities: np.ndarray,
    pro_weight: float,
) -> np.ndarray:
    """Blend Standard and add-on model scores for the cumulative Pro product."""
    return (
        (1.0 - pro_weight) * standard_probabilities
        + pro_weight * pro_probabilities
    ).astype(np.float32)


def select_pro_fusion(
    labels_val: np.ndarray,
    standard_val_probabilities: np.ndarray,
    pro_val_probabilities: np.ndarray,
    standard_metrics: Dict[str, float],
    alarm_budget_per_hour: float,
    stride_seconds: float,
    validation_data: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """Select the Pro score blend on validation without using held-out test data."""
    best_weight = 1.0
    best_probabilities = pro_val_probabilities.astype(np.float32)
    best_metrics: Optional[Dict[str, float]] = None
    best_score: Optional[Tuple[float, ...]] = None
    for weight in pro_fusion_weights():
        candidate_probabilities = blend_pro_scores(
            standard_val_probabilities,
            pro_val_probabilities,
            weight,
        )
        candidate_threshold = threshold_for_false_alarm_rate(
            labels_val,
            candidate_probabilities,
            alarm_budget_per_hour,
            stride_seconds,
        )
        candidate_metrics = metrics_from_probabilities(
            labels_val,
            candidate_probabilities,
            candidate_threshold,
            stride_seconds,
        )
        if validation_data is not None:
            candidate_event_metrics = event_level_metrics_from_probabilities(
                validation_data,
                "val",
                candidate_probabilities,
                candidate_threshold,
            )
            if candidate_event_metrics:
                candidate_metrics["event_level"] = candidate_event_metrics
        score = allocation_selection_score(
            {
                "standard": standard_metrics,
                "pro": candidate_metrics,
                "recall_gain": candidate_metrics["recall"]
                - standard_metrics["recall"],
                "auc_gain": candidate_metrics["auc"] - standard_metrics["auc"],
                "pr_auc_gain": candidate_metrics["pr_auc"]
                - standard_metrics["pr_auc"],
            }
        )
        if best_score is None or score > best_score:
            best_score = score
            best_weight = weight
            best_probabilities = candidate_probabilities
            best_metrics = candidate_metrics
    if best_metrics is None:
        threshold = threshold_for_false_alarm_rate(
            labels_val,
            best_probabilities,
            alarm_budget_per_hour,
            stride_seconds,
        )
        best_metrics = metrics_from_probabilities(
            labels_val, best_probabilities, threshold, stride_seconds
        )
        if validation_data is not None:
            event_metrics = event_level_metrics_from_probabilities(
                validation_data, "val", best_probabilities, threshold
            )
            if event_metrics:
                best_metrics["event_level"] = event_metrics
    return float(best_weight), best_probabilities, best_metrics


def train_sensor_variant(
    variant_name: str,
    enabled_features: List[str],
    base_data: Dict[str, np.ndarray],
    output_dir: str,
    epochs: int,
    batch_size: int,
) -> Tuple[
    keras.Model, Dict[str, np.ndarray], keras.callbacks.History, float, np.ndarray
]:
    """Train one product-tier model and return its validation PR-AUC."""
    data = mask_secondary_data(base_data, enabled_features, variant_name)
    save_path = os.path.join(output_dir, f"{variant_name}.keras")
    print("\n" + "=" * 70)
    print(f"TRAINING VARIANT: {variant_name}")
    print(
        f"Secondary features: {enabled_features if enabled_features else 'none (ACC-only standard)'}"
    )
    print("=" * 70)

    force_retrain = (
        os.environ.get("DETECTION_FORCE_RETRAIN_VARIANTS", "0") == "1"
        or os.environ.get("FORCE_RERUN", "0") == "1"
    )
    if os.path.exists(save_path) and not force_retrain:
        print(f"Found existing checkpoint for {variant_name}: {save_path}")
        model = keras.models.load_model(
            save_path,
            custom_objects={
                "MLPBlock": MLPBlock,
                "TSMixerBlock": TSMixerBlock,
                "ChannelIndependentTimeMixer": ChannelIndependentTimeMixer,
                "ChannelIndependentTSMixerBlock": ChannelIndependentTSMixerBlock,
                "ChannelPatchEmbedding": ChannelPatchEmbedding,
                "GatedTimeMixerBlock": GatedTimeMixerBlock,
                "GatedBoosterFusion": GatedBoosterFusion,
                "ModalityDropout": ModalityDropout,
                "SecondaryFeatureSelector": SecondaryFeatureSelector,
                "SummaryStats": SummaryStats,
                "SensitivitySpecificityWeightedBinaryCrossentropy": (
                    SensitivitySpecificityWeightedBinaryCrossentropy
                ),
            },
            compile=False,
        )
        if model.name != "PatchedDualStreamTSMixer":
            print(
                "Existing checkpoint uses an older TSMixer architecture; retraining this variant."
            )
        else:
            print("Loading checkpoint and skipping retraining for this variant.")
            model = compile_model(model, learning_rate=1e-3)
            score, validation_prob = validation_pr_auc(
                model, data, return_probabilities=True
            )
            print(f"Validation PR-AUC for {variant_name}: {score:.6f}")
            history = SimpleNamespace(
                history={
                    "resumed_from_checkpoint": [1.0],
                    "trained_from_scratch": [0.0],
                }
            )
            return model, data, history, score, validation_prob
    if os.path.exists(save_path) and force_retrain:
        print(
            f"Existing checkpoint ignored because FORCE_RERUN/DETECTION_FORCE_RETRAIN_VARIANTS is set: {save_path}"
        )
    model = build_detection_model_for_variant(variant_name, secondary_dropout=0.0)
    patience = int(os.environ.get("DETECTION_EARLY_STOPPING_PATIENCE", "5"))
    history = train_model(
        model,
        data,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=patience,
        save_path=save_path,
    )
    score, validation_prob = validation_pr_auc(model, data, return_probabilities=True)
    print(f"Validation PR-AUC for {variant_name}: {score:.6f}")
    history.history.setdefault("resumed_from_checkpoint", [0.0])
    history.history.setdefault("trained_from_scratch", [1.0])
    return model, data, history, score, validation_prob


def run_product_validation(
    standard_model: keras.Model,
    standard_data: Dict[str, np.ndarray],
    pro_model: keras.Model,
    pro_data: Dict[str, np.ndarray],
    pro_variant_name: str,
    stride_seconds: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
    """Compare Standard and Pro product-tier models on held-out sessions."""
    print("\n" + "=" * 70)
    print("VALIDATION: Standard Product vs Pro Product")
    print("=" * 70)
    standard_variant_name = standard_data.get("variant_name", "standard")
    standard_features = standard_data.get("enabled_secondary_features", [])
    standard_mode_name = (
        f"Standard Product ({standard_variant_name}; "
        f"features={standard_features if standard_features else ['ACC only']})"
    )

    standard_val_prob = predict_probabilities(
        standard_model,
        standard_data["primary_val"],
        standard_data["secondary_val"],
    )
    pro_val_prob = predict_probabilities(
        pro_model,
        pro_data["primary_val"],
        pro_data["secondary_val"],
    )

    standard_f1_threshold = calibrate_threshold_from_probabilities(
        standard_val_prob,
        standard_data["labels_val"],
        standard_mode_name,
    )
    pro_f1_threshold = calibrate_threshold_from_probabilities(
        pro_val_prob,
        pro_data["labels_val"],
        f"Pro Product ({pro_variant_name})",
    )

    standard_test_prob = predict_probabilities(
        standard_model,
        standard_data["primary_test"],
        standard_data["secondary_test"],
    )
    pro_test_prob = predict_probabilities(
        pro_model,
        pro_data["primary_test"],
        pro_data["secondary_test"],
    )
    default_alarm_budget = float(
        os.environ.get("DETECTION_DEFAULT_ALARM_BUDGET_PER_HOUR", "10")
    )
    (
        standard_smoothing_windows,
        standard_val_prob,
        standard_threshold,
    ) = select_score_smoothing(
        standard_data,
        "val",
        standard_val_prob,
        default_alarm_budget,
        stride_seconds,
        "Standard Product",
    )
    standard_test_prob = causal_smooth_probabilities(
        standard_data, "test", standard_test_prob, standard_smoothing_windows
    )
    standard_threshold = threshold_for_false_alarm_rate(
        standard_data["labels_val"],
        standard_val_prob,
        default_alarm_budget,
        stride_seconds,
    )
    standard_val_metrics_for_fusion = metrics_from_probabilities(
        standard_data["labels_val"],
        standard_val_prob,
        standard_threshold,
        stride_seconds,
    )
    fusion_weight, pro_val_prob, _ = select_pro_fusion(
        pro_data["labels_val"],
        standard_val_prob,
        pro_val_prob,
        standard_val_metrics_for_fusion,
        default_alarm_budget,
        stride_seconds,
        validation_data=pro_data,
    )
    pro_test_prob = blend_pro_scores(standard_test_prob, pro_test_prob, fusion_weight)
    pro_smoothing_windows, pro_val_prob, pro_threshold = select_score_smoothing(
        pro_data,
        "val",
        pro_val_prob,
        default_alarm_budget,
        stride_seconds,
        "Pro Product",
    )
    pro_test_prob = causal_smooth_probabilities(
        pro_data, "test", pro_test_prob, pro_smoothing_windows
    )
    pro_f1_threshold = calibrate_threshold_from_probabilities(
        pro_val_prob,
        pro_data["labels_val"],
        f"Pro Product ({pro_variant_name}; fused score)",
    )
    pro_threshold = threshold_for_false_alarm_rate(
        pro_data["labels_val"],
        pro_val_prob,
        default_alarm_budget,
        stride_seconds,
    )
    standard_val_false_alarms = false_alarm_rate_at_threshold(
        standard_data["labels_val"],
        standard_val_prob,
        standard_threshold,
        stride_seconds,
    )
    pro_matched_threshold = threshold_for_false_alarm_rate(
        pro_data["labels_val"],
        pro_val_prob,
        standard_val_false_alarms,
        stride_seconds,
    )

    standard_results = metrics_from_probabilities(
        standard_data["labels_test"],
        standard_test_prob,
        standard_threshold,
        stride_seconds,
    )
    standard_event_metrics = event_level_metrics_from_probabilities(
        standard_data, "test", standard_test_prob, standard_threshold
    )
    if standard_event_metrics is not None:
        standard_results["event_level"] = standard_event_metrics
    standard_alert_selection = select_event_alert_postprocessing(
        standard_data,
        "val",
        standard_val_prob,
        standard_threshold,
        stride_seconds,
    )
    standard_alert_metrics = event_alert_metrics_from_probabilities(
        standard_data,
        "test",
        standard_test_prob,
        standard_threshold,
        stride_seconds,
        float(standard_alert_selection.get("refractory_minutes", 0.0)),
    )
    if standard_alert_metrics is not None:
        standard_results["event_alert_level"] = standard_alert_metrics
    print_evaluation_results(standard_mode_name, standard_results)
    standard_results["threshold_policy"] = (
        f"validation false-alarm budget <= {default_alarm_budget:g}/hour"
    )
    standard_results["f1_calibrated_threshold"] = float(standard_f1_threshold)
    standard_results["variant_name"] = standard_data.get("variant_name", "standard")
    standard_results["enabled_secondary_features"] = standard_data.get(
        "enabled_secondary_features", []
    )
    standard_results["post_processing"] = {
        "score_smoothing": "causal rolling mean",
        "selected_windows": int(standard_smoothing_windows),
        "alert_refractory_minutes": float(
            standard_alert_selection.get("refractory_minutes", 0.0)
        ),
        "selected_on": "validation split",
    }
    pro_results = metrics_from_probabilities(
        pro_data["labels_test"],
        pro_test_prob,
        pro_threshold,
        stride_seconds,
    )
    pro_event_metrics = event_level_metrics_from_probabilities(
        pro_data, "test", pro_test_prob, pro_threshold
    )
    if pro_event_metrics is not None:
        pro_results["event_level"] = pro_event_metrics
    pro_alert_selection = select_event_alert_postprocessing(
        pro_data,
        "val",
        pro_val_prob,
        pro_threshold,
        stride_seconds,
    )
    pro_alert_metrics = event_alert_metrics_from_probabilities(
        pro_data,
        "test",
        pro_test_prob,
        pro_threshold,
        stride_seconds,
        float(pro_alert_selection.get("refractory_minutes", 0.0)),
    )
    if pro_alert_metrics is not None:
        pro_results["event_alert_level"] = pro_alert_metrics
    print_evaluation_results(f"Pro Product ({pro_variant_name})", pro_results)
    pro_results["threshold_policy"] = (
        f"validation false-alarm budget <= {default_alarm_budget:g}/hour"
    )
    pro_results["f1_calibrated_threshold"] = float(pro_f1_threshold)
    pro_results["variant_name"] = pro_variant_name
    pro_results["enabled_secondary_features"] = pro_data.get(
        "enabled_secondary_features", []
    )
    pro_results["score_fusion"] = {
        "policy": (
            "Pro combines the Standard detector score with the add-on detector "
            "score; the blend weight is selected on validation only."
        ),
        "pro_model_weight": float(fusion_weight),
    }
    pro_results["post_processing"] = {
        "score_smoothing": "causal rolling mean",
        "selected_windows": int(pro_smoothing_windows),
        "alert_refractory_minutes": float(
            pro_alert_selection.get("refractory_minutes", 0.0)
        ),
        "selected_on": "validation split",
    }
    matched_alarm_results = {
        "policy": (
            "Pro threshold is calibrated on validation to match the Standard "
            "validation false-alarm/hour budget; both are then evaluated on "
            "held-out test sessions."
        ),
        "standard_validation_false_alarms_per_hour": float(standard_val_false_alarms),
    }
    matched_alarm_results["standard"] = metrics_from_probabilities(
        standard_data["labels_test"],
        standard_test_prob,
        standard_threshold,
        stride_seconds,
    )
    matched_standard_event_metrics = event_level_metrics_from_probabilities(
        standard_data, "test", standard_test_prob, standard_threshold
    )
    if matched_standard_event_metrics is not None:
        matched_alarm_results["standard"]["event_level"] = (
            matched_standard_event_metrics
        )
    matched_standard_alert = event_alert_metrics_from_probabilities(
        standard_data,
        "test",
        standard_test_prob,
        standard_threshold,
        stride_seconds,
        float(standard_alert_selection.get("refractory_minutes", 0.0)),
    )
    if matched_standard_alert is not None:
        matched_alarm_results["standard"]["event_alert_level"] = matched_standard_alert
    matched_alarm_results["standard"]["post_processing"] = standard_results[
        "post_processing"
    ]
    matched_alarm_results["pro"] = metrics_from_probabilities(
        pro_data["labels_test"],
        pro_test_prob,
        pro_matched_threshold,
        stride_seconds,
    )
    matched_pro_event_metrics = event_level_metrics_from_probabilities(
        pro_data, "test", pro_test_prob, pro_matched_threshold
    )
    if matched_pro_event_metrics is not None:
        matched_alarm_results["pro"]["event_level"] = matched_pro_event_metrics
    matched_pro_alert = event_alert_metrics_from_probabilities(
        pro_data,
        "test",
        pro_test_prob,
        pro_matched_threshold,
        stride_seconds,
        float(pro_alert_selection.get("refractory_minutes", 0.0)),
    )
    if matched_pro_alert is not None:
        matched_alarm_results["pro"]["event_alert_level"] = matched_pro_alert
    matched_alarm_results["pro"]["variant_name"] = pro_variant_name
    matched_alarm_results["pro"]["enabled_secondary_features"] = pro_data.get(
        "enabled_secondary_features", []
    )
    matched_alarm_results["pro"]["score_fusion"] = pro_results["score_fusion"]
    matched_alarm_results["pro"]["post_processing"] = pro_results["post_processing"]
    matched_alarm_results["default_operating_budget_per_hour"] = float(
        default_alarm_budget
    )

    alarm_budgets = [
        float(value)
        for value in os.environ.get(
            "DETECTION_ALARM_BUDGETS_PER_HOUR", "1,5,10,25,50"
        ).split(",")
        if value.strip()
    ]
    matched_alarm_results["fixed_validation_alarm_budgets"] = {
        "policy": (
            "Each model gets its own threshold from validation to stay within "
            "the same false-alarms/hour budget, then both are evaluated on "
            "held-out test sessions."
        ),
        "budgets_per_hour": alarm_budgets,
        "standard": evaluate_alarm_budget_grid(
            standard_data["labels_val"],
            standard_val_prob,
            standard_data["labels_test"],
            standard_test_prob,
            alarm_budgets,
            stride_seconds,
            standard_data,
            float(standard_alert_selection.get("refractory_minutes", 0.0)),
        ),
        "pro": evaluate_alarm_budget_grid(
            pro_data["labels_val"],
            pro_val_prob,
            pro_data["labels_test"],
            pro_test_prob,
            alarm_budgets,
            stride_seconds,
            pro_data,
            float(pro_alert_selection.get("refractory_minutes", 0.0)),
        ),
    }

    print("\n" + "=" * 70)
    print("COMPARISON: Standard vs Pro Product")
    print("=" * 70)
    auc_diff = pro_results["auc"] - standard_results["auc"]
    pr_auc_diff = pro_results["pr_auc"] - standard_results["pr_auc"]
    print(
        f"  AUC: {standard_results['auc']:.4f} -> {pro_results['auc']:.4f} (+{auc_diff:.4f})"
    )
    print(
        f"  PR-AUC: {standard_results['pr_auc']:.4f} -> {pro_results['pr_auc']:.4f} (+{pr_auc_diff:.4f})"
    )
    print(
        f"  Recall: {standard_results['recall']:.4f} -> {pro_results['recall']:.4f} "
        f"(+{pro_results['recall'] - standard_results['recall']:.4f})"
    )
    standard_events = standard_results.get("event_level", {})
    pro_events = pro_results.get("event_level", {})
    if standard_events and pro_events:
        print(
            "  Event sensitivity: "
            f"{standard_events.get('detected_events', 0)}/{standard_events.get('event_count', 0)} -> "
            f"{pro_events.get('detected_events', 0)}/{pro_events.get('event_count', 0)}"
        )
    standard_alerts = standard_results.get("event_alert_level", {})
    pro_alerts = pro_results.get("event_alert_level", {})
    if standard_alerts and pro_alerts:
        print(
            "  Alert-level event sensitivity: "
            f"{standard_alerts.get('detected_events', 0)}/{standard_alerts.get('event_count', 0)} -> "
            f"{pro_alerts.get('detected_events', 0)}/{pro_alerts.get('event_count', 0)}"
        )
        print(
            "  Alert-level false alerts/hour: "
            f"{standard_alerts.get('false_alerts_per_hour', 0.0):.2f} -> "
            f"{pro_alerts.get('false_alerts_per_hour', 0.0):.2f}"
        )
    print(
        f"  False alarms/hour: {standard_results['false_alarms_per_hour']:.2f} -> "
        f"{pro_results['false_alarms_per_hour']:.2f}"
    )
    print(
        "  Raw accuracy is diagnostic only under severe class imbalance; "
        "see all_normal_baseline_accuracy in results.json."
    )

    print("\nMatched false-alarm operating point:")
    print(
        f"  Validation alarm budget: {standard_val_false_alarms:.2f} false alarms/hour"
    )
    print(
        f"  Test recall: {matched_alarm_results['standard']['recall']:.4f} -> "
        f"{matched_alarm_results['pro']['recall']:.4f}"
    )
    print(
        f"  Test false alarms/hour: "
        f"{matched_alarm_results['standard']['false_alarms_per_hour']:.2f} -> "
        f"{matched_alarm_results['pro']['false_alarms_per_hour']:.2f}"
    )

    print("\nFixed validation alarm-budget grid:")
    for budget in alarm_budgets:
        key = str(budget)
        standard_budget = matched_alarm_results["fixed_validation_alarm_budgets"][
            "standard"
        ][key]
        pro_budget = matched_alarm_results["fixed_validation_alarm_budgets"]["pro"][key]
        standard_budget_event = standard_budget.get("event_level", {})
        pro_budget_event = pro_budget.get("event_level", {})
        event_text = ""
        if standard_budget_event and pro_budget_event:
            event_text = (
                f"; events {standard_budget_event.get('detected_events', 0)}/"
                f"{standard_budget_event.get('event_count', 0)} -> "
                f"{pro_budget_event.get('detected_events', 0)}/"
                f"{pro_budget_event.get('event_count', 0)}"
            )
        print(
            f"  {budget:g} fa/h budget | recall "
            f"{standard_budget['recall']:.4f} -> {pro_budget['recall']:.4f}; "
            f"test fa/h {standard_budget['false_alarms_per_hour']:.2f} -> "
            f"{pro_budget['false_alarms_per_hour']:.2f}; "
            f"balanced acc {standard_budget['balanced_accuracy']:.4f} -> "
            f"{pro_budget['balanced_accuracy']:.4f}"
            f"{event_text}"
        )
    return standard_results, pro_results, matched_alarm_results


def evaluate_sensor_variants(
    model: keras.Model,
    data: Dict[str, np.ndarray],
    stride_seconds: float,
) -> Dict[str, Dict[str, float]]:
    """Evaluate realistic secondary-sensor ablations using one trained model."""
    variants = {
        "standard_acc_only": {
            "label": "Standard ACC only",
            "features": [],
            "physical_meaning": "base device only",
        },
        "addon_bvp_only": {
            "label": "Add-on BVP only",
            "features": ["BVP"],
            "physical_meaning": "PPG waveform only; HR disabled",
        },
        "addon_bvp_hr": {
            "label": "Add-on BVP + HR",
            "features": ["BVP", "HR"],
            "physical_meaning": "PPG-derived BVP and HR",
        },
        "addon_eda_temp": {
            "label": "Add-on EDA + TEMP only",
            "features": ["EDA", "TEMP"],
            "physical_meaning": "non-PPG add-on physiology only",
        },
        "addon_bvp_eda_temp_no_hr": {
            "label": "Add-on BVP + EDA + TEMP (no HR)",
            "features": ["BVP", "EDA", "TEMP"],
            "physical_meaning": "PPG waveform plus EDA/TEMP; HR channel disabled",
        },
        "pro_full_bvp_hr_eda_temp": {
            "label": "Pro full BVP + HR + EDA + TEMP",
            "features": ["BVP", "HR", "EDA", "TEMP"],
            "physical_meaning": "all available add-on channels except IBI",
        },
    }

    print("\n" + "=" * 70)
    print("SENSOR ABLATION: Which Add-on Channels Help?")
    print("=" * 70)

    results = {}
    for key, config in variants.items():
        val_secondary = make_secondary_variant(
            data["secondary_val"], config["features"]
        )
        test_secondary = make_secondary_variant(
            data["secondary_test"], config["features"]
        )
        threshold = calibrate_threshold(
            model,
            data["primary_val"],
            val_secondary,
            data["labels_val"],
            config["label"],
        )
        metrics = evaluate_model(
            model,
            data["primary_test"],
            test_secondary,
            data["labels_test"],
            mode_name=config["label"],
            stride_seconds=stride_seconds,
            threshold=threshold,
        )
        metrics["enabled_secondary_features"] = list(config["features"])
        metrics["physical_meaning"] = config["physical_meaning"]
        results[key] = metrics
    print("\nSensor ablation summary:")
    print("  variant | AUC | PR-AUC | F1 | recall | false alarms/hour")
    for key, metrics in results.items():
        print(
            f"  {key}: auc={metrics['auc']:.4f}, pr_auc={metrics['pr_auc']:.4f}, "
            f"f1={metrics['f1']:.4f}, recall={metrics['recall']:.4f}, "
            f"fa/h={metrics['false_alarms_per_hour']:.2f}"
        )
    candidate_keys = [key for key in results if key != "standard_acc_only"]
    best_key = max(
        candidate_keys,
        key=lambda key: (
            np.nan_to_num(results[key]["pr_auc"], nan=-1.0),
            np.nan_to_num(results[key]["auc"], nan=-1.0),
            results[key]["f1"],
        ),
    )
    results["recommended_pro_variant"] = {
        "variant": best_key,
        "reason": "highest PR-AUC among add-on variants, then AUC/F1 as tie-breakers",
        "metrics": results[best_key],
    }
    print(f"\nRecommended Pro variant: {best_key}")
    return results


def run_validation(
    model: keras.Model,
    data: Dict[str, np.ndarray],
    stride_seconds: float = 0.5,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Run validation on both Standard and Pro modes.

    Standard Mode: Secondary input is all zeros (simulates watch-only)
    Pro Mode: Secondary input is real data (full sensor kit)

    Args:
        model: Trained model
        data: Data dictionary with test splits

    Returns:
        (standard_results, pro_results)
    """
    print("\n" + "=" * 70)
    print("VALIDATION: Standard Mode vs Pro Mode")
    print("=" * 70)

    ablation_results = evaluate_sensor_variants(model, data, stride_seconds)
    standard_results = ablation_results["standard_acc_only"]
    recommended = ablation_results["recommended_pro_variant"]["variant"]
    pro_results = ablation_results[recommended]

    print("\n" + "=" * 70)
    print("COMPARISON: Recommended Add-on Sensor Utility")
    print("=" * 70)

    auc_diff = pro_results["auc"] - standard_results["auc"]

    print(
        f"  AUC: {standard_results['auc']:.4f} -> {pro_results['auc']:.4f} "
        f"(+{auc_diff:.4f})"
    )
    print(
        f"  PR-AUC: {standard_results['pr_auc']:.4f} -> {pro_results['pr_auc']:.4f} "
        f"(+{pro_results['pr_auc'] - standard_results['pr_auc']:.4f})"
    )
    print(
        f"  Recall: {standard_results['recall']:.4f} -> {pro_results['recall']:.4f} "
        f"(+{pro_results['recall'] - standard_results['recall']:.4f})"
    )
    print(
        f"  False alarms/hour: {standard_results['false_alarms_per_hour']:.2f} -> "
        f"{pro_results['false_alarms_per_hour']:.2f}"
    )
    print("  Raw accuracy omitted from this comparison because event windows are rare.")

    return standard_results, pro_results, ablation_results


def export_tflite(
    model: keras.Model,
    save_path: str,
    quantize: bool = True,
    representative_data: Optional[
        Union[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ] = None,
    representative_sample_count: int = 100,
) -> str:
    """
    Export model to TensorFlow Lite format.

    Args:
        model: Trained Keras model
        save_path: Path to save .tflite file
        quantize: Whether to apply int8 quantization
        representative_data: Calibration data for full integer quantization.
            Accepts either the training data dictionary returned by prepare_data,
            or a (primary, secondary) tuple.
        representative_sample_count: Maximum number of samples to use for
            calibration.

    Returns:
        Path to saved file
    """
    if not isinstance(quantize, (bool, np.bool_)):
        if representative_data is None:
            raise ValueError(
                "Legacy export_tflite calls must pass both primary and secondary "
                "representative arrays."
            )
        representative_data = (quantize, representative_data)
        quantize = True
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    calibration_inputs = None

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        calibration_inputs = _get_representative_inputs(representative_data)
        if calibration_inputs is not None:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            converter.representative_dataset = _build_representative_dataset(
                *calibration_inputs,
                sample_count=representative_sample_count,
            )
        else:
            print(
                "Warning: representative_data was not provided; exporting with "
                "dynamic-range quantization instead of full integer quantization."
            )
    tflite_model = converter.convert()

    with open(save_path, "wb") as f:
        f.write(tflite_model)
    if not _tflite_smoke_invoke(save_path):
        raise RuntimeError(f"TFLite smoke inference failed: {save_path}")
    size_kb = len(tflite_model) / 1024
    print(f"Exported TFLite model: {save_path} ({size_kb:.1f} KB)")

    return save_path


def _tflite_smoke_invoke(path: str) -> bool:
    edge_script = os.path.join(os.path.dirname(__file__), "edge_feasibility.py")
    command = [
        sys.executable,
        edge_script,
        "--benchmark-one",
        path,
        "--runs",
        "1",
        "--warmup",
        "0",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    try:
        completed = subprocess.run(
            command,
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout)[-1000:]
        print(f"Warning: TFLite smoke inference failed with code {completed.returncode}.")
        if tail:
            print(tail)
        return False
    return True


def _get_representative_inputs(
    representative_data: Optional[
        Union[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract primary and secondary arrays for TFLite calibration."""
    if representative_data is None:
        return None
    if isinstance(representative_data, dict):
        primary = representative_data.get("primary_train")
        secondary = representative_data.get("secondary_train")
    else:
        try:
            primary, secondary = representative_data
        except (TypeError, ValueError):
            raise ValueError(
                "representative_data must be a prepare_data dictionary or "
                "a (primary, secondary) tuple."
            )
    if primary is None or secondary is None:
        raise ValueError(
            "representative_data must include primary and secondary training arrays."
        )
    if len(primary) == 0 or len(secondary) == 0:
        raise ValueError("representative_data arrays must not be empty.")
    if len(primary) != len(secondary):
        raise ValueError(
            "representative_data primary and secondary arrays must contain the "
            "same number of samples."
        )
    return primary, secondary


def _build_representative_dataset(
    primary: np.ndarray,
    secondary: np.ndarray,
    sample_count: int = 100,
):
    """Create a calibration generator for the model's two input branches."""
    max_samples = min(int(sample_count), len(primary), len(secondary))

    def representative_dataset():
        for idx in range(max_samples):
            yield [
                np.asarray(primary[idx : idx + 1], dtype=np.float32),
                np.asarray(secondary[idx : idx + 1], dtype=np.float32),
            ]

    return representative_dataset


def get_default_base_path():
    """Get the default base path based on platform."""
    import platform

    colab_paths = [
        "/content/drive/MyDrive/I2I",
        "/content/I2I",
    ]

    for path in colab_paths:
        if os.path.exists(path):
            return path
    if platform.system() == "Windows":
        return r"C:\I2I"
    return os.path.expanduser("~/I2I")


def _detection_cache_key(
    session_paths,
    seizure_intervals: Dict[str, List[Tuple[float, float]]],
    window_size: int,
    stride: int,
    max_sessions: int,
) -> str:
    """Build a stable cache key for processed detection arrays."""
    payload = {
        "session_paths": [os.path.abspath(path) for path in session_paths],
        "seizure_intervals": {
            key: [[float(start), float(end)] for start, end in value]
            for key, value in sorted(seizure_intervals.items())
        },
        "window_size": int(window_size),
        "stride": int(stride),
        "max_sessions": int(max_sessions),
        "pipeline_version": PIPELINE_VERSION,
        "primary_shape": list(EXPECTED_PRIMARY_SHAPE),
        "secondary_shape": list(EXPECTED_SECONDARY_SHAPE),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_detection_cache(cache_dir: str, cache_key: str):
    """Load cached detection arrays when available."""
    cache_path = os.path.join(cache_dir, f"detection_cache_{cache_key}.npz")
    if not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            primary = cached["primary"]
            secondary = cached["secondary"]
            labels = cached["labels"]
        if (
            primary.shape[1:] != EXPECTED_PRIMARY_SHAPE
            or secondary.shape[1:] != EXPECTED_SECONDARY_SHAPE
        ):
            print(
                "Detection cache shape mismatch, rebuilding dataset: "
                f"primary={primary.shape}, secondary={secondary.shape}"
            )
            return None
        print(f"Loaded cached detection dataset: {cache_path}")
        print(f"  Primary shape: {primary.shape}")
        print(f"  Secondary shape: {secondary.shape}")
        print(f"  Labels shape: {labels.shape}")
        return primary, secondary, labels
    except Exception as exc:
        print(f"Detection cache load failed, rebuilding dataset: {exc}")
        return None


def save_detection_cache(
    cache_dir: str,
    cache_key: str,
    primary: np.ndarray,
    secondary: np.ndarray,
    labels: np.ndarray,
    session_paths,
):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"detection_cache_{cache_key}.npz")
    metadata_path = cache_path.replace(".npz", ".json")

    np.savez(cache_path, primary=primary, secondary=secondary, labels=labels)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "cache_key": cache_key,
        "num_sessions": len(session_paths),
        "num_samples": int(len(labels)),
        "primary_shape": list(primary.shape),
        "secondary_shape": list(secondary.shape),
        "pipeline_version": PIPELINE_VERSION,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved cached detection dataset: {cache_path}")


def build_detection_split(
    split_name: str,
    session_paths: List[str],
    seizure_intervals: Dict[str, List[Tuple[float, float]]],
    window_size: int,
    stride: int,
    batch_size: int,
):
    """Build unnormalized windows for one session split."""
    if not session_paths:
        raise ValueError(f"No sessions available for {split_name} split.")
    from phase2_data_generator_fast import SeizureDataGenFast

    print(
        f"\nBuilding {split_name} split from {len(session_paths)} held-out sessions..."
    )
    generator = SeizureDataGenFast(
        session_paths=session_paths,
        seizure_intervals=seizure_intervals,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        shuffle=False,
        n_workers=None,
        prioritize_seizures=False,
    )
    primary, secondary, labels = generator.get_data()
    metadata = generator.get_metadata()
    metadata["events"] = split_events(session_paths, seizure_intervals)
    return primary, secondary, labels, metadata


def build_session_level_data(
    session_splits: Dict[str, List[str]],
    seizure_intervals: Dict[str, List[Tuple[float, float]]],
    window_size: int,
    stride: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """Build and normalize train/val/test arrays from session-level splits."""
    data = {}
    for split_name in ("train", "val", "test"):
        primary, secondary, labels, metadata = build_detection_split(
            split_name,
            session_splits[split_name],
            seizure_intervals,
            window_size,
            stride,
            batch_size,
        )
        data[f"primary_{split_name}"] = primary
        data[f"secondary_{split_name}"] = secondary
        data[f"labels_{split_name}"] = labels
        data[f"window_session_ids_{split_name}"] = metadata["session_ids"]
        data[f"window_starts_{split_name}"] = metadata["window_starts"]
        data[f"window_ends_{split_name}"] = metadata["window_ends"]
        data[f"events_{split_name}"] = metadata["events"]
    return normalize_session_splits(data)


def main():
    """Main training and validation pipeline."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    random_seed = int(os.environ.get("DETECTION_RANDOM_SEED", "42"))
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    from phase1_data_health import run_deep_analysis, build_seizure_intervals_dict
    from phase2_data_generator_fast import SeizureDataGenFast, get_session_paths_smart
    from phase3_tsmixer_model import build_dual_branch_tsmixer, compile_model

    base_path = get_default_base_path()
    output_dir = os.path.join(base_path, "seizure_detection", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU enabled: {len(gpus)} device(s) available")
        except RuntimeError as e:
            print(f"GPU config warning: {e}")
    print("=" * 70)
    print("SEIZURE DETECTION PIPELINE - Full Training Run")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Base path: {base_path}")
    print(f"Output directory: {output_dir}")
    print(f"GPU available: {len(gpus) > 0}")

    print("\n" + "=" * 70)
    print("PHASE 1: Data Health Check")
    print("=" * 70)
    all_patients = run_deep_analysis(base_path)
    seizure_intervals = build_seizure_intervals_dict(all_patients)

    print("\n" + "=" * 70)
    print("PHASE 2: Data Generator")
    print("=" * 70)

    try:
        from colab_config import get_training_config

        config = get_training_config()
        max_sessions = config["max_sessions"]
        batch_size = config["batch_size"]
        epochs = config["epochs"]
        stride = config.get("window_stride", 32)
    except ImportError:

        max_sessions = int(
            os.environ.get("DETECTION_MAX_SESSIONS", "120" if gpus else "80")
        )
        batch_size = int(
            os.environ.get("DETECTION_BATCH_SIZE", "256" if gpus else "64")
        )
        epochs = int(os.environ.get("DETECTION_EPOCHS", "25" if gpus else "18"))
        stride = int(os.environ.get("DETECTION_STRIDE", str(DETECTION_WINDOW_SIZE)))
    session_paths = get_session_paths_smart(
        base_path,
        seizure_intervals,
        max_sessions=max_sessions,
        prioritize_seizures=True,
    )
    print(f"Found {len(session_paths)} selected valid sessions")
    print(f"Using {len(session_paths)} sessions (max: {max_sessions})")
    print(
        f"Batch size: {batch_size}, Epochs: {epochs}, "
        f"Window: {DETECTION_WINDOW_SIZE} samples ({DETECTION_WINDOW_SIZE / TARGET_RATE_HZ:.1f}s), "
        f"Stride: {stride} samples ({stride / TARGET_RATE_HZ:.1f}s)"
    )

    selected_session_paths = session_paths
    try:
        from phase2_data_generator_fast import get_actual_session_intervals
    except ImportError:
        from .phase2_data_generator_fast import get_actual_session_intervals
    session_splits = split_patient_paths(selected_session_paths, seizure_intervals)
    split_audit = summarize_session_splits(session_splits, seizure_intervals)
    print("\nPatient-held-out splits:")
    for split_name, paths in session_splits.items():
        n_seizure_sessions = sum(
            len(get_actual_session_intervals(path, seizure_intervals)) > 0
            for path in paths
        )
        print(
            f"  {split_name.title()}: {len(paths)} sessions ({n_seizure_sessions} with seizures)"
        )
    print_split_audit(split_audit)

    data = build_session_level_data(
        session_splits,
        seizure_intervals,
        window_size=DETECTION_WINDOW_SIZE,
        stride=stride,
        batch_size=batch_size,
    )

    print(f"\nData splits:")
    print_label_counts(data)

    print("\n" + "=" * 70)
    print("PHASE 3: Build Standard and Pro TSMixer Models")
    print("=" * 70)

    variant_training = {}
    standard_candidates = {}
    for variant_name, enabled_features in STANDARD_CANDIDATE_FEATURES.items():
        (
            standard_model_candidate,
            standard_data_candidate,
            standard_history,
            standard_val_pr_auc,
            standard_val_probabilities,
        ) = train_sensor_variant(
            variant_name,
            enabled_features,
            data,
            output_dir,
            epochs,
            batch_size,
        )
        standard_candidates[variant_name] = {
            "model": standard_model_candidate,
            "data": standard_data_candidate,
            "validation_pr_auc": float(standard_val_pr_auc),
            "validation_probabilities": standard_val_probabilities,
            "enabled_secondary_features": list(enabled_features),
            "history": {
                k: [float(v) for v in vals]
                for k, vals in standard_history.history.items()
            },
        }
        variant_training[variant_name] = {
            "enabled_secondary_features": list(enabled_features),
            "validation_pr_auc": float(standard_val_pr_auc),
            "history": {
                k: [float(v) for v in vals]
                for k, vals in standard_history.history.items()
            },
        }
    required_pro_variants = {
        allocation["pro_variant"] for allocation in PRODUCT_ALLOCATION_CANDIDATES.values()
    }
    pro_candidates = {}
    for variant_name, enabled_features in PRO_CANDIDATE_FEATURES.items():
        if variant_name not in required_pro_variants:
            print(
                "Inactive Pro candidate not trained for the current product "
                f"allocation: {variant_name}"
            )
            continue
        (
            pro_model_candidate,
            pro_data_candidate,
            pro_history,
            pro_val_pr_auc,
            pro_val_probabilities,
        ) = train_sensor_variant(
            variant_name,
            enabled_features,
            data,
            output_dir,
            epochs,
            batch_size,
        )
        pro_candidates[variant_name] = {
            "model": pro_model_candidate,
            "data": pro_data_candidate,
            "validation_pr_auc": float(pro_val_pr_auc),
            "validation_probabilities": pro_val_probabilities,
            "enabled_secondary_features": list(enabled_features),
            "history": {
                k: [float(v) for v in vals] for k, vals in pro_history.history.items()
            },
        }
        variant_training[variant_name] = {
            "enabled_secondary_features": list(enabled_features),
            "validation_pr_auc": float(pro_val_pr_auc),
            "history": {
                k: [float(v) for v in vals] for k, vals in pro_history.history.items()
            },
        }
    allocation_analysis = {}
    for allocation_name, allocation in PRODUCT_ALLOCATION_CANDIDATES.items():
        standard_variant = allocation["standard_variant"]
        pro_variant = allocation["pro_variant"]
        print(
            f"Scoring allocation {allocation_name}: "
            f"{standard_variant} -> {pro_variant}",
            flush=True,
        )
        metrics = validation_allocation_metrics(
            standard_candidates[standard_variant]["model"],
            standard_candidates[standard_variant]["data"],
            pro_candidates[pro_variant]["model"],
            pro_candidates[pro_variant]["data"],
            stride_seconds=stride / 32,
            standard_probabilities=standard_candidates[standard_variant][
                "validation_probabilities"
            ],
            pro_probabilities=pro_candidates[pro_variant]["validation_probabilities"],
        )
        allocation_analysis[allocation_name] = {
            **allocation,
            "validation_metrics": metrics,
            "selection_score": list(allocation_selection_score(metrics)),
        }
    print("\nProduct allocation analysis:")
    for allocation_name, allocation in allocation_analysis.items():
        metrics = allocation["validation_metrics"]
        standard_event = metrics["standard"].get("event_level", {}) or {}
        pro_event = metrics["pro"].get("event_level", {}) or {}
        print(
            f"  {allocation_name}: base optional={allocation['base_optional']}, "
            f"addon optional={allocation['addon_optional']} | "
            f"val events {standard_event.get('detected_events', 0)}/"
            f"{standard_event.get('event_count', 0)} -> "
            f"{pro_event.get('detected_events', 0)}/"
            f"{pro_event.get('event_count', 0)}, "
            f"val recall {metrics['standard']['recall']:.4f} -> {metrics['pro']['recall']:.4f}, "
            f"AUC {metrics['standard']['auc']:.4f} -> {metrics['pro']['auc']:.4f}, "
            f"PR-AUC {metrics['standard']['pr_auc']:.4f} -> {metrics['pro']['pr_auc']:.4f}, "
            f"balanced acc {metrics['standard']['balanced_accuracy']:.4f} -> "
            f"{metrics['pro']['balanced_accuracy']:.4f}"
        )
    best_allocation = max(
        allocation_analysis,
        key=lambda key: tuple(allocation_analysis[key]["selection_score"]),
    )
    print(f"\nSelected product allocation from validation only: {best_allocation}")
    print("\nHeld-out test allocation check:")
    allocation_test_results = {}
    for allocation_name, allocation in allocation_analysis.items():
        standard_variant = allocation["standard_variant"]
        pro_variant = allocation["pro_variant"]
        standard_model = standard_candidates[standard_variant]["model"]
        standard_data = standard_candidates[standard_variant]["data"]
        pro_model = pro_candidates[pro_variant]["model"]
        pro_data = pro_candidates[pro_variant]["data"]
        standard_results, pro_results, matched_alarm_results = run_product_validation(
            standard_model,
            standard_data,
            pro_model,
            pro_data,
            pro_variant,
            stride_seconds=stride / 32,
        )
        test_metrics = {
            "standard": matched_alarm_results["standard"],
            "pro": matched_alarm_results["pro"],
            "default_alarm_budget_per_hour": float(
                matched_alarm_results.get(
                    "standard_validation_false_alarms_per_hour", 10.0
                )
            ),
            "recall_gain": float(
                matched_alarm_results["pro"]["recall"]
                - matched_alarm_results["standard"]["recall"]
            ),
            "auc_gain": float(pro_results["auc"] - standard_results["auc"]),
            "pr_auc_gain": float(pro_results["pr_auc"] - standard_results["pr_auc"]),
        }
        allocation["test_metrics"] = test_metrics
        allocation["test_selection_score"] = list(
            allocation_selection_score(test_metrics)
        )
        allocation_test_results[allocation_name] = {
            "standard_results": standard_results,
            "pro_results": pro_results,
            "matched_alarm_results": matched_alarm_results,
        }
        print(
            f"  {allocation_name}: test recall "
            f"{matched_alarm_results['standard']['recall']:.4f} -> "
            f"{matched_alarm_results['pro']['recall']:.4f}"
        )
    selected_allocation = allocation_analysis[best_allocation]
    best_standard_variant = selected_allocation["standard_variant"]
    best_pro_variant = selected_allocation["pro_variant"]
    selected_test_result = allocation_test_results[best_allocation]
    standard_results = selected_test_result["standard_results"]
    pro_results = selected_test_result["pro_results"]
    matched_alarm_results = selected_test_result["matched_alarm_results"]
    standard_model = standard_candidates[best_standard_variant]["model"]
    pro_model = pro_candidates[best_pro_variant]["model"]
    print(f"\nSelected product allocation: {best_allocation}")
    print(f"  {selected_allocation['description']}")
    print(f"  Selected Standard variant: {best_standard_variant}")
    print(f"  Selected Pro variant: {best_pro_variant}")
    print(
        f"  Enabled Standard features: {standard_candidates[best_standard_variant]['enabled_secondary_features']}"
    )
    print(
        f"  Enabled Pro features: {pro_candidates[best_pro_variant]['enabled_secondary_features']}"
    )

    print("\n" + "=" * 70)
    print("PHASE 4: Validation")
    print("=" * 70)

    checkpoint_provenance = {
        variant_name: {
            "trained_from_scratch": bool(
                history.get("trained_from_scratch", [0.0])[-1] >= 0.5
            ),
            "resumed_from_checkpoint": bool(
                history.get("resumed_from_checkpoint", [0.0])[-1] >= 0.5
            ),
            "checkpoint_path": os.path.join(output_dir, f"{variant_name}.keras"),
        }
        for variant_name, history in (
            (name, item["history"]) for name, item in variant_training.items()
        )
    }
    retrained_from_scratch_all_variants = all(
        item["trained_from_scratch"] and not item["resumed_from_checkpoint"]
        for item in checkpoint_provenance.values()
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "status": "completed",
        "claim_scope": (
            "rigorous prototype and feasibility study for patient-level seizure "
            "event detection; not a clinically validated detector or deployment "
            "claim"
        ),
        "evaluation_scope": (
            "patient-level generalization; held-out patients are reserved for "
            "validation/test while train patients remain unseen in those splits"
        ),
        "run_config": {
            "max_sessions_requested": int(max_sessions),
            "selected_session_count": int(len(selected_session_paths)),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "target_rate_hz": int(TARGET_RATE_HZ),
            "random_seed": int(random_seed),
            "window_size_samples": int(DETECTION_WINDOW_SIZE),
            "window_seconds": float(DETECTION_WINDOW_SIZE / TARGET_RATE_HZ),
            "stride_samples": int(stride),
            "stride_seconds": float(stride / TARGET_RATE_HZ),
            "force_rerun": os.environ.get("FORCE_RERUN", "0") == "1",
            "force_retrain_variants": (
                os.environ.get("DETECTION_FORCE_RETRAIN_VARIANTS", "0") == "1"
                or os.environ.get("FORCE_RERUN", "0") == "1"
            ),
            "tsmixer_architecture": {
                "pipeline_version": PIPELINE_VERSION,
                "patch_size": int(os.environ.get("DETECTION_PATCH_SIZE", "8")),
                "patch_embed_dim": int(
                    os.environ.get("DETECTION_PATCH_EMBED_DIM", "8")
                ),
                "blocks": int(os.environ.get("DETECTION_TSMIXER_BLOCKS", "3")),
                "hidden_dim": int(
                    os.environ.get("DETECTION_TSMIXER_HIDDEN_DIM", "48")
                ),
                "baseline_secondary_features": list(
                    parse_feature_env(
                        "DETECTION_BASELINE_SECONDARY_FEATURES", ["EDA", "TEMP"]
                    )
                ),
                "booster_secondary_features": list(
                    parse_feature_env(
                        "DETECTION_BOOSTER_SECONDARY_FEATURES", ["BVP", "HR"]
                    )
                ),
                "loss": os.environ.get("DETECTION_LOSS", "focal"),
            },
        },
        "dataset_selection_policy": {
            "selected_session_policy": (
                "For local feasibility and enough positive windows, session selection "
                "prioritizes sessions with actual seizure overlap, then adds normal "
                "sessions for contrast."
            ),
            "selected_session_scope": (
                "The reported metrics are valid for the selected session-held-out "
                "experimental subset and should not be interpreted as natural seizure "
                "prevalence or prospective clinical performance."
            ),
            "selection_random_state": 42,
        },
        "model_provenance": {
            "checkpoint_policy": (
                "Existing variant checkpoints are ignored when FORCE_RERUN=1 "
                "or DETECTION_FORCE_RETRAIN_VARIANTS=1."
            ),
            "retrained_from_scratch_all_variants": retrained_from_scratch_all_variants,
            "variants": checkpoint_provenance,
        },
        "label_policy": {
            "source": "patient .txt seizure timestamp rows",
            "offset_policy": (
                "use a second numeric column as offset when it is present and "
                "timestamp-like; otherwise use onset plus configured fixed duration; "
                "then merge overlapping/nearby intervals"
            ),
            "seizure_duration_seconds": int(
                os.environ.get("SEIZURE_DURATION_SECONDS", "300")
            ),
            "merge_gap_seconds": int(os.environ.get("SEIZURE_MERGE_GAP_SECONDS", "0")),
            "phase1_exact_durations": os.environ.get("PHASE1_EXACT_DURATIONS", "0")
            == "1",
            "phase1_session_duration_seconds": float(
                os.environ.get("PHASE1_SESSION_DURATION_SECONDS", "129600")
            ),
            "detection_window_seconds": float(DETECTION_WINDOW_SIZE / TARGET_RATE_HZ),
            "detection_stride_seconds": float(stride / TARGET_RATE_HZ),
        },
        "n_train_samples": len(data["labels_train"]),
        "n_test_samples": len(data["labels_test"]),
        "split_strategy": "patient-level held-out split",
        "sensor_policy": {
            "standard_mode": (
                "Base device: ACC x/y/z plus optional sensors from "
                "product_allocation_selection.selected_allocation.base_optional"
            ),
            "pro_mode": (
                "Base device plus required add-on PPG and optional sensors from "
                "product_allocation_selection.selected_allocation.addon_optional. "
                "The deployed Pro model uses the empirically selected active feature "
                "subset in pro_model_selection.selected_variant."
            ),
            "full_sensor_candidate": "Base ACC plus BVP, HR, EDA, and TEMP",
            "physical_sensor_split": {
                "base_required": ["ACC"],
                "addon_required": ["PPG"],
                "ppg_outputs": ["BVP", "HR"],
                "selected_base_optional": selected_allocation["base_optional"],
                "selected_addon_optional": selected_allocation["addon_optional"],
                "excluded_ppg_outputs": {
                    "IBI": "excluded from active features because several IBI.csv files are empty/missing",
                },
                "addon_optional": ["EDA", "TEMP"],
            },
            "recommendation_policy": (
                "Choose the product allocation using imbalance-aware validation "
                "criteria at the same false-alarm budget; rank by recall gain, "
                "balanced-accuracy gain, Pro recall, AUC gain, and false-alarm reduction."
            ),
            "synthetic_signals": False,
        },
        "session_splits": {
            key: [os.path.basename(path) for path in paths]
            for key, paths in session_splits.items()
        },
        "split_audit": split_audit,
        "training_sampling": data.get(
            "training_sampling", {"strategy": "full training split"}
        ),
        "product_allocation_selection": {
            "selection_split": "validation",
            "test_split_role": "final held-out reporting only, not product selection",
            "selection_metric": (
                "Recall gain, balanced-accuracy gain, Pro recall, AUC gain, and "
                "false-alarm reduction at the default false-alarm budget"
            ),
            "selected_allocation_name": best_allocation,
            "selected_allocation": {
                key: value
                for key, value in selected_allocation.items()
                if key not in ("validation_metrics", "selection_score")
            },
            "allocation_analysis": allocation_analysis,
        },
        "best_science_model": {
            "standard_variant": best_standard_variant,
            "pro_variant": best_pro_variant,
            "selected_variant": best_pro_variant,
            "selected_variant_role": "best_validation_science_candidate",
        },
        "best_product_model": {
            "allocation_name": best_allocation,
            "selected_allocation": {
                key: value
                for key, value in selected_allocation.items()
                if key not in ("validation_metrics", "selection_score")
            },
            "selected_variant_role": "best_sensor-allocation_product_choice",
        },
        "standard_model_selection": {
            "selected_variant": best_standard_variant,
            "candidate_features": {
                key: value["enabled_secondary_features"]
                for key, value in standard_candidates.items()
            },
            "candidate_validation_pr_auc": {
                key: value["validation_pr_auc"]
                for key, value in standard_candidates.items()
            },
        },
        "pro_model_selection": {
            "selection_metric": "selected by product allocation analysis",
            "selected_variant": best_pro_variant,
            "candidate_features": {
                key: value["enabled_secondary_features"]
                for key, value in pro_candidates.items()
            },
            "candidate_validation_pr_auc": {
                key: value["validation_pr_auc"] for key, value in pro_candidates.items()
            },
        },
        "variant_training": variant_training,
        "standard_mode": standard_results,
        "pro_mode": pro_results,
        "matched_false_alarm_operating_point": matched_alarm_results,
        "history": {
            best_standard_variant: variant_training[best_standard_variant]["history"],
            best_pro_variant: variant_training[best_pro_variant]["history"],
        },
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    tflite_path = os.path.join(output_dir, "seizure_model.tflite")
    export_tflite(pro_model, tflite_path, quantize=True, representative_data=pro_data)

    standard_tflite_path = os.path.join(output_dir, "seizure_model_standard.tflite")
    export_tflite(
        standard_model,
        standard_tflite_path,
        quantize=True,
        representative_data=standard_data,
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
