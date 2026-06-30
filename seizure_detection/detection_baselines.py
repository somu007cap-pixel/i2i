"""
Baseline model comparisons for seizure anomaly detection.

This script uses the same session-level data preparation path as the main
TSMixer detector and writes a separate final-semester baseline report. It is
intended to answer evaluator feedback without changing the submitted mid-sem
artifacts or the primary detection pipeline.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.get_logger().setLevel("ERROR")

from phase1_data_health import build_seizure_intervals_dict, run_deep_analysis
from phase2_data_generator_fast import get_session_paths_smart
from phase4_validation import (
    DETECTION_WINDOW_SIZE,
    EXPECTED_PRIMARY_SHAPE,
    EXPECTED_SECONDARY_SHAPE,
    PREDICT_BATCH_SIZE,
    TARGET_RATE_HZ,
    build_balanced_training_view,
    build_session_level_data,
    compute_class_weights,
    event_level_metrics_from_probabilities,
    get_default_base_path,
    metrics_from_probabilities,
    split_session_paths,
    summarize_session_splits,
    threshold_for_false_alarm_rate,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "seizure_detection" / "outputs"
BASELINE_DIR = OUTPUT_DIR / "baselines"
REPORT_PATH = BASELINE_DIR / "baseline_comparison_report.json"
SUMMARY_PATH = BASELINE_DIR / "BASELINE_COMPARISON_SUMMARY.md"


def set_seed() -> int:
    seed = int(os.environ.get("DETECTION_RANDOM_SEED", "42"))
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return seed


def combine_inputs(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    return np.concatenate([primary, secondary], axis=-1).astype(np.float32)


def compile_supervised_model(model: keras.Model) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(os.environ.get("BASELINE_LEARNING_RATE", "0.001"))
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_lstm(input_shape: Tuple[int, int]) -> keras.Model:
    inputs = layers.Input(shape=input_shape, name="sensor_window")
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.LSTM(32, dropout=0.15, recurrent_dropout=0.0)(x)
    x = layers.Dense(24, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return compile_supervised_model(keras.Model(inputs, outputs, name="LSTMBaseline"))


def build_cnn_lstm(input_shape: Tuple[int, int]) -> keras.Model:
    inputs = layers.Input(shape=input_shape, name="sensor_window")
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(24, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return compile_supervised_model(keras.Model(inputs, outputs, name="CNNLSTMBaseline"))


def build_transformer(input_shape: Tuple[int, int]) -> keras.Model:
    inputs = layers.Input(shape=input_shape, name="sensor_window")
    x = layers.Dense(32, activation="relu")(inputs)
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=16, dropout=0.1)(x, x)
    x = layers.LayerNormalization()(x + attention)
    ff = layers.Dense(64, activation="relu")(x)
    ff = layers.Dropout(0.1)(ff)
    ff = layers.Dense(32)(ff)
    x = layers.LayerNormalization()(x + ff)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(24, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return compile_supervised_model(
        keras.Model(inputs, outputs, name="CompactTransformerBaseline")
    )


def build_autoencoder(input_shape: Tuple[int, int]) -> keras.Model:
    inputs = layers.Input(shape=input_shape, name="sensor_window")
    x = layers.Conv1D(24, 5, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = layers.Conv1D(12, 3, padding="same", activation="relu")(x)
    encoded = layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = layers.Conv1D(12, 3, padding="same", activation="relu")(encoded)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(24, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Cropping1D(cropping=(0, max(0, x.shape[1] - input_shape[0])))(x)
    outputs = layers.Conv1D(input_shape[-1], 3, padding="same", activation="linear")(x)
    model = keras.Model(inputs, outputs, name="SequenceAutoencoderBaseline")
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(os.environ.get("BASELINE_LEARNING_RATE", "0.001"))
        ),
        loss="mse",
    )
    return model


def train_supervised(
    name: str,
    model: keras.Model,
    data: Dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
) -> keras.callbacks.History:
    train_data = build_balanced_training_view(
        data,
        negative_ratio=int(os.environ.get("BASELINE_NEGATIVE_RATIO", "5")),
        max_train_windows=int(os.environ.get("BASELINE_MAX_TRAIN_WINDOWS", "250000")),
    )
    train_x = combine_inputs(train_data["primary_train"], train_data["secondary_train"])
    class_weights = compute_class_weights(train_data["labels_train"])
    checkpoint = BASELINE_DIR / f"{name}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=int(os.environ.get("BASELINE_PATIENCE", "4")),
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]
    return model.fit(
        train_x,
        train_data["labels_train"],
        validation_data=(data["x_val"], data["labels_val"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )


def train_autoencoder(
    model: keras.Model,
    data: Dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
) -> keras.callbacks.History:
    normal_idx = np.flatnonzero(data["labels_train"] <= 0.5)
    max_windows = int(os.environ.get("BASELINE_AE_MAX_NORMAL_WINDOWS", "250000"))
    if len(normal_idx) > max_windows:
        rng = np.random.default_rng(42)
        normal_idx = rng.choice(normal_idx, size=max_windows, replace=False)
    x_train = data["x_train"][normal_idx]
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(os.environ.get("BASELINE_PATIENCE", "4")),
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            BASELINE_DIR / "autoencoder.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
    return model.fit(
        x_train,
        x_train,
        validation_data=(data["x_val"], data["x_val"]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )


def supervised_probabilities(model: keras.Model, x: np.ndarray) -> np.ndarray:
    return model.predict(x, batch_size=PREDICT_BATCH_SIZE, verbose=0).reshape(-1)


def autoencoder_scores(model: keras.Model, x: np.ndarray) -> np.ndarray:
    reconstructed = model.predict(x, batch_size=PREDICT_BATCH_SIZE, verbose=0)
    return np.mean(np.square(x - reconstructed), axis=(1, 2)).astype(np.float32)


def evaluate_probabilities(
    labels_val: np.ndarray,
    labels_test: np.ndarray,
    probabilities_val: np.ndarray,
    probabilities_test: np.ndarray,
    data: Dict[str, np.ndarray],
    stride_seconds: float,
) -> Dict:
    alarm_budget = float(os.environ.get("DETECTION_DEFAULT_ALARM_BUDGET_PER_HOUR", "10"))
    threshold = threshold_for_false_alarm_rate(
        labels_val,
        probabilities_val,
        alarm_budget,
        stride_seconds,
    )
    metrics = metrics_from_probabilities(
        labels_test,
        probabilities_test,
        threshold,
        stride_seconds,
    )
    metrics["event_level"] = event_level_metrics_from_probabilities(
        data,
        "test",
        probabilities_test,
        threshold,
    )
    metrics["threshold_policy"] = (
        f"validation false-alarm budget <= {alarm_budget:g}/hour"
    )
    return metrics


def build_data() -> Tuple[Dict[str, np.ndarray], Dict]:
    base_path = get_default_base_path()
    all_patients = run_deep_analysis(base_path)
    seizure_intervals = build_seizure_intervals_dict(all_patients)
    max_sessions = int(os.environ.get("DETECTION_MAX_SESSIONS", "80"))
    batch_size = int(os.environ.get("DETECTION_BATCH_SIZE", "64"))
    stride = int(os.environ.get("DETECTION_STRIDE", str(DETECTION_WINDOW_SIZE)))
    session_paths = get_session_paths_smart(
        base_path,
        seizure_intervals,
        max_sessions=max_sessions,
        prioritize_seizures=True,
    )
    session_splits = split_session_paths(session_paths, seizure_intervals)
    data = build_session_level_data(
        session_splits,
        seizure_intervals,
        window_size=DETECTION_WINDOW_SIZE,
        stride=stride,
        batch_size=batch_size,
    )
    for split in ("train", "val", "test"):
        data[f"x_{split}"] = combine_inputs(
            data[f"primary_{split}"],
            data[f"secondary_{split}"],
        )
    metadata = {
        "base_path": base_path,
        "max_sessions": max_sessions,
        "selected_sessions": len(session_paths),
        "session_split_summary": summarize_session_splits(
            session_splits, seizure_intervals
        ),
        "window_size_samples": DETECTION_WINDOW_SIZE,
        "window_seconds": float(DETECTION_WINDOW_SIZE / TARGET_RATE_HZ),
        "stride_samples": stride,
        "stride_seconds": float(stride / TARGET_RATE_HZ),
        "input_shape": list(data["x_train"].shape[1:]),
    }
    return data, metadata


def run() -> Dict:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    seed = set_seed()
    epochs = int(os.environ.get("BASELINE_EPOCHS", "12"))
    batch_size = int(os.environ.get("BASELINE_BATCH_SIZE", "128"))
    data, metadata = build_data()
    input_shape = tuple(data["x_train"].shape[1:])
    report = {
        "timestamp": datetime.now().isoformat(),
        "scope": (
            "Final-semester baseline comparison for wearable seizure anomaly "
            "detection using the same session-level split as the TSMixer pipeline."
        ),
        "random_seed": seed,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "metadata": metadata,
        "baselines": {},
    }
    builders = {
        "lstm": build_lstm,
        "cnn_lstm": build_cnn_lstm,
        "compact_transformer": build_transformer,
    }
    for name, builder in builders.items():
        print("\n" + "=" * 70)
        print(f"TRAINING BASELINE: {name}")
        print("=" * 70)
        start = time.perf_counter()
        model = builder(input_shape)
        history = train_supervised(name, model, data, epochs, batch_size)
        val_prob = supervised_probabilities(model, data["x_val"])
        test_prob = supervised_probabilities(model, data["x_test"])
        metrics = evaluate_probabilities(
            data["labels_val"],
            data["labels_test"],
            val_prob,
            test_prob,
            data,
            metadata["stride_seconds"],
        )
        report["baselines"][name] = {
            "model_name": model.name,
            "parameters": int(model.count_params()),
            "training_seconds": float(time.perf_counter() - start),
            "history": {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            },
            "metrics": metrics,
        }
    print("\n" + "=" * 70)
    print("TRAINING BASELINE: autoencoder")
    print("=" * 70)
    start = time.perf_counter()
    autoencoder = build_autoencoder(input_shape)
    history = train_autoencoder(autoencoder, data, epochs, batch_size)
    val_scores = autoencoder_scores(autoencoder, data["x_val"])
    test_scores = autoencoder_scores(autoencoder, data["x_test"])
    metrics = evaluate_probabilities(
        data["labels_val"],
        data["labels_test"],
        val_scores,
        test_scores,
        data,
        metadata["stride_seconds"],
    )
    report["baselines"]["autoencoder"] = {
        "model_name": autoencoder.name,
        "parameters": int(autoencoder.count_params()),
        "training_seconds": float(time.perf_counter() - start),
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "metrics": metrics,
        "score_note": "Reconstruction error is used as the anomaly score.",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary(report)
    print(f"\nSaved baseline report: {REPORT_PATH}")
    print(f"Saved baseline summary: {SUMMARY_PATH}")
    return report


def write_summary(report: Dict) -> None:
    lines: List[str] = [
        "# Baseline Comparison Summary",
        "",
        "Scope: same session-level data split as the detection pipeline. Metrics are "
        "computed on held-out test sessions using thresholds selected on validation.",
        "",
        "| Baseline | Parameters | ROC-AUC | PR-AUC | Recall | Event sensitivity | False alarms/hour |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in report["baselines"].items():
        metrics = item["metrics"]
        event = metrics.get("event_level", {})
        lines.append(
            "| {name} | {params:,} | {auc:.4f} | {pr_auc:.4f} | {recall:.4f} | "
            "{detected}/{events} | {fa:.2f} |".format(
                name=name,
                params=item["parameters"],
                auc=metrics["auc"],
                pr_auc=metrics["pr_auc"],
                recall=metrics["recall"],
                detected=event.get("detected_events", 0),
                events=event.get("event_count", 0),
                fa=metrics["false_alarms_per_hour"],
            )
        )
    lines += [
        "",
        "These baselines are comparison references, not product claims. Raw accuracy is "
        "not used as the main success metric because seizure windows are rare.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
