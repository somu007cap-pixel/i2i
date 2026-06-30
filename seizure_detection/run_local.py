"""
Local Seizure Prediction Pipeline Runner
========================================
Runs the SOTA Seizure Prediction pipeline on the local machine.
Automatically adapts to GPU/CPU and handles all phases.
"""

import os
import sys
import subprocess
import time
import shutil
import json
from typing import Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)


def has_seizure_session(
    path: str, seizure_intervals: Dict[str, List[Tuple[float, float]]]
) -> bool:
    try:
        from phase2_data_generator_fast import get_actual_session_intervals
    except ImportError:
        from .phase2_data_generator_fast import get_actual_session_intervals
    return len(get_actual_session_intervals(path, seizure_intervals)) > 0


def get_patient_id(session_path: str) -> str:
    parts = os.path.abspath(session_path).split(os.sep)
    for part in reversed(parts):
        if part.startswith("Mayo_"):
            return part
    return "unknown"


def summarize_sessions(
    session_paths: List[str],
    seizure_intervals: Dict[str, List[Tuple[float, float]]],
) -> Dict:
    try:
        from phase2_data_generator_fast import get_actual_session_intervals
    except ImportError:
        from .phase2_data_generator_fast import get_actual_session_intervals
    patients = {}
    seizure_sessions = 0
    episode_keys = set()
    for path in session_paths:
        patients[get_patient_id(path)] = patients.get(get_patient_id(path), 0) + 1
        episodes = get_actual_session_intervals(path, seizure_intervals)
        if episodes:
            seizure_sessions += 1
            for start, end in episodes:
                episode_keys.add((round(float(start), 3), round(float(end), 3)))
    return {
        "sessions": len(session_paths),
        "seizure_sessions": seizure_sessions,
        "seizure_episodes": len(episode_keys),
        "patients": patients,
    }


def select_balanced_sessions(
    session_paths: List[str],
    seizure_intervals: Dict[str, List[Tuple[float, float]]],
    max_sessions: int,
    random_state: int = 42,
) -> List[str]:
    """Prioritize seizure sessions, then add normal sessions for contrast."""
    import numpy as np

    rng = np.random.default_rng(random_state)
    seizure_sessions = [
        p for p in session_paths if has_seizure_session(p, seizure_intervals)
    ]
    normal_sessions = [
        p for p in session_paths if not has_seizure_session(p, seizure_intervals)
    ]
    rng.shuffle(seizure_sessions)
    rng.shuffle(normal_sessions)

    if max_sessions <= 0:
        return seizure_sessions + normal_sessions
    n_seizure = min(len(seizure_sessions), max(1, max_sessions // 2))
    selected = seizure_sessions[:n_seizure]
    selected += normal_sessions[: max_sessions - len(selected)]
    if len(selected) < max_sessions:
        selected += seizure_sessions[n_seizure:max_sessions]
    rng.shuffle(selected)
    return selected


def split_sessions(
    session_paths: List[str],
    seizure_intervals: Dict[str, List[Tuple[float, float]]],
    val_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split at session level while trying to keep seizure sessions in both sets."""
    import numpy as np

    rng = np.random.default_rng(random_state)
    seizure_sessions = [
        p for p in session_paths if has_seizure_session(p, seizure_intervals)
    ]
    normal_sessions = [
        p for p in session_paths if not has_seizure_session(p, seizure_intervals)
    ]
    rng.shuffle(seizure_sessions)
    rng.shuffle(normal_sessions)

    def split_group(paths: List[str]):
        if len(paths) < 2:
            return paths, []
        n_val = max(1, int(round(len(paths) * val_size)))
        return paths[n_val:], paths[:n_val]

    sz_train, sz_val = split_group(seizure_sessions)
    normal_train, normal_val = split_group(normal_sessions)
    train = sz_train + normal_train
    val = sz_val + normal_val
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def normalize_from_train(X_train, X_val):
    """Fit sequence normalization on train only."""
    import numpy as np

    mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


def subset_prediction_data(X, labels, max_sequences: int, random_state: int = 42):
    """Keep rare positives across horizons and sample negatives for CPU-safe training."""
    import numpy as np

    n = len(X)
    if max_sequences <= 0 or n <= max_sequences:
        return (
            X.astype(np.float32, copy=False),
            labels,
            {
                "strategy": "full split",
                "original_sequences": int(n),
                "selected_sequences": int(n),
            },
        )
    positive_mask = np.zeros(n, dtype=bool)
    for horizon_labels in labels.values():
        positive_mask |= np.asarray(horizon_labels).astype(bool)
    positive_idx = np.flatnonzero(positive_mask)
    negative_idx = np.flatnonzero(~positive_mask)
    rng = np.random.default_rng(random_state)

    if len(positive_idx) >= max_sequences:
        selected_idx = rng.choice(positive_idx, size=max_sequences, replace=False)
    else:
        n_negative = min(len(negative_idx), max_sequences - len(positive_idx))
        sampled_negative_idx = rng.choice(negative_idx, size=n_negative, replace=False)
        selected_idx = np.concatenate([positive_idx, sampled_negative_idx])
    rng.shuffle(selected_idx)
    subset_labels = {
        horizon: np.asarray(horizon_labels)[selected_idx]
        for horizon, horizon_labels in labels.items()
    }
    return (
        X[selected_idx].astype(np.float32, copy=False),
        subset_labels,
        {
            "strategy": "all positives across horizons plus sampled negatives",
            "original_sequences": int(n),
            "selected_sequences": int(len(selected_idx)),
            "positive_union_sequences": int(len(positive_idx)),
            "max_sequences": int(max_sequences),
        },
    )


def write_prediction_report(
    output_dir: str,
    status: str,
    reason: str,
    config,
    train_sessions,
    val_sessions,
    split_audit,
    train_labels=None,
    val_labels=None,
    metrics=None,
    extra_info=None,
) -> None:
    """Write a prediction report even when training exits early."""
    import numpy as np

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pipeline_version": "realistic_acc_ppg_no_ibi_v10",
        "status": status,
        "reason": reason,
        "split_strategy": "session-level held-out validation",
        "evaluation_scope": (
            "personalized/session-level generalization; validation holds out "
            "sessions but may contain patients seen during training"
        ),
        "sensor_policy": (
            "Detection selected the product split ACC+TEMP for Standard and "
            "PPG-derived BVP/HR plus EDA for the Pro add-on. Prediction uses "
            "the real available ACC, BVP, HR, EDA, and TEMP feature stack to "
            "explore pre-ictal risk ranking. IBI is excluded because several "
            "IBI.csv files are empty/missing. No synthetic label-derived features."
        ),
        "train_sessions": [os.path.basename(path) for path in train_sessions],
        "val_sessions": [os.path.basename(path) for path in val_sessions],
        "split_audit": split_audit,
        "metrics": metrics or {},
    }
    if extra_info:
        report.update(extra_info)
    if config is not None:
        report["prediction_horizons"] = list(config.prediction_horizons)
    if train_labels is not None:
        report["train_label_counts"] = {
            str(horizon): {
                "total": int(len(labels)),
                "positives": int(np.sum(labels)),
            }
            for horizon, labels in train_labels.items()
        }
    if val_labels is not None:
        report["val_label_counts"] = {
            str(horizon): {
                "total": int(len(labels)),
                "positives": int(np.sum(labels)),
            }
            for horizon, labels in val_labels.items()
        }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def install_dependencies():
    """Install required packages."""
    print("Checking dependencies...")
    try:
        import tensorflow
        import sklearn
        import tqdm
        import pandas
        import matplotlib

        print("[OK] Dependencies already installed.")
    except ImportError:
        print("[WARN] Dependencies missing. Installing...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "tensorflow",
                "scikit-learn",
                "tqdm",
                "pandas",
                "matplotlib",
            ]
        )
        print("[OK] Dependencies installed.")


def run_pipeline():
    print("\n" + "=" * 60)
    print("STARTING LOCAL SEIZURE PREDICTION PIPELINE")
    print("=" * 60)

    import numpy as np
    import tensorflow as tf
    import warnings

    warnings.filterwarnings("ignore")

    try:
        from prediction_data_generator import (
            PredictionConfig,
            SeizurePredictionDataGen,
            get_session_paths,
        )
        from prediction_model import build_seizure_prediction_model
        from phase1_data_health import run_deep_analysis, build_seizure_intervals_dict
    except ImportError as e:
        print(f"[ERROR] Failed to import modules: {e}")
        import traceback

        traceback.print_exc()
        return
    print("\n[Hardware Detection]")
    gpus = tf.config.list_physical_devices("GPU")
    HAS_GPU = len(gpus) > 0

    if HAS_GPU:
        print(f"[OK] GPU DETECTED: {gpus[0]}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
            print("   - Enabled mixed precision (float16)")
        except Exception as e:
            print(f"   - GPU config warning: {e}")
        BATCH_SIZE = 128
        NUM_SESSIONS = int(os.environ.get("PREDICTION_MAX_SESSIONS", "80"))
        EPOCHS = 20
    else:
        print("[WARN] NO GPU - Using CPU Mode")
        print("   - Reduced model size and data for feasibility")
        BATCH_SIZE = 32
        NUM_SESSIONS = int(os.environ.get("PREDICTION_MAX_SESSIONS", "40"))
        EPOCHS = int(os.environ.get("PREDICTION_EPOCHS", "10"))
    OUTPUT_DIR = os.path.join(BASE_PATH, "prediction_outputs_local")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model_local.keras")

    print(f"\n[Configuration]")
    print(f"   Base Path: {BASE_PATH}")
    print(f"   Output Dir: {OUTPUT_DIR}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Max Sessions: {NUM_SESSIONS}")
    print(f"   Epochs: {EPOCHS}")

    print("\n[Phase 1: Data Health Check]")
    print("   Scanning patient folders...")
    try:
        all_patients = run_deep_analysis(BASE_PATH)
        seizure_intervals = build_seizure_intervals_dict(all_patients)
        print(f"   [OK] Found seizure data from {len(all_patients)} patients")
    except Exception as e:
        print(f"   [ERROR] Phase 1 failed: {e}")
        import traceback

        traceback.print_exc()
        return
    print("\n[Data Pipeline]")
    print("   Finding valid sessions...")
    session_paths = get_session_paths(BASE_PATH)

    if not session_paths:
        print("   [ERROR] No valid sessions found!")
        print("   Looking for folders with: ACC.csv, BVP.csv, EDA.csv, TEMP.csv")
        return
    print(f"   [OK] Found {len(session_paths)} valid sessions")

    sample_sessions = select_balanced_sessions(
        session_paths, seizure_intervals, NUM_SESSIONS
    )
    train_sessions, val_sessions = split_sessions(sample_sessions, seizure_intervals)
    print(f"   Selected {len(sample_sessions)} sessions for prediction")
    print(
        f"   Train sessions: {len(train_sessions)} ({sum(has_seizure_session(p, seizure_intervals) for p in train_sessions)} with seizures)"
    )
    print(
        f"   Val sessions: {len(val_sessions)} ({sum(has_seizure_session(p, seizure_intervals) for p in val_sessions)} with seizures)"
    )
    split_audit = {
        "train": summarize_sessions(train_sessions, seizure_intervals),
        "val": summarize_sessions(val_sessions, seizure_intervals),
    }
    print(f"   Train patients: {split_audit['train']['patients']}")
    print(f"   Val patients: {split_audit['val']['patients']}")

    print("\n[Creating Data Generator]")
    config = PredictionConfig(prediction_horizons=[300, 900, 1800], sequence_length=60)

    try:
        train_gen = SeizurePredictionDataGen(
            session_paths=train_sessions,
            seizure_intervals=seizure_intervals,
            config=config,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_gen = SeizurePredictionDataGen(
            session_paths=val_sessions,
            seizure_intervals=seizure_intervals,
            config=config,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        print(f"   [OK] Data generator ready")
    except Exception as e:
        print(f"   [ERROR] Data generation failed: {e}")
        import traceback

        traceback.print_exc()
        return
    print("\n[Loading Data]")
    try:
        X_train, train_labels = train_gen.get_multi_horizon_data()
        X_val, val_labels = val_gen.get_multi_horizon_data()
    except Exception as e:
        print(f"   [ERROR] Error getting data: {e}")
        import traceback

        traceback.print_exc()
        return
    if len(X_train) == 0 or len(X_val) == 0:
        print("   [ERROR] Not enough train/validation sequences generated!")
        write_prediction_report(
            OUTPUT_DIR,
            status="skipped",
            reason="not enough train/validation sequences generated",
            config=config,
            train_sessions=train_sessions,
            val_sessions=val_sessions,
            split_audit=split_audit,
            train_labels=train_labels,
            val_labels=val_labels,
        )
        return
    full_dataset_counts = {
        "train_sequences": int(len(X_train)),
        "val_sequences": int(len(X_val)),
        "train_positives": {
            str(horizon): int(np.sum(train_labels[horizon]))
            for horizon in config.prediction_horizons
        },
        "val_positives": {
            str(horizon): int(np.sum(val_labels[horizon]))
            for horizon in config.prediction_horizons
        },
    }

    X_val_full = X_val
    val_labels_full = {
        horizon: np.asarray(labels).copy() for horizon, labels in val_labels.items()
    }

    max_train_sequences = int(
        os.environ.get(
            "PREDICTION_MAX_TRAIN_SEQUENCES", "30000" if not HAS_GPU else "0"
        )
    )
    max_val_sequences = int(
        os.environ.get("PREDICTION_MAX_VAL_SEQUENCES", "15000" if not HAS_GPU else "0")
    )
    evaluate_full_validation = (
        os.environ.get("PREDICTION_EVALUATE_FULL_VAL", "1") == "1"
    )
    X_train, train_labels, train_subset = subset_prediction_data(
        X_train,
        train_labels,
        max_train_sequences,
        random_state=42,
    )
    X_val, val_labels, val_subset = subset_prediction_data(
        X_val,
        val_labels,
        max_val_sequences,
        random_state=43,
    )
    print("   Dataset usage:")
    print(f"     Full train sequences: {full_dataset_counts['train_sequences']}")
    print(f"     Full val sequences: {full_dataset_counts['val_sequences']}")
    print(f"     Used train sequences: {len(X_train)} ({train_subset['strategy']})")
    print(f"     Used val sequences: {len(X_val)} ({val_subset['strategy']})")
    print(
        f"     Final metric evaluation: "
        f"{'full validation split' if evaluate_full_validation else 'training validation subset'}"
    )

    X_train, X_val, norm_mean, norm_std = normalize_from_train(X_train, X_val)
    if evaluate_full_validation:
        X_val_eval = X_val_full.astype(np.float32, copy=False)
        X_val_eval -= norm_mean
        X_val_eval /= norm_std
        eval_labels = val_labels_full
        eval_subset = {
            "strategy": "full validation split",
            "original_sequences": int(len(X_val_full)),
            "selected_sequences": int(len(X_val_full)),
        }
    else:
        X_val_eval = X_val
        eval_labels = val_labels
        eval_subset = val_subset
    y_train_5 = train_labels[300]
    y_train_15 = train_labels[900]
    y_train_30 = train_labels[1800]
    y_val_5 = val_labels[300]
    y_val_15 = val_labels[900]
    y_val_30 = val_labels[1800]

    print(f"   Train: {len(X_train)} | Val: {len(X_val)}")
    for horizon, y_train, y_val in (
        (300, y_train_5, y_val_5),
        (900, y_train_15, y_val_15),
        (1800, y_train_30, y_val_30),
    ):
        print(
            f"   Horizon {horizon//60} min positives: "
            f"train={int(np.sum(y_train))}/{len(y_train)}, "
            f"val={int(np.sum(y_val))}/{len(y_val)}"
        )
    if sum(np.sum(train_labels[h]) for h in config.prediction_horizons) == 0:
        print(
            "   [ERROR] No pre-ictal training positives. Cannot train an honest prediction model."
        )
        write_prediction_report(
            OUTPUT_DIR,
            status="skipped",
            reason="no pre-ictal training positives after exclusions",
            config=config,
            train_sessions=train_sessions,
            val_sessions=val_sessions,
            split_audit=split_audit,
            train_labels=train_labels,
            val_labels=val_labels,
        )
        return
    print("\n[Model Building]")
    input_shape = X_train.shape[1:]
    print(f"   Input shape: {input_shape}")

    model_kwargs = {
        "d_model": 128 if HAS_GPU else 64,
        "num_heads": 4 if HAS_GPU else 2,
        "num_layers": 3 if HAS_GPU else 1,
        "dropout_rate": 0.2,
        "prediction_horizons": [300, 900, 1800],
    }

    try:
        model = build_seizure_prediction_model(input_shape=input_shape, **model_kwargs)
        print(f"   [OK] Model created")
    except Exception as e:
        print(f"   [ERROR] Model creation failed: {e}")
        import traceback

        traceback.print_exc()
        return
    from tensorflow.keras.losses import BinaryFocalCrossentropy

    try:
        focal_loss = BinaryFocalCrossentropy(gamma=2.0, from_logits=False)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={300: focal_loss, 900: focal_loss, 1800: focal_loss},
            loss_weights={300: 0.3, 900: 0.4, 1800: 0.3},
            metrics={
                300: ["accuracy", tf.keras.metrics.AUC(name="auc_5min")],
                900: ["accuracy", tf.keras.metrics.AUC(name="auc_15min")],
                1800: ["accuracy", tf.keras.metrics.AUC(name="auc_30min")],
            },
        )
        print(f"   [OK] Model compiled")
    except Exception as e:
        print(f"   [ERROR] Compilation failed: {e}")
        import traceback

        traceback.print_exc()
        return
    print("\n[Training]")
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_900_auc_15min",
            patience=3,
            restore_best_weights=True,
            verbose=1,
            mode="max",
        ),
        ModelCheckpoint(
            CHECKPOINT_PATH,
            monitor="val_900_auc_15min",
            save_best_only=True,
            save_weights_only=False,
            verbose=0,
            mode="max",
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0),
    ]

    history = model.fit(
        X_train,
        {300: y_train_5, 900: y_train_15, 1800: y_train_30},
        validation_data=(X_val, {300: y_val_5, 900: y_val_15, 1800: y_val_30}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    print(f"   [OK] Training complete")
    print(f"   [OK] Best model saved to {CHECKPOINT_PATH}")

    try:
        import json
        import pandas as pd
        from sklearn.metrics import (
            roc_auc_score,
            precision_score,
            recall_score,
            confusion_matrix,
            precision_recall_curve,
        )

        print("\n[Post-training Evaluation]")
        preds = {}
        metrics = {}
        horizon_map = {
            300: eval_labels[300],
            900: eval_labels[900],
            1800: eval_labels[1800],
        }

        print(
            "   [OK] Running predictions on "
            f"{'full validation split' if evaluate_full_validation else 'validation subset'}..."
        )
        y_pred_dict = model.predict(X_val_eval, verbose=0)

        for h in (300, 900, 1800):
            y_true = np.array(horizon_map[h])

            if isinstance(y_pred_dict, dict):
                yhat = y_pred_dict.get(h)
                if yhat is None:
                    yhat = y_pred_dict.get(str(h))
            else:
                yhat = y_pred_dict
            y_prob = np.array(yhat).reshape(-1)
            y_pred = (y_prob >= 0.5).astype(int)
            preds[h] = y_prob.tolist()

            auc = None
            if len(np.unique(y_true)) == 2:
                auc = float(roc_auc_score(y_true, y_prob))
            prec = float(precision_score(y_true, y_pred, zero_division=0))
            rec = float(recall_score(y_true, y_pred, zero_division=0))
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

            calibrated = {}
            if len(np.unique(y_true)) == 2:
                precision_curve, recall_curve, thresholds = precision_recall_curve(
                    y_true, y_prob
                )
                if len(thresholds) > 0:
                    f1_curve = (
                        2
                        * precision_curve[:-1]
                        * recall_curve[:-1]
                        / (precision_curve[:-1] + recall_curve[:-1] + 1e-8)
                    )
                    best_idx = int(np.nanargmax(f1_curve))
                    best_threshold = float(thresholds[best_idx])
                    best_pred = (y_prob >= best_threshold).astype(int)
                    calibrated = {
                        "threshold": best_threshold,
                        "precision": float(
                            precision_score(y_true, best_pred, zero_division=0)
                        ),
                        "recall": float(
                            recall_score(y_true, best_pred, zero_division=0)
                        ),
                        "f1": float(f1_curve[best_idx]),
                        "confusion_matrix": confusion_matrix(
                            y_true, best_pred, labels=[0, 1]
                        ).tolist(),
                        "note": "threshold selected on validation for calibrated operating-point reporting",
                    }
            top_risk = {}
            positive_total = max(int(np.sum(y_true)), 1)
            order = np.argsort(y_prob)[::-1]
            for pct in (0.01, 0.05, 0.10):
                k = max(1, int(round(len(y_true) * pct)))
                selected = order[:k]
                hits = int(np.sum(y_true[selected]))
                top_risk[f"top_{int(pct * 100)}pct"] = {
                    "selected": int(k),
                    "positives_captured": hits,
                    "recall": float(hits / positive_total),
                    "precision": float(hits / k),
                }
            metrics[h] = {
                "auc": auc,
                "precision": prec,
                "recall": rec,
                "confusion_matrix": cm,
                "fixed_threshold": 0.5,
                "calibrated_threshold_metrics": calibrated,
                "top_risk_metrics": top_risk,
                "n_train": int(len(train_labels[h])),
                "n_val": int(len(y_true)),
                "train_positives": int(np.sum(train_labels[h])),
                "val_positives": int(np.sum(y_true)),
                "evaluation_split": (
                    "full_validation"
                    if evaluate_full_validation
                    else "sampled_validation_subset"
                ),
                "evaluation_note": (
                    "AUC unavailable because validation contains one class"
                    if auc is None
                    else "AUC computed on held-out sessions"
                ),
            }

            df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred})
            df.to_csv(os.path.join(OUTPUT_DIR, f"predictions_{h}.csv"), index=False)
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline_version": "realistic_acc_ppg_no_ibi_v10",
            "status": "completed",
            "reason": "trained and evaluated",
            "split_strategy": "session-level held-out validation",
            "evaluation_scope": (
                "personalized/session-level generalization; validation holds out "
                "sessions but may contain patients seen during training"
            ),
            "label_policy": {
                "source": "patient .txt onset timestamps",
                "offset_policy": "onset plus configured fixed duration, then merge overlapping/nearby intervals",
                "seizure_duration_seconds": int(
                    os.environ.get("SEIZURE_DURATION_SECONDS", "300")
                ),
                "merge_gap_seconds": int(
                    os.environ.get("SEIZURE_MERGE_GAP_SECONDS", "0")
                ),
                "phase1_exact_durations": os.environ.get("PHASE1_EXACT_DURATIONS", "0")
                == "1",
                "phase1_session_duration_seconds": float(
                    os.environ.get("PHASE1_SESSION_DURATION_SECONDS", "129600")
                ),
                "prediction_exclusions": (
                    "ictal, post-ictal, and pre-ictal samples too close to a "
                    "previous seizure are excluded from training/evaluation"
                ),
            },
            "sensor_policy": (
                "Detection selected the product split ACC+TEMP for Standard and "
                "PPG-derived BVP/HR plus EDA for the Pro add-on. Prediction uses "
                "the real available ACC, BVP, HR, EDA, and TEMP feature stack to "
                "explore pre-ictal risk ranking. IBI is excluded because several "
                "IBI.csv files are empty/missing. No synthetic label-derived features."
            ),
            "train_sessions": [os.path.basename(path) for path in train_sessions],
            "val_sessions": [os.path.basename(path) for path in val_sessions],
            "split_audit": split_audit,
            "dataset_usage": {
                "full_dataset_counts": full_dataset_counts,
                "train_subset": train_subset,
                "val_subset": val_subset,
                "final_metric_evaluation": eval_subset,
                "final_metrics_use_full_validation": bool(evaluate_full_validation),
                "cpu_subset_note": (
                    "On CPU, training uses all rare pre-ictal sequences across "
                    "horizons plus sampled negatives to keep one-click training "
                    "runtime and memory bounded. Final validation metrics use the "
                    "full validation split by default unless "
                    "PREDICTION_EVALUATE_FULL_VAL=0."
                ),
            },
            "metrics": metrics,
        }
        with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"   [OK] Saved metrics and predictions to {OUTPUT_DIR}")
    except Exception as e:
        print(f"   [ERROR] Post-training evaluation failed: {e}")
        import traceback

        traceback.print_exc()
    print("\n" + "=" * 60)
    print("[OK] PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"   Model: {CHECKPOINT_PATH}")
    print(f"   Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    install_dependencies()
    run_pipeline()
