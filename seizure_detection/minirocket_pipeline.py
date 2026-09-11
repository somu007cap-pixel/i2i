# minirocket_pipeline.py
"""MiniRocket baseline for the Empatica seizure‑detection data.

The script:
1. Loads the Empatica tensors via ``build_data()`` (train/val/test splits).
2. Reshapes the data to the 2‑D shape expected by ``pyts`` MiniRocket
   (samples × timesteps).  The model treats each sensor channel as an
   independent time‑series and concatenates the extracted features.
3. Fits a linear classifier (LogisticRegression with ``solver='lbfgs'``) on the
   MiniRocket features.
4. Evaluates AUC on the validation and test sets and prints a short report.

Usage
-----
    python minirocket_pipeline.py

The script assumes the repository root is ``C:\\I2I`` (the workspace URI) and
that the ``seizure_detection`` package is importable (it already is on the
PYTHONPATH because the repo root is added automatically when you run a script
from that directory).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# MiniRocket from pyts
from pyts.transformation import MiniRocket

# ---------------------------------------------------------------------------
# 1️⃣ Load Empatica data (the same function used by the existing pipeline)
# ---------------------------------------------------------------------------
# ``build_data`` returns a dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
# Each X is shaped (samples, channels, timesteps).

from seizure_detection.detection_baselines import build_data

print("[info] Loading Empatica data …")
data = build_data()  # uses default parameters (train/val/test split)

# Helper to reshape for MiniRocket: combine channel and timestep dimensions
def reshape_for_minirocket(X: np.ndarray) -> tuple[np.ndarray, int, int]:
    # X shape: (n_samples, n_channels, n_timesteps)
    # MiniRocket expects (n_samples, n_timesteps) per channel, so we reshape to
    # (n_samples * n_channels, n_timesteps) and later aggregate.
    n_samples, n_channels, n_timesteps = X.shape
    X_reshaped = X.transpose(0, 2, 1).reshape(-1, n_timesteps)  # (samples*channels, timesteps)
    return X_reshaped, n_samples, n_channels

# ---------------------------------------------------------------------------
# 2️⃣ Fit MiniRocket on the training data
# ---------------------------------------------------------------------------
X_train_raw, n_train_samples, n_channels = reshape_for_minirocket(data["X_train"])
print(f"[info] Training MiniRocket on {X_train_raw.shape[0]} series (" 
      f"{n_train_samples} samples × {n_channels} channels) …")

# Number of random kernels – 5 000 is a good trade‑off for CPU speed.
mr = MiniRocket(num_kernels=5000, random_state=42)
mr.fit(X_train_raw)

# Transform training data → feature matrix (samples*channels, n_kernels)
train_features = mr.transform(X_train_raw)
# Aggregate back to per‑sample level (mean across channels)
train_features = train_features.reshape(n_train_samples, n_channels, -1).mean(axis=1)

# ---------------------------------------------------------------------------
# 3️⃣ Train a linear classifier
# ---------------------------------------------------------------------------
clf = LogisticRegression(max_iter=1000, solver="lbfgs")
clf.fit(train_features, data["y_train"])  # y is already binary (0/1)

# ---------------------------------------------------------------------------
# 4️⃣ Evaluate on validation and test sets
# ---------------------------------------------------------------------------

def evaluate(split_name: str, X_raw: np.ndarray, y_true: np.ndarray) -> None:
    X_reshaped, n_samples, _ = reshape_for_minirocket(X_raw)
    feats = mr.transform(X_reshaped)
    feats = feats.reshape(n_samples, n_channels, -1).mean(axis=1)
    prob = clf.predict_proba(feats)[:, 1]
    auc = roc_auc_score(y_true, prob)
    print(f"[{split_name}] AUC = {auc:.4f}")

print("[info] Evaluating …")
evaluate("val", data["X_val"], data["y_val"])
evaluate("test", data["X_test"], data["y_test"])

print("[done] MiniRocket baseline finished.")
