"""
Experimental Pre-Ictal Risk Data Generator
===========================================
Data pipeline for future-work pre-ictal risk modeling.

Current status: this module is retained for exploratory final-milestone work.
It is not part of the mid-semester detection-only claim.

State labels:
- Interictal: Normal state (no seizure coming)
- Pre-ictal: Seizure coming within prediction horizon
- Ictal: During seizure
- Post-ictal: Recovery period after seizure

Risk horizons:
- 5 minutes: Short-term warning
- 15 minutes: Medium-term (optimal for intervention)
- 30 minutes: Long-term planning
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.interpolate import interp1d
from tqdm import tqdm

PREDICTION_FEATURE_VERSION = "real_empatica_acc_ppg_no_ibi_v10"
PREDICTION_FEATURE_DIM = 224


@dataclass
class PredictionConfig:
    """Configuration for seizure prediction pipeline."""

    prediction_horizons: List[int] = None

    preictal_duration: int = 1800

    postictal_duration: int = 1800

    default_horizon: int = 900

    min_interictal_gap: int = 3600

    window_size: int = 32
    window_stride: int = 32

    sequence_length: int = 60

    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [300, 900, 1800]


class SeizurePredictionDataGen:
    """
    Data generator for experimental pre-ictal risk modeling.

    Difference from detection:
    - Labels windows as pre-ictal candidates before annotated seizure onset
    - Supports risk-ranking experiments across multiple horizons
    - Supports multiple prediction horizons
    """

    TARGET_RATE = 32

    def __init__(
        self,
        session_paths: List[str],
        seizure_intervals: Dict[str, List[Tuple[float, float]]],
        config: PredictionConfig = None,
        batch_size: int = 64,
        shuffle: bool = True,
        cache_dir: Optional[str] = None,
        cache_enabled: bool = True,
    ):
        self.session_paths = session_paths
        self.seizure_intervals = seizure_intervals
        self.config = config or PredictionConfig()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prediction_outputs_local",
            "cache",
        )

        self._sequences = []
        self._labels = {}
        self._timestamps = []

        for horizon in self.config.prediction_horizons:
            self._labels[horizon] = []
        self._prepare_dataset()

    def _config_signature(self) -> str:
        """Create a stable signature for the cache inputs."""
        payload = {
            "session_paths": sorted(self.session_paths),
            "seizure_intervals": {
                key: [[float(start), float(end)] for start, end in value]
                for key, value in sorted(self.seizure_intervals.items())
            },
            "config": {
                "feature_version": PREDICTION_FEATURE_VERSION,
                "prediction_horizons": list(self.config.prediction_horizons),
                "preictal_duration": self.config.preictal_duration,
                "postictal_duration": self.config.postictal_duration,
                "default_horizon": self.config.default_horizon,
                "min_interictal_gap": self.config.min_interictal_gap,
                "window_size": self.config.window_size,
                "window_stride": self.config.window_stride,
                "sequence_length": self.config.sequence_length,
            },
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self) -> str:
        return os.path.join(
            self.cache_dir, f"prediction_cache_{self._config_signature()}.npz"
        )

    def _load_cache(self) -> bool:
        """Load cached arrays if available."""
        if not self.cache_enabled:
            return False
        cache_path = self._cache_path()
        if not os.path.exists(cache_path):
            return False
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                self._sequences = cached["sequences"]
                self._timestamps = cached["timestamps"]
                for horizon in self.config.prediction_horizons:
                    self._labels[horizon] = cached[f"labels_{horizon}"]
            expected_shape = (self.config.sequence_length, PREDICTION_FEATURE_DIM)
            if self._sequences.shape[1:] != expected_shape:
                print(
                    "Prediction cache shape mismatch, rebuilding dataset: "
                    f"found={self._sequences.shape}, expected=(*, {expected_shape[0]}, {expected_shape[1]})"
                )
                self._sequences = []
                self._timestamps = []
                self._labels = {
                    horizon: [] for horizon in self.config.prediction_horizons
                }
                return False
            print(f"Loaded cached prediction dataset: {cache_path}")
            return True
        except Exception as exc:
            print(f"Cache load failed, rebuilding dataset: {exc}")
            return False

    def _save_cache(self) -> None:
        if not self.cache_enabled:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._cache_path()
        metadata_path = cache_path.replace(".npz", ".json")
        payload = {
            "session_paths": sorted(self.session_paths),
            "cache_signature": self._config_signature(),
            "feature_version": PREDICTION_FEATURE_VERSION,
            "num_sequences": int(len(self._sequences)),
            "feature_dim": PREDICTION_FEATURE_DIM,
            "prediction_horizons": list(self.config.prediction_horizons),
        }

        arrays = {
            "sequences": self._sequences,
            "timestamps": self._timestamps,
        }
        for horizon in self.config.prediction_horizons:
            arrays[f"labels_{horizon}"] = self._labels[horizon]
        np.savez_compressed(cache_path, **arrays)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved cached prediction dataset: {cache_path}")

    def _read_sensor_file(
        self, filepath: str
    ) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
        """Read Empatica E4 sensor file."""
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()
            if len(lines) < 3:
                return None, None, None
            start_time = float(lines[0].strip().split(",")[0])
            sample_rate = float(lines[1].strip().split(",")[0])

            data = []
            for line in lines[2:]:
                if line.strip():
                    values = [float(v) for v in line.strip().split(",")]
                    data.append(values)
            return start_time, sample_rate, np.array(data)
        except Exception as e:
            return None, None, None

    def _resample_to_target(
        self,
        data: np.ndarray,
        original_rate: float,
        start_time: float,
        target_timestamps: np.ndarray,
    ) -> np.ndarray:
        """Resample sensor data to target timestamps."""
        n_samples = len(data)
        original_timestamps = start_time + np.arange(n_samples) / original_rate

        if data.ndim == 1:
            data = data.reshape(-1, 1)
        resampled = np.zeros((len(target_timestamps), data.shape[1]))
        for i in range(data.shape[1]):
            interp_func = interp1d(
                original_timestamps,
                data[:, i],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            resampled[:, i] = interp_func(target_timestamps)
        return resampled

    def _derive_hr_from_bvp(self, bvp: np.ndarray) -> np.ndarray:
        """Fallback HR-like feature from BVP when HR.csv is absent."""
        series = bvp.reshape(-1).astype(np.float32)
        if len(series) == 0:
            return np.empty((0, 1), dtype=np.float32)
        centered = series - np.median(series)
        scale = float(np.std(centered))
        prominence = max(scale * 0.25, 1e-6)
        min_distance = max(1, int(self.TARGET_RATE * 0.35))
        peaks, _ = signal.find_peaks(
            centered, distance=min_distance, prominence=prominence
        )
        peak_indicator = np.zeros(len(series), dtype=np.float32)
        peak_indicator[peaks] = 1.0
        window = max(1, int(self.TARGET_RATE * 8.0))
        kernel = np.ones(window, dtype=np.float32)
        peak_count = np.convolve(peak_indicator, kernel, mode="same")
        window_seconds = window / self.TARGET_RATE
        hr = (peak_count / window_seconds) * 60.0
        return hr.reshape(-1, 1).astype(np.float32)

    def _get_prediction_label(
        self, timestamp: float, seizure_times: List[Tuple[float, float]], horizon: int
    ) -> int:
        """
        Get prediction label for a timestamp.

        Labels:
        0 = Interictal (safe, no seizure coming within horizon)
        1 = Pre-ictal (seizure coming within horizon) <- PREDICTION TARGET
        2 = Ictal (during seizure)
        3 = Post-ictal (recovery, usually excluded)

        For binary prediction, we map: 0,3 -> 0 (negative), 1 -> 1 (positive)
        We exclude ictal (2) from prediction training.
        """
        for sz_start, sz_end in seizure_times:
            if sz_start <= timestamp <= sz_end:
                return 2
            if sz_end < timestamp <= sz_end + self.config.postictal_duration:
                return 3
        for sz_start, sz_end in seizure_times:
            preictal_start = sz_start - self.config.preictal_duration
            if preictal_start <= timestamp < sz_start:
                previous_ends = [
                    prev_end for _, prev_end in seizure_times if prev_end < timestamp
                ]
                if (
                    previous_ends
                    and timestamp - max(previous_ends) < self.config.min_interictal_gap
                ):
                    return 3
                time_to_seizure = sz_start - timestamp
                if time_to_seizure <= horizon:
                    return 1
        return 0

    def _load_session(self, session_path: str) -> Optional[Dict]:
        """Load and process a single session for prediction."""
        session_id = os.path.basename(session_path)

        sensors = {}
        start_times = {}
        rates = {}

        for sensor, filename in [
            ("ACC", "ACC.csv"),
            ("BVP", "BVP.csv"),
            ("HR", "HR.csv"),
            ("EDA", "EDA.csv"),
            ("TEMP", "TEMP.csv"),
        ]:
            filepath = os.path.join(session_path, filename)
            if os.path.exists(filepath):
                start, rate, data = self._read_sensor_file(filepath)
                if data is not None and len(data) > 0:
                    sensors[sensor] = data
                    start_times[sensor] = start
                    rates[sensor] = rate
        if "ACC" not in sensors:
            return None
        acc_start = start_times["ACC"]
        acc_duration = len(sensors["ACC"]) / rates["ACC"]
        n_samples = int(acc_duration * self.TARGET_RATE)

        timestamps = acc_start + np.arange(n_samples) / self.TARGET_RATE

        resampled = {}
        for sensor, data in sensors.items():
            resampled[sensor] = self._resample_to_target(
                data, rates[sensor], start_times[sensor], timestamps
            )
        acc = resampled.get("ACC", np.zeros((n_samples, 3)))
        bvp = resampled.get("BVP", np.zeros((n_samples, 1)))
        hr = resampled.get("HR")
        if hr is None:
            hr = self._derive_hr_from_bvp(bvp[:, 0])
        eda = resampled.get("EDA", np.zeros((n_samples, 1)))
        temp = resampled.get("TEMP", np.zeros((n_samples, 1)))

        primary = acc
        secondary = np.hstack([bvp, hr, eda, temp])

        session_start = float(timestamps[0])
        session_end = float(timestamps[-1])
        seizure_times = [
            (start, end)
            for start, end in self.seizure_intervals.get(session_id, [])
            if start <= session_end and end >= session_start
        ]
        if not seizure_times:
            print(f"    [WARNING] No seizures found for session {session_id}")
        return {
            "timestamps": timestamps,
            "primary": primary,
            "secondary": secondary,
            "seizure_times": seizure_times,
            "session_id": session_id,
        }

    def _create_sequences(self, session_data: Dict) -> None:
        """Create sequences for temporal prediction model."""
        timestamps = session_data["timestamps"]
        primary = session_data["primary"]
        secondary = session_data["secondary"]
        seizure_times = session_data["seizure_times"]

        window_size = self.config.window_size
        stride = self.config.window_stride
        seq_len = self.config.sequence_length

        n_samples = len(timestamps)

        windows_primary = []
        windows_secondary = []
        window_timestamps = []

        for i in range(0, n_samples - window_size + 1, stride):
            win_p = primary[i : i + window_size]
            win_s = secondary[i : i + window_size]
            win_t = timestamps[i + window_size // 2]

            windows_primary.append(win_p)
            windows_secondary.append(win_s)
            window_timestamps.append(win_t)
        if len(windows_primary) < seq_len:
            return
        windows_primary = np.array(windows_primary)
        windows_secondary = np.array(windows_secondary)
        window_timestamps = np.array(window_timestamps)

        for i in range(0, len(windows_primary) - seq_len + 1, seq_len // 2):
            seq_p = windows_primary[i : i + seq_len]
            seq_s = windows_secondary[i : i + seq_len]
            seq_t = window_timestamps[i : i + seq_len]

            seq_p_flat = seq_p.reshape(seq_len, -1)
            seq_s_flat = seq_s.reshape(seq_len, -1)

            combined = np.concatenate([seq_p_flat, seq_s_flat], axis=1)

            center_time = seq_t[-1]

            labels_valid = True
            horizon_labels = {}

            for horizon in self.config.prediction_horizons:
                label = self._get_prediction_label(center_time, seizure_times, horizon)

                if label in (2, 3):
                    labels_valid = False
                    break
                binary_label = 1 if label == 1 else 0
                horizon_labels[horizon] = binary_label
            if labels_valid:
                self._sequences.append(combined)
                self._timestamps.append(center_time)
                for horizon, label in horizon_labels.items():
                    self._labels[horizon].append(label)

    def _prepare_dataset(self) -> None:
        """Prepare the full dataset."""
        if self._load_cache():
            print(f"\nDataset prepared from cache:")
            print(f"  Total sequences: {len(self._sequences)}")
            print(f"  Sequence shape: {self._sequences.shape}")
            return
        print(f"Loading {len(self.session_paths)} sessions for PREDICTION...")

        for session_path in tqdm(self.session_paths, desc="Processing sessions"):
            session_data = self._load_session(session_path)
            if session_data is not None:
                self._create_sequences(session_data)
        self._sequences = np.array(self._sequences)
        self._timestamps = np.array(self._timestamps)
        for horizon in self.config.prediction_horizons:
            self._labels[horizon] = np.array(self._labels[horizon])
        print(f"\nDataset prepared:")
        print(f"  Total sequences: {len(self._sequences)}")
        print(f"  Sequence shape: {self._sequences.shape}")

        all_positives = 0
        for horizon in self.config.prediction_horizons:
            n_preictal = np.sum(self._labels[horizon] == 1)
            pct = (
                100 * n_preictal / len(self._labels[horizon])
                if len(self._labels[horizon]) > 0
                else 0
            )
            print(f"  Horizon {horizon//60}min: {n_preictal} pre-ictal ({pct:.2f}%)")
            all_positives += n_preictal
        if all_positives == 0:
            print(f"\nWARNING: No pre-ictal samples found!")
            print(f"   This may indicate:")
            print(f"   1. Seizure label file format mismatch")
            print(f"   2. Seizure times don't overlap with session data")
            print(f"   3. All samples are interictal (safe) periods")
        self._save_cache()

    def normalize(self) -> None:
        """Normalize sequences."""
        if len(self._sequences) == 0:
            return
        mean = np.mean(self._sequences, axis=(0, 1), keepdims=True)
        std = np.std(self._sequences, axis=(0, 1), keepdims=True) + 1e-8
        self._sequences = (self._sequences - mean) / std

    def get_data(self, horizon: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a specific prediction horizon.

        Args:
            horizon: Prediction horizon in seconds (default: 900 = 15min)

        Returns:
            (sequences, labels)
        """
        if horizon is None:
            horizon = self.config.default_horizon
        return self._sequences, self._labels[horizon]

    def get_multi_horizon_data(self) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Get data with labels for all prediction horizons."""
        return self._sequences, self._labels


def get_session_paths(base_path: str) -> List[str]:
    """Get all valid session paths."""
    session_paths = []

    for patient_folder in os.listdir(base_path):
        patient_path = os.path.join(base_path, patient_folder)
        if not os.path.isdir(patient_path) or not patient_folder.startswith("Mayo_"):
            continue
        for item in os.listdir(patient_path):
            item_path = os.path.join(patient_path, item)
            if os.path.isdir(item_path) and "_" in item:
                required = ["ACC.csv", "BVP.csv", "EDA.csv", "TEMP.csv"]
                if all(os.path.exists(os.path.join(item_path, f)) for f in required):
                    session_paths.append(item_path)
        chronic_path = os.path.join(patient_path, "Empatica_Chronic")
        if os.path.exists(chronic_path):
            for item in os.listdir(chronic_path):
                item_path = os.path.join(chronic_path, item)
                if os.path.isdir(item_path) and "_" in item:
                    required = ["ACC.csv", "BVP.csv", "EDA.csv", "TEMP.csv"]
                    if all(
                        os.path.exists(os.path.join(item_path, f)) for f in required
                    ):
                        session_paths.append(item_path)
    return session_paths
