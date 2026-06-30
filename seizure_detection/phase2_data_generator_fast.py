"""
Phase 2: OPTIMIZED Data Generator with Preprocessing
=====================================================
High-performance SeizureDataGen optimized for ~1 day runtime.

Optimizations:
- Vectorized resampling (100-1000x faster than per-sample loops)
- Multiprocessing for parallel session loading
- Numpy stride tricks for fast windowing
- Smart session selection (prioritize seizure sessions)
- Progress logging with ETA
- SOTA DSP: Orientation-invariant ACC vector magnitude, 2-8Hz clonic bandpass, phasic EDA SCR, local session z-score centering.
"""

import os
import sys
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


@dataclass
class SensorData:
    """Container for a single sensor's data."""

    timestamps: np.ndarray
    values: np.ndarray
    sample_rate: float
    start_time: float


def fast_read_sensor_file(filepath: str) -> Optional[SensorData]:
    """
    Read Empatica E4 sensor CSV file using numpy for speed.
    ~10x faster than line-by-line reading.
    """
    try:

        with open(filepath, "r") as f:
            line1 = f.readline().strip()
            line2 = f.readline().strip()
        start_time = float(line1.split(",")[0])
        sample_rate = float(line2.split(",")[0])

        try:
            values = np.loadtxt(filepath, delimiter=",", skiprows=2)
            if values.ndim == 0:
                values = np.array([values])
        except Exception:
            with open(filepath, "r") as f:
                lines = f.readlines()[2:]
            if not lines:
                return None
            values = np.array(
                [[float(x) for x in line.strip().split(",")] for line in lines]
            )

        if len(values) == 0:
            return None
        duration = len(values) / sample_rate
        timestamps = np.linspace(
            start_time, start_time + duration, len(values), endpoint=False
        )

        return SensorData(
            timestamps=timestamps,
            values=values,
            sample_rate=sample_rate,
            start_time=start_time,
        )
    except Exception as e:
        return None


def fast_downsample_2x(values: np.ndarray) -> np.ndarray:
    """
    Fast 2:1 downsampling (64Hz -> 32Hz) using reshape+mean.
    ~1000x faster than per-sample loop.
    """
    n = len(values)

    if n % 2 == 1:
        values = values[:-1]
    if values.ndim == 1:
        return values.reshape(-1, 2).mean(axis=1)
    else:

        return values.reshape(-1, 2, values.shape[1]).mean(axis=1)


def fast_upsample_8x(values: np.ndarray, n_target: int) -> np.ndarray:
    """
    Fast 1:8 upsampling (4Hz -> 32Hz) using numpy repeat.
    ~100x faster than scipy interp1d.
    """
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    upsampled = np.repeat(values, 8, axis=0)

    if len(upsampled) >= n_target:
        return upsampled[:n_target]
    else:

        padding = np.tile(upsampled[-1:], (n_target - len(upsampled), 1))
        return np.vstack([upsampled, padding])


def fast_resample(
    data: SensorData, target_length: int, target_rate: float = 32.0
) -> np.ndarray:
    """
    Fast resampling to target rate using vectorized operations.
    """
    values = data.values
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if data.sample_rate == target_rate:

        if len(values) >= target_length:
            return values[:target_length]
        else:
            padding = np.tile(values[-1:], (target_length - len(values), 1))
            return np.vstack([values, padding])
    elif data.sample_rate == 64.0 and target_rate == 32.0:

        downsampled = fast_downsample_2x(values)
        if len(downsampled) >= target_length:
            return downsampled[:target_length]
        else:
            padding = np.tile(downsampled[-1:], (target_length - len(downsampled), 1))
            return np.vstack([downsampled, padding])
    elif data.sample_rate == 4.0 and target_rate == 32.0:

        return fast_upsample_8x(values, target_length)
    else:

        resampled = signal.resample(values, target_length, axis=0)
        return resampled


def read_hr_file(filepath: str, target_timestamps: np.ndarray) -> Optional[np.ndarray]:
    """Read Empatica HR.csv and interpolate it to ACC timestamps."""
    if not os.path.exists(filepath):
        return None
    data = fast_read_sensor_file(filepath)
    if data is None:
        return None
    values = data.values
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if len(values) == 0:
        return None
    resampled = np.zeros((len(target_timestamps), values.shape[1]), dtype=np.float32)
    for idx in range(values.shape[1]):
        resampled[:, idx] = np.interp(
            target_timestamps,
            data.timestamps,
            values[:, idx],
            left=values[0, idx],
            right=values[-1, idx],
        )
    return resampled.astype(np.float32)


def fast_create_windows(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Create overlapping windows using numpy stride tricks.
    ~50x faster than Python loop.
    """
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size) // stride + 1

    if n_windows <= 0:
        return np.array([]).reshape(0, window_size, n_features)
    shape = (n_windows, window_size, n_features)
    strides = (data.strides[0] * stride, data.strides[0], data.strides[1])

    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    return windows.copy()


def fast_create_labels(
    timestamps: np.ndarray, seizure_intervals: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Create binary labels using vectorized operations.
    """
    labels = np.zeros(len(timestamps), dtype=np.float32)

    for sz_start, sz_end in seizure_intervals:
        mask = (timestamps >= sz_start) & (timestamps <= sz_end)
        labels[mask] = 1.0
    return labels


def derive_hr_from_bvp(bvp_values: np.ndarray, sample_rate: float = 32.0) -> np.ndarray:
    """Derive an HR-like fallback from BVP when HR.csv is absent."""
    series = bvp_values.reshape(-1).astype(np.float32)
    if len(series) == 0:
        return np.empty((0, 1), dtype=np.float32)
    centered = series - np.median(series)
    scale = float(np.std(centered))
    prominence = max(scale * 0.25, 1e-6)
    min_distance = max(1, int(sample_rate * 0.35))
    peaks, _ = signal.find_peaks(centered, distance=min_distance, prominence=prominence)

    peak_indicator = np.zeros(len(series), dtype=np.float32)
    peak_indicator[peaks] = 1.0

    window = max(1, int(sample_rate * 8.0))
    kernel = np.ones(window, dtype=np.float32)
    peak_count = np.convolve(peak_indicator, kernel, mode="same")

    window_seconds = window / sample_rate
    hr_proxy = (peak_count / window_seconds) * 60.0
    return hr_proxy.reshape(-1, 1).astype(np.float32)


def preprocess_session_signals(
    acc_resampled: np.ndarray,
    bvp_resampled: np.ndarray,
    hr_resampled: np.ndarray,
    eda_resampled: np.ndarray,
    temp_resampled: np.ndarray,
    sample_rate: float = 32.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SOTA physiological signal processing:
    1. ACC: Compute Euclidean vector magnitude (L2 norm minus 1g ~ 64.0), apply 2-8 Hz Butterworth bandpass filter for clonic shaking, and compute rolling energy envelope.
    2. EDA: Extract phasic Skin Conductance Response (SCR) using 0.05 Hz high-pass filter.
    3. Session-level z-score normalization on both primary and secondary features.
    """
    nyq = 0.5 * sample_rate

    # 1. Primary (ACC features: magnitude, 2-8 Hz clonic bandpass, rolling energy envelope)
    if acc_resampled.shape[1] >= 3:
        acc_mag = np.sqrt(np.sum(acc_resampled[:, :3] ** 2, axis=1, keepdims=True))
    else:
        acc_mag = np.abs(acc_resampled)

    try:
        b_bp, a_bp = signal.butter(
            3, [2.0 / nyq, min(8.0 / nyq, 0.95)], btype="bandpass"
        )
        acc_clonic = signal.filtfilt(b_bp, a_bp, acc_mag.flatten()).reshape(-1, 1)
    except Exception:
        acc_clonic = acc_mag.copy()

    win_len = max(1, int(sample_rate * 1.0))
    kernel = np.ones(win_len, dtype=np.float32) / win_len
    acc_energy = np.sqrt(
        np.convolve((acc_clonic.flatten()) ** 2, kernel, mode="same") + 1e-8
    ).reshape(-1, 1)

    primary_raw = np.hstack([acc_mag, acc_clonic, acc_energy]).astype(np.float32)
    p_mean = np.mean(primary_raw, axis=0, keepdims=True)
    p_std = np.std(primary_raw, axis=0, keepdims=True) + 1e-8
    primary = (primary_raw - p_mean) / p_std

    # 2. Secondary (BVP, HR, EDA phasic SCR, TEMP features)
    bvp_col = bvp_resampled.reshape(-1, 1)
    hr_col = hr_resampled.reshape(-1, 1)
    temp_col = temp_resampled.reshape(-1, 1)

    eda_flat = eda_resampled.flatten()
    try:
        b_hp, a_hp = signal.butter(2, 0.05 / nyq, btype="highpass")
        eda_phasic = signal.filtfilt(b_hp, a_hp, eda_flat).reshape(-1, 1)
    except Exception:
        eda_phasic = eda_resampled.reshape(-1, 1)

    secondary_raw = np.hstack([bvp_col, hr_col, eda_phasic, temp_col]).astype(
        np.float32
    )
    s_mean = np.mean(secondary_raw, axis=0, keepdims=True)
    s_std = np.std(secondary_raw, axis=0, keepdims=True) + 1e-8
    secondary = (secondary_raw - s_mean) / s_std

    return primary.astype(np.float32), secondary.astype(np.float32)


def load_single_session(
    args: Tuple,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load and preprocess a single session. Designed for multiprocessing.

    Returns:
        (primary_windows, secondary_windows, labels, n_seizure_windows,
        session_ids, window_starts, window_ends) or None
    """
    session_path, seizure_intervals_for_session, window_size, stride = args
    session_id = os.path.basename(session_path)

    try:

        acc_data = fast_read_sensor_file(os.path.join(session_path, "ACC.csv"))
        bvp_data = fast_read_sensor_file(os.path.join(session_path, "BVP.csv"))
        eda_data = fast_read_sensor_file(os.path.join(session_path, "EDA.csv"))
        temp_data = fast_read_sensor_file(os.path.join(session_path, "TEMP.csv"))

        if any(d is None for d in [acc_data, bvp_data, eda_data, temp_data]):
            return None
        target_length = len(acc_data.values)
        target_timestamps = acc_data.timestamps

        acc_resampled = acc_data.values
        if acc_resampled.ndim == 1:
            acc_resampled = acc_resampled.reshape(-1, 1)
        bvp_resampled = fast_resample(bvp_data, target_length)
        eda_resampled = fast_resample(eda_data, target_length)
        temp_resampled = fast_resample(temp_data, target_length)

        hr_resampled = read_hr_file(
            os.path.join(session_path, "HR.csv"), target_timestamps
        )
        if hr_resampled is None:
            hr_resampled = derive_hr_from_bvp(bvp_resampled[:, 0], sample_rate=32.0)
            
        primary, secondary = preprocess_session_signals(
            acc_resampled,
            bvp_resampled,
            hr_resampled,
            eda_resampled,
            temp_resampled,
            sample_rate=32.0,
        )

        session_start = float(target_timestamps[0])
        session_end = float(target_timestamps[-1])
        seizure_intervals_for_session = [
            (start, end)
            for start, end in seizure_intervals_for_session
            if start <= session_end and end >= session_start
        ]

        sample_labels = fast_create_labels(
            target_timestamps, seizure_intervals_for_session
        )

        primary_windows = fast_create_windows(primary, window_size, stride)
        secondary_windows = fast_create_windows(secondary, window_size, stride)
        label_windows = fast_create_windows(
            sample_labels.reshape(-1, 1), window_size, stride
        )

        window_labels = (label_windows.max(axis=(1, 2)) > 0.5).astype(np.float32)

        n_seizure = int(window_labels.sum())
        n_windows = len(window_labels)
        session_ids = np.array([session_id] * n_windows, dtype=object)
        window_starts = np.array(
            [target_timestamps[idx * stride] for idx in range(n_windows)], dtype=np.float64
        )
        window_ends = np.array(
            [
                target_timestamps[idx * stride + window_size - 1]
                for idx in range(n_windows)
            ],
            dtype=np.float64,
        )

        return (
            primary_windows,
            secondary_windows,
            window_labels,
            n_seizure,
            session_ids,
            window_starts,
            window_ends,
        )
    except Exception as e:
        print(f"Error processing {session_id}: {e}", flush=True)
        return None


class SeizureDataGenFast:
    """
    High-performance Data Generator for Seizure Detection.

    Optimizations:
    - Vectorized resampling (100-1000x faster)
    - Multiprocessing for parallel session loading
    - Numpy stride tricks for fast windowing
    - Smart session selection
    """

    TARGET_RATE = 32

    def __init__(
        self,
        session_paths: List[str],
        seizure_intervals: Dict[str, List[Tuple[float, float]]],
        window_size: int = 32,
        stride: int = 16,
        batch_size: int = 64,
        shuffle: bool = True,
        n_workers: int = None,
        prioritize_seizures: bool = True,
    ):
        self.session_paths = session_paths
        self.seizure_intervals = seizure_intervals
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.prioritize_seizures = prioritize_seizures

        if prioritize_seizures:
            self._sort_sessions_by_seizures()
        self._prepare_dataset()

    def _sort_sessions_by_seizures(self):
        """Sort sessions so seizure-containing sessions come first."""

        def has_seizure(path):
            return len(get_actual_session_intervals(path, self.seizure_intervals)) > 0

        seizure_sessions = [p for p in self.session_paths if has_seizure(p)]
        normal_sessions = [p for p in self.session_paths if not has_seizure(p)]

        print(f"  Sessions with seizures: {len(seizure_sessions)}")
        print(f"  Sessions without seizures: {len(normal_sessions)}")

        self.session_paths = seizure_sessions + normal_sessions

    def _prepare_dataset(self):
        """Prepare all windows using multiprocessing."""
        print(f"\nPreparing dataset from {len(self.session_paths)} sessions...")
        print(f"  Using {self.n_workers} CPU cores for parallel processing")
        print(f"  Window size: {self.window_size}, Stride: {self.stride}")

        start_time = time.time()

        args_list = []
        for path in self.session_paths:
            seizures = get_actual_session_intervals(path, self.seizure_intervals)
            args_list.append((path, seizures, self.window_size, self.stride))
        all_primary = []
        all_secondary = []
        all_labels = []
        all_session_ids = []
        all_window_starts = []
        all_window_ends = []
        total_seizure_windows = 0
        processed = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(load_single_session, args): i
                for i, args in enumerate(args_list)
            }

            for future in as_completed(futures):
                idx = futures[future]
                session_id = os.path.basename(self.session_paths[idx])
                processed += 1

                try:
                    result = future.result()
                    if result is not None:
                        if len(result) == 4:
                            primary_w, secondary_w, labels_w, n_sz = result
                            session_ids_w = np.array(
                                [session_id] * len(labels_w), dtype=object
                            )
                            window_starts_w = np.full(len(labels_w), np.nan)
                            window_ends_w = np.full(len(labels_w), np.nan)
                        else:
                            (
                                primary_w,
                                secondary_w,
                                labels_w,
                                n_sz,
                                session_ids_w,
                                window_starts_w,
                                window_ends_w,
                            ) = result
                        all_primary.append(primary_w)
                        all_secondary.append(secondary_w)
                        all_labels.append(labels_w)
                        all_session_ids.append(session_ids_w)
                        all_window_starts.append(window_starts_w)
                        all_window_ends.append(window_ends_w)
                        total_seizure_windows += n_sz

                        elapsed = time.time() - start_time
                        rate = processed / elapsed
                        remaining = (
                            (len(self.session_paths) - processed) / rate
                            if rate > 0
                            else 0
                        )

                        print(
                            f"  [{processed}/{len(self.session_paths)}] {session_id}: "
                            f"{len(labels_w)} windows, {n_sz} seizure | "
                            f"ETA: {remaining/60:.1f} min",
                            flush=True,
                        )
                    else:
                        failed += 1
                        print(
                            f"  [{processed}/{len(self.session_paths)}] {session_id}: SKIP",
                            flush=True,
                        )
                except Exception as e:
                    failed += 1
                    print(
                        f"  [{processed}/{len(self.session_paths)}] {session_id}: ERROR - {e}",
                        flush=True,
                    )
        if all_primary:
            self._windows_primary = np.concatenate(all_primary, axis=0)
            self._windows_secondary = np.concatenate(all_secondary, axis=0)
            self._labels = np.concatenate(all_labels, axis=0)
            self._window_session_ids = np.concatenate(all_session_ids, axis=0)
            self._window_starts = np.concatenate(all_window_starts, axis=0)
            self._window_ends = np.concatenate(all_window_ends, axis=0)
        else:
            self._windows_primary = np.array([]).reshape(0, self.window_size, 3)
            self._windows_secondary = np.array([]).reshape(0, self.window_size, 4)
            self._labels = np.array([])
            self._window_session_ids = np.array([], dtype=object)
            self._window_starts = np.array([], dtype=np.float64)
            self._window_ends = np.array([], dtype=np.float64)
        elapsed = time.time() - start_time
        print(f"\n  Completed in {elapsed/60:.1f} minutes")
        print(f"  Processed: {processed - failed}/{len(self.session_paths)} sessions")
        print(f"  Total windows: {len(self._labels):,}")
        print(f"  Seizure windows: {total_seizure_windows:,}")
        print(f"  Normal windows: {len(self._labels) - total_seizure_windows:,}")
        print(f"  Primary shape: {self._windows_primary.shape}")
        print(f"  Secondary shape: {self._windows_secondary.shape}")

    def normalize(self):
        """Normalize features to zero mean and unit variance."""
        # Note: Sessions are already locally z-scored inside load_single_session.
        # We record global statistics for compatibility without shifting session anomalies.
        if len(self._windows_primary) > 0:
            self._primary_mean = np.zeros((1, 1, self._windows_primary.shape[-1]), dtype=np.float32)
            self._primary_std = np.ones((1, 1, self._windows_primary.shape[-1]), dtype=np.float32)
        else:
            self._primary_mean = np.zeros((1, 1, 3), dtype=np.float32)
            self._primary_std = np.ones((1, 1, 3), dtype=np.float32)

        if len(self._windows_secondary) > 0:
            self._secondary_mean = np.zeros((1, 1, self._windows_secondary.shape[-1]), dtype=np.float32)
            self._secondary_std = np.ones((1, 1, self._windows_secondary.shape[-1]), dtype=np.float32)
        else:
            self._secondary_mean = np.zeros((1, 1, 4), dtype=np.float32)
            self._secondary_std = np.ones((1, 1, 4), dtype=np.float32)

        print(
            f"  Normalized: Session-level z-scoring applied per recording session."
        )

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get all data as numpy arrays."""
        return self._windows_primary, self._windows_secondary, self._labels

    def get_metadata(self) -> Dict[str, np.ndarray]:
        """Get per-window metadata aligned with get_data arrays."""
        return {
            "session_ids": self._window_session_ids,
            "window_starts": self._window_starts,
            "window_ends": self._window_ends,
        }

    def __len__(self) -> int:
        return len(self._labels) // self.batch_size


def fast_count_data_rows(filepath: str) -> int:
    """Count CSV data rows after the two Empatica header rows."""
    try:
        with open(filepath, "rb") as handle:
            line_count = 0
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                line_count += chunk.count(b"\n")
        return max(0, line_count - 2)
    except Exception:
        return 0


def get_actual_session_bounds(session_path: str) -> Optional[Tuple[float, float]]:
    """Return exact ACC-derived session start/end timestamps."""
    acc_path = os.path.join(session_path, "ACC.csv")
    try:
        with open(acc_path, "r") as handle:
            start_time = float(handle.readline().strip().split(",")[0])
            sample_rate = float(handle.readline().strip().split(",")[0])
        n_samples = fast_count_data_rows(acc_path)
        if n_samples <= 0 or sample_rate <= 0:
            return None
        return start_time, start_time + (n_samples / sample_rate)
    except Exception:
        return None


def get_actual_session_intervals(
    session_path: str,
    seizure_intervals: Dict[str, List[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    """Filter session interval candidates using exact ACC-derived bounds."""
    session_id = os.path.basename(session_path)
    candidates = seizure_intervals.get(session_id, [])
    if not candidates:
        return []
    bounds = get_actual_session_bounds(session_path)
    if bounds is None:
        return []
    session_start, session_end = bounds
    return [
        (start, end)
        for start, end in candidates
        if start < session_end and end > session_start
    ]


def get_session_paths_smart(
    base_path: str,
    seizure_intervals: Dict[str, List[Tuple[float, float]]],
    max_sessions: int = 50,
    prioritize_seizures: bool = True,
    random_state: int = 42,
) -> List[str]:
    """
    Get session paths with smart selection:
    - Prioritize sessions containing seizures
    - Balance seizure vs non-seizure sessions
    """
    all_sessions = []

    for patient_folder in os.listdir(base_path):
        patient_path = os.path.join(base_path, patient_folder)
        if not os.path.isdir(patient_path) or not patient_folder.startswith("Mayo_"):
            continue
        for item in os.listdir(patient_path):
            item_path = os.path.join(patient_path, item)
            if os.path.isdir(item_path) and "_" in item:
                required = ["ACC.csv", "BVP.csv", "EDA.csv", "TEMP.csv"]
                if all(os.path.exists(os.path.join(item_path, f)) for f in required):
                    has_seizure = (
                        len(get_actual_session_intervals(item_path, seizure_intervals))
                        > 0
                    )
                    all_sessions.append((item_path, has_seizure))
        chronic_path = os.path.join(patient_path, "Empatica_Chronic")
        if os.path.exists(chronic_path):
            for item in os.listdir(chronic_path):
                item_path = os.path.join(chronic_path, item)
                if os.path.isdir(item_path) and "_" in item:
                    required = ["ACC.csv", "BVP.csv", "EDA.csv", "TEMP.csv"]
                    if all(
                        os.path.exists(os.path.join(item_path, f)) for f in required
                    ):
                        has_seizure = (
                            len(
                                get_actual_session_intervals(
                                    item_path, seizure_intervals
                                )
                            )
                            > 0
                        )
                        all_sessions.append((item_path, has_seizure))
    seizure_sessions = [p for p, has_sz in all_sessions if has_sz]
    normal_sessions = [p for p, has_sz in all_sessions if not has_sz]

    rng = np.random.default_rng(random_state)
    rng.shuffle(seizure_sessions)
    rng.shuffle(normal_sessions)

    print(f"\nSmart session selection:")
    print(f"  Total sessions: {len(all_sessions)}")
    print(f"  Sessions with actual seizure overlap: {len(seizure_sessions)}")
    print(f"  Sessions without actual seizure overlap: {len(normal_sessions)}")

    if prioritize_seizures:
        seizure_fraction = float(
            os.environ.get("DETECTION_SEIZURE_SESSION_FRACTION", "0.7")
        )
        seizure_fraction = min(max(seizure_fraction, 0.1), 1.0)
        target_seizure = min(
            len(seizure_sessions), int(round(max_sessions * seizure_fraction))
        )
        if seizure_sessions and target_seizure == 0:
            target_seizure = 1
        selected = seizure_sessions[:target_seizure]
        remaining = max_sessions - len(selected)
        selected += normal_sessions[:remaining]
        if len(selected) < max_sessions:
            selected += seizure_sessions[target_seizure:max_sessions]
    else:

        all_paths = [p for p, _ in all_sessions]
        rng.shuffle(all_paths)
        selected = all_paths[:max_sessions]
    print(f"  Selected: {len(selected)} sessions")

    return selected


SeizureDataGen = SeizureDataGenFast


def get_session_paths(base_path: str) -> List[str]:
    """Get all valid session paths from base directory."""
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
