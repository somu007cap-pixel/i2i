"""
Phase 1: Data Health Check & Label Parsing
==========================================
Deep analysis of Empatica E4 seizure detection dataset from Zenodo.

This script:
1. Recursively scans folders for model-usable sessions and six-signal availability
2. Parses seizure ground truth files (one onset timestamp per line)
3. Verifies sampling rates match expected values
4. Generates a comprehensive data health report
"""

import os
import glob
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np

DEFAULT_SEIZURE_DURATION_SECONDS = int(
    os.environ.get("SEIZURE_DURATION_SECONDS", "300")
)
SEIZURE_MERGE_GAP_SECONDS = int(os.environ.get("SEIZURE_MERGE_GAP_SECONDS", "0"))
PHASE1_EXACT_DURATIONS = os.environ.get("PHASE1_EXACT_DURATIONS", "0") == "1"
PHASE1_SESSION_DURATION_SECONDS = float(
    os.environ.get("PHASE1_SESSION_DURATION_SECONDS", "129600")
)
PHASE1_CACHE_VERSION = "phase1_sensor_export_quality_v7"


@dataclass
class SessionInfo:
    """Information about a single recording session."""

    session_id: str
    path: str
    start_timestamp: float
    duration_seconds: float
    has_acc: bool = False
    has_bvp: bool = False
    has_eda: bool = False
    has_temp: bool = False
    has_hr: bool = False
    has_ibi: bool = False
    has_tags: bool = False
    has_info: bool = False
    acc_samples: int = 0
    bvp_samples: int = 0
    eda_samples: int = 0
    temp_samples: int = 0
    hr_samples: int = 0
    acc_rate: float = 0.0
    bvp_rate: float = 0.0
    eda_rate: float = 0.0
    temp_rate: float = 0.0
    hr_rate: float = 0.0
    is_valid: bool = False
    has_complete_six_signals: bool = False


@dataclass
class PatientData:
    """All data for a single patient."""

    patient_id: str
    sessions: List[SessionInfo] = field(default_factory=list)
    seizure_intervals: List[Tuple[float, float]] = field(default_factory=list)
    raw_seizure_onsets: int = 0
    total_seizures: int = 0
    total_duration_hours: float = 0.0


def read_sensor_file(
    filepath: str,
) -> Tuple[Optional[float], Optional[float], Optional[np.ndarray]]:
    """
    Read Empatica E4 sensor CSV file.

    Returns:
        (start_timestamp, sample_rate, data_array) or (None, None, None) on error
    """
    try:
        with open(filepath, "r") as f:
            first_line = f.readline()
            second_line = f.readline()

            if not first_line or not second_line:
                return None, None, None
            start_timestamp = float(first_line.strip().split(",")[0])

            sample_rate = float(second_line.strip().split(",")[0])

            if (
                PHASE1_EXACT_DURATIONS
                and os.path.basename(filepath).upper() == "ACC.CSV"
            ):
                sample_count = sum(1 for line in f if line.strip())
            else:
                sample_count = 1
        if sample_count == 0:
            return start_timestamp, sample_rate, np.array([])
        return start_timestamp, sample_rate, np.empty((sample_count, 0))
    except Exception as e:
        print(f"  [ERROR] Reading {filepath}: {e}")
        return None, None, None


def merge_seizure_intervals(
    intervals: List[Tuple[float, float]],
    merge_gap_seconds: int = SEIZURE_MERGE_GAP_SECONDS,
) -> List[Tuple[float, float]]:
    """Merge overlapping or near-adjacent onset-derived seizure intervals."""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda item: item[0])
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + merge_gap_seconds:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def parse_seizure_label_rows(label_file: str) -> List[Tuple[float, Optional[float]]]:
    """
    Read label rows from a patient label file.

    Supported formats:
    - one numeric value: onset timestamp
    - two numeric values: onset and offset/duration-like second field
    """
    rows: List[Tuple[float, Optional[float]]] = []
    try:
        with open(label_file, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        for line in lines:
            parts = [part.strip() for part in re.split(r"[,\s]+", line) if part.strip()]
            if not parts:
                continue
            try:
                onset = float(parts[0])
            except ValueError:
                continue
            extra: Optional[float] = None
            if len(parts) >= 2:
                try:
                    extra = float(parts[1])
                except ValueError:
                    extra = None
            rows.append((onset, extra))
        return sorted(rows, key=lambda item: item[0])
    except Exception as e:
        print(f"  [ERROR] Parsing labels {label_file}: {e}")
        return []


def parse_seizure_labels(label_file: str) -> List[Tuple[float, float]]:
    """
    Parse seizure ground truth file.

    Format: Each line is a seizure ONSET timestamp.
    Assumes seizures last a configurable duration because these files do not
    contain offsets. Override with SEIZURE_DURATION_SECONDS if better clinical
    duration annotations become available.
    Returns list of (start_time, end_time) tuples.
    """
    rows = parse_seizure_label_rows(label_file)
    intervals: List[Tuple[float, float]] = []
    for onset_ts, second_field in rows:
        if second_field is None or second_field <= onset_ts:
            end_ts = onset_ts + DEFAULT_SEIZURE_DURATION_SECONDS
        elif second_field - onset_ts <= 24 * 3600:
            end_ts = second_field
        else:
            end_ts = onset_ts + DEFAULT_SEIZURE_DURATION_SECONDS
        intervals.append((onset_ts, end_ts))
    return merge_seizure_intervals(intervals)


def analyze_session(session_path: str) -> SessionInfo:
    """Analyze a single session folder."""
    session_id = os.path.basename(session_path)
    info = SessionInfo(
        session_id=session_id, path=session_path, start_timestamp=0, duration_seconds=0
    )

    required_files = {
        "ACC": os.path.join(session_path, "ACC.csv"),
        "BVP": os.path.join(session_path, "BVP.csv"),
        "EDA": os.path.join(session_path, "EDA.csv"),
        "TEMP": os.path.join(session_path, "TEMP.csv"),
        "HR": os.path.join(session_path, "HR.csv"),
    }
    info.has_ibi = os.path.exists(os.path.join(session_path, "IBI.csv"))
    info.has_tags = os.path.exists(os.path.join(session_path, "tags.csv"))
    info.has_info = os.path.exists(os.path.join(session_path, "info.txt"))

    for sensor, filepath in required_files.items():
        if os.path.exists(filepath):
            start_ts, rate, data = read_sensor_file(filepath)

            if sensor == "ACC":
                info.has_acc = True
                info.acc_rate = rate or 0
                info.acc_samples = len(data) if data is not None else 0
                if start_ts:
                    info.start_timestamp = start_ts
            elif sensor == "BVP":
                info.has_bvp = True
                info.bvp_rate = rate or 0
                info.bvp_samples = len(data) if data is not None else 0
            elif sensor == "EDA":
                info.has_eda = True
                info.eda_rate = rate or 0
                info.eda_samples = len(data) if data is not None else 0
            elif sensor == "TEMP":
                info.has_temp = True
                info.temp_rate = rate or 0
                info.temp_samples = len(data) if data is not None else 0
            elif sensor == "HR":
                info.has_hr = True
                info.hr_rate = rate or 0
                info.hr_samples = len(data) if data is not None else 0
    if PHASE1_EXACT_DURATIONS and info.acc_samples > 0 and info.acc_rate > 0:
        info.duration_seconds = info.acc_samples / info.acc_rate
    elif info.has_acc:
        info.duration_seconds = PHASE1_SESSION_DURATION_SECONDS
        info.acc_samples = (
            int(PHASE1_SESSION_DURATION_SECONDS * info.acc_rate) if info.acc_rate else 0
        )
    info.has_complete_six_signals = all(
        [
            info.has_acc,
            info.has_bvp,
            info.has_hr,
            info.has_ibi,
            info.has_eda,
            info.has_temp,
        ]
    )

    info.is_valid = all([info.has_acc, info.has_bvp, info.has_eda, info.has_temp])

    return info


def find_label_file(patient_folder: str) -> Optional[str]:
    """Find the seizure label .txt file for a patient folder."""

    txt_files = glob.glob(os.path.join(patient_folder, "*.txt"))

    for f in txt_files:
        basename = os.path.basename(f).lower()
        if basename not in ["info.txt", "readme.txt"]:
            return f
    chronic_folder = os.path.join(patient_folder, "Empatica_Chronic")
    if os.path.exists(chronic_folder):
        txt_files = glob.glob(os.path.join(chronic_folder, "*.txt"))
        for f in txt_files:
            basename = os.path.basename(f).lower()
            if basename not in ["info.txt", "readme.txt"]:
                return f
    return None


def scan_patient_folder(patient_folder: str) -> PatientData:
    """Scan a patient folder for all sessions and labels."""
    patient_id = os.path.basename(patient_folder)
    patient_data = PatientData(patient_id=patient_id)

    label_file = find_label_file(patient_folder)
    if label_file:
        print(f"\n  [LABELS] Found: {os.path.basename(label_file)}")
        raw_onsets = [item[0] for item in parse_seizure_label_rows(label_file)]
        patient_data.seizure_intervals = parse_seizure_labels(label_file)
        patient_data.raw_seizure_onsets = len(raw_onsets)
        patient_data.total_seizures = len(patient_data.seizure_intervals)

        print(f"  First 5 rows of {os.path.basename(label_file)}:")
        with open(label_file, "r") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"    {line.strip()}")
    session_folders = _find_session_folders(patient_folder)

    print(f"  [SESSIONS] Found {len(session_folders)} session folders")

    valid_count = 0
    for session_path in sorted(session_folders):
        session_info = analyze_session(session_path)
        patient_data.sessions.append(session_info)
        if session_info.is_valid:
            valid_count += 1
            patient_data.total_duration_hours += session_info.duration_seconds / 3600
    complete_six = sum(1 for s in patient_data.sessions if s.has_complete_six_signals)
    complete_metadata = sum(
        1 for s in patient_data.sessions if s.has_tags and s.has_info
    )
    print(f"  [VALID] {valid_count}/{len(session_folders)} sessions are model-usable")
    print(
        f"  [COMPLETE EXPORT] {complete_six}/{len(session_folders)} sessions have ACC + PPG(BVP/HR/IBI) + EDA + TEMP"
    )
    print(
        f"  [METADATA] {complete_metadata}/{len(session_folders)} sessions have tags.csv/info.txt"
    )

    return patient_data


def verify_sampling_rates(patient_data: PatientData) -> Dict[str, List[float]]:
    """Verify sampling rates across all sessions."""
    rates = {"ACC": [], "BVP": [], "HR": [], "EDA": [], "TEMP": []}

    for session in patient_data.sessions:
        if session.is_valid:
            rates["ACC"].append(session.acc_rate)
            rates["BVP"].append(session.bvp_rate)
            if session.has_hr:
                rates["HR"].append(session.hr_rate)
            rates["EDA"].append(session.eda_rate)
            rates["TEMP"].append(session.temp_rate)
    return rates


def patient_to_dict(patient_data: PatientData) -> Dict:
    """Serialize PatientData for the Phase 1 metadata cache."""
    return {
        "patient_id": patient_data.patient_id,
        "sessions": [asdict(session) for session in patient_data.sessions],
        "seizure_intervals": [
            [float(start), float(end)] for start, end in patient_data.seizure_intervals
        ],
        "raw_seizure_onsets": int(patient_data.raw_seizure_onsets),
        "total_seizures": int(patient_data.total_seizures),
        "total_duration_hours": float(patient_data.total_duration_hours),
    }


def patient_from_dict(payload: Dict) -> PatientData:
    """Deserialize PatientData from the Phase 1 metadata cache."""
    return PatientData(
        patient_id=payload["patient_id"],
        sessions=[SessionInfo(**session) for session in payload.get("sessions", [])],
        seizure_intervals=[
            (float(start), float(end))
            for start, end in payload.get("seizure_intervals", [])
        ],
        raw_seizure_onsets=int(
            payload.get("raw_seizure_onsets", payload.get("total_seizures", 0))
        ),
        total_seizures=int(payload.get("total_seizures", 0)),
        total_duration_hours=float(payload.get("total_duration_hours", 0.0)),
    )


def phase1_cache_key(base_path: str) -> str:
    """Build a cache key from lightweight filesystem metadata and label policy."""
    entries = {
        "version": PHASE1_CACHE_VERSION,
        "base_path": os.path.abspath(base_path),
        "seizure_duration_seconds": DEFAULT_SEIZURE_DURATION_SECONDS,
        "merge_gap_seconds": SEIZURE_MERGE_GAP_SECONDS,
        "exact_durations": PHASE1_EXACT_DURATIONS,
        "session_duration_seconds": PHASE1_SESSION_DURATION_SECONDS,
        "patients": [],
    }

    for patient_folder in sorted(glob.glob(os.path.join(base_path, "Mayo_*"))):
        if not os.path.isdir(patient_folder):
            continue
        patient_entry = {
            "name": os.path.basename(patient_folder),
            "label": None,
            "sessions": [],
        }
        label_file = find_label_file(patient_folder)
        if label_file and os.path.exists(label_file):
            stat = os.stat(label_file)
            patient_entry["label"] = {
                "path": os.path.abspath(label_file),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        for session_path in sorted(_find_session_folders(patient_folder)):
            session_entry = {"name": os.path.basename(session_path), "files": []}
            for filename in (
                "ACC.csv",
                "BVP.csv",
                "HR.csv",
                "IBI.csv",
                "EDA.csv",
                "TEMP.csv",
                "tags.csv",
                "info.txt",
            ):
                file_path = os.path.join(session_path, filename)
                if os.path.exists(file_path):
                    stat = os.stat(file_path)
                    session_entry["files"].append(
                        (filename, stat.st_size, stat.st_mtime)
                    )
                else:
                    session_entry["files"].append((filename, None, None))
            patient_entry["sessions"].append(session_entry)
        entries["patients"].append(patient_entry)
    raw = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def phase1_cache_path(base_path: str) -> str:
    cache_dir = os.path.join(base_path, "seizure_detection", "outputs", "cache")
    return os.path.join(cache_dir, f"phase1_health_{phase1_cache_key(base_path)}.json")


def load_phase1_cache(base_path: str) -> Optional[Dict[str, PatientData]]:
    """Load cached Phase 1 metadata when enabled and valid."""
    if os.environ.get("PHASE1_CACHE", "1") != "1":
        return None
    cache_path = phase1_cache_path(base_path)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != PHASE1_CACHE_VERSION:
            return None
        patients = {
            patient_id: patient_from_dict(patient_payload)
            for patient_id, patient_payload in payload.get("patients", {}).items()
        }
        print(f"\nLoaded Phase 1 metadata cache: {cache_path}")
        return patients
    except Exception as exc:
        print(f"\n[WARN] Phase 1 cache load failed, rescanning: {exc}")
        return None


def save_phase1_cache(base_path: str, all_patients: Dict[str, PatientData]) -> None:
    """Save Phase 1 metadata for faster repeated runs."""
    if os.environ.get("PHASE1_CACHE", "1") != "1":
        return
    cache_path = phase1_cache_path(base_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "version": PHASE1_CACHE_VERSION,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "label_policy": {
            "seizure_duration_seconds": DEFAULT_SEIZURE_DURATION_SECONDS,
            "merge_gap_seconds": SEIZURE_MERGE_GAP_SECONDS,
            "exact_durations": PHASE1_EXACT_DURATIONS,
            "session_duration_seconds": PHASE1_SESSION_DURATION_SECONDS,
            "row_format": "onset or onset+offset columns when available",
        },
        "patients": {
            patient_id: patient_to_dict(patient_data)
            for patient_id, patient_data in all_patients.items()
        },
    }
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved Phase 1 metadata cache: {cache_path}")


def _find_session_folders(patient_folder: str) -> List[str]:
    """Find Empatica session folders for a patient."""
    session_folders = []
    for item in os.listdir(patient_folder):
        item_path = os.path.join(patient_folder, item)
        if os.path.isdir(item_path) and "_" in item:
            session_folders.append(item_path)
    chronic_folder = os.path.join(patient_folder, "Empatica_Chronic")
    if os.path.exists(chronic_folder):
        for item in os.listdir(chronic_folder):
            item_path = os.path.join(chronic_folder, item)
            if os.path.isdir(item_path) and "_" in item:
                session_folders.append(item_path)
    return session_folders


def print_analysis_summary(all_patients: Dict[str, PatientData]) -> None:
    """Print a compact global summary for scanned or cached Phase 1 metadata."""
    total_sessions = sum(len(p.sessions) for p in all_patients.values())
    valid_sessions = sum(
        sum(1 for s in p.sessions if s.is_valid) for p in all_patients.values()
    )
    total_hours = sum(p.total_duration_hours for p in all_patients.values())
    total_raw_onsets = sum(p.raw_seizure_onsets for p in all_patients.values())
    total_seizures = sum(p.total_seizures for p in all_patients.values())

    print(f"\n{'='*70}")
    print("GLOBAL SUMMARY")
    print("=" * 70)
    print(f"  Patients: {len(all_patients)}")
    print(f"  Total sessions: {total_sessions}")
    print(f"  Valid sessions: {valid_sessions}")
    recording_label = (
        "Total recording" if PHASE1_EXACT_DURATIONS else "Estimated recording"
    )
    print(f"  {recording_label}: {total_hours:.1f} hours")
    print(f"  Total raw seizure onset rows: {total_raw_onsets}")
    print(f"  Total merged seizure episodes: {total_seizures}")

    seizure_intervals_dict = build_seizure_intervals_dict(all_patients)
    overlap_label = (
        "Sessions with seizures"
        if PHASE1_EXACT_DURATIONS
        else "Sessions with possible seizure overlap"
    )
    print(f"\n  {overlap_label}: {len(seizure_intervals_dict)}")


def run_deep_analysis(base_path: str) -> Dict[str, PatientData]:
    """
    Run complete deep analysis on the dataset.

    Args:
        base_path: Root folder containing patient folders (e.g., Mayo_1110, Mayo_1869)

    Returns:
        Dictionary mapping patient_id -> PatientData
    """
    print("=" * 70)
    print("DEEP DATA ANALYSIS - Empatica E4 Seizure Detection Dataset")
    print("=" * 70)
    print(
        "\nLabel policy: patient .txt files are parsed as seizure onsets; "
        "if a second numeric column exists it is treated as the offset when it "
        "looks like an absolute timestamp, otherwise "
        f"offset = onset + {DEFAULT_SEIZURE_DURATION_SECONDS}s "
        f"and overlapping intervals are merged with gap <= {SEIZURE_MERGE_GAP_SECONDS}s "
        "(override with SEIZURE_DURATION_SECONDS and SEIZURE_MERGE_GAP_SECONDS)."
    )
    if PHASE1_EXACT_DURATIONS:
        print("Phase 1 duration policy: exact ACC row counts enabled.")
    else:
        print(
            "Phase 1 duration policy: fast header-only scan using "
            f"{PHASE1_SESSION_DURATION_SECONDS:.0f}s conservative session span; "
            "Phase 2 uses actual selected-session samples."
        )
    cached = load_phase1_cache(base_path)
    if cached is not None:
        print_analysis_summary(cached)
        return cached
    all_patients: Dict[str, PatientData] = {}

    patient_folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.startswith("Mayo_"):
            patient_folders.append(item_path)
    print(f"\nFound {len(patient_folders)} patient folders")

    seizure_times_folder = os.path.join(base_path, "SeizureTimesOnly")
    additional_labels = {}
    if os.path.exists(seizure_times_folder):
        print(f"\n[INFO] Found SeizureTimesOnly folder with additional labels")
        for txt_file in glob.glob(os.path.join(seizure_times_folder, "*.txt")):
            patient_id = os.path.splitext(os.path.basename(txt_file))[0]
            additional_labels[patient_id] = parse_seizure_labels(txt_file)
            print(
                f"  - {patient_id}: {len(additional_labels[patient_id])} seizure events"
            )
    for patient_folder in sorted(patient_folders):
        patient_id = os.path.basename(patient_folder)
        print(f"\n{'='*70}")
        print(f"PATIENT: {patient_id}")
        print("=" * 70)

        patient_data = scan_patient_folder(patient_folder)
        all_patients[patient_id] = patient_data

        rates = verify_sampling_rates(patient_data)
        print(f"\n  [SAMPLING RATE VERIFICATION]")
        for sensor, rate_list in rates.items():
            if rate_list:
                unique_rates = set(rate_list)
                expected = {
                    "ACC": 32.0,
                    "BVP": 64.0,
                    "HR": 1.0,
                    "EDA": 4.0,
                    "TEMP": 4.0,
                }
                status = "OK" if unique_rates == {expected[sensor]} else "WARN"
                print(
                    f"    {sensor}: {unique_rates} Hz (expected: {expected[sensor]} Hz) {status}"
                )
        print(f"\n  [SUMMARY]")
        print(f"    Total sessions: {len(patient_data.sessions)}")
        print(
            f"    Valid sessions: {sum(1 for s in patient_data.sessions if s.is_valid)}"
        )
        recording_label = (
            "Total recording" if PHASE1_EXACT_DURATIONS else "Estimated recording"
        )
        print(f"    {recording_label}: {patient_data.total_duration_hours:.1f} hours")
        print(f"    Raw seizure onset rows: {patient_data.raw_seizure_onsets}")
        print(f"    Merged seizure episodes: {patient_data.total_seizures}")

        if patient_data.seizure_intervals:
            print(f"    Sample intervals (first 3):")
            for i, (start, end) in enumerate(patient_data.seizure_intervals[:3]):
                duration = end - start
                print(f"      [{i+1}] {start:.0f} -> {end:.0f} ({duration:.0f}s)")
    print_analysis_summary(all_patients)
    save_phase1_cache(base_path, all_patients)

    return all_patients


def build_seizure_intervals_dict(
    all_patients: Dict[str, PatientData],
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Build the seizure_intervals dictionary mapping session_id to seizure intervals.
    """
    seizure_intervals = {}

    for patient_id, patient_data in all_patients.items():
        for session in patient_data.sessions:
            if session.is_valid:
                session_start = session.start_timestamp
                session_end = session_start + session.duration_seconds

                matching = []
                for sz_start, sz_end in patient_data.seizure_intervals:
                    if sz_start < session_end and sz_end > session_start:
                        matching.append((sz_start, sz_end))
                if matching:
                    seizure_intervals[session.session_id] = matching
    return seizure_intervals


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


if __name__ == "__main__":
    import sys

    base_path = get_default_base_path()
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    print(f"Scanning: {base_path}\n")

    all_patients = run_deep_analysis(base_path)

    seizure_intervals = build_seizure_intervals_dict(all_patients)

    print(f"\n{'='*70}")
    print("SEIZURE_INTERVALS DICTIONARY")
    print("=" * 70)
    print(f"seizure_intervals = {{")
    for session_id, intervals in list(seizure_intervals.items())[:10]:
        print(f"    '{session_id}': {intervals},")
    if len(seizure_intervals) > 10:
        print(f"    ... ({len(seizure_intervals) - 10} more sessions)")
    print("}")
