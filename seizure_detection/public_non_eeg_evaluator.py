"""Subject-held-out public evaluation using retained non-EEG signals only.

This is intentionally a conservative external lane. It never reads EEG EDFs,
and it does not claim that public ECG/EMG/MOV sensors are identical to Mayo's
Empatica channels. Event TSVs are the ground truth and are joined by recording
identity before windows are sampled.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path

import numpy as np


def _features(path: str, starts: list[float], window: float) -> np.ndarray:
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    rows = []
    for start in starts:
        stop = min(start + window, raw.times[-1])
        if stop <= start:
            continue
        first = max(0, int(round(start * raw.info["sfreq"])))
        last = max(first + 1, int(round(stop * raw.info["sfreq"])))
        data = raw.get_data(start=first, stop=last, verbose="ERROR")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # Public EDF files vary in channel count. Keep a fixed, disclosed
        # wearable schema so recordings can be concatenated safely.
        modality = Path(path).stem.rsplit("_", 1)[-1].lower()
        target_channels = {"mov": 6, "ecg": 1, "emg": 2}[modality]
        if data.shape[0] < target_channels:
            data = np.pad(data, ((0, target_channels - data.shape[0]), (0, 0)))
        elif data.shape[0] > target_channels:
            data = data[:target_channels]
        rows.append(np.concatenate([
            np.mean(data, axis=1), np.std(data, axis=1),
            np.sqrt(np.mean(data * data, axis=1)),
        ]))
    return np.asarray(rows, dtype=np.float32)


def _window_dataset(records: list[dict], track: str, max_windows: int = 6000):
    rng = np.random.default_rng(42)
    modality_order = ("mov", "ecg", "emg") if track == "pro" else ("mov",)
    feature_dims = {"mov": 18, "ecg": 3, "emg": 6}
    X, y, groups = [], [], []
    for record in records:
        signals = record["signals"]
        paths = {modality: signals[modality] for modality in modality_order if modality in signals}
        if not paths:
            continue
        import mne
        raws = [mne.io.read_raw_edf(path, preload=False, verbose="ERROR") for path in paths.values()]
        duration = min(float(raw.times[-1]) for raw in raws)
        if duration < 10:
            continue
        starts = np.arange(0, max(0.0, duration - 10), 10.0).tolist()
        intervals = record["seizure_intervals"]
        labels = [int(any(a < start + 10 and b > start for a, b in intervals)) for start in starts]
        pos = [i for i, value in enumerate(labels) if value]
        neg = [i for i, value in enumerate(labels) if not value]
        if len(neg) > max(20, len(pos) * 3):
            neg = rng.choice(neg, size=max(20, len(pos) * 3), replace=False).tolist()
        selected = pos + neg

        blocks = {}
        for modality, path in paths.items():
            block = _features(path, [starts[i] for i in selected], 10.0)
            if len(block):
                blocks[modality] = block

        if not blocks:
            continue

        n = min(len(block) for block in blocks.values())
        combined = []
        for modality in modality_order:
            if modality in blocks:
                combined.append(blocks[modality][:n])
            else:
                combined.append(np.zeros((n, feature_dims[modality]), dtype=np.float32))

        block = np.concatenate(combined, axis=1)
        target_width = sum(feature_dims[modality] for modality in modality_order)
        if block.shape[1] < target_width:
            block = np.pad(block, ((0, 0), (0, target_width - block.shape[1])))
        elif block.shape[1] > target_width:
            block = block[:, :target_width]
        X.append(block)
        y.extend(labels[i] for i in selected[:n])
        groups.extend([record["subject"]] * n)

        if sum(len(item) for item in X) >= max_windows:
            break
    if not X:
        return np.empty((0, 0)), np.empty(0), np.empty(0, dtype=str)
    rows = min(max_windows, sum(len(item) for item in X))
    matrix = np.concatenate(X, axis=0)[:rows]
    return matrix, np.asarray(y[:rows]), np.asarray(groups[:rows])


def _metrics(X, y, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    subjects = sorted(set(groups))
    test_subjects = {s for s in subjects if int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 5 == 0}
    train = np.asarray([g not in test_subjects for g in groups])
    if train.sum() == 0 or (~train).sum() == 0 or len(np.unique(y[train])) < 2:
        return {"status": "insufficient_subject_split"}
    result = {"status": "pass", "test_subjects": len(test_subjects), "test_windows": int((~train).sum())}
    for name, model in {
        "logistic_regression": LogisticRegression(max_iter=300, class_weight="balanced"),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, random_state=42),
    }.items():
        model.fit(X[train], y[train])
        score = model.predict_proba(X[~train])[:, 1]
        result[name] = {
            "roc_auc": float(roc_auc_score(y[~train], score)) if len(np.unique(y[~train])) > 1 else None,
            "pr_auc": float(average_precision_score(y[~train], score)),
            "positive_test_windows": int(y[~train].sum()),
        }
    return result


def subject_id(path: Path) -> str:
    for token in path.name.split("_"):
        if token.startswith("sub-"):
            return token[4:]
    return "unknown"


def event_intervals(event_file: Path) -> list[tuple[float, float]]:
    intervals = []
    with event_file.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            event = str(row.get("eventType", "")).strip().lower()
            if not event.startswith("sz"):
                continue
            try:
                onset = float(row.get("onset", ""))
                duration = max(0.0, float(row.get("duration", "")))
            except (TypeError, ValueError):
                continue
            intervals.append((onset, onset + duration))
    return intervals


def discover(root: Path) -> list[dict]:
    # Index once; per-annotation recursive globs make a 2,850-file audit
    # needlessly expensive on Windows.
    signal_index = {}
    for path in root.rglob("*.edf"):
        if path.name.lower().endswith("_eeg.edf"):
            raise RuntimeError(f"EEG file present in retained public tree: {path}")
        tokens = path.stem.split("_")
        if not tokens:
            continue
        modality = tokens[-1].lower()
        if modality not in {"mov", "ecg", "emg"}:
            continue
        signal_index[(path.parent.parent, path.stem.removesuffix(f"_{modality}"), modality)] = str(path)
    records = []
    for event in sorted(root.rglob("*_events.tsv")):
        stem = event.name.removesuffix("_events.tsv")
        files = {}
        for modality in ("mov", "ecg", "emg"):
            match = signal_index.get((event.parent.parent, stem, modality))
            if match:
                files[modality] = match
        if not files:
            continue
        records.append({
            "recording": stem,
            "subject": subject_id(event),
            "event_file": str(event),
            "signals": files,
            "seizure_intervals": event_intervals(event),
        })
    return records


def stable_split(subject: str) -> str:
    # Stable across machines and reruns; no random leakage through split drift.
    value = int(hashlib.sha256(subject.encode()).hexdigest()[:8], 16) % 100
    return "test" if value < 20 else "validation" if value < 40 else "train"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-recordings", type=int, default=0,
                        help="Smoke limit; zero means all mapped recordings")
    parser.add_argument("--train-metrics", action="store_true",
                        help="Extract non-EEG windows and train public Standard/Pro baselines")
    parser.add_argument("--max-windows", type=int, default=6000)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    records = discover(root)
    if args.max_recordings:
        records = records[:args.max_recordings]
    modalities = sorted({m for record in records for m in record["signals"]})
    report = {
        "status": "pass" if records else "incomplete",
        "study_scope": "public external evaluation; no EEG EDFs read",
        "standard_track": "MOV ACC channels where available",
        "pro_track": "MOV ACC plus ECG and EMG where available",
        "ground_truth": "per-recording *_events.tsv; seizure rows are eventType values beginning with sz",
        "recordings_considered": len(records),
        "subjects_considered": len({r["subject"] for r in records}),
        "modalities_present": modalities,
        "recordings_with_seizure_labels": sum(bool(r["seizure_intervals"]) for r in records),
        "split_counts": {},
        "records": records,
        "metrics_status": "not_run",
    }
    for record in records:
        split = stable_split(record["subject"])
        report["split_counts"][split] = report["split_counts"].get(split, 0) + 1
    if args.train_metrics:
        report["metrics"] = {}
        for track in ("standard", "pro"):
            X, y, groups = _window_dataset(records, track, args.max_windows)
            report["metrics"][track] = _metrics(X, y, groups) if len(X) else {"status": "no_windows"}
        report["metrics_status"] = "trained_non_eeg_subject_held_out_models"
    output.mkdir(parents=True, exist_ok=True)
    (output / "public_non_eeg_evaluation_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "PUBLIC_NON_EEG_EVALUATION.md").write_text(
        "# Public Non-EEG Evaluation\n\n"
        f"- Recordings: {len(records)}\n"
        f"- Subjects: {report['subjects_considered']}\n"
        f"- Modalities consumed: `{', '.join(modalities)}`\n"
        f"- Recordings with seizure labels: {report['recordings_with_seizure_labels']}\n"
        f"- Subject split counts: `{json.dumps(report['split_counts'])}`\n\n"
        "This gate verifies the public labels are joined to retained ECG/EMG/MOV data. "
        "It does not use EEG and does not substitute public sensor performance for Mayo performance.\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
