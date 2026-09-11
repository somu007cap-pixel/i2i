"""Auxiliary EDF benchmark utilities for the public SeizeIT2 dataset.

This module keeps the public benchmark separate from the Empatica clinical
pipeline. It focuses on:

* discovering EDF recordings under a BIDS-style dataset root,
* extracting lightweight header metadata,
* sampling a short preview from each recording for smoke testing,
* writing a compact summary artifact that can be cited in the thesis.

The dataset shipped in EDF format does not expose the same sensor stack or
label structure as the Mayo/Empatica pipeline, so this module does not try to
force it into the seizure-classification training flow.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import mne
except ImportError as exc:  # pragma: no cover - surfaced in preflight if missing
    mne = None
    _MNE_IMPORT_ERROR = exc
else:
    _MNE_IMPORT_ERROR = None


@dataclass
class EdfRecordingSummary:
    """Compact description of one EDF recording."""

    path: str
    subject: str
    session: str
    task: str
    run: str
    modality: str
    sfreq: float
    n_channels: int
    channel_names: list[str]
    duration_seconds: float
    sampling_scheme: str


def _require_mne() -> None:
    if mne is None:
        raise RuntimeError(
            "mne is required to inspect EDF files. Install it or keep the "
            "public benchmark lane disabled."
        ) from _MNE_IMPORT_ERROR


def discover_edf_files(root: str | Path) -> list[Path]:
    """Return all EDF files under a benchmark root."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return []
    return sorted(root_path.rglob("*.edf"))


def _parse_bids_tokens(path: Path) -> dict[str, str]:
    tokens = {
        "subject": "unknown",
        "session": "unknown",
        "task": "unknown",
        "run": "unknown",
        "modality": path.stem.split("_")[-1],
    }
    for part in path.stem.split("_"):
        if part.startswith("sub-"):
            tokens["subject"] = part[4:]
        elif part.startswith("ses-"):
            tokens["session"] = part[4:]
        elif part.startswith("task-"):
            tokens["task"] = part[5:]
        elif part.startswith("run-"):
            tokens["run"] = part[4:]
    return tokens


def summarize_edf_file(path: str | Path, preview_seconds: float = 10.0) -> EdfRecordingSummary:
    """Read EDF header metadata and sample a short data preview."""
    _require_mne()
    edf_path = Path(path)
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    tokens = _parse_bids_tokens(edf_path)
    duration_seconds = float(raw.n_times / raw.info["sfreq"]) if raw.info["sfreq"] else 0.0
    sampling_scheme = "continuous" if duration_seconds > 0 else "unknown"
    preview_end = min(raw.n_times, int(math.ceil(preview_seconds * raw.info["sfreq"])))
    if preview_end > 0:
        preview = raw.get_data(start=0, stop=preview_end, verbose="ERROR")
        preview_energy = float(np.mean(np.square(preview)))
        sampling_scheme = f"preview_rms={preview_energy:.6f}"
    return EdfRecordingSummary(
        path=str(edf_path),
        subject=tokens["subject"],
        session=tokens["session"],
        task=tokens["task"],
        run=tokens["run"],
        modality=tokens["modality"],
        sfreq=float(raw.info["sfreq"]),
        n_channels=int(raw.info["nchan"]),
        channel_names=list(raw.ch_names),
        duration_seconds=duration_seconds,
        sampling_scheme=sampling_scheme,
    )


def build_public_benchmark_summary(root: str | Path, output_dir: str | Path) -> dict:
    """Summarize the public benchmark EDFs without merging them into training."""
    _require_mne()
    root_path = Path(root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    edf_files = discover_edf_files(root_path)
    summaries = [summarize_edf_file(path) for path in edf_files]

    subject_counts: dict[str, int] = {}
    modality_counts: dict[str, int] = {}
    sfreq_counts: dict[str, int] = {}
    total_duration = 0.0
    channel_sets = set()

    for item in summaries:
        subject_counts[item.subject] = subject_counts.get(item.subject, 0) + 1
        modality_counts[item.modality] = modality_counts.get(item.modality, 0) + 1
        sfreq_key = f"{item.sfreq:g}"
        sfreq_counts[sfreq_key] = sfreq_counts.get(sfreq_key, 0) + 1
        total_duration += item.duration_seconds
        channel_sets.add(tuple(item.channel_names))

    report = {
        "status": "completed" if summaries else "empty",
        "dataset_root": str(root_path),
        "edf_file_count": len(summaries),
        "unique_subject_count": len(subject_counts),
        "modality_counts": modality_counts,
        "sampling_frequencies": sfreq_counts,
        "total_duration_hours": round(total_duration / 3600.0, 3),
        "unique_channel_layouts": len(channel_sets),
        "recordings": [asdict(item) for item in summaries[:200]],
    }

    (output_path / "public_benchmark_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_path / "public_benchmark_summary.md").write_text(
        "\n".join(
            [
                "# Public Benchmark Summary",
                "",
                f"- Dataset root: `{root_path}`",
                f"- EDF files: {report['edf_file_count']}",
                f"- Subjects: {report['unique_subject_count']}",
                f"- Total duration: {report['total_duration_hours']} hours",
                f"- Modalities: {json.dumps(modality_counts)}",
                f"- Sampling frequencies: {json.dumps(sfreq_counts)}",
            ]
        ),
        encoding="utf-8",
    )
    return report

