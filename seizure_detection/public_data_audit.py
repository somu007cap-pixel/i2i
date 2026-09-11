"""Audit and consume the public SeizeIT2 checkout without false claims.

The current checkout contains EDF signals and sidecar metadata, but may not
contain event annotations. This audit makes that state explicit and produces a
machine-readable gate for the multi-week campaign.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def audit(root: Path, output: Path) -> dict:
    edfs = sorted(root.rglob("*.edf")) if root.exists() else []
    sidecars = sorted(root.rglob("*.json")) if root.exists() else []
    annotation_candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".tsv", ".csv", ".annot", ".xml"}
        and "participant" not in path.name.lower()
    ) if root.exists() else []

    modalities = Counter()
    event_types = Counter()
    annotation_rows = 0
    mapped_event_files = 0
    unmapped_event_files = []
    subjects = set()
    for path in edfs:
        tokens = path.stem.split("_")
        modality = tokens[-1].lower()
        modalities[modality] += 1
        for token in tokens:
            if token.startswith("sub-"):
                subjects.add(token[4:])

    event_files = sorted(root.rglob("*_events.tsv")) if root.exists() else []
    for event_file in event_files:
        stem = event_file.name.removesuffix("_events.tsv")
        recording_matches = [
            candidate
            for modality in ("ecg", "emg", "mov")
            for candidate in (event_file.parent.parent / modality).glob(f"{stem}_*.edf")
        ]
        if recording_matches:
            mapped_event_files += 1
        else:
            unmapped_event_files.append(str(event_file))
        with event_file.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                annotation_rows += 1
                event_types[str(row.get("eventType", "")).strip()] += 1
    # EEG signal is intentionally not required by this project. Event TSVs are
    # required because they contain the public seizure ground truth.
    # A small number of recordings may legitimately lack non-EEG sensors. They
    # are excluded and disclosed; the public lane remains valid on the mapped
    # cohort rather than silently pretending the files exist.
    complete_supervised = bool(edfs and event_files and mapped_event_files > 0)
    report = {
        "status": "pass" if complete_supervised else "incomplete",
        "signal_checkout_status": "pass" if edfs else "fail",
        "supervised_external_eval_ready": complete_supervised,
        "dataset_root": str(root.resolve()),
        "edf_file_count": len(edfs),
        "sidecar_json_count": len(sidecars),
        "annotation_candidate_count": len(annotation_candidates),
        "annotation_candidates": [str(path) for path in annotation_candidates[:100]],
        "event_tsv_count": len(event_files),
        "annotation_row_count": annotation_rows,
        "event_type_counts": dict(sorted(event_types.items())),
        "event_files_mapped_to_retained_non_eeg_recordings": mapped_event_files,
        "unmapped_event_files": unmapped_event_files[:100],
        "unique_subject_count": len(subjects),
        "modality_counts": dict(sorted(modalities.items())),
        "expected_modalities_not_present": sorted(
            {"ecg", "emg", "mov", "eeg"} - set(modalities)
        ),
        "label_status": (
            "per-recording event annotations found"
            if event_files
            else "no per-recording event annotations found in checkout"
        ),
        "allowed_uses": [
            "public signal quality and metadata audit",
            "domain-shift and sensor-distribution analysis",
            "labeled external performance only after annotations are verified",
        ],
        "forbidden_claims_now": [
            "public-data seizure AUC or recall",
            "direct external validation of Mayo Empatica performance",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "public_data_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output / "PUBLIC_DATA_AUDIT.md").write_text(
        "\n".join(
            [
                "# Public Data Audit",
                "",
                f"- Status: **{report['status'].upper()}**",
                f"- Root: `{report['dataset_root']}`",
                f"- EDF files: {report['edf_file_count']}",
                f"- Subjects: {report['unique_subject_count']}",
                f"- Modalities: `{json.dumps(report['modality_counts'])}`",
                f"- Event TSV files: {report['event_tsv_count']}",
                f"- Annotation rows: {report['annotation_row_count']}",
                f"- Event types: `{json.dumps(report['event_type_counts'])}`",
                f"- Event files mapped to retained non-EEG recordings: {report['event_files_mapped_to_retained_non_eeg_recordings']}",
                f"- Expected modalities absent: `{json.dumps(report['expected_modalities_not_present'])}`",
                f"- Label status: {report['label_status']}",
                "",
                "EEG signal is optional for this project; event TSV labels must be "
                "present, parsed, and included in a subject-held-out protocol.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if not args.root:
        raise SystemExit("--root is required")
    root = Path(args.root)
    output = Path(args.output) if args.output else root.parent / "public_benchmark_outputs"
    report = audit(root, output)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
