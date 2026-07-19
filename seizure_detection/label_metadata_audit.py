"""
Audit seizure label files for onset-only versus onset-offset metadata.

The detector can use exact offsets only when the source label files contain a
second numeric field that behaves like an offset. Otherwise the study must stay
honest and report configurable onset-derived intervals.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from phase1_data_health import find_label_file, parse_seizure_label_rows


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "seizure_detection" / "outputs"
JSON_PATH = OUTPUT_DIR / "label_metadata_audit.json"
SUMMARY_PATH = OUTPUT_DIR / "LABEL_METADATA_AUDIT.md"


def classify_second_field(onset: float, second: float | None) -> str:
    if second is None:
        return "onset_only"
    if second <= onset:
        return "duration_or_invalid"
    if second - onset <= 24 * 3600:
        return "usable_offset"
    return "implausible_offset"


def run() -> Dict:
    base_path = Path(__file__).resolve().parents[1]
    patient_dirs = sorted(path for path in base_path.glob("Mayo_*") if path.is_dir())
    patients: List[Dict] = []
    totals = {
        "patients": 0,
        "label_files": 0,
        "rows": 0,
        "onset_only_rows": 0,
        "usable_offset_rows": 0,
        "duration_or_invalid_rows": 0,
        "implausible_offset_rows": 0,
    }
    for patient_dir in patient_dirs:
        label_file = find_label_file(str(patient_dir))
        if not label_file:
            patients.append(
                {
                    "patient_id": patient_dir.name,
                    "label_file": None,
                    "rows": 0,
                    "usable_offset_rows": 0,
                }
            )
            continue
        rows = parse_seizure_label_rows(label_file)
        counts = {
            "onset_only": 0,
            "usable_offset": 0,
            "duration_or_invalid": 0,
            "implausible_offset": 0,
        }
        examples = []
        for onset, second in rows:
            kind = classify_second_field(onset, second)
            counts[kind] += 1
            if len(examples) < 5:
                examples.append({"onset": onset, "second_field": second, "classification": kind})
        totals["patients"] += 1
        totals["label_files"] += 1
        totals["rows"] += len(rows)
        totals["onset_only_rows"] += counts["onset_only"]
        totals["usable_offset_rows"] += counts["usable_offset"]
        totals["duration_or_invalid_rows"] += counts["duration_or_invalid"]
        totals["implausible_offset_rows"] += counts["implausible_offset"]
        patients.append(
            {
                "patient_id": patient_dir.name,
                "label_file": str(Path(label_file).relative_to(base_path)),
                "rows": len(rows),
                "usable_offset_rows": counts["usable_offset"],
                "onset_only_rows": counts["onset_only"],
                "duration_or_invalid_rows": counts["duration_or_invalid"],
                "implausible_offset_rows": counts["implausible_offset"],
                "examples": examples,
            }
        )
    payload = {
        "timestamp": datetime.now().isoformat(),
        "base_path": str(base_path),
        "label_policy_conclusion": (
            "usable_onset_offset_labels_available"
            if totals["usable_offset_rows"] > 0
            else "onset_only_labels_detected"
        ),
        "totals": totals,
        "patients": patients,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Label Metadata Audit",
        "",
        f"- Label files scanned: {totals['label_files']}",
        f"- Label rows scanned: {totals['rows']}",
        f"- Rows with usable onset-offset fields: {totals['usable_offset_rows']}",
        f"- Onset-only rows: {totals['onset_only_rows']}",
        "",
    ]
    if totals["usable_offset_rows"] > 0:
        lines.append(
            "Conclusion: at least some source rows contain usable offset fields; "
            "the label pipeline may use them for those events."
        )
    else:
        lines.append(
            "Conclusion: no usable onset-offset rows were found. Final reports "
            "should describe labels as onset-derived intervals, not exact seizure "
            "boundaries."
        )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY_PATH)
    return payload


if __name__ == "__main__":
    run()
