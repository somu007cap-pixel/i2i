"""Audit whether a completed campaign is scientifically reportable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    findings = []
    public = manifest["phases"].get("public_external_evaluation", {})
    public_report = manifest_path.parents[3] / "public_benchmark_outputs" / "public_non_eeg_evaluation_manifest.json"
    public_report = public_report.resolve()
    if public_report.exists():
        public_data = json.loads(public_report.read_text(encoding="utf-8"))
        metrics = public_data.get("metrics", {})
        metrics_ready = all(metrics.get(track, {}).get("status") == "pass" for track in ("standard", "pro"))
        if public_data.get("metrics_status") != "trained_non_eeg_subject_held_out_models" or not metrics_ready:
            findings.append({"severity": "blocking", "issue": "Public data is mapped and split, but no external model metrics were produced."})
    else:
        findings.append({"severity": "blocking", "issue": "Public evaluation manifest is missing."})
    if public.get("status") != "completed":
        findings.append({"severity": "blocking", "issue": "Public external evaluation phase did not complete."})
    findings.extend([
        {"severity": "blocking", "issue": "Mayo frozen test contains only six seizure events; event sensitivity is too uncertain for a strong claim."},
        {"severity": "blocking", "issue": "False-alarm burden must be reported at the selected operating point and remains above a practical target in recent runs."},
        {"severity": "warning", "issue": "Standard versus Pro must be compared over paired seeds and identical validation alarm budgets; a small AUC delta is not evidence of Pro benefit."},
    ])
    report = {
        "campaign_status": manifest.get("status"),
        "scientific_release_status": "not_ready",
        "findings": findings,
        "required_before_submission": [
            "train and evaluate public non-EEG Standard and Pro feature models with subject-held-out metrics",
            "expand or justify the frozen Mayo event holdout and report patient-level uncertainty",
            "pre-register validation operating-point selection and report paired Standard/Pro results",
            "retain failure cases and calibration/alarm curves",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
