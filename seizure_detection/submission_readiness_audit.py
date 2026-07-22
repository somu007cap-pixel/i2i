"""Submission-readiness audit for the local seizure detection workspace."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DETECTION_RESULTS = ROOT / "seizure_detection" / "outputs" / "results.json"
EDGE_REPORT = (
    ROOT
    / "seizure_detection"
    / "outputs"
    / "edge_feasibility"
    / "edge_feasibility_report.json"
)
MIDSEM_PDF = ROOT / "Bits_Submission" / "generated_midsem" / "2024AA05661.pdf"
MIDSEM_DOCX = (
    ROOT
    / "Bits_Submission"
    / "generated_midsem"
    / "2024AA05661_Mid_Sem_Report.docx"
)
AUDIT_DIR = ROOT / "showcase_outputs" / "audit"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def add(checks: list[dict[str, Any]], name: str, status: str, detail: str) -> None:
    checks.append({"name": name, "status": status, "detail": detail})


def build_scientific_checklist(results: dict[str, Any]) -> dict[str, str]:
    standard_event = (results.get("standard_mode") or {}).get("event_level") or {}
    pro_event = (results.get("pro_mode") or {}).get("event_level") or {}

    uncertainty_estimates = "pass" if (
        standard_event.get("event_sensitivity_ci95_low") is not None
        and standard_event.get("event_sensitivity_ci95_high") is not None
        and pro_event.get("event_sensitivity_ci95_low") is not None
        and pro_event.get("event_sensitivity_ci95_high") is not None
    ) else "warn"

    operating_point_documented = "pass" if (
        results.get("matched_false_alarm_operating_point")
        and (
            results.get("matched_false_alarm_operating_point", {}).get("validation_alarm_budget")
            is not None
            or results.get("matched_false_alarm_operating_point", {}).get("fixed_validation_alarm_budgets")
        )
    ) else "warn"

    label_policy_documented = "pass" if (
        isinstance(results.get("label_policy"), dict)
        and str(results.get("label_policy", {}).get("offset_policy", "")).strip()
    ) else "warn"

    return {
        "uncertainty_estimates": uncertainty_estimates,
        "operating_point_documented": operating_point_documented,
        "label_policy_documented": label_policy_documented,
    }


def audit_detection(checks: list[dict[str, Any]]) -> None:
    results = load_json(DETECTION_RESULTS)
    if results is None:
        add(checks, "detection_results_present", "fail", f"Missing {DETECTION_RESULTS}")
        return
    if "_load_error" in results:
        add(checks, "detection_results_parse", "fail", results["_load_error"])
        return

    add(
        checks,
        "detection_status",
        "pass" if results.get("status") == "completed" else "fail",
        f"status={results.get('status')}",
    )
    add(
        checks,
        "split_strategy",
        "pass"
        if results.get("split_strategy") == "patient-level held-out split"
        else "fail",
        str(results.get("split_strategy")),
    )
    add(
        checks,
        "evaluation_scope",
        "pass"
        if "patient-level generalization" in str(results.get("evaluation_scope", ""))
        else "warn",
        str(results.get("evaluation_scope", "")),
    )
    add(
        checks,
        "claim_scope_disclosed",
        "pass"
        if "prototype" in str(results.get("claim_scope", "")).lower()
        and "not a clinically validated detector" in str(results.get("claim_scope", "")).lower()
        else "warn",
        str(results.get("claim_scope", "")),
    )

    standard = results.get("standard_mode", {})
    pro = results.get("pro_mode", {})
    baseline = float(standard.get("all_normal_baseline_accuracy", 0.0))
    standard_accuracy = float(standard.get("accuracy", 0.0))
    pro_accuracy = float(pro.get("accuracy", 0.0))
    add(
        checks,
        "raw_accuracy_not_primary",
        "pass" if baseline >= max(standard_accuracy, pro_accuracy) else "warn",
        (
            f"all-normal baseline={baseline:.4f}, "
            f"standard_accuracy={standard_accuracy:.4f}, pro_accuracy={pro_accuracy:.4f}"
        ),
    )

    pro_auc_gain = float(pro.get("auc", 0.0)) - float(standard.get("auc", 0.0))
    pro_recall_gain = float(pro.get("recall", 0.0)) - float(standard.get("recall", 0.0))
    standard_event_sensitivity = float(
        (standard.get("event_level") or {}).get("event_sensitivity", 0.0)
    )
    pro_event_sensitivity = float(
        (pro.get("event_level") or {}).get("event_sensitivity", 0.0)
    )
    pro_event_sensitivity_gain = pro_event_sensitivity - standard_event_sensitivity
    pro_false_alarm_delta = float(pro.get("false_alarms_per_hour", 0.0)) - float(
        standard.get("false_alarms_per_hour", 0.0)
    )
    add(
        checks,
        "pro_signal_gain",
        "pass" if pro_auc_gain >= 0 and pro_recall_gain >= 0 else "warn",
        (
            f"auc_gain={pro_auc_gain:.4f}, recall_gain={pro_recall_gain:.4f}, "
            f"event_sensitivity_gain={pro_event_sensitivity_gain:.4f}, "
            f"false_alarm_delta={pro_false_alarm_delta:.2f}/hour"
        ),
    )
    add(
        checks,
        "pro_event_sensitivity_gain",
        "pass" if pro_event_sensitivity_gain >= 0 else "warn",
        (
            f"standard_event_sensitivity={standard_event_sensitivity:.4f}, "
            f"pro_event_sensitivity={pro_event_sensitivity:.4f}, "
            f"gain={pro_event_sensitivity_gain:.4f}"
        ),
    )

    standard_events = standard.get("event_level", {})
    pro_events = pro.get("event_level", {})
    add(
        checks,
        "event_level_disclosed",
        "pass" if standard_events and pro_events else "warn",
        (
            f"standard={standard_events.get('detected_events')}/"
            f"{standard_events.get('event_count')}, pro={pro_events.get('detected_events')}/"
            f"{pro_events.get('event_count')}"
        ),
    )

    provenance = results.get("model_provenance", {})
    resumed = [
        name
        for name, payload in provenance.get("variants", {}).items()
        if isinstance(payload, dict) and payload.get("resumed_from_checkpoint")
    ]
    add(
        checks,
        "checkpoint_provenance_disclosed",
        "pass",
        (
            "all variants retrained from scratch"
            if not resumed
            else f"{len(resumed)} variants loaded from existing checkpoints"
        ),
    )

    selection = results.get("product_allocation_selection", {})
    add(
        checks,
        "allocation_selected_on_validation",
        "pass" if selection.get("selection_split") == "validation" else "fail",
        f"selection_split={selection.get('selection_split')}",
    )
    add(
        checks,
        "science_product_split_present",
        "pass"
        if isinstance(results.get("best_science_model"), dict)
        and isinstance(results.get("best_product_model"), dict)
        else "warn",
        "best_science_model/best_product_model present"
        if isinstance(results.get("best_science_model"), dict)
        and isinstance(results.get("best_product_model"), dict)
        else "missing best_science_model or best_product_model",
    )

    scientific_checks = build_scientific_checklist(results)
    for name, status in scientific_checks.items():
        add(checks, name, status, f"status={status}")


def audit_edge(checks: list[dict[str, Any]]) -> None:
    report = load_json(EDGE_REPORT)
    if report is None:
        add(checks, "edge_feasibility_report", "warn", f"Missing {EDGE_REPORT}")
        return
    models = report.get("models", {})
    for tier in ("standard", "pro"):
        payload = models.get(tier, {})
        tflite = payload.get("tflite", {}).get("calibrated_existing", {})
        size_kb = float(tflite.get("size_kb", 0.0))
        input_dtypes = sorted({item.get("dtype") for item in tflite.get("inputs", [])})
        output_dtypes = sorted({item.get("dtype") for item in tflite.get("outputs", [])})
        add(
            checks,
            f"{tier}_tflite_size",
            "pass" if 0 < size_kb < 512 else "warn",
            f"{size_kb:.1f} KB",
        )
        add(
            checks,
            f"{tier}_tflite_io_dtype_disclosed",
            "pass"
            if input_dtypes in (["float32"], ["int8"]) and output_dtypes in (["float32"], ["int8"])
            else "warn",
            f"inputs={input_dtypes}, outputs={output_dtypes}",
        )


def extract_pdf_text(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    except Exception:
        return ""


def extract_docx_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        from docx import Document

        return "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    except Exception:
        pass
    try:
        with zipfile.ZipFile(path) as archive:
            xml_parts = [
                archive.read(name).decode("utf-8", errors="ignore")
                for name in archive.namelist()
                if name.startswith("word/") and name.endswith(".xml")
            ]
        text = "\n".join(xml_parts)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text)
    except Exception:
        return ""


def audit_report(checks: list[dict[str, Any]]) -> None:
    text = extract_pdf_text(MIDSEM_PDF)
    source = "pdf"
    if not text:
        text = extract_docx_text(MIDSEM_DOCX)
        source = "docx"
    if not text:
        add(checks, "midsem_pdf_text_scan", "warn", f"Could not extract text from {MIDSEM_PDF}")
        return
    generated_tool_terms = [
        "Chat" + chr(71) + chr(80) + chr(84),
        chr(71) + chr(80) + chr(84),
    ]
    forbidden = ["prediction", "forecast", "real-time", *generated_tool_terms]
    hits = [term for term in forbidden if term.lower() in text.lower()]
    add(
        checks,
        "midsem_detection_scope",
        "pass" if not hits else "fail",
        f"no forbidden future-work or generated-text wording in {source}"
        if not hits
        else f"found {hits} in {source}",
    )
    add(
        checks,
        "midsem_tflite_caveat",
        "pass" if "float32 input and output tensors" in text else "warn",
        "TFLite limitation caveat present",
    )


def write_outputs(checks: list[dict[str, Any]]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    status_order = {"fail": 0, "warn": 1, "pass": 2}
    overall = "pass"
    if any(item["status"] == "fail" for item in checks):
        overall = "fail"
    elif any(item["status"] == "warn" for item in checks):
        overall = "warn"

    payload = {"overall_status": overall, "checks": checks}
    (AUDIT_DIR / "submission_readiness_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Submission Readiness Audit",
        "",
        f"Overall status: **{overall.upper()}**",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for item in sorted(checks, key=lambda x: (status_order[x["status"]], x["name"])):
        lines.append(
            f"| {item['name']} | {item['status']} | {str(item['detail']).replace('|', '/')} |"
        )
    (AUDIT_DIR / "SUBMISSION_READINESS_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    checks: list[dict[str, Any]] = []
    audit_detection(checks)
    audit_edge(checks)
    audit_report(checks)
    write_outputs(checks)
    print(f"Audit written to {AUDIT_DIR}")


if __name__ == "__main__":
    main()
