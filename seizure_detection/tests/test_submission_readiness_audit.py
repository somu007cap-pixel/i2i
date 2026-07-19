import unittest

from seizure_detection.submission_readiness_audit import (
    audit_detection,
    build_scientific_checklist,
)


class ScientificChecklistTests(unittest.TestCase):
    def test_reports_missing_uncertainty_and_operating_point(self) -> None:
        results = {
            "standard_mode": {"event_level": {"detected_events": 2, "event_count": 4}},
            "pro_mode": {"event_level": {"detected_events": 3, "event_count": 4}},
            "label_policy": {"offset_policy": "use onset plus fixed duration"},
        }

        checklist = build_scientific_checklist(results)

        self.assertEqual(checklist["uncertainty_estimates"], "warn")
        self.assertEqual(checklist["operating_point_documented"], "warn")
        self.assertEqual(checklist["label_policy_documented"], "pass")

    def test_passes_when_event_ci_and_operating_point_exist(self) -> None:
        results = {
            "standard_mode": {
                "event_level": {
                    "detected_events": 2,
                    "event_count": 4,
                    "event_sensitivity_ci95_low": 0.1,
                    "event_sensitivity_ci95_high": 0.9,
                }
            },
            "pro_mode": {
                "event_level": {
                    "detected_events": 3,
                    "event_count": 4,
                    "event_sensitivity_ci95_low": 0.2,
                    "event_sensitivity_ci95_high": 0.8,
                }
            },
            "label_policy": {"offset_policy": "use onset plus fixed duration"},
            "matched_false_alarm_operating_point": {"validation_alarm_budget": 10.0},
        }

        checklist = build_scientific_checklist(results)

        self.assertEqual(checklist["uncertainty_estimates"], "pass")
        self.assertEqual(checklist["operating_point_documented"], "pass")
        self.assertEqual(checklist["label_policy_documented"], "pass")

    def test_audit_discloses_event_sensitivity_gain_separately(self) -> None:
        results = {
            "status": "completed",
            "split_strategy": "patient-level held-out split",
            "evaluation_scope": "patient-level generalization",
            "claim_scope": "rigorous prototype; not a clinically validated detector",
            "standard_mode": {
                "accuracy": 0.98,
                "all_normal_baseline_accuracy": 0.99,
                "auc": 0.50,
                "recall": 0.20,
                "false_alarms_per_hour": 10.0,
                "event_level": {
                    "detected_events": 2,
                    "event_count": 4,
                    "event_sensitivity": 0.50,
                    "event_sensitivity_ci95_low": 0.10,
                    "event_sensitivity_ci95_high": 0.90,
                },
            },
            "pro_mode": {
                "accuracy": 0.97,
                "auc": 0.60,
                "recall": 0.10,
                "false_alarms_per_hour": 12.0,
                "event_level": {
                    "detected_events": 3,
                    "event_count": 4,
                    "event_sensitivity": 0.75,
                    "event_sensitivity_ci95_low": 0.20,
                    "event_sensitivity_ci95_high": 0.95,
                },
            },
            "model_provenance": {"variants": {}},
            "product_allocation_selection": {"selection_split": "validation"},
            "best_science_model": {},
            "best_product_model": {},
            "label_policy": {"offset_policy": "fixed duration"},
            "matched_false_alarm_operating_point": {
                "fixed_validation_alarm_budgets": {}
            },
        }

        import seizure_detection.submission_readiness_audit as audit_module

        old_loader = audit_module.load_json
        try:
            audit_module.load_json = lambda path: results
            checks = []
            audit_detection(checks)
        finally:
            audit_module.load_json = old_loader

        by_name = {item["name"]: item for item in checks}
        self.assertEqual(by_name["pro_signal_gain"]["status"], "warn")
        self.assertEqual(by_name["pro_event_sensitivity_gain"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
