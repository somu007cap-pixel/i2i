import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seizure_detection import research_campaign


class ResearchCampaignTests(unittest.TestCase):
    def test_phases_have_unique_names_and_commands(self):
        names = [phase["name"] for phase in research_campaign.PHASES]
        self.assertEqual(len(names), len(set(names)))
        for phase in research_campaign.PHASES:
            self.assertIn("description", phase)
            if phase["kind"] == "detection":
                self.assertTrue(phase["args"])
                self.assertNotIn("--resume", phase["args"])

    def test_campaign_includes_scientific_quality_gate(self):
        kinds = {phase["kind"] for phase in research_campaign.PHASES}
        self.assertIn("quality_audit", kinds)

    def test_public_phase_forces_non_eeg_metrics(self):
        phase = next(item for item in research_campaign.PHASES if item["kind"] == "public_evaluation")
        self.assertTrue(phase.get("requires_metrics"))

    def test_manifest_is_durable_and_preserves_holdout_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "campaign_manifest.json"
            with patch.object(research_campaign, "CAMPAIGN_ROOT", root), patch.object(
                research_campaign, "MANIFEST", manifest_path
            ):
                manifest = research_campaign.load_manifest()
                research_campaign.save_manifest(manifest)
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertIn("test split", loaded["holdout_policy"])
                self.assertEqual(set(loaded["phases"]), {
                    phase["name"] for phase in research_campaign.PHASES
                })


if __name__ == "__main__":
    unittest.main()
