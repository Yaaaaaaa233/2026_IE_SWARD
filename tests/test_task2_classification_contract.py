import copy
import json
import unittest
from pathlib import Path

from task2.classification.validate_classification import validate_contract


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "task2"
    / "classification"
    / "classification_v1.json"
)


class Task2ClassificationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_published_contract_passes(self):
        self.assertEqual([], validate_contract(self.contract))

    def test_duplicate_code_fails(self):
        bad = copy.deepcopy(self.contract)
        bad["event_mappings"][1]["code"] = bad["event_mappings"][0]["code"]
        self.assertTrue(any("event codes must be unique" in item for item in validate_contract(bad)))

    def test_missing_class_boundary_fails(self):
        bad = copy.deepcopy(self.contract)
        bad["catalog_classes"][0]["boundary"] = ""
        self.assertTrue(any("lacks a one-sentence boundary" in item for item in validate_contract(bad)))

    def test_blind_spot_requires_equipped_cohort_and_lane_change_denominator(self):
        bad = copy.deepcopy(self.contract)
        item = next(row for row in bad["event_mappings"] if row["code"] == "60292")
        item["applicability_conditions"] = []
        item["exposure_denominator"] = "mileage_km"
        self.assertTrue(any("blind-spot event 60292" in item for item in validate_contract(bad)))

    def test_reserved_class_cannot_be_enabled_before_quality_gate(self):
        bad = copy.deepcopy(self.contract)
        item = next(row for row in bad["catalog_classes"] if row["id"] == "C29")
        item["enabled"] = True
        self.assertTrue(any("reserved catalog class C29" in item for item in validate_contract(bad)))

    def test_future_label_contract_cannot_be_weakened(self):
        bad = copy.deepcopy(self.contract)
        bad["label_contract"]["near_miss_min_count"] = 1
        self.assertTrue(any("ADR-0006/ADR-0007" in item for item in validate_contract(bad)))


if __name__ == "__main__":
    unittest.main()
