"""The final-fit adapter must reject proxy leakage and incomplete history."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from models.task1_final.prepare_submission import digest
from models.task1_final.run_g8v6_fullfit import (
    MEMBERS, prepare_member, read_labels, read_member, read_vehicles,
    validate_score_manifest,
)


class FullFitContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.vehicles = [f"V{i:03}" for i in range(500)]
        self.vehicle_path = self.root / "vehicles.csv"
        pd.DataFrame({"gpsno": self.vehicles}).to_csv(self.vehicle_path, index=False)
        sample = [f"{v}_20260621_20d" for v in self.vehicles]
        self.labels_path = self.root / "labels.csv"
        pd.DataFrame({"sample_id": sample, "y": [1] * 54 + [0] * 446,
                      "label_version": "label_v2_record_count_20260923",
                      "label_window": "2026-06-21/2026-07-31"}).to_csv(self.labels_path, index=False)
        self.score_paths = {}
        for name in MEMBERS:
            p = self.root / f"{name}.csv"
            pd.DataFrame({"sample_id": sample, "gpsno": self.vehicles,
                          "f": range(500)}).to_csv(p, index=False)
            self.score_paths[name] = p
        self.manifest = {
            "role": "future_features", "as_of": "2026-08-01",
            "feature_start": "2026-06-01", "feature_end_exclusive": "2026-08-01",
            "member_sha256": {name: digest(path) for name, path in self.score_paths.items()},
            "source_coverage": {name: "2026-07-31" for name in
                                ("events_clean", "vehicle_day", "trajectory", "imu")},
        }
        self.manifest_path = self.root / "score_manifest.json"
        self.save_manifest()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_locked_labels_and_vehicle_identity(self) -> None:
        vehicles = read_vehicles(self.vehicle_path)
        labels = read_labels(self.labels_path, vehicles)
        self.assertEqual(int(labels.y.sum()), 54)
        frame = read_member(self.score_paths["g1"], vehicles, score=False,
                            expected_sample_ids=labels.sample_id)
        self.assertEqual(list(frame), ["f"])

    def test_vehicle_order_is_preserved_for_seeded_replay(self) -> None:
        reverse = list(reversed(self.vehicles))
        pd.DataFrame({"gpsno": reverse}).to_csv(self.vehicle_path, index=False)
        self.assertEqual(read_vehicles(self.vehicle_path), reverse)

    def test_sixty_day_manifest_is_rejected(self) -> None:
        self.manifest["feature_end_exclusive"] = "2026-07-31"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "feature_end_exclusive"):
            validate_score_manifest(self.manifest_path, self.score_paths)

    def test_missing_july31_source_and_wrong_file_hash_are_rejected(self) -> None:
        self.manifest["source_coverage"]["vehicle_day"] = "2026-07-30"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "July 31"):
            validate_score_manifest(self.manifest_path, self.score_paths)
        self.manifest["source_coverage"]["vehicle_day"] = "2026-07-31"
        self.manifest["member_sha256"]["g1"] = "0" * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_score_manifest(self.manifest_path, self.score_paths)

    def test_inference_label_and_missing_training_feature_are_rejected(self) -> None:
        p = self.score_paths["g1"]
        frame = pd.read_csv(p)
        frame["y"] = 0
        frame.to_csv(p, index=False)
        with self.assertRaisesRegex(ValueError, "label or fold"):
            read_member(p, self.vehicles, score=True)
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare_member(pd.DataFrame({"a": range(500), "b": range(500)}),
                           pd.DataFrame({"a": range(500)}))


if __name__ == "__main__":
    unittest.main()
