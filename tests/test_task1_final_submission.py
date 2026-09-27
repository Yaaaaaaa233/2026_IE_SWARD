"""Synthetic contract checks for a future-only Task 1 submission."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from models.task1_final.prepare_submission import digest, package, validate_submission


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class SubmissionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.vehicles = self.root / "vehicles.csv"
        self.predictions = self.root / "predictions.csv"
        self.provenance = self.root / "provenance.json"
        self.out = self.root / "out"
        write_csv(self.vehicles, ["gpsno"], [{"gpsno": f"V{i:03}"} for i in range(500)])
        write_csv(self.predictions, ["gpsno", "risk_prob"],
                  [{"gpsno": f"V{i:03}", "risk_prob": i / 1000} for i in range(500)])
        h = "a" * 64
        self.manifest = {
            "prediction_role": "future_inference",
            "model_version": "fixed_synthetic",
            "as_of": "2026-08-01",
            "feature_start": "2026-06-01",
            "feature_end_exclusive": "2026-08-01",
            "training_feature_end_exclusive": "2026-06-21",
            "training_label_start": "2026-06-21",
            "training_label_end_exclusive": "2026-07-31",
            "model_sha256": h,
            "inference_features_sha256": h,
            "training_features_sha256": h,
            "training_labels_sha256": h,
            "feature_schema_sha256": h,
            "predictions_sha256": digest(self.predictions),
        }
        self.save_manifest()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def save_manifest(self) -> None:
        self.provenance.write_text(json.dumps(self.manifest), encoding="utf-8")

    def run_package(self) -> Path:
        return package(self.predictions, self.vehicles, self.provenance, self.out)

    def test_valid_future_inference_is_deterministic_and_lf(self) -> None:
        output = self.run_package()
        first = output.read_bytes()
        self.assertNotIn(b"\r", first)
        self.assertFalse(first.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(first.count(b"\n"), 501)
        validate_submission(output, {f"V{i:03}" for i in range(500)})
        with output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(sum(int(r["accident"]) for r in rows), 100)
        self.assertEqual(self.run_package().read_bytes(), first)
        receipt = json.loads((self.out / "submission_receipt.json").read_text())
        self.assertEqual(receipt["submission_sha256"], digest(output))

    def test_oof_role_and_proxy_cutoff_are_rejected(self) -> None:
        self.manifest["prediction_role"] = "oof"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "future_inference"):
            self.run_package()
        self.manifest["prediction_role"] = "future_inference"
        self.manifest["as_of"] = "2026-06-21"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "as_of"):
            self.run_package()

    def test_prediction_hash_and_vehicle_set_are_checked(self) -> None:
        self.predictions.write_text(self.predictions.read_text() + "V999,0.2\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.run_package()
        self.manifest["predictions_sha256"] = digest(self.predictions)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "500 rows"):
            self.run_package()

    def test_label_column_and_nonfinite_probability_are_rejected(self) -> None:
        rows = [{"gpsno": f"V{i:03}", "risk_prob": "nan" if i == 0 else i / 1000,
                 "y": 0} for i in range(500)]
        write_csv(self.predictions, ["gpsno", "risk_prob", "y"], rows)
        self.manifest["predictions_sha256"] = digest(self.predictions)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "expected columns"):
            self.run_package()
        write_csv(self.predictions, ["gpsno", "risk_prob"],
                  [{"gpsno": r["gpsno"], "risk_prob": r["risk_prob"]} for r in rows])
        self.manifest["predictions_sha256"] = digest(self.predictions)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "finite"):
            self.run_package()

    def test_crlf_submission_is_rejected(self) -> None:
        output = self.run_package()
        output.write_bytes(output.read_bytes().replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(ValueError, "LF"):
            validate_submission(output, {f"V{i:03}" for i in range(500)})


if __name__ == "__main__":
    unittest.main()
