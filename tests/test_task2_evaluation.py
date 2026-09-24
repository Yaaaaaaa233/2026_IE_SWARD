import json
import unittest
from pathlib import Path

from task2.evaluation.baselines import (
    probability_to_safety_score,
    transparent_presence_baseline,
)
from task2.evaluation.evaluate_scores import evaluate_scores
from task2.evaluation.human_review import validate_human_review


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_CONTRACT = json.loads(
    (ROOT / "task2" / "evaluation" / "evaluation_contract_v1.json").read_text(
        encoding="utf-8"
    )
)

SYNTHETIC_MANIFEST = {
    "as_of_time": "2026-06-21T00:00:00",
    "feature_window": {
        "start": "2026-06-01T00:00:00",
        "end": "2026-06-21T00:00:00"
    },
    "label_window": {
        "start": "2026-06-21T00:00:00",
        "end": "2026-07-31T00:00:00"
    },
    "timezone": "Asia/Shanghai",
    "label_version": "label_v2_record_count_20260923",
    "split_version": "split_v2_record_strat5_seed42",
    "score_version": "synthetic_test",
    "all_fit_operations_train_fold_only": True,
    "uses_task1_probability": True,
    "task1_predictions_are_oof": True,
    "target_count": 10
}
CLASSIFICATION_CONTRACT = json.loads(
    (ROOT / "task2" / "classification" / "classification_v1.json").read_text(
        encoding="utf-8"
    )
)


def synthetic_score_rows():
    rows = []
    for index in range(10):
        score = float(index * 10 + 5)
        rows.append(
            {
                "gpsno": f"vehicle-{index:02d}",
                "y": 1 if index < 5 else 0,
                "safety_score": score,
                "rule_recomputed_score": score,
                "task1_risk_probability": 1.0 - score / 100.0,
                "observation_status": "observed",
                "vehicle_type": "synthetic-type",
                "energy_type": "synthetic-energy",
                "coverage_group": "complete",
                "fold": str(index % 5),
            }
        )
    return rows


def synthetic_human_review_rows(rating=4):
    case_types = [
        "low_score_future_event",
        "high_score_no_event",
        "high_score_future_event",
        "low_exposure_or_missing",
        "trend_improve_or_worsen",
    ]
    rows = []
    questions = EVALUATION_CONTRACT["human_review"]["questions"]
    for reviewer in range(5):
        for case in range(10):
            row = {
                "reviewer_id": f"reviewer-{reviewer}",
                "case_id": f"case-{case}",
                "case_type": case_types[case // 2],
            }
            row.update({question: rating for question in questions})
            rows.append(row)
    return rows


class Task2EvaluationTests(unittest.TestCase):
    def test_q1_probability_conversion(self):
        self.assertEqual(100.0, probability_to_safety_score(0.0))
        self.assertEqual(25.0, probability_to_safety_score(0.75))
        with self.assertRaises(ValueError):
            probability_to_safety_score(1.1)

    def test_q0_is_transparent_and_conservative_for_low_evidence(self):
        result = transparent_presence_baseline(
            {"41001": 2, "41006": 1, "11804": 2},
            {"41001", "41003", "30002", "30000", "11401"},
            CLASSIFICATION_CONTRACT,
        )
        self.assertEqual("observed", result["observation_status"])
        self.assertEqual("low", result["evidence_confidence"])
        self.assertTrue(result["historical_outcome_triggered"])
        self.assertFalse(result["historical_outcome_cap_applied"])
        self.assertEqual(["41006"], result["device_quality_flags"])

        low_evidence = transparent_presence_baseline(
            {}, {"41001"}, CLASSIFICATION_CONTRACT
        )
        self.assertEqual("insufficient_evidence", low_evidence["observation_status"])
        self.assertEqual(50.0, low_evidence["safety_score"])

    def test_score_evaluation_perfect_order_and_monotonic_quintiles(self):
        report = evaluate_scores(
            synthetic_score_rows(), EVALUATION_CONTRACT, SYNTHETIC_MANIFEST
        )
        self.assertEqual("pass", report["status"])
        self.assertEqual(1.0, report["roc_auc"])
        self.assertTrue(report["quintiles"]["monotonic_nonincreasing"])
        self.assertEqual(1.0, report["task1_consistency"]["spearman"])
        self.assertEqual("reference_only_not_a_selection_gate", report["task1_consistency"]["role"])

    def test_duplicate_vehicle_and_recomputation_mismatch_fail(self):
        rows = synthetic_score_rows()
        rows[1]["gpsno"] = rows[0]["gpsno"]
        rows[2]["rule_recomputed_score"] += 1
        report = evaluate_scores(rows, EVALUATION_CONTRACT, SYNTHETIC_MANIFEST)
        self.assertEqual("fail", report["status"])
        self.assertTrue(any("duplicate gpsno" in item for item in report["errors"]))
        self.assertTrue(any("recomputation mismatch" in item for item in report["errors"]))

    def test_low_evidence_vehicle_cannot_receive_false_high_score(self):
        rows = synthetic_score_rows()
        rows[0]["observation_status"] = "insufficient_evidence"
        rows[0]["safety_score"] = 95
        rows[0]["rule_recomputed_score"] = 95
        report = evaluate_scores(rows, EVALUATION_CONTRACT, SYNTHETIC_MANIFEST)
        self.assertEqual("fail", report["status"])
        self.assertTrue(any("falsely high score" in item for item in report["errors"]))

    def test_protocol_rejects_future_features_and_non_oof_task1_scores(self):
        bad_manifest = dict(SYNTHETIC_MANIFEST)
        bad_manifest["feature_window"] = {
            "start": "2026-06-01T00:00:00",
            "end": "2026-06-22T00:00:00",
        }
        bad_manifest["task1_predictions_are_oof"] = False
        report = evaluate_scores(
            synthetic_score_rows(), EVALUATION_CONTRACT, bad_manifest
        )
        self.assertEqual("fail", report["status"])
        self.assertTrue(any("time windows" in item for item in report["errors"]))
        self.assertTrue(any("out-of-fold" in item for item in report["errors"]))

    def test_complete_human_review_passes(self):
        report = validate_human_review(
            synthetic_human_review_rows(), EVALUATION_CONTRACT
        )
        self.assertEqual("pass", report["status"])
        self.assertEqual(4.0, report["overall_mean"])
        self.assertEqual(250, report["rating_count"])

    def test_incomplete_or_low_human_review_fails(self):
        rows = synthetic_human_review_rows()
        rows.pop()
        rows[0][EVALUATION_CONTRACT["human_review"]["questions"][0]] = 2
        report = validate_human_review(rows, EVALUATION_CONTRACT)
        self.assertEqual("fail", report["status"])
        self.assertTrue(any("complete" in item for item in report["errors"]))
        self.assertTrue(any("every single rating" in item for item in report["errors"]))


if __name__ == "__main__":
    unittest.main()
