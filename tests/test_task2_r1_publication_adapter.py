import unittest
import json
from pathlib import Path

from task2.deduction.engine import DeductionRuleset, VehicleScoring
from task2.evaluation.evaluate_scores import evaluate_scores
from task2.evaluation.r1_publication import build_r1_publication_rows


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_CONTRACT = json.loads(
    (ROOT / "task2" / "evaluation" / "evaluation_contract_v1.json").read_text(
        encoding="utf-8"
    )
)


def ruleset():
    return DeductionRuleset(
        rule_version="synthetic-r1-publication-v1",
        denominator_kind="synthetic_exposure",
        bins_by_class={},
        prior_rate_by_class={},
        class_group={},
        class_budget={},
    )


class R1PublicationAdapterTests(unittest.TestCase):
    def test_observed_score_is_unchanged_and_recomputable(self):
        scoring = VehicleScoring(
            gpsno="observed-vehicle",
            safety_score=100.0,
            evaluable_class_count=3,
            insufficient_evidence=False,
        )
        rows = build_r1_publication_rows(
            [scoring], ruleset(), {"observed-vehicle": 0}, {"observed-vehicle": "f0"}
        )
        row = rows[0]
        self.assertEqual(100.0, row["safety_score"])
        self.assertEqual(row["safety_score"], row["rule_recomputed_score"])
        self.assertEqual("observed", row["observation_status"])
        self.assertEqual("raw_rule_score", row["publication_policy"])

    def test_low_evidence_raw_hundred_is_published_as_conservative_prior(self):
        scoring = VehicleScoring(
            gpsno="low-evidence-vehicle",
            safety_score=100.0,
            evaluable_class_count=0,
            insufficient_evidence=True,
        )
        rows = build_r1_publication_rows(
            [scoring],
            ruleset(),
            {"low-evidence-vehicle": 1},
            {"low-evidence-vehicle": "f1"},
            low_evidence_prior=50.0,
        )
        row = rows[0]
        self.assertEqual(100.0, row["raw_rule_score"])
        self.assertEqual(50.0, row["safety_score"])
        self.assertEqual(50.0, row["rule_recomputed_score"])
        self.assertEqual("insufficient_evidence", row["observation_status"])
        self.assertEqual("low_evidence_prior", row["publication_policy"])

    def test_vehicle_sets_and_inputs_are_strict(self):
        scoring = VehicleScoring(
            gpsno="vehicle-a",
            safety_score=100.0,
            evaluable_class_count=0,
            insufficient_evidence=True,
        )
        with self.assertRaisesRegex(ValueError, "labels must cover exactly"):
            build_r1_publication_rows(
                [scoring], ruleset(), {"different": 0}, {"vehicle-a": "f0"}
            )
        with self.assertRaisesRegex(ValueError, "low_evidence_prior"):
            build_r1_publication_rows(
                [scoring], ruleset(), {"vehicle-a": 0}, {"vehicle-a": "f0"},
                low_evidence_prior=101,
            )

    def test_published_rows_pass_the_r4_evaluator_without_false_high_scores(self):
        scorings = []
        labels = {}
        folds = {}
        for index in range(10):
            gpsno = f"synthetic-publication-{index}"
            low_evidence = index < 5
            scorings.append(
                VehicleScoring(
                    gpsno=gpsno,
                    safety_score=100.0,
                    evaluable_class_count=0 if low_evidence else 3,
                    insufficient_evidence=low_evidence,
                )
            )
            labels[gpsno] = 1 if low_evidence else 0
            folds[gpsno] = f"f{index % 5}"
        rows = build_r1_publication_rows(scorings, ruleset(), labels, folds)
        manifest = {
            "as_of_time": "2026-06-21T00:00:00",
            "feature_window": {
                "start": "2026-06-01T00:00:00",
                "end": "2026-06-21T00:00:00",
            },
            "label_window": {
                "start": "2026-06-21T00:00:00",
                "end": "2026-07-31T00:00:00",
            },
            "timezone": "Asia/Shanghai",
            "label_version": "label_v2_record_count_20260923",
            "split_version": "split_v2_record_strat5_seed42",
            "score_version": "synthetic-r1-publication-v1",
            "all_fit_operations_train_fold_only": True,
            "uses_task1_probability": False,
            "task1_predictions_are_oof": False,
            "target_count": 10,
        }
        report = evaluate_scores(rows, EVALUATION_CONTRACT, manifest)
        self.assertEqual("pass", report["status"], report.get("errors"))
        self.assertTrue(report["primary_gate_pass"])
        self.assertEqual(1.0, report["roc_auc"])


if __name__ == "__main__":
    unittest.main()
