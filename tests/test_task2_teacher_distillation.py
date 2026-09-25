import json
import unittest
from pathlib import Path

from task2.teacher_distillation.model import (
    FeatureSpec,
    build_feature_specs,
    cross_fit_distillation,
    fit_ruleset,
    recompute_score,
    score_vehicle,
)
from task2.evaluation.evaluate_scores import evaluate_scores


ROOT = Path(__file__).resolve().parents[1]


def feature_specs():
    return (
        FeatureSpec("C01", "行为一", "G1", "P1", "rate_a"),
        FeatureSpec("C02", "行为二", "G1", "P1", "rate_b"),
        FeatureSpec("C03", "行为三", "G2", "P2", "rate_c"),
    )


def synthetic_population():
    rows = []
    teacher = {}
    folds = {}
    labels = {}
    for repeat in range(2):
        for first in range(10):
            for second in range(10):
                gpsno = f"synthetic-{repeat}-{first}-{second}"
                correlated = min(9, first + ((second + repeat) % 3 == 0))
                probability = (5.0 + 35.0 * first / 9.0 + 25.0 * second / 9.0) / 100.0
                rows.append(
                    {
                        "gpsno": gpsno,
                        "exposure": 100.0,
                        "rate_a": float(first),
                        "rate_b": float(correlated),
                        "rate_c": float(second),
                    }
                )
                teacher[gpsno] = probability
                folds[gpsno] = f"f{(first * 3 + second + repeat) % 5}"
                labels[gpsno] = int(probability >= 0.4)
    return rows, teacher, folds, labels


class TeacherDistillationTests(unittest.TestCase):
    def test_contract_builds_twenty_deductible_features(self):
        contract = json.loads(
            (ROOT / "task2" / "classification" / "classification_v1.json").read_text(
                encoding="utf-8"
            )
        )
        specs = build_feature_specs(contract)
        self.assertEqual(20, len(specs))
        self.assertEqual({"G1", "G2", "G3", "G4", "G5"}, {s.semantic_group for s in specs})
        self.assertNotIn("rate_11803_per_1000km", {s.column for s in specs})
        self.assertNotIn("rate_41006_per_1000km", {s.column for s in specs})

    def test_fit_is_monotone_recomputable_and_reduces_dimensions(self):
        rows, teacher, _, _ = synthetic_population()
        rules = fit_ruleset(
            rows,
            teacher,
            feature_specs(),
            teacher_model_version="synthetic-oof-v1",
            teacher_scores_are_oof=True,
        )
        self.assertEqual({"G1", "G2"}, set(rules.group_weights))
        self.assertLess(len(rules.group_weights), len(rules.feature_specs))
        low = score_vehicle(
            {"gpsno": "low", "exposure": 100, "rate_a": 0, "rate_b": 0, "rate_c": 0},
            rules,
        )
        high_a = score_vehicle(
            {"gpsno": "high-a", "exposure": 100, "rate_a": 9, "rate_b": 0, "rate_c": 0},
            rules,
        )
        high_all = score_vehicle(
            {"gpsno": "high-all", "exposure": 100, "rate_a": 9, "rate_b": 9, "rate_c": 9},
            rules,
        )
        self.assertGreaterEqual(low.safety_score, high_a.safety_score)
        self.assertGreaterEqual(high_a.safety_score, high_all.safety_score)
        self.assertAlmostEqual(high_all.safety_score, recompute_score(high_all, rules), places=10)
        self.assertAlmostEqual(
            sum(high_all.class_points.values()), sum(high_all.group_deductions.values()), places=10
        )

    def test_cross_fit_tracks_teacher_and_reports_actual_label_metrics_separately(self):
        rows, teacher, folds, labels = synthetic_population()
        result = cross_fit_distillation(
            rows,
            teacher,
            folds,
            feature_specs(),
            teacher_model_version="synthetic-oof-v1",
            teacher_scores_are_oof=True,
            actual_labels=labels,
        )
        self.assertEqual(len(rows), len(result.rows))
        self.assertGreater(result.metrics["teacher_risk_spearman"], 0.8)
        self.assertGreater(result.metrics["top_risk_overlap"], 0.6)
        self.assertTrue(result.metrics["quintile_direction_pass"])
        self.assertTrue(result.metrics["actual_outcome_validated"])
        self.assertFalse(result.metrics["teacher_fidelity_only"])
        self.assertGreater(result.metrics["actual_auc"], 0.8)
        for row in result.rows:
            self.assertAlmostEqual(row["safety_score"], row["rule_recomputed_score"], places=10)
        contract = json.loads(
            (ROOT / "task2" / "evaluation" / "evaluation_contract_v1.json").read_text(
                encoding="utf-8"
            )
        )
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
            "label_version": contract["dependencies"]["label_version"],
            "split_version": contract["dependencies"]["split_version"],
            "score_version": "D0-synthetic-v1",
            "all_fit_operations_train_fold_only": True,
            "uses_task1_probability": True,
            "task1_predictions_are_oof": True,
            "target_count": len(rows),
        }
        report = evaluate_scores(result.rows, contract, manifest)
        self.assertEqual("pass", report["status"], report.get("errors"))

    def test_low_exposure_uses_prior_and_bad_inputs_fail_closed(self):
        rows, teacher, folds, _ = synthetic_population()
        rules = fit_ruleset(
            rows,
            teacher,
            feature_specs(),
            teacher_model_version="synthetic-oof-v1",
            teacher_scores_are_oof=True,
            min_exposure=1.0,
            low_evidence_prior=50.0,
        )
        scoring = score_vehicle(
            {"gpsno": "stopped", "exposure": 0, "rate_a": 0, "rate_b": 0, "rate_c": 0},
            rules,
        )
        self.assertEqual("insufficient_evidence", scoring.observation_status)
        self.assertEqual(50.0, scoring.safety_score)
        with self.assertRaisesRegex(ValueError, "out-of-fold"):
            fit_ruleset(
                rows,
                teacher,
                feature_specs(),
                teacher_model_version="bad-in-sample",
                teacher_scores_are_oof=False,
            )
        with self.assertRaisesRegex(ValueError, "folds must cover exactly"):
            cross_fit_distillation(
                rows,
                teacher,
                {key: value for key, value in list(folds.items())[1:]},
                feature_specs(),
                teacher_model_version="synthetic-oof-v1",
                teacher_scores_are_oof=True,
            )


if __name__ == "__main__":
    unittest.main()
