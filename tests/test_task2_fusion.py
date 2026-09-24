"""Synthetic acceptance tests for the R3 fusion harness (T2R-005).

All scores, labels and probabilities below are fabricated with fixed seeds.
The tests assert fusion mechanics, manifest consistency (probability column
declared iff attached), same-MOE grid behaviour, disagreement detection and
the D20 talking-point format.
"""

import json
import random
import unittest
from pathlib import Path

from task2.evaluation.evaluate_scores import evaluate_scores
from task2.fusion import (
    dimension_talking_point,
    fuse_scores,
    rank_disagreements,
    run_alpha_grid,
)

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_CONTRACT = json.loads(
    (ROOT / "task2" / "evaluation" / "evaluation_contract_v1.json").read_text(
        encoding="utf-8"
    )
)

MANIFEST_BASE = {
    "as_of_time": "2026-06-21T00:00:00",
    "feature_window": {"start": "2026-06-01T00:00:00", "end": "2026-06-21T00:00:00"},
    "label_window": {"start": "2026-06-21T00:00:00", "end": "2026-07-31T00:00:00"},
    "timezone": "Asia/Shanghai",
    "label_version": "label_v2_record_count_20260923",
    "split_version": "split_v2_record_strat5_seed42",
    "score_version": "T2R-005-synthetic",
    "all_fit_operations_train_fold_only": True,
    "uses_task1_probability": False,
    "task1_predictions_are_oof": False,
    "target_count": 50,
}


def _synthetic_scores(seed: int = 20260925):
    """Risky vehicles get low Q0/Q1 (high risk probability), safe the reverse."""

    rng = random.Random(seed)
    q0: dict[str, float] = {}
    q1: dict[str, float] = {}
    labels: dict[str, int] = {}
    folds: dict[str, str] = {}
    for index in range(50):
        gpsno = f"synthetic-vehicle-{index:03d}"
        risky = index < 25
        labels[gpsno] = 1 if risky else 0
        folds[gpsno] = f"f{index % 5}"
        q0[gpsno] = round(
            (rng.uniform(40, 58) if risky else rng.uniform(78, 92)), 3
        )
        probability = rng.uniform(0.55, 0.85) if risky else rng.uniform(0.08, 0.3)
        q1[gpsno] = round(100.0 * (1.0 - probability), 3)
    return q0, q1, labels, folds


class FuseTests(unittest.TestCase):
    def setUp(self):
        self.q0, self.q1, self.labels, self.folds = _synthetic_scores()

    def test_formula_and_endpoints(self):
        fused = fuse_scores(self.q0, self.q1, 0.5)
        for gpsno in self.q0:
            self.assertAlmostEqual(
                fused[gpsno], 0.5 * self.q0[gpsno] + 0.5 * self.q1[gpsno]
            )
        self.assertEqual(fuse_scores(self.q0, self.q1, 1.0), dict(self.q0))
        self.assertEqual(fuse_scores(self.q0, self.q1, 0.0), dict(self.q1))

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            fuse_scores(self.q0, self.q1, 1.5)
        with self.assertRaises(ValueError):
            fuse_scores(self.q0, {**self.q1, "extra": 50.0}, 0.5)
        with self.assertRaises(ValueError):
            fuse_scores({**self.q0, "synthetic-vehicle-000": 120.0}, self.q1, 0.5)


class AlphaGridTests(unittest.TestCase):
    def setUp(self):
        self.q0, self.q1, self.labels, self.folds = _synthetic_scores()

    def test_every_arm_passes_the_same_gate_and_manifest_stays_consistent(self):
        result = run_alpha_grid(
            self.q0, self.q1, self.labels, self.folds,
            EVALUATION_CONTRACT, MANIFEST_BASE,
            alphas=(1.0, 0.7, 0.5, 0.3, 0.0),
        )
        self.assertEqual(len(result["runs"]), 5)
        for run in result["runs"]:
            self.assertEqual(run["report"]["status"], "pass", run["report"].get("errors"))
            self.assertTrue(run["report"]["primary_gate_pass"])
        selection = result["selection"]
        self.assertIsNotNone(selection)
        self.assertIn(selection["alpha_q0"], (1.0, 0.7, 0.5, 0.3, 0.0))

    def test_pure_rule_arm_carries_no_probability_column(self):
        result = run_alpha_grid(
            self.q0, self.q1, self.labels, self.folds,
            EVALUATION_CONTRACT, MANIFEST_BASE,
            alphas=(1.0,),
        )
        run = result["runs"][0]
        self.assertEqual(run["alpha_q0"], 1.0)
        self.assertEqual(run["report"]["status"], "pass")

    def test_probability_declaration_mismatch_is_rejected_by_evaluation(self):
        # Attach the probability column while declaring it unused -> hard check fails.
        rows = []
        fused = fuse_scores(self.q0, self.q1, 0.5)
        for gpsno in sorted(fused):
            rows.append(
                {
                    "gpsno": gpsno,
                    "y": self.labels[gpsno],
                    "safety_score": fused[gpsno],
                    "fold": self.folds[gpsno],
                    "task1_risk_probability": 1.0 - self.q1[gpsno] / 100.0,
                }
            )
        manifest = dict(MANIFEST_BASE)
        manifest["target_count"] = len(rows)
        report = evaluate_scores(rows, EVALUATION_CONTRACT, manifest)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(
            any("task1" in error for error in report["errors"]), report["errors"]
        )


class DisagreementTests(unittest.TestCase):
    def test_top_disagreements_ranked_by_max_gap(self):
        q0 = {"a": 10.0, "b": 50.0, "c": 90.0}
        q1 = {"a": 90.0, "b": 50.0, "c": 10.0}
        fused = {"a": 50.0, "b": 50.0, "c": 50.0}
        result = rank_disagreements(q0, q1, fused, top_k=2)
        self.assertEqual(result["vehicle_count"], 3)
        self.assertEqual(len(result["top_disagreements"]), 2)
        top = result["top_disagreements"][0]
        self.assertIn(top["gpsno"], ("a", "c"))
        self.assertEqual(top["max_rank_gap"], 2)

    def test_perfect_agreement_has_zero_gap(self):
        scores = {name: float(value) for value, name in enumerate("abcdef", start=1)}
        result = rank_disagreements(scores, scores, scores)
        self.assertEqual(set(), {e["gpsno"] for e in result["top_disagreements"] if e["max_rank_gap"] > 0})


class TalkingPointTests(unittest.TestCase):
    def test_format_and_validation(self):
        self.assertEqual(
            dimension_talking_point("跟车距离", 80.4),
            "您的跟车距离风险高于80%的同行司机",
        )
        with self.assertRaises(ValueError):
            dimension_talking_point("  ", 50.0)
        with self.assertRaises(ValueError):
            dimension_talking_point("夜间行驶", 100.0)


if __name__ == "__main__":
    unittest.main()
