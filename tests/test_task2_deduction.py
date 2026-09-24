"""Synthetic acceptance tests for the R1 deduction engine (T2R-004).

No real data: vehicles, events and labels below are fabricated with fixed
seeds.  The tests assert the charter invariants (monotonicity, exact
recomputation, value domain) and the two downstream interfaces
(T2-R4 evaluation rows, T2-R2 adapter payload, including a live run of the
operations dynamic layer).
"""

import json
import random
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from task2.deduction import (
    aggregate_group_deduction,
    build_adapter_payload,
    build_lookups,
    build_scores_rows,
    eb_smooth_rate,
    fit_bins,
    fit_ruleset,
    load_contract,
    recompute_score_from_ledger,
    score_vehicle,
    segment_events,
    validate_adapter_payload,
)
from task2.deduction.engine import fold_prior_rate
from task2.evaluation.evaluate_scores import evaluate_scores
from task2.operations.dynamic_score import run_payload

ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION = load_contract()
LOOKUPS = build_lookups(CLASSIFICATION)

SYNTHETIC_POLICY = json.loads(
    (ROOT / "task2" / "deduction" / "deduction_policy_v1.json").read_text(encoding="utf-8")
)
OPERATIONAL_POLICY = json.loads(
    (ROOT / "task2" / "operations" / "operational_policy_v1.json").read_text(
        encoding="utf-8"
    )
)
EVALUATION_CONTRACT = json.loads(
    (ROOT / "task2" / "evaluation" / "evaluation_contract_v1.json").read_text(
        encoding="utf-8"
    )
)

SYNTHETIC_MANIFEST = {
    "as_of_time": "2026-06-21T00:00:00",
    "feature_window": {"start": "2026-06-01T00:00:00", "end": "2026-06-21T00:00:00"},
    "label_window": {"start": "2026-06-21T00:00:00", "end": "2026-07-31T00:00:00"},
    "timezone": "Asia/Shanghai",
    "label_version": "label_v2_record_count_20260923",
    "split_version": "split_v2_record_strat5_seed42",
    "score_version": "T2R-004-synthetic-e2e",
    "all_fit_operations_train_fold_only": True,
    "uses_task1_probability": False,
    "task1_predictions_are_oof": False,
    "target_count": 50,
}

BASE_TIME = datetime(2026, 6, 1, 8, 0, 0)
SCORING_CODES = ("41004", "30017", "30005", "11401")  # G2/G3/G4/G5, one class each


class ContractLoaderTests(unittest.TestCase):
    def test_twenty_four_codes_with_expected_roles(self):
        self.assertEqual(len(LOOKUPS["by_code"]), 24)
        roles = {}
        for row in LOOKUPS["by_code"].values():
            roles[row["scoring_role"]] = roles.get(row["scoring_role"], 0) + 1
        self.assertEqual(
            roles,
            {"behavior": 17, "context_warning": 3, "evidence_quality": 2, "outcome_gate": 2},
        )

    def test_chain_windows_and_speed_neighbours(self):
        windows = {
            entry["chain_id"]: entry["window_seconds"]
            for entry in LOOKUPS["chain_by_code"].values()
        }
        self.assertEqual(windows, {"RC1": 120.0, "RC2": 120.0, "RC3": 120.0, "RC4": 120.0, "RC5": 600.0})
        self.assertEqual(
            LOOKUPS["speed_neighbor_codes"],
            ["11401", "11402", "11403", "11405", "11406"],
        )


class SegmentationTests(unittest.TestCase):
    def test_fatigue_chain_merges_within_window_and_keeps_strongest(self):
        events = [
            {"gpsno": "syn-1", "code": "41002", "time": BASE_TIME, "severity": 1.0},
            {"gpsno": "syn-1", "code": "41001", "time": BASE_TIME + timedelta(seconds=60), "severity": 3.0},
            {"gpsno": "syn-1", "code": "41029", "time": BASE_TIME + timedelta(seconds=119), "severity": 2.0},
        ]
        segments = segment_events(events, LOOKUPS)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["representative_code"], "41001")
        self.assertEqual(segments[0]["member_count"], 3)
        self.assertEqual(segments[0]["merged_codes"], ["41001", "41002", "41029"])

    def test_gap_beyond_window_splits_segments(self):
        events = [
            {"gpsno": "syn-1", "code": "41002", "time": BASE_TIME},
            {"gpsno": "syn-1", "code": "41002", "time": BASE_TIME + timedelta(seconds=121)},
        ]
        self.assertEqual(len(segment_events(events, LOOKUPS)), 2)

    def test_device_chain_uses_longer_window_and_outcome_codes_are_skipped(self):
        events = [
            {"gpsno": "syn-1", "code": "41006", "time": BASE_TIME},
            {"gpsno": "syn-1", "code": "41021", "time": BASE_TIME + timedelta(seconds=599)},
            {"gpsno": "syn-1", "code": "11803", "time": BASE_TIME + timedelta(seconds=1)},
            {"gpsno": "syn-1", "code": "11804", "time": BASE_TIME + timedelta(seconds=2)},
        ]
        segments = segment_events(events, LOOKUPS)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["chain_id"], "RC5")

    def test_speed_codes_use_sixty_second_neighbour_window(self):
        events = [
            {"gpsno": "syn-1", "code": "11401", "time": BASE_TIME},
            {"gpsno": "syn-1", "code": "11406", "time": BASE_TIME + timedelta(seconds=60)},
            {"gpsno": "syn-1", "code": "11402", "time": BASE_TIME + timedelta(seconds=121)},
        ]
        segments = segment_events(events, LOOKUPS)
        self.assertEqual(len(segments), 2)


class SmoothingAndBinsTests(unittest.TestCase):
    def test_eb_smoothing_pulls_toward_prior_and_is_monotone(self):
        prior = fold_prior_rate([(4, 100), (6, 100)])  # 0.05
        low = eb_smooth_rate(0, 100, prior, 10)
        high = eb_smooth_rate(8, 100, prior, 10)
        self.assertGreater(low, 0.0)
        self.assertLess(low, prior + 0.01)
        self.assertGreater(high, low)
        self.assertLess(high, 0.08)

    def test_pava_enforces_monotone_risks_and_compresses(self):
        samples = (
            [(0.0, 0)] * 3
            + [(1.0, 1)] * 2
            + [(2.0, 0)] * 2   # non-monotone dip
            + [(3.0, 1)] * 3
        )
        spec = fit_bins(samples, n_bins=4)
        self.assertEqual(len(spec.risks), len(spec.boundaries) + 1)
        self.assertEqual(list(spec.risks), sorted(spec.risks))
        self.assertLess(spec.risks[spec.bin_index(0.0)], spec.risks[spec.bin_index(3.0)])

    def test_degenerate_samples_collapse_to_single_bin(self):
        spec = fit_bins([(1.0, 0), (1.0, 1)], n_bins=4)
        self.assertEqual(len(spec.risks), 1)

    def test_group_aggregation_formula_and_monotonicity(self):
        points = {"a": 6.0, "b": 4.0, "c": 2.0}
        self.assertAlmostEqual(aggregate_group_deduction(points), 6.0 + 0.25 * 3.0)
        self.assertAlmostEqual(aggregate_group_deduction({"a": 5.0}), 5.0)
        self.assertLessEqual(
            aggregate_group_deduction(points),
            aggregate_group_deduction({**points, "b": 5.0}),
        )


def _synthetic_population(seed: int = 20260925):
    """50 vehicles, 5 folds, risky half carries many segments and label 1."""

    rng = random.Random(seed)
    vehicles = []
    for index in range(50):
        risky = index < 25
        gpsno = f"synthetic-vehicle-{index:03d}"
        events = []
        per_class = rng.randint(8, 14) if risky else rng.randint(0, 2)
        for k in range(per_class):
            code = SCORING_CODES[rng.randrange(len(SCORING_CODES))]
            moment = BASE_TIME + timedelta(days=k % 18, minutes=11 * k)
            events.append({"gpsno": gpsno, "code": code, "time": moment})
        exposure = rng.uniform(300, 700) if risky else rng.uniform(600, 900)
        segments = segment_events(events, LOOKUPS)
        vehicles.append(
            {
                "gpsno": gpsno,
                "exposure": round(exposure, 3),
                "events": events,
                "segments": segments,
                "label": 1 if risky else 0,
                "fold": f"f{index % 5}",
            }
        )
    return vehicles


class ScoringInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.population = _synthetic_population()
        cls.ruleset = fit_ruleset(
            [dict(row) for row in cls.population], LOOKUPS, SYNTHETIC_POLICY
        )

    def test_ruleset_covers_expected_groups_and_budget_split(self):
        groups = {self.ruleset.class_group[cls] for cls in self.ruleset.bins_by_class}
        self.assertTrue({"G2", "G3", "G4", "G5"}.issubset(groups))
        self.assertNotIn("G6", groups)  # evidence-quality codes carry no budget
        for cls, budget in self.ruleset.class_budget.items():
            group = self.ruleset.class_group[cls]
            if group != "G6":
                self.assertGreater(budget, 0.0)

    def test_monotonicity_removing_evidence_never_lowers_score(self):
        rng = random.Random(7)
        for row in self.population[:8]:
            full = score_vehicle(row, self.ruleset, LOOKUPS)
            reduced = dict(row)
            keep = sorted({seg["segment_id"] for seg in row["segments"]})[
                : max(0, len(row["segments"]) - 1)
            ]
            reduced["segments"] = [
                seg for seg in row["segments"] if seg["segment_id"] in set(keep)
            ]
            if len(reduced["segments"]) == len(row["segments"]):
                continue
            partial = score_vehicle(reduced, self.ruleset, LOOKUPS)
            self.assertGreaterEqual(partial.safety_score, full.safety_score)

    def test_value_domain_and_recomputation(self):
        for row in self.population:
            scoring = score_vehicle(row, self.ruleset, LOOKUPS)
            self.assertTrue(0.0 <= scoring.safety_score <= 100.0)
            recomputed = recompute_score_from_ledger(scoring, self.ruleset)
            self.assertEqual(recomputed, scoring.safety_score)
            for entry in scoring.ledger:
                self.assertEqual(entry["rule_version"], self.ruleset.rule_version)

    def test_low_exposure_vehicle_is_flagged_not_fabricated(self):
        row = {
            "gpsno": "synthetic-vehicle-low",
            "exposure": 0.0,
            "segments": self.population[0]["segments"],
            "label": 1,
        }
        scoring = score_vehicle(row, self.ruleset, LOOKUPS)
        self.assertTrue(scoring.insufficient_evidence)
        self.assertEqual(scoring.safety_score, 100.0)  # raw static score; publishing guards apply


class DownstreamInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.population = _synthetic_population()
        labels = {row["gpsno"]: row["label"] for row in cls.population}
        folds = {row["gpsno"]: row["fold"] for row in cls.population}
        scorings = []
        for fold in ("f0", "f1", "f2", "f3", "f4"):
            training = [row for row in cls.population if row["fold"] != fold]
            ruleset = fit_ruleset(training, LOOKUPS, SYNTHETIC_POLICY)
            for row in cls.population:
                if row["fold"] == fold:
                    scorings.append(score_vehicle(row, ruleset, LOOKUPS))
        cls.scorings = scorings
        cls.rows = build_scores_rows(scorings, cls._last_ruleset(), labels, folds)

    @staticmethod
    def _last_ruleset():
        training = [row for row in _synthetic_population() if row["fold"] != "f4"]
        return fit_ruleset(training, LOOKUPS, SYNTHETIC_POLICY)

    def test_recomputed_column_matches_published_exactly(self):
        for row in self.rows:
            self.assertEqual(row["safety_score"], row["rule_recomputed_score"])

    def test_evaluation_primary_gate_passes_on_synthetic_population(self):
        report = evaluate_scores(self.rows, EVALUATION_CONTRACT, SYNTHETIC_MANIFEST)
        self.assertEqual(report["status"], "pass", report.get("errors"))
        self.assertTrue(report["primary_gate_pass"])
        self.assertGreater(report["roc_auc"], 0.75)
        self.assertTrue(report["fold_direction_consistent"])

    def test_adapter_payload_runs_through_dynamic_layer(self):
        segments_by_gpsno = {row["gpsno"]: row["segments"] for row in self.population}
        exposure_days = {}
        for row in self.population:
            days = []
            for day in range(1, 21):
                amount = 0.0 if day == 20 else round(row["exposure"] / 19.0, 3)
                days.append({"date": f"2026-06-{day:02d}", "exposure": amount})
            exposure_days[row["gpsno"]] = days
        payload = build_adapter_payload(
            self.scorings,
            self._last_ruleset(),
            segments_by_gpsno,
            exposure_days,
            "2026-06-20",
            upstream_score_version="T2R-004-synthetic-v1",
        )
        self.assertEqual(validate_adapter_payload(payload), [])
        result = run_payload(payload, OPERATIONAL_POLICY)
        self.assertEqual(result["vehicle_count"], 50)
        self.assertTrue(result["vehicles"])

    def test_adapter_contract_violations_are_rejected(self):
        scoring = self.scorings[0]
        ruleset = self._last_ruleset()
        gpsno = scoring.gpsno
        source = next(row for row in self.population if row["gpsno"] == gpsno)

        def segments_on(day: int):
            return [
                dict(seg, start_time=seg["start_time"].replace(day=day))
                for seg in source["segments"]
            ]

        good_days = [
            {"date": "2026-06-05", "exposure": 10.0},
            {"date": "2026-06-06", "exposure": 0.0},
        ]
        with self.assertRaises(ValueError):  # valid deduction on zero-exposure day
            build_adapter_payload(
                [scoring], ruleset, {gpsno: segments_on(6)}, {gpsno: good_days},
                "2026-06-20", upstream_score_version="v",
            )

        with self.assertRaises(ValueError):  # event outside day records
            build_adapter_payload(
                [scoring], ruleset, {gpsno: segments_on(7)}, {gpsno: good_days},
                "2026-06-20", upstream_score_version="v",
            )

        with self.assertRaises(ValueError):  # empty version tag
            build_adapter_payload(
                [scoring], ruleset, {gpsno: segments_on(5)}, {gpsno: good_days},
                "2026-06-20", upstream_score_version="",
            )


if __name__ == "__main__":
    unittest.main()
