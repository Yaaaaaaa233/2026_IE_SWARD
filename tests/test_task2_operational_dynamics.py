import copy
import json
import unittest
from pathlib import Path

from task2.operations.dynamic_score import classify_time_band, run_payload, run_vehicle


ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads(
    (ROOT / "task2" / "operations" / "operational_policy_v1.json").read_text(
        encoding="utf-8"
    )
)


def event(event_id, event_time, deduction, status="valid"):
    return {
        "event_id": event_id,
        "event_time": event_time,
        "base_deduction": deduction,
        "review_status": status,
    }


class Task2OperationalDynamicsTests(unittest.TestCase):
    def test_official_time_band_boundaries(self):
        self.assertEqual("night", classify_time_band("2026-09-01T05:59:59"))
        self.assertEqual("morning", classify_time_band("2026-09-01T06:00:00"))
        self.assertEqual("day", classify_time_band("2026-09-01T09:00:00"))
        self.assertEqual("dusk", classify_time_band("2026-09-01T18:00:00"))
        self.assertEqual("night", classify_time_band("2026-09-01T21:00:00"))

    def test_stopped_day_freezes_previous_score(self):
        vehicle = {
            "gpsno": "synthetic-1",
            "initial_score": 70,
            "initial_score_is_previous_valid": True,
            "days": [
                {"date": "2026-09-01", "exposure": 10},
                {"date": "2026-09-02", "exposure": 0},
            ],
            "events": [],
        }
        result = run_vehicle(vehicle, "2026-09-02", POLICY)
        first, second = result["history"]
        self.assertEqual(first["safety_score"], second["safety_score"])
        self.assertEqual(0.0, second["score_change"])
        self.assertEqual("stopped_frozen", second["observation_status"])

    def test_never_observed_vehicle_uses_conservative_prior(self):
        vehicle = {
            "gpsno": "synthetic-2",
            "initial_score": 95,
            "initial_score_is_previous_valid": False,
            "days": [{"date": "2026-09-01", "exposure": 0}],
            "events": [],
        }
        result = run_vehicle(vehicle, "2026-09-01", POLICY)
        self.assertEqual(50.0, result["safety_score"])
        self.assertEqual("no_exposure_conservative", result["observation_status"])

    def test_safe_exposure_allows_limited_recovery_after_risk(self):
        vehicle = {
            "gpsno": "synthetic-3",
            "initial_score": 80,
            "initial_score_is_previous_valid": True,
            "days": [
                {"date": "2026-09-01", "exposure": 10},
                {"date": "2026-09-02", "exposure": 10},
            ],
            "events": [event("risk-1", "2026-09-01T12:00:00", 50)],
        }
        result = run_vehicle(vehicle, "2026-09-02", POLICY)
        first, second = result["history"]
        self.assertLess(first["safety_score"], 80)
        self.assertGreater(second["safety_score"], first["safety_score"])
        self.assertLessEqual(second["score_change"], 5.0)

    def test_pending_exempt_and_expired_events_are_distinct(self):
        vehicle = {
            "gpsno": "synthetic-4",
            "initial_score": 80,
            "initial_score_is_previous_valid": True,
            "days": [
                {"date": "2026-09-01", "exposure": 10},
                {"date": "2026-09-09", "exposure": 10},
            ],
            "events": [
                event("expired", "2026-09-01T22:00:00", 20, "pending"),
                event("pending", "2026-09-09T22:00:00", 20, "pending"),
                event("exempt", "2026-09-09T12:00:00", 20, "exempt"),
            ],
        }
        result = run_vehicle(vehicle, "2026-09-10", POLICY)
        resolutions = {item["event_id"]: item for item in result["event_resolution"]}
        self.assertEqual("auto_valid_after_deadline", resolutions["expired"]["resolved_status"])
        self.assertEqual("pending", resolutions["pending"]["resolved_status"])
        self.assertEqual("exempt", resolutions["exempt"]["resolved_status"])
        self.assertEqual(["expired"], result["history"][0]["included_event_ids"])
        self.assertEqual(["pending"], result["history"][1]["pending_event_ids"])
        self.assertEqual(["exempt"], result["history"][1]["exempt_event_ids"])

    def test_configurable_time_and_persistence_multipliers(self):
        policy = copy.deepcopy(POLICY)
        policy["dynamics"]["time_coefficients"]["night"] = 1.5
        policy["dynamics"]["persistence_step"] = 0.1
        policy["dynamics"]["persistence_multiplier_cap"] = 1.5
        vehicle = {
            "gpsno": "synthetic-5",
            "initial_score": 90,
            "initial_score_is_previous_valid": True,
            "days": [
                {"date": "2026-09-01", "exposure": 10},
                {"date": "2026-09-02", "exposure": 10},
            ],
            "events": [
                event("night-1", "2026-09-01T22:00:00", 10),
                event("night-2", "2026-09-02T22:00:00", 10),
            ],
        }
        result = run_vehicle(vehicle, "2026-09-02", policy)
        self.assertEqual(15.0, result["history"][0]["base_risk_points"])
        self.assertAlmostEqual(1.1, result["history"][1]["persistence_multiplier"])
        self.assertAlmostEqual(16.5, result["history"][1]["adjusted_daily_risk"])

    def test_zero_exposure_day_cannot_carry_included_risk(self):
        vehicle = {
            "gpsno": "synthetic-6",
            "initial_score": 70,
            "initial_score_is_previous_valid": True,
            "days": [{"date": "2026-09-01", "exposure": 0}],
            "events": [event("risk-1", "2026-09-01T12:00:00", 10)],
        }
        with self.assertRaisesRegex(ValueError, "zero-exposure"):
            run_vehicle(vehicle, "2026-09-01", POLICY)

    def test_payload_rejects_duplicate_vehicle_ids(self):
        vehicle = {
            "gpsno": "synthetic-7",
            "initial_score": 50,
            "initial_score_is_previous_valid": False,
            "days": [],
            "events": [],
        }
        with self.assertRaisesRegex(ValueError, "duplicate gpsno"):
            run_payload(
                {"as_of_date": "2026-09-01", "vehicles": [vehicle, vehicle]}, POLICY
            )


if __name__ == "__main__":
    unittest.main()
