import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from models.task1_final.build_g8_score_features import join_imu_features, restore_training_transitions


class ScoreFeatureTests(unittest.TestCase):
    def test_displaced_training_transition_uses_real_history_count(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.csv"
            pd.DataFrame([
                ["evt_1", "car-a", "1", "2026-07-31 01:00:00"],
                ["evt_2", "car-a", "2", "2026-07-31 01:00:03"],
                ["evt_3", "car-a", "1", "2026-07-31 01:00:05"],
                ["evt_4", "car-a", "2", "2026-07-31 01:00:07"],
                ["evt_5", "car-b", "2", "2026-07-31 01:00:00"],
            ], columns=["row_id", "gpsno", "event_type", "start_time"]).to_csv(events, index=False)
            train = root / "train.csv"
            train.write_text("sample_id,gpsno,tr_1_2\n")
            score = pd.DataFrame({"sample_id": ["a", "b"], "gpsno": ["car-a", "car-b"]})
            out, restored = restore_training_transitions(score, events, train)
            self.assertEqual(restored, ["tr_1_2"])
            self.assertEqual(out.tr_1_2.tolist(), [2.0, 0.0])

    def test_nontransition_missing_column_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.csv"
            train.write_text("sample_id,gpsno,missing_rate\n")
            score = pd.DataFrame({"sample_id": ["a"], "gpsno": ["car-a"]})
            with self.assertRaisesRegex(ValueError, "non-transition"):
                restore_training_transitions(score, root / "absent.csv", train)

    def test_imu_join_keeps_frozen_training_count_column(self):
        labels = pd.DataFrame({"sample_id": ["car-a_20260621_20d", "car-b_20260621_20d"]})
        imu = pd.DataFrame({"gpsno": ["car-a"], "imu_rows": [20], "impact_p50": [0.05]})
        result = join_imu_features(labels, imu)
        self.assertEqual(result.imu_rows.iloc[0], 20)
        self.assertTrue(pd.isna(result.imu_rows.iloc[1]))
        self.assertEqual(result.imu_missing.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
