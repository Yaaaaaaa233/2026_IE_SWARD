from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from feature_engineering.experiments.feat014.run import G1_COLUMNS, extract_g1_features, read_oof
from feature_engineering.experiments.feat014.diagnose import top100_change
from feature_engineering.experiments.feat014.fusion import logit_mean


class Feat014ContractTests(unittest.TestCase):
    def samples(self):
        return pd.DataFrame({"sample_id": ["s1", "s2"], "gpsno": ["1", "2"]})

    def test_vehicle_day_cutoff_is_exclusive_and_rows_are_conserved(self):
        daily = pd.DataFrame({
            "gpsno": ["1", "1", "1", "2"],
            "date": ["2026-06-01", "2026-06-20", "2026-06-21", "2026-06-19"],
            "evt_segments30": [2, 4, 90, 1], "evt_days_flag": [1, 1, 1, 1],
            "traj_km": [10.0, 20.0, 1.0, 0.0], "traj_hours": [1.0, 2.0, 1.0, 0.0],
        })
        got, audit = extract_g1_features(daily, self.samples(), cutoff="2026-06-21")
        self.assertEqual(len(got), 2)
        self.assertEqual(got.sample_id.tolist(), ["s1", "s2"])
        self.assertAlmostEqual(got.loc[0, G1_COLUMNS[0]], 200.0)
        self.assertAlmostEqual(got.loc[0, G1_COLUMNS[1]], 2.0)
        self.assertEqual(audit["excluded_at_or_after_cutoff_rows"], 1)
        self.assertTrue(pd.isna(got.loc[1, G1_COLUMNS[0]]))
        self.assertTrue(pd.isna(got.loc[1, G1_COLUMNS[4]]))

    def test_oof_gpsno_read_as_numeric_input_still_binds_by_string_key(self):
        frame = pd.DataFrame({"sample_id": ["s1", "s2"], "gpsno": ["01", "02"],
                              "y": [0, 1], "fold": [0, 1]})
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"oof.csv"
            pd.DataFrame({"sample_id":["s1","s2"],"gpsno":["01","02"],"y":[0,1],"fold":[0,1],"probability":[.1,.9]}).to_csv(path,index=False)
            # Explicit dtype preserves IDs with leading zeros, as the runner does.
            self.assertEqual(read_oof(path,frame).tolist(),[.1,.9])

    def test_duplicate_daily_key_and_unknown_vehicle_fail_closed(self):
        daily = pd.DataFrame({
            "gpsno": ["1", "1"], "date": ["2026-06-02", "2026-06-02"],
            "evt_segments30": [1, 1], "evt_days_flag": [1, 1],
            "traj_km": [1.0, 1.0], "traj_hours": [1.0, 1.0],
        })
        with self.assertRaisesRegex(ValueError, "duplicate vehicle/date"):
            extract_g1_features(daily, self.samples(), cutoff="2026-06-21")
        daily = daily.iloc[:1].copy()
        daily.loc[:, "gpsno"] = "outside"
        with self.assertRaisesRegex(ValueError, "outside the locked cohort"):
            extract_g1_features(daily, self.samples(), cutoff="2026-06-21")

    def test_oof_alignment_rejects_reordered_rows(self):
        frame = pd.DataFrame({"sample_id": ["s1", "s2"], "gpsno": ["1", "2"],
                              "y": [0, 1], "fold": [0, 1]})
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oof.csv"
            pd.DataFrame({"sample_id": ["s2", "s1"], "gpsno": ["2", "1"],
                          "y": [1, 0], "fold": [1, 0], "probability": [.8, .2]}).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,"sample order/identity"):
                read_oof(path,frame)

    def test_top100_change_separates_fixed_positive_gains_and_false_positive_changes(self):
        y = [1, 0, 1, 0, 1]
        reference = {0, 1, 3}
        candidate = {0, 2, 4}
        got = top100_change(y, candidate, reference)
        self.assertEqual(got["reference_false_positives_removed"], 2)
        self.assertEqual(got["candidate_false_positives_added"], 0)
        self.assertEqual(got["new_true_positives_in_top100"], 2)
        self.assertEqual(got["true_positives_dropped_from_top100"], 0)
        self.assertEqual(got["top100_overlap"], 1)

    def test_fixed_logit_mean_is_deterministic_and_rejects_invalid_weights(self):
        got = logit_mean([np.array([0.25]), np.array([0.75])], [0.5, 0.5])
        self.assertAlmostEqual(float(got[0]), 0.5)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            logit_mean([np.array([0.25]), np.array([0.75])], [0.2, 0.2])


if __name__ == "__main__": unittest.main()
