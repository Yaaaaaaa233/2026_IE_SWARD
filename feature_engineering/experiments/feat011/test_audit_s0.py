from __future__ import annotations

import unittest

import pandas as pd

from audit_s0 import NIGHT_SUMMARY, audit_frame, make_views, run_self_tests, sha256_frame, validate_column_views


class S0AuditTests(unittest.TestCase):
    def test_synthetic_bad_case_suite_rejects_all_registered_failures(self):
        result = run_self_tests()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["cases_passed"], result["cases_expected"])

    def test_night_summary_columns_stay_in_both_views(self):
        columns = ["sample_id", "gpsno", "y", "fold", *sorted(NIGHT_SUMMARY),
                   "f3_night_collision_warn_deep_rate"]
        full, slim, removed = make_views(columns)
        validate_column_views(full, slim, removed)
        self.assertTrue(NIGHT_SUMMARY.issubset(slim))
        self.assertEqual(removed, ["f3_night_collision_warn_deep_rate"])


if __name__ == "__main__":
    unittest.main()
