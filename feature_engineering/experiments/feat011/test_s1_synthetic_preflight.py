from __future__ import annotations

import copy
import unittest

import numpy as np

from s1_synthetic_preflight import (
    EXPECTED_WEIGHTS,
    audit_equal_logit_fusion,
    equal_logit_fusion,
    materialize_preregistration,
    run_negative_checks,
    synthetic_data,
    validate_fold_map,
    validate_preregistration,
)


class S1SyntheticPreflightTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.folds, self.groups = synthetic_data()
        self.prereg = materialize_preregistration(self.x, self.y, self.folds, self.groups)

    def test_registered_factor_matrix_is_valid(self):
        validate_preregistration(self.prereg)
        validate_fold_map(self.x, self.y, self.folds, self.groups, self.prereg)

    def test_configuration_mutation_breaks_fingerprint(self):
        changed = copy.deepcopy(self.prereg)
        changed["arms"]["A"]["model_spec"]["max_bins"] = 63
        with self.assertRaisesRegex(ValueError, "fingerprint drift"):
            validate_preregistration(changed)

    def test_fusion_is_fixed_and_reproducible(self):
        p_a = np.array([0.0, 0.25, 0.9, 1.0])
        p_d = np.array([0.1, 0.75, 0.6, 1.0])
        fused = equal_logit_fusion(p_a, p_d)
        np.testing.assert_array_equal(fused, equal_logit_fusion(p_a, p_d, EXPECTED_WEIGHTS))
        with self.assertRaisesRegex(ValueError, "weights"):
            equal_logit_fusion(p_a, p_d, {"A": 0.4, "D": 0.6})
        audit_equal_logit_fusion(fused, p_a, p_d, self.prereg)
        with self.assertRaisesRegex(ValueError, "does not equal"):
            audit_equal_logit_fusion(fused + 0.001, p_a, p_d, self.prereg)

    def test_registered_bad_fixtures_are_rejected(self):
        result = run_negative_checks()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["rejected_count"], 3)


if __name__ == "__main__":
    unittest.main()
