"""Leakage-focused FEAT-009 S1 synthetic tests; no competition data is read."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from feature_engineering.experiments.task1_feature_modeling.modeling import fit_transformer, select_and_fit_fold


def fixture(n=100):
    rng = np.random.default_rng(51)
    y = np.tile([0, 1], n // 2)
    latent = y + rng.normal(0, 1, n)
    x = pd.DataFrame({
        "signal": latent,
        "signal_copy": latent * 2,
        "noise": rng.normal(size=n),
        "category": np.where(np.arange(n) % 3, "diesel", "electric"),
        "cohort_source": rng.normal(size=n),
    })
    groups = pd.DataFrame({
        "gpsno": [f"v{i}" for i in range(n)],
        "energy_type": np.where(np.arange(n) % 2, "diesel", "electric"),
        "highway_ratio": np.linspace(0.01, 0.99, n),
    })
    folds = np.arange(n) % 5
    return x, groups, y, folds


def estimator_factory(spec):
    return LogisticRegression(C=spec["C"], solver="liblinear", random_state=42)


class FoldLocalTransformTests(unittest.TestCase):
    def test_transformer_runs_without_unverified_portrait_group_keys(self):
        rng = np.random.default_rng(11)
        x = pd.DataFrame({"x": rng.normal(size=24)})
        groups = pd.DataFrame({"sample_id": [f"s{i}" for i in range(24)],
                               "gpsno": [f"g{i}" for i in range(24)]})
        fitted = fit_transformer(x.iloc[:18], groups.iloc[:18], cohort_sources=[])
        result = fitted.transform(x.iloc[18:], groups.iloc[18:])
        self.assertEqual(result.shape, (6, 1))

    def test_heldout_covariates_cannot_change_fitted_transform(self):
        x, groups, _, folds = fixture()
        train = np.flatnonzero(folds != 0)
        heldout = np.flatnonzero(folds == 0)
        a = fit_transformer(x.iloc[train], groups.iloc[train], ["cohort_source"])
        changed_x = x.copy()
        changed_x.loc[heldout, "signal"] = 1e8
        changed_x.loc[heldout, "cohort_source"] = -1e8
        changed_groups = groups.copy()
        changed_groups.loc[heldout, "highway_ratio"] = np.linspace(100, 200, len(heldout))
        b = fit_transformer(changed_x.iloc[train], changed_groups.iloc[train], ["cohort_source"])
        self.assertEqual(a.selected, b.selected)
        self.assertEqual(a.training_fingerprint, b.training_fingerprint)
        self.assertEqual(a.cohort_median, b.cohort_median)
        np.testing.assert_allclose(a.transform(x.iloc[train], groups.iloc[train]),
                                   b.transform(x.iloc[train], groups.iloc[train]))

    def test_outer_validation_labels_cannot_change_selection_or_predictions(self):
        x, groups, y, folds = fixture()
        candidates = ({"name": "simple", "C": 0.2}, {"name": "flexible", "C": 3.0})
        first = select_and_fit_fold("synthetic", x, y, folds, 0, groups,
                                    ["cohort_source"], candidates, estimator_factory)
        changed_y = y.copy()
        changed_y[folds == 0] = 1 - changed_y[folds == 0]
        second = select_and_fit_fold("synthetic", x, changed_y, folds, 0, groups,
                                     ["cohort_source"], candidates, estimator_factory)
        self.assertTrue(first["test_indices_disjoint"])
        self.assertEqual(first["selected_candidate"], second["selected_candidate"])
        self.assertEqual(first["inner_auc"], second["inner_auc"])
        self.assertEqual(first["selected_features"], second["selected_features"])
        self.assertEqual(first["train_fingerprint"], second["train_fingerprint"])
        np.testing.assert_allclose(first["prediction"], second["prediction"])

    def test_fold_selector_drops_training_constant_and_later_correlated_column(self):
        x, groups, _, folds = fixture()
        train = np.flatnonzero(folds != 0)
        x["constant"] = 1.0
        transform = fit_transformer(x.iloc[train], groups.iloc[train], ["cohort_source"])
        self.assertNotIn("constant", transform.selected)
        self.assertIn("signal", transform.selected)
        self.assertNotIn("signal_copy", transform.selected)


if __name__ == "__main__":
    unittest.main()
