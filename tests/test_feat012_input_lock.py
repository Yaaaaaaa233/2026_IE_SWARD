from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from feature_engineering.experiments.feat012.run_exploration import load_locked_inputs


class LockedInputTests(unittest.TestCase):
    def build_fixture(self, root: Path, *, future: bool = False):
        families = [
            ("evt_signal", "evt"), ("f1_base", "f1"), ("f2_base", "f2"),
            ("f2r2_base", "f2r2"), ("accident_x", "accident"),
            ("f3_hist_x", "f3_hist"), ("f3_chain_x", "f3_chain"),
            ("f3_event_x", "f3_event"), ("f3_fatigue_x", "f3_fatigue"),
            ("f3_score_x", "f3_score"), ("f3_traj_x", "f3_traj"),
            ("traj_x", "traj"), ("f3_night_detail", "f3_night"),
        ]
        if future:
            families.append(("future_feature", "future"))
        rows = []
        for i in range(10):
            row = {"sample_id": f"s{i}", "gpsno": f"g{i}", "y": i % 2, "fold": i // 2,
                   "label_version": "label_v2_record_count_20260923", "split_version": "split_v2_record_strat5_seed42"}
            row.update({name: float(i + j) for j, (name, _) in enumerate(families)})
            rows.append(row)
        frame = pd.DataFrame(rows)
        inp, manifest_path, ledger_path = root / "f3.csv", root / "manifest.json", root / "ledger.csv"
        frame.to_csv(inp, index=False, lineterminator="\n")
        ledger = [{"column": c, "role": "predictor", "family": family, "visibility_decision": "retain_candidate",
                   "in_full_f3": True, "in_slim_f3": family != "f3_night"} for c, family in families]
        pd.DataFrame(ledger).to_csv(ledger_path, index=False, lineterminator="\n")
        manifest = {"protocol": "label_v2_record_count_20260923", "split_version": "split_v2_record_strat5_seed42",
                    "outputs": {"f3_model_input.csv": {"sha256": hashlib.sha256(inp.read_bytes()).hexdigest()}}}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return inp, manifest_path, ledger_path

    def test_accepts_locked_source_and_constructs_registered_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.build_fixture(Path(tmp))
            frame, manifest, ledger, views, groups, y, folds, features = load_locked_inputs(*paths)
            self.assertEqual(len(frame), 10)
            self.assertEqual(len(features), 13)
            self.assertEqual(len(views["history"]), 10)
            self.assertEqual(len(views["slim_night"]), 12)
            self.assertEqual(set(folds), {0, 1, 2, 3, 4})

    def test_rejects_input_changed_after_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.build_fixture(Path(tmp))
            frame = pd.read_csv(paths[0]); frame.loc[0, "evt_signal"] += 1
            frame.to_csv(paths[0], index=False, lineterminator="\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_locked_inputs(*paths)

    def test_rejects_future_derived_feature_even_if_ledger_claims_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.build_fixture(Path(tmp), future=True)
            with self.assertRaisesRegex(ValueError, "future-derived"):
                load_locked_inputs(*paths)


if __name__ == "__main__":
    unittest.main()
