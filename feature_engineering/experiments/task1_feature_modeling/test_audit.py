"""Synthetic S0 audit checks; no competition data is read."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


AUDIT = Path(__file__).with_name("audit.py")


def write_table(path: Path, rows: list[dict]) -> None:
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class AuditCliTests(unittest.TestCase):
    def _fixture(self, root: Path, mismatch: bool) -> Path:
        base = [
            {"sample_id": f"s{i}", "gpsno": f"v{i}", "y": i % 2, "fold": i % 5,
             "as_of": "2026-06-21T00:00:00", "lookback_days": 20, "safe_feature": i * 0.1}
            for i in range(10)
        ]
        f3 = [dict(row, f3_traj_speed_p50=i * 0.2) for i, row in enumerate(base)]
        if mismatch:
            f3[2]["y"] = 1 - f3[2]["y"]
        v5 = [{"sample_id": r["sample_id"], "gpsno": r["gpsno"], "y": r["y"],
               "fold": r["fold"], "p_v5": 0.2 + 0.1 * r["y"]} for r in base]
        rf = [{"sample_id": r["sample_id"], "gpsno": r["gpsno"], "y": r["y"],
               "fold": r["fold"], "p_random_forest": 0.25 + 0.1 * r["y"]} for r in base]
        for name, rows in (("f1.csv", base), ("f3.csv", f3), ("v5.csv", v5), ("rf.csv", rf)):
            write_table(root / name, rows)
        manifest = {"rows": 10, "feature_count": 2, "base_kept": ["safe_feature"],
                    "new_kept": ["f3_traj_speed_p50"], "input_fingerprint": "PENDING",
                    "window": {"as_of": "2026-06-21T00:00:00", "lookback_days": 20, "horizon_days": 40},
                    "inputs": {}}
        (root / "f3_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        output = root / "output"
        config = root / "config.toml"
        entries = {
            "f1_model_input": root / "f1.csv", "f3_model_input": root / "f3.csv",
            "v5_oof": root / "v5.csv", "rf_oof": root / "rf.csv",
            "f3_manifest": root / "f3_manifest.json", "output_dir": output,
        }
        config.write_text("\n".join(f'{k} = "{v}"' for k, v in entries.items()) + "\n", encoding="utf-8")
        return config

    def test_aligned_synthetic_inputs_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._fixture(root, mismatch=False)
            result = subprocess.run([sys.executable, str(AUDIT), "--config", str(config), "--skip-figures"],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads((root / "output" / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(report["machine_status"], "pass")

    def test_label_mismatch_is_rejected_with_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._fixture(root, mismatch=True)
            result = subprocess.run([sys.executable, str(AUDIT), "--config", str(config), "--skip-figures"],
                                    capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("y differs from F1", result.stderr)


if __name__ == "__main__":
    unittest.main()
