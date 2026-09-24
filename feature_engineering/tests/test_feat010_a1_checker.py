import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "feature_engineering/experiments/task1_feature_modeling"))

from check_feat010_a1 import payload_hash, sha256, verify


def _write_run(root: Path) -> None:
    f3_dir = root / "f3"
    f3_dir.mkdir(parents=True)
    keys = pd.DataFrame({
        "sample_id": [f"S{i:03d}_20260801_61d" for i in range(500)],
        "gpsno": [f"G{i:03d}" for i in range(500)],
    })
    candidate = keys.assign(as_of="2026-08-01 00:00:00", lookback_days=61,
                            traj_km_61d=0.0, traj_hours_61d=0.0, imu_rows_61d=0)
    candidate.to_csv(root / "feature_61d_unlabeled.csv", index=False, lineterminator="\n")
    pd.DataFrame({"field": ["traj_km_61d", "traj_hours_61d", "imu_rows_61d"],
                  "status": ["included_candidate"] * 3}).to_csv(
        root / "window_contract.csv", index=False, lineterminator="\n")
    keys.assign(f3_feature=0.0).to_csv(f3_dir / "f3_new_features.csv", index=False, lineterminator="\n")
    keys.assign(traj_km_61d=0.0, traj_hours_61d=0.0).to_csv(
        f3_dir / "f3_window_exposure.csv", index=False, lineterminator="\n")
    keys.assign(imu_rows_window=0).to_csv(
        f3_dir / "f3_window_observation.csv", index=False, lineterminator="\n")
    inventory = {"trajectory": [], "imu": []}
    (root / "audit.json").write_text(json.dumps({
        "machine_status": "pass", "labels_or_folds_read": False,
        "labels_or_folds_in_candidate": False,
        "time_contract": {"timezone": "Asia/Shanghai",
                          "start_inclusive": "2026-06-01 00:00:00",
                          "as_of_exclusive": "2026-08-01 00:00:00", "lookback_days": 61},
        "raw_shard_inventory": inventory,
        "synthetic_checks": {"window": True},
    }), encoding="utf-8")
    for name in ("a1_window_separation.png", "a1_feature_distribution.png"):
        (root / name).write_bytes(b"synthetic image placeholder")
    for name in ("acceptance.md", "resume_record.json", "attempt-1-failure.json"):
        (root / name).write_text("synthetic evidence\n", encoding="utf-8")
    outputs = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            outputs[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (root / "manifest.json").write_text(json.dumps({
        "candidate_excludes_labels_and_folds": True,
        "raw_shard_inventory_fingerprint": payload_hash(inventory),
        "output_sha256": outputs,
    }), encoding="utf-8")


def test_feat010_a1_checker_accepts_synthetic_contract(tmp_path):
    _write_run(tmp_path)
    assert verify(tmp_path) == []


def test_feat010_a1_checker_rejects_fold_leak_even_with_updated_hashes(tmp_path):
    _write_run(tmp_path)
    path = tmp_path / "feature_61d_unlabeled.csv"
    candidate = pd.read_csv(path)
    candidate["fold"] = 0
    candidate.to_csv(path, index=False, lineterminator="\n")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"]["feature_61d_unlabeled.csv"] = {
        "bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    errors = verify(tmp_path)
    assert any("target metadata" in error for error in errors)


def test_feat010_a1_checker_rejects_tampered_output_hash(tmp_path):
    _write_run(tmp_path)
    with (tmp_path / "f3/f3_window_observation.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    errors = verify(tmp_path)
    assert any("SHA-256 mismatch" in error for error in errors)
