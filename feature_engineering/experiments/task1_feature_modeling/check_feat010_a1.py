#!/usr/bin/env python3
"""Independently verify the private, unlabeled FEAT-010 A1 output contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


META_COLUMNS = {"sample_id", "gpsno", "as_of", "lookback_days"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_hash(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def verify(run_dir: Path) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    audit_path = run_dir / "audit.json"
    manifest_path = run_dir / "manifest.json"
    candidate_path = run_dir / "feature_61d_unlabeled.csv"
    contract_path = run_dir / "window_contract.csv"
    f3_dir = run_dir / "f3"
    required = [audit_path, manifest_path, candidate_path, contract_path,
                f3_dir / "f3_new_features.csv", f3_dir / "f3_window_exposure.csv",
                f3_dir / "f3_window_observation.csv"]
    missing = [path.name for path in required if not path.is_file()]
    require(not missing, f"required outputs missing: {', '.join(missing)}")
    if missing:
        return errors

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = pd.read_csv(candidate_path, dtype={"gpsno": str, "sample_id": str})
    contract = pd.read_csv(contract_path)
    f3 = pd.read_csv(f3_dir / "f3_new_features.csv", dtype={"gpsno": str, "sample_id": str})
    exposure = pd.read_csv(f3_dir / "f3_window_exposure.csv", dtype={"gpsno": str, "sample_id": str})
    observation = pd.read_csv(f3_dir / "f3_window_observation.csv", dtype={"gpsno": str, "sample_id": str})

    require(audit.get("machine_status") == "pass", "A1 runner did not report machine pass")
    require(audit.get("labels_or_folds_read") is False, "audit says labels/folds were read")
    require(audit.get("labels_or_folds_in_candidate") is False,
            "audit says labels/folds are present in candidate")
    require(len(candidate) == 500, "candidate row count is not the fixed 500-vehicle roster")
    for column in ("sample_id", "gpsno"):
        require(column in candidate, f"candidate lacks key column {column}")
        if column in candidate:
            require(not candidate[column].duplicated().any(), f"candidate has duplicate {column}")
    forbidden = {"y", "fold", "label", "label_window", "label_version", "target"}
    leaked = sorted(column for column in candidate.columns
                    if column.lower() in forbidden or column.lower().startswith("label_"))
    require(not leaked, "candidate contains target metadata columns")
    require(set(candidate.columns) - META_COLUMNS <= set(contract["field"].astype(str)),
            "candidate feature fields are missing from the window contract")
    require(not (contract["status"].astype(str) == "fail_unclassified").any(),
            "window contract contains unclassified fields")

    window = audit.get("time_contract", {})
    require(window.get("timezone") == "Asia/Shanghai", "time zone does not match contract")
    require(window.get("start_inclusive") == "2026-06-01 00:00:00",
            "window start does not match contract")
    require(window.get("as_of_exclusive") == "2026-08-01 00:00:00",
            "as_of boundary does not match contract")
    require(window.get("lookback_days") == 61, "lookback is not 61 days")
    if "as_of" in candidate:
        require((candidate["as_of"].astype(str) == "2026-08-01 00:00:00").all(),
                "candidate rows have inconsistent as_of values")
    if "lookback_days" in candidate:
        require((pd.to_numeric(candidate["lookback_days"], errors="coerce") == 61).all(),
                "candidate rows have inconsistent lookback values")

    keys = set(candidate["sample_id"].astype(str))
    for label, frame in (("F3", f3), ("exposure", exposure), ("observation", observation)):
        require(len(frame) == len(candidate), f"{label} output row count differs from candidate")
        require(frame["sample_id"].astype(str).is_unique, f"{label} output has duplicate sample IDs")
        require(set(frame["sample_id"].astype(str)) == keys,
                f"{label} output sample IDs differ from candidate")
    require({"traj_km_61d", "traj_hours_61d"} <= set(exposure.columns),
            "raw F3 exposure sidecar lacks 61-day distance/time columns")
    require("imu_rows_window" in observation, "IMU observation sidecar lacks row count")
    if "imu_rows_61d" in candidate and "imu_rows_window" in observation:
        candidate_rows = candidate.set_index("sample_id")["imu_rows_61d"].sort_index()
        observed_rows = observation.set_index("sample_id")["imu_rows_window"].sort_index()
        require(candidate_rows.equals(observed_rows), "candidate IMU row count differs from sidecar")
    for name in ("a1_window_separation.png", "a1_feature_distribution.png", "acceptance.md",
                 "resume_record.json", "attempt-1-failure.json"):
        require((run_dir / name).is_file(), f"run evidence missing: {name}")
    require(manifest.get("candidate_excludes_labels_and_folds") is True,
            "manifest does not affirm label/fold exclusion")
    require(manifest.get("raw_shard_inventory_fingerprint") ==
            payload_hash(audit.get("raw_shard_inventory", {})),
            "raw source inventory fingerprint mismatch")

    expected_outputs = manifest.get("output_sha256", {})
    for relative, expected_meta in expected_outputs.items():
        path = run_dir / relative
        require(path.is_file(), f"manifest output missing: {relative}")
        if path.is_file():
            require(path.stat().st_size == expected_meta.get("bytes"),
                    f"manifest byte size mismatch: {relative}")
            require(sha256(path) == expected_meta.get("sha256"),
                    f"manifest SHA-256 mismatch: {relative}")
    actual_files = {str(path.relative_to(run_dir)) for path in run_dir.rglob("*")
                    if path.is_file() and path.name != "manifest.json"}
    require(actual_files == set(expected_outputs),
            "manifest output inventory differs from files in run directory")

    synthetic = audit.get("synthetic_checks", {})
    require(bool(synthetic) and all(value is True for value in synthetic.values()),
            "one or more A1 synthetic checks failed")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    errors = verify(args.run_dir.expanduser().resolve())
    if errors:
        print(json.dumps({"machine_status": "fail", "errors": errors}, ensure_ascii=False))
        return 1
    print(json.dumps({"machine_status": "pass", "checks": "contract, keys, windows, sidecars, synthetic cases, hashes"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
