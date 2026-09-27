"""Validate a real August-1 Task 1 inference and package the official CSV.

This is an output adapter, not a model trainer. It deliberately refuses proxy
OOF files and requires a provenance manifest from the upstream model runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

EXPECTED_AS_OF = "2026-08-01"
EXPECTED_START = "2026-06-01"
EXPECTED_TRAIN_AS_OF = "2026-06-21"
EXPECTED_TRAIN_LABEL_END = "2026-07-31"
EXPECTED_ROWS = 500
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path.name}: UTF-8 BOM is forbidden")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f"{path.name}: expected columns {columns}, got {reader.fieldnames}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name}: malformed row")
    return rows


def validate_manifest(manifest: dict, predictions: Path) -> None:
    required = {
        "prediction_role": "future_inference",
        "as_of": EXPECTED_AS_OF,
        "feature_start": EXPECTED_START,
        "feature_end_exclusive": EXPECTED_AS_OF,
        "training_feature_end_exclusive": EXPECTED_TRAIN_AS_OF,
        "training_label_start": EXPECTED_TRAIN_AS_OF,
        "training_label_end_exclusive": EXPECTED_TRAIN_LABEL_END,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} must be {expected!r}")
    version = manifest.get("model_version")
    if not isinstance(version, str) or not version.strip() or "oof" in version.lower():
        raise ValueError("manifest model_version must identify a fitted future model")
    for key in ("model_sha256", "inference_features_sha256", "training_features_sha256",
                "training_labels_sha256", "feature_schema_sha256", "predictions_sha256"):
        value = manifest.get(key)
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            raise ValueError(f"manifest {key} must be a SHA-256 hex digest")
    if digest(predictions) != manifest["predictions_sha256"]:
        raise ValueError("prediction file does not match manifest hash")


def validate_vehicles(rows: list[dict[str, str]]) -> set[str]:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"vehicle list must contain {EXPECTED_ROWS} rows")
    ids = [row["gpsno"].strip() for row in rows]
    if any(not value for value in ids) or len(set(ids)) != EXPECTED_ROWS:
        raise ValueError("vehicle list contains empty or duplicate gpsno")
    return set(ids)


def validate_predictions(rows: list[dict[str, str]], vehicles: set[str]) -> list[tuple[str, float]]:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"predictions must contain {EXPECTED_ROWS} rows")
    if {row["gpsno"].strip() for row in rows} != vehicles:
        raise ValueError("prediction gpsno set differs from target vehicles")
    if len({row["gpsno"].strip() for row in rows}) != EXPECTED_ROWS:
        raise ValueError("duplicate prediction gpsno")
    result = []
    for row in rows:
        try:
            probability = float(row["risk_prob"])
        except ValueError as exc:
            raise ValueError("risk_prob is not numeric") from exc
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("risk_prob must be finite and within [0, 1]")
        result.append((row["gpsno"].strip(), probability))
    return sorted(result)


def validate_submission(path: Path, vehicles: set[str]) -> None:
    raw = path.read_bytes()
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise ValueError("submission must use LF line endings")
    rows = read_csv(path, ["gpsno", "accident", "risk_prob"])
    if len(rows) != EXPECTED_ROWS or {r["gpsno"] for r in rows} != vehicles:
        raise ValueError("submission does not cover the exact vehicle set")
    if len({r["gpsno"] for r in rows}) != EXPECTED_ROWS:
        raise ValueError("submission contains duplicate gpsno")
    if any(row["accident"] not in {"0", "1"} for row in rows):
        raise ValueError("accident must be binary")
    validate_predictions([{"gpsno": r["gpsno"], "risk_prob": r["risk_prob"]} for r in rows], vehicles)


def package(predictions: Path, vehicles_path: Path, provenance_path: Path, output_dir: Path) -> Path:
    vehicles = validate_vehicles(read_csv(vehicles_path, ["gpsno"]))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    validate_manifest(provenance, predictions)
    ordered = validate_predictions(read_csv(predictions, ["gpsno", "risk_prob"]), vehicles)
    # accident is only a format field; use a deterministic 100-car display flag.
    high = {gpsno for gpsno, _ in sorted(ordered, key=lambda pair: (-pair[1], pair[0]))[:100]}
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "forecast_result.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["gpsno", "accident", "risk_prob"])
        writer.writerows((gpsno, int(gpsno in high), repr(probability)) for gpsno, probability in ordered)
    validate_submission(output, vehicles)
    receipt = {
        "prediction_role": "future_inference",
        "model_version": provenance["model_version"],
        "as_of": EXPECTED_AS_OF,
        "vehicle_count": EXPECTED_ROWS,
        "submission_sha256": digest(output),
        "predictions_sha256": digest(predictions),
        "vehicles_sha256": digest(vehicles_path),
        "provenance_sha256": digest(provenance_path),
    }
    (output_dir / "submission_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--vehicles", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(package(args.predictions, args.vehicles, args.provenance, args.output_dir))


if __name__ == "__main__":
    main()
