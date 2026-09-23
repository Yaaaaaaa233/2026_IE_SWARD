#!/usr/bin/env python3
"""Build versioned Task 1 proxy labels and vehicle folds from controlled CSVs.

This is an internal historical-window evaluation input, not the unknown official
August/September labels. Source and output CSVs must remain outside public Git.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path


LABEL_VERSION = "label_v2_record_count_20260923"
SPLIT_VERSION = "split_v2_record_strat5_seed42"
EVENT_CODES = {"11803", "11804"}
EVENT_ID_COLUMNS = ("gpsno", "event_type", "event_name", "start_time", "speed", "lat", "lng")


def _rows(path: Path, required: set[str]):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        yield from reader


def _day(value: str) -> date:
    return date.fromisoformat(value)


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        raise ValueError("Timestamp timezone must be resolved by the source contract")
    return parsed


def load_samples(features: Path, vehicles: Path, as_of: date, expected_vehicles: int) -> list[tuple[str, str]]:
    targets = [row["gpsno"].strip() for row in _rows(vehicles, {"gpsno"})]
    if len(targets) != expected_vehicles or len(set(targets)) != len(targets) or not all(targets):
        raise ValueError("target_vehicles must have the expected number of unique nonempty gpsno")
    samples = []
    for row in _rows(features, {"sample_id", "gpsno", "as_of"}):
        if _day(row["as_of"]) != as_of:
            raise ValueError("features.as_of differs from requested proxy cutoff")
        samples.append((row["sample_id"].strip(), row["gpsno"].strip()))
    if (len(samples) != expected_vehicles or not all(sid and gpsno for sid, gpsno in samples)
            or len({sid for sid, _ in samples}) != len(samples)):
        raise ValueError("features must have one unique sample_id per target vehicle")
    if len({gpsno for _, gpsno in samples}) != len(samples) or {g for _, g in samples} != set(targets):
        raise ValueError("features and target_vehicles gpsno sets differ or contain duplicates")
    return samples


def count_label_events(events: Path, gpsnos: set[str], as_of: date, end_exclusive: date,
                       source_version: str | None = None) -> tuple[dict[str, Counter], int]:
    start = datetime.combine(as_of, time.min)
    stop = datetime.combine(end_exclusive, time.min)
    counts = {gpsno: Counter() for gpsno in gpsnos}
    seen_rows: set[tuple[str, ...]] = set()
    seen_ids: set[str] = set()
    exact_duplicates = 0
    required = set(EVENT_ID_COLUMNS) | {"row_id"}
    if source_version is not None:
        required.add("source_version")
    for row in _rows(events, required):
        code = row["event_type"].strip()
        if code not in EVENT_CODES:
            continue
        gpsno = row["gpsno"].strip()
        if gpsno not in gpsnos:
            continue
        if not row["start_time"].strip():
            raise ValueError("Target label event has missing start_time")
        occurred = _timestamp(row["start_time"].strip())
        if not start <= occurred < stop:
            continue
        if source_version is not None and row["source_version"].strip() != source_version:
            raise ValueError("Label event source_version differs from features.source_version")
        row_id = row["row_id"].strip()
        if not row_id or row_id in seen_ids:
            raise ValueError("Label-window event has missing or duplicate row_id")
        seen_ids.add(row_id)
        signature = tuple(row[column].strip() for column in EVENT_ID_COLUMNS)
        if signature in seen_rows:
            exact_duplicates += 1
            continue
        seen_rows.add(signature)
        counts[gpsno][code] += 1
    return counts, exact_duplicates


def make_label_rows(samples: list[tuple[str, str]], counts: dict[str, Counter],
                    as_of: date, end_exclusive: date, source_version: str) -> list[dict[str, str | int]]:
    window = f"{as_of.isoformat()}/{end_exclusive.isoformat()}"
    horizon = (end_exclusive - as_of).days
    return [
        {"sample_id": sample_id, "label_window": window, "horizon_days": horizon,
         "y": int(counts[gpsno]["11803"] >= 1 or counts[gpsno]["11804"] >= 2),
         "label_status": "log_complete", "label_version": LABEL_VERSION,
         "source_version": source_version}
        for sample_id, gpsno in samples
    ]


def make_splits(samples: list[tuple[str, str]], labels: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    folds, seed = 5, 42  # Both are encoded in SPLIT_VERSION.
    if len(samples) != len(labels):
        raise ValueError("samples and labels length mismatch")
    groups = {0: [], 1: []}
    for (_, gpsno), label in zip(samples, labels):
        groups[int(label["y"])].append(gpsno)
    if min(len(groups[0]), len(groups[1])) < folds:
        raise ValueError("Each class must contain at least one vehicle per fold")
    mapping = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda g: (hashlib.sha256(f"{seed}:{g}".encode()).digest(), g))
        for index, gpsno in enumerate(ordered):
            mapping[gpsno] = index % folds
    return [{"gpsno": gpsno, "fold": mapping[gpsno], "split_version": SPLIT_VERSION}
            for _, gpsno in samples]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(events: Path, features: Path, vehicles: Path, output_dir: Path,
          as_of: date, end_exclusive: date, expected_vehicles: int = 500) -> None:
    if (end_exclusive - as_of).days != 40:
        raise ValueError("Task 1 proxy label horizon must be exactly 40 days")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing protocol output: {output_dir}")
    samples = load_samples(features, vehicles, as_of, expected_vehicles)
    versions = {row["source_version"].strip() for row in _rows(features, {"source_version"})}
    if len(versions) != 1 or not next(iter(versions)):
        raise ValueError("features.source_version must have one unique nonempty value")
    source_version = versions.pop()
    counts, duplicate_count = count_label_events(events, {gpsno for _, gpsno in samples},
                                                 as_of, end_exclusive, source_version)
    labels = make_label_rows(samples, counts, as_of, end_exclusive, source_version)
    splits = make_splits(samples, labels)
    output_dir.mkdir(parents=True)
    label_path = output_dir / "labels.csv"
    split_path = output_dir / "splits.csv"
    _write_csv(label_path, ["sample_id", "label_window", "horizon_days", "y", "label_status",
                            "label_version", "source_version"], labels)
    _write_csv(split_path, ["gpsno", "fold", "split_version"], splits)
    positive = sum(int(row["y"]) for row in labels)
    fold_counts = {str(fold): {"positive": 0, "negative": 0} for fold in range(5)}
    for label, split in zip(labels, splits):
        fold_counts[str(split["fold"])]["positive" if label["y"] else "negative"] += 1
    manifest = {
        "purpose": "internal_proxy_backtest_only; not official future labels or an evaluated model",
        "decision": "ADR-0006 + ADR-0007", "label_version": LABEL_VERSION,
        "split_version": SPLIT_VERSION, "as_of": as_of.isoformat(),
        "end_exclusive": end_exclusive.isoformat(), "horizon_days": 40,
        "event_unit": "one exact-deduplicated retained event row per occurrence",
        "target_vehicles": len(samples), "positive_vehicles": positive,
        "label_event_exact_duplicates_removed": duplicate_count,
        "fold_counts": fold_counts, "seed": 42,
        "source_sha256": {path.name: _sha256(path) for path in (events, features, vehicles)},
        "output_sha256": {path.name: _sha256(path) for path in (label_path, split_path)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True, help="Controlled events_clean.csv")
    parser.add_argument("--features", type=Path, required=True, help="Matching proxy features.csv")
    parser.add_argument("--vehicles", type=Path, required=True, help="Matching target_vehicles.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="New ignored/controlled output directory")
    parser.add_argument("--as-of", type=_day, required=True, help="Proxy prediction cutoff, YYYY-MM-DD")
    parser.add_argument("--end-exclusive", type=_day, required=True, help="Proxy label right boundary, YYYY-MM-DD")
    args = parser.parse_args()
    build(args.events, args.features, args.vehicles, args.output_dir, args.as_of, args.end_exclusive)
    print("Proxy labels, folds and manifest written to controlled output; no model was evaluated.")


if __name__ == "__main__":
    main()
