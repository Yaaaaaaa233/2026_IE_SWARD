#!/usr/bin/env python3
"""Re-audit FEAT-009 v2 labels, folds, source identities, and time-window evidence.

All detailed tables and figures are written to the ignored local outputs directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import tomllib
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-feat010")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml


REPO = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
LABEL_DIR = REPO / "outputs/protocol_20260923/label_v2_record_proxy"
FEAT009_DIR = REPO / "outputs/feat-009/v2"
FEATURE_CUTOFF = datetime(2026, 6, 21)
FEATURE_START = datetime(2026, 6, 1)
LABEL_START = datetime(2026, 6, 21)
LABEL_END = datetime(2026, 7, 31)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_proxy_module():
    path = REPO / "tools/build_task1_proxy_labels.py"
    spec = importlib.util.spec_from_file_location("feat010_proxy_label_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the registered proxy label builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False, **kwargs)


def audit_shard_samples(directory: Path, column_order: list[str], has_header: bool,
                        delimiter: str, label: str) -> list[dict]:
    files = sorted(directory.glob("part-*"))
    if not files:
        return [{"source": label, "status": "missing_shards"}]
    selected = sorted(set([0, len(files) // 2, len(files) - 1]))
    if "data_time" not in column_order:
        return [{"source": label, "status": "no_data_time_column_in_config"}]
    time_index = column_order.index("data_time")
    sampled = []
    for ix in selected:
        path = files[ix]
        kwargs = {"sep": delimiter, "nrows": 1000, "dtype": str}
        if has_header:
            frame = pd.read_csv(path, usecols=["data_time"], **kwargs)
        else:
            frame = pd.read_csv(path, header=None, names=column_order,
                                usecols=[time_index], **kwargs)
            if "data_time" not in frame:
                frame = frame.rename(columns={time_index: "data_time"})
        values = pd.to_datetime(frame["data_time"], errors="coerce")
        good = values.dropna()
        if good.empty:
            before = inside = after = 0
            lo = hi = None
        else:
            before = int((good < pd.Timestamp(FEATURE_START)).sum())
            inside = int(((good >= pd.Timestamp(FEATURE_START)) &
                          (good < pd.Timestamp(FEATURE_CUTOFF))).sum())
            after = int((good >= pd.Timestamp(FEATURE_CUTOFF)).sum())
            lo, hi = str(good.min()), str(good.max())
        sampled.append({"source": label, "shard_ordinal": ix + 1, "shard_count": len(files),
                        "rows_sampled": int(len(frame)), "valid_timestamps": int(len(good)),
                        "sample_min_time": lo, "sample_max_time": hi,
                        "before_feature_window": before, "inside_20d_feature_window": inside,
                        "at_or_after_as_of": after})
    return sampled


def render_figures(out: Path, category_counts: dict[str, int], daily: Counter,
                   fold_counts: dict[str, dict[str, int]], rows: int, positives: int,
                   run_id: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    labels = ["Accident ≥1", "No accident; near-miss 0", "No accident; near-miss 1", "No accident; near-miss ≥2"]
    keys = ["accident_positive", "zero_event_negative", "one_nearmiss_negative", "threshold_nearmiss_positive"]
    colors = ["#6d9f7a", "#8fa3b7", "#d4b56a", "#4b8da8"]
    axes[0].bar(range(len(keys)), [category_counts.get(k, 0) for k in keys], color=colors)
    axes[0].set_xticks(range(len(keys)), labels, rotation=24, ha="right")
    axes[0].set_ylabel("Vehicles")
    axes[0].set_title("Label composition")
    axes[0].text(0.02, 0.96, f"n={rows}, positives={positives}", transform=axes[0].transAxes,
                 va="top", fontsize=8)
    dates = sorted(daily)
    axes[1].bar([date.fromisoformat(d) for d in dates], [daily[d] for d in dates],
                width=0.85, color="#7393ad")
    axes[1].set_title("Valid 11803 / 11804 records by day")
    axes[1].set_ylabel("Exact-deduplicated records")
    axes[1].tick_params(axis="x", rotation=30)
    folds = sorted(fold_counts, key=int)
    x = range(len(folds))
    axes[2].bar(x, [fold_counts[f]["negative"] for f in folds], label="negative", color="#8fa3b7")
    axes[2].bar(x, [fold_counts[f]["positive"] for f in folds],
                bottom=[fold_counts[f]["negative"] for f in folds], label="positive", color="#6d9f7a")
    axes[2].set_xticks(list(x), [f"Fold {f}" for f in folds])
    axes[2].set_ylabel("Vehicles")
    axes[2].set_title("Frozen split composition")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"FEAT-010 A0 | Controlled-data audit | {run_id}\n"
                 "Proxy window [Jun 21, Jul 31); source-level audit, not model evaluation", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "a0_label_and_split_audit.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.barh([1], [20], left=[0], height=0.35, color="#6da5c4", label="feature window")
    ax.barh([0], [40], left=[20], height=0.35, color="#e7a56b", label="label window")
    ax.axvline(20, color="#444", linestyle="--", linewidth=1)
    ax.set_xlim(0, 62)
    ax.set_yticks([0, 1], ["Label", "Feature"])
    ax.set_xticks([0, 20, 60], ["Jun 1", "Jun 21", "Jul 31"])
    ax.set_xlabel("Days from Jun 1 (schematic scale; right edges excluded)")
    ax.set_title("A0 time contract | feature [Jun 1, Jun 21), label [Jun 21, Jul 31)")
    ax.text(0.5, -0.28,
            "Feature and label intervals meet at the cutoff; the label interval is not a feature source.",
            transform=ax.transAxes, ha="center", color="#8b4e35", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out / "a0_window_contract.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(REPO / "outputs/feat-010/a0/20260924-r1"))
    parser.add_argument("--feat009-config", default=str(REPO / "configs/feat-009.local.toml"))
    parser.add_argument("--base-config", default=str(REPO / "feature_engineering/configs/pipeline.local.yaml"))
    parser.add_argument("--f3-config", default=str(REPO / "feature_engineering/configs/pipeline_f3r5_v1.local.yaml"))
    args = parser.parse_args()
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    proxy = load_proxy_module()
    feat_cfg_path = Path(args.feat009_config).expanduser().resolve()
    feat_cfg = tomllib.loads(feat_cfg_path.read_text(encoding="utf-8"))
    base_cfg_path = Path(args.base_config).expanduser().resolve()
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    f3_cfg_path = Path(args.f3_config).expanduser().resolve()
    f3_cfg = yaml.safe_load(f3_cfg_path.read_text(encoding="utf-8"))
    data_root = resolve(base_cfg_path.parent.parent, base_cfg["data_root"])
    tables = data_root / base_cfg.get("table_layer", "v4_assessed")
    source_paths = {
        "events": tables / "events_clean.csv",
        "features": tables / "features.csv",
        "vehicles": tables / "target_vehicles.csv",
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing controlled source inputs: {missing}")

    saved_manifest_path = LABEL_DIR / "manifest.json"
    saved_manifest = json.loads(saved_manifest_path.read_text(encoding="utf-8"))
    saved_labels_path, saved_splits_path = LABEL_DIR / "labels.csv", LABEL_DIR / "splits.csv"
    expected_hashes = saved_manifest["source_sha256"]
    actual_hashes = {path.name: sha256(path) for path in source_paths.values()}
    hash_match = {name: actual_hashes[path.name] == expected_hashes.get(path.name)
                  for name, path in source_paths.items()}
    if not all(hash_match.values()):
        raise ValueError("Configured source bytes differ from the registered label manifest")

    samples = proxy.load_samples(source_paths["features"], source_paths["vehicles"],
                                 date(2026, 6, 21), 500)
    sample_to_gps = dict(samples)
    source_version = next(iter({row["source_version"].strip()
                                for row in proxy._rows(source_paths["features"], {"source_version"})}))
    counts, exact_duplicates = proxy.count_label_events(
        source_paths["events"], set(sample_to_gps.values()), date(2026, 6, 21), date(2026, 7, 31),
        source_version=source_version)
    regenerated_labels = proxy.make_label_rows(samples, counts, date(2026, 6, 21), date(2026, 7, 31), source_version)
    regenerated_splits = proxy.make_splits(samples, regenerated_labels)
    saved_labels = list(proxy._rows(saved_labels_path, {"sample_id", "label_window", "horizon_days", "y",
                                                        "label_status", "label_version", "source_version"}))
    saved_labels = [{**row, "y": int(row["y"]), "horizon_days": int(row["horizon_days"])}
                    for row in saved_labels]
    saved_splits = list(proxy._rows(saved_splits_path, {"gpsno", "fold", "split_version"}))
    label_match = regenerated_labels == saved_labels
    split_match = [{**row, "fold": int(row["fold"])} for row in regenerated_splits] == [
        {**row, "fold": int(row["fold"])} for row in saved_splits]

    labels_by_gps: dict[str, dict] = {}
    categories = Counter()
    for sid, gps in samples:
        accident_count = counts[gps]["11803"]
        near_count = counts[gps]["11804"]
        y = int(accident_count >= 1 or near_count >= 2)
        labels_by_gps[gps] = {"sample_id": sid, "y": y, "accident_count": accident_count,
                              "nearmiss_count": near_count}
        if accident_count >= 1:
            categories["accident_positive"] += 1
        elif near_count == 0:
            categories["zero_event_negative"] += 1
        elif near_count == 1:
            categories["one_nearmiss_negative"] += 1
        else:
            categories["threshold_nearmiss_positive"] += 1
    positive_count = sum(row["y"] for row in labels_by_gps.values())
    fold_by_gps = {str(row["gpsno"]): int(row["fold"]) for row in saved_splits}
    fold_counts: dict[str, dict[str, int]] = {str(f): {"positive": 0, "negative": 0} for f in range(5)}
    for gps, item in labels_by_gps.items():
        fold_counts[str(fold_by_gps[gps])]["positive" if item["y"] else "negative"] += 1

    # Rebind current FEAT-009 v2 inputs and compare them to labels/splits recomputed from source.
    v2_alignment = {}
    for name in ("f0", "f1", "f3"):
        path = FEAT009_DIR / "y1" / f"{name}_model_input.csv"
        frame = _read_csv(path)
        row_errors = []
        if len(frame) != 500 or frame["sample_id"].duplicated().any() or frame["gpsno"].duplicated().any():
            row_errors.append("vehicle_rows_or_keys_invalid")
        for row in frame[["sample_id", "gpsno", "y", "fold"]].itertuples(index=False):
            gps = str(row.gpsno)
            expected = labels_by_gps.get(gps)
            if expected is None or expected["sample_id"] != str(row.sample_id):
                row_errors.append("sample_vehicle_binding_mismatch")
                break
            if int(row.y) != expected["y"] or int(row.fold) != fold_by_gps[gps]:
                row_errors.append("label_or_fold_mismatch")
                break
        if "label_version" not in frame or set(frame["label_version"].astype(str)) != {saved_manifest["label_version"]}:
            row_errors.append("label_version_mismatch")
        if "split_version" not in frame or set(frame["split_version"].astype(str)) != {saved_manifest["split_version"]}:
            row_errors.append("split_version_mismatch")
        v2_alignment[name] = {"rows": len(frame), "unique_sample_id": bool(frame.sample_id.is_unique),
                              "unique_gpsno": bool(frame.gpsno.is_unique), "status": "pass" if not row_errors else "fail",
                              "errors": sorted(set(row_errors))}

    # Verify the original 20-day source interface metadata and the F3 declared window.
    raw_inputs = {}
    f0_value = feat_cfg.get("f0_model_input")
    if f0_value:
        raw_inputs["f0"] = Path(f0_value).expanduser().resolve()
    else:
        raw_inputs["f0"] = resolve(base_cfg_path.parent.parent, base_cfg["output_dir"]) / "model_interface/model_input.csv"
    raw_inputs["f1"] = Path(feat_cfg["f1_model_input"]).expanduser().resolve()
    raw_inputs["f3"] = Path(feat_cfg["f3_model_input"]).expanduser().resolve()
    source_windows = {}
    for name, path in raw_inputs.items():
        frame = _read_csv(path, usecols=lambda c: c in {"sample_id", "gpsno", "as_of", "lookback_days"})
        as_of = pd.to_datetime(frame["as_of"], errors="coerce") if "as_of" in frame else pd.Series(dtype="datetime64[ns]")
        lookback = pd.to_numeric(frame["lookback_days"], errors="coerce") if "lookback_days" in frame else pd.Series(dtype=float)
        source_windows[name] = {
            "rows": len(frame), "unique_sample_id": bool(frame.sample_id.is_unique),
            "as_of_values": sorted({str(v) for v in as_of.dropna().dt.strftime("%Y-%m-%d %H:%M:%S")}),
            "lookback_days_values": sorted({float(v) for v in lookback.dropna().unique()}),
            "status": "pass" if len(frame) == 500 and as_of.notna().all() and
                      set(as_of.dt.strftime("%Y-%m-%d")) == {"2026-06-21"} and
                      set(lookback.dropna().astype(float)) == {20.0} else "fail",
        }
    f3_manifest_path = Path(feat_cfg["f3_manifest"]).expanduser().resolve()
    f3_manifest = json.loads(f3_manifest_path.read_text(encoding="utf-8"))
    source_windows["f3_manifest"] = f3_manifest.get("window", {})
    f3_events_path = resolve(f3_cfg_path.parent, f3_cfg["events_path"])
    source_windows["f3_event_source"] = {
        "configured_source_exists": f3_events_path.is_file(),
        "configured_source_sha256": sha256(f3_events_path) if f3_events_path.is_file() else None,
        "same_bytes_as_proxy_label_source": (f3_events_path.is_file() and
                                               sha256(f3_events_path) == actual_hashes["events_clean.csv"]),
        "schema": list(pd.read_csv(f3_events_path, nrows=0).columns) if f3_events_path.is_file() else [],
        "note": "F3 uses canonical event_id/gpsno/event_time/scenario input; F1 and proxy labels use v4_assessed/events_clean.csv.",
    }

    # Source timestamp extent and event-day summary; row-level samples remain local.
    gpsnos = set(sample_to_gps.values())
    daily_events: Counter[str] = Counter()
    source_period_counts: Counter[str] = Counter()
    sample_gps = {}
    for category, predicate in [
        ("accident_positive", lambda a, n: a >= 1),
        ("threshold_nearmiss_positive", lambda a, n: a == 0 and n >= 2),
        ("one_nearmiss_negative", lambda a, n: a == 0 and n == 1),
        ("zero_event_negative", lambda a, n: a == 0 and n == 0),
    ]:
        chosen = next((gps for gps, item in labels_by_gps.items()
                       if predicate(item["accident_count"], item["nearmiss_count"])), None)
        if chosen is not None:
            sample_gps[category] = chosen
    sample_events = []
    label_seen = set()
    event_min = event_max = None
    with source_paths["events"].open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("event_type", "").strip() not in {"11803", "11804"} or row.get("gpsno", "").strip() not in gpsnos:
                continue
            raw_time = row.get("start_time", "").strip()
            if not raw_time:
                continue
            occurred = datetime.fromisoformat(raw_time)
            event_min = occurred if event_min is None or occurred < event_min else event_min
            event_max = occurred if event_max is None or occurred > event_max else event_max
            if FEATURE_START <= occurred < FEATURE_CUTOFF:
                source_period_counts["feature_window_events"] += 1
            elif LABEL_START <= occurred < LABEL_END:
                source_period_counts["proxy_label_window_events"] += 1
                signature = tuple(row.get(c, "").strip() for c in proxy.EVENT_ID_COLUMNS)
                duplicate = signature in label_seen
                label_seen.add(signature)
                daily_events[occurred.date().isoformat()] += int(not duplicate)
                gps = row["gpsno"].strip()
                if gps in sample_gps.values():
                    sample_events.append({
                        "review_category": next(k for k, v in sample_gps.items() if v == gps),
                        "sample_id": sample_to_gps and labels_by_gps[gps]["sample_id"],
                        "gpsno": gps, "row_id": row.get("row_id", "").strip(),
                        "event_type": row.get("event_type", "").strip(), "event_time": raw_time,
                        "exact_duplicate_within_review_sample": duplicate,
                        "expected_vehicle_label": labels_by_gps[gps]["y"],
                        "count_unit": "exact-deduplicated retained event record",
                    })
            elif occurred >= LABEL_END:
                source_period_counts["on_or_after_label_end"] += 1
            else:
                source_period_counts["before_feature_window"] += 1

    f3_event_time_counts = Counter()
    f3_event_time_min = f3_event_time_max = None
    for chunk in pd.read_csv(f3_events_path, usecols=["gpsno", "event_time"], chunksize=200_000,
                             dtype={"gpsno": "string"}, low_memory=False):
        times = pd.to_datetime(chunk["event_time"], errors="coerce")
        valid = times.notna()
        if not valid.any():
            continue
        good = times[valid]
        f3_event_time_min = good.min() if f3_event_time_min is None else min(f3_event_time_min, good.min())
        f3_event_time_max = good.max() if f3_event_time_max is None else max(f3_event_time_max, good.max())
        target_rows = chunk.loc[valid, "gpsno"].astype(str).isin(gpsnos).to_numpy()
        target_times = good.to_numpy()[target_rows]
        ts = pd.Series(target_times)
        f3_event_time_counts["target_rows_before_feature_window"] += int((ts < pd.Timestamp(FEATURE_START)).sum())
        f3_event_time_counts["target_rows_in_20d_feature_window"] += int(
            ((ts >= pd.Timestamp(FEATURE_START)) & (ts < pd.Timestamp(FEATURE_CUTOFF))).sum())
        f3_event_time_counts["target_rows_at_or_after_as_of"] += int((ts >= pd.Timestamp(FEATURE_CUTOFF)).sum())
    source_windows["f3_event_source_time_scan"] = {
        "min_time": None if f3_event_time_min is None else str(f3_event_time_min),
        "max_time": None if f3_event_time_max is None else str(f3_event_time_max),
        "target_event_rows_by_period": dict(f3_event_time_counts),
        "scan": "full timestamp-column scan; F3 pipeline applies [as_of-lookback, as_of) before aggregation",
    }
    write_csv(out / "label_sample_review.csv", sample_events,
              ["review_category", "sample_id", "gpsno", "row_id", "event_type", "event_time",
               "exact_duplicate_within_review_sample", "expected_vehicle_label", "count_unit"])

    f3_dir = Path(f3_cfg["trajectory_dir"]).expanduser()
    if not f3_dir.is_absolute():
        f3_dir = (f3_cfg_path.parent / f3_dir).resolve()
    imu_dir = Path(f3_cfg["imu_dir"]).expanduser()
    if not imu_dir.is_absolute():
        imu_dir = (f3_cfg_path.parent / imu_dir).resolve()
    shard_samples = audit_shard_samples(f3_dir, list(f3_cfg.get("trajectory_column_order", [])),
                                        bool(f3_cfg.get("shard_has_header", True)),
                                        str(f3_cfg.get("shard_delimiter", "\t")), "trajectory")
    shard_samples += audit_shard_samples(imu_dir, list(f3_cfg.get("imu_column_order", [])),
                                        bool(f3_cfg.get("imu_has_header", False)),
                                        str(f3_cfg.get("imu_delimiter", "\t")), "imu")
    write_csv(out / "source_timestamp_shard_samples.csv", shard_samples)

    fold_rows = [{"fold": int(fold), **values} for fold, values in sorted(fold_counts.items(), key=lambda x: int(x[0]))]
    write_csv(out / "fold_counts.csv", fold_rows)
    category_json = dict(categories)
    write_csv(out / "label_categories.csv", [{"category": k, "vehicles": v} for k, v in category_json.items()])
    render_figures(out, category_json, daily_events, fold_counts, len(samples), positive_count, out.name)

    checks = {
        "source_sha256_matches_label_manifest": all(hash_match.values()),
        "regenerated_labels_equal_saved_labels": label_match,
        "regenerated_splits_equal_saved_splits": split_match,
        "500_unique_target_vehicles": len(samples) == 500 and len({g for _, g in samples}) == 500,
        "all_folds_have_both_classes": all(v["positive"] > 0 and v["negative"] > 0 for v in fold_counts.values()),
        "feat009_v2_inputs_match_new_labels_and_folds": all(v["status"] == "pass" for v in v2_alignment.values()),
        "original_20d_input_metadata_consistent": all(source_windows[k].get("status") == "pass"
                                                       for k in ("f0", "f1", "f3")),
        "f3_manifest_declares_20d_cutoff": source_windows["f3_manifest"].get("as_of", "").startswith("2026-06-21") and float(source_windows["f3_manifest"].get("lookback_days", -1)) == 20.0,
    }
    errors = [name for name, passed in checks.items() if not passed]
    status = "pass" if not errors else "fail"
    report = {
        "task": "FEAT-010-A0", "run_id": out.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine_status": status, "human_review": "pending", "data_evaluation_owner_review": "pending",
        "protocol": saved_manifest.get("label_version"), "split_version": saved_manifest.get("split_version"),
        "feature_window": "[2026-06-01,2026-06-21)", "label_window": "[2026-06-21,2026-07-31)",
        "rows": len(samples), "positive_vehicles": positive_count, "exact_duplicate_rows_removed": exact_duplicates,
        "label_categories": category_json, "fold_counts": fold_counts,
        "source_identities": {name: {"sha256": actual_hashes[path.name], "matches_registered_manifest": hash_match[name]}
                              for name, path in source_paths.items()},
        "source_event_timestamp_extent": {"min": None if event_min is None else event_min.isoformat(),
                                           "max": None if event_max is None else event_max.isoformat(),
                                           "window_record_counts": dict(source_period_counts),
                                           "note": "event source exact scan; timestamps are local naive values interpreted as Asia/Shanghai per project contract"},
        "raw_shard_timestamp_samples": shard_samples,
        "v2_input_alignment": v2_alignment, "original_source_windows": source_windows,
        "checks": checks, "errors": errors,
        "limitations": [
            "Trajectory and IMU raw timestamps are sampled from first/middle/last shard, not exhaustively rescanned.",
            "F0/F1/F3 source window code and metadata are checked; field-by-field raw row lineage still needs human review.",
            "This is label/source/fold audit only; no model metric is recalculated and no baseline is frozen.",
        ],
        "inputs_sha256": {name: sha256(path) for name, path in source_paths.items()},
    }
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "acceptance.md").write_text(
        "# FEAT-010 A0 受控验收摘要\n\n"
        f"- 运行：`{out.name}`；协议：`{saved_manifest.get('label_version')}` / `{saved_manifest.get('split_version')}`。\n"
        "- 范围：新版代理标签、固定车辆折分、FEAT-009 v2 输入绑定及原始时间窗口证据。\n"
        "- 实现／数据契约：`" + status + "`（见 `audit.json` 的机器断言和限制）。\n"
        "- 实验效果：`not_run`；未训练模型、未复算 AUC，也未冻结公共基线。\n"
        "- 验收与证据：机器结果已写入本目录；叶安字段来源复核及数据／评估责任人标签折分复核待记录。\n"
        "- 关键图：`a0_label_and_split_audit.png`、`a0_window_contract.png`。抽查行只在 `label_sample_review.csv` 本地受控保存。\n"
        "- 限制：轨迹与 IMU 仅按方案抽取首／中／末分片样本时间；完整时间过滤依赖代码窗口掩码，不以样本抽查冒充全量证明。\n",
        encoding="utf-8")
    print(json.dumps({"output": str(out), "machine_status": status, "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
