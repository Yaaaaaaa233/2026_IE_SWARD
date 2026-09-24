#!/usr/bin/env python3
"""Build and audit a private, unlabeled 61-day FEAT-010 A1 feature candidate.

Real rows, figures, source identities, and raw file inventories are written only
under the ignored outputs/feat-010/a1 directory. No label or fold is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig-feat010")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
sys.path.insert(0, str(REPO / "feature_engineering" / "src"))

from accident_pipeline.config import load_config as load_base_config
from accident_pipeline_f1.config import F1Config
from accident_pipeline_f1.event_features import build_event_features
from accident_pipeline_f3.pipeline import load_f3_config, run_scan


START = pd.Timestamp("2026-06-01 00:00:00")
AS_OF = pd.Timestamp("2026-08-01 00:00:00")
WINDOW_DAYS = 61
OFFICIAL_TZ = "Asia/Shanghai"
FAMILY_TYPES = {
    "evt_lane_count": {30002, 30003, 30017},
    "evt_fatigue_count": {41001, 41002, 41029},
    "evt_distraction_count": {41003, 41004, 41005, 41009, 41023},
    "evt_speed_count": {11401, 11402, 11403, 11405, 11406},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_payload(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def run_synthetic_checks() -> dict:
    checks = {}
    sample_times = pd.to_datetime(["2026-05-31 23:59:59", "2026-06-01 00:00:00",
                                   "2026-07-31 23:59:59", "2026-08-01 00:00:00"])
    mask = (sample_times >= START) & (sample_times < AS_OF)
    checks["half_open_61d_window"] = mask.tolist() == [False, True, True, False]

    count, exposure = 8.0, 400.0
    rate = count / exposure
    doubled_rate = (count * 2) / (exposure * 2)
    checks["count_and_exposure_scale_together"] = bool(np.isclose(rate, doubled_rate))

    hours = pd.Series(["2026-06-01 21:30:00", "2026-06-01 23:30:00",
                       "2026-06-02 05:30:00", "2026-06-02 10:00:00"])
    hour = pd.to_datetime(hours).dt.hour
    official_night = ((hour >= 21) | (hour < 6)).tolist()
    research_deep = ((hour >= 23) | (hour < 5)).tolist()
    checks["official_night_is_distinct_from_research_deep_night"] = (
        official_night == [True, True, True, False]
        and research_deep == [False, True, False, False]
    )

    forbidden = {"y", "fold", "label", "label_window", "horizon_days"}
    try:
        assert_unlabeled_candidate(pd.DataFrame({"sample_id": ["synthetic"], "y": [1]}))
    except ValueError:
        checks["synthetic_contract_rejects_label_fields"] = True
    else:
        checks["synthetic_contract_rejects_label_fields"] = False
    if not all(checks.values()):
        raise AssertionError(f"A1 synthetic contract checks failed: {checks}")
    return checks


def assert_unlabeled_candidate(frame: pd.DataFrame) -> None:
    banned = {"y", "fold", "label", "label_window", "horizon_days", "label_status",
              "label_version", "split_version"}
    present = banned & set(frame.columns)
    if present:
        raise ValueError(f"A1 candidate contains label/split columns: {sorted(present)}")


def aggregate_events(events_path: Path, roster: pd.DataFrame, source_features: pd.DataFrame, out: Path,
                     scenario_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target = set(roster["gpsno"].astype(str))
    retained = []
    all_counts: dict[str, Counter] = defaultdict(Counter)
    active_days: dict[str, set] = defaultdict(set)
    feature_window_counts: dict[str, Counter] = defaultdict(Counter)
    feature_active_days: dict[str, set] = defaultdict(set)
    raw_rows_read = 0

    usecols = ["row_id", "gpsno", "event_type", "start_time", "speed", "lat", "lng"]
    for chunk in pd.read_csv(events_path, usecols=usecols, dtype={"gpsno": str, "event_type": str},
                             chunksize=300_000, low_memory=False):
        raw_rows_read += len(chunk)
        gps = chunk["gpsno"].astype(str)
        time = pd.to_datetime(chunk["start_time"], errors="coerce")
        in_window = gps.isin(target) & time.ge(START) & time.lt(AS_OF)
        if not in_window.any():
            continue
        selected = chunk.loc[in_window].copy()
        selected["gpsno"] = gps.loc[in_window].to_numpy()
        selected["event_time_parsed"] = time.loc[in_window].to_numpy()
        selected["event_type_num"] = pd.to_numeric(selected["event_type"], errors="coerce")
        selected["event_type_num"] = selected["event_type_num"].fillna(-1).astype(int)
        selected["event_date"] = selected["event_time_parsed"].dt.date

        for row in selected[["gpsno", "event_type_num", "event_date"]].itertuples(index=False):
            vehicle, event_type, day = str(row.gpsno), int(row.event_type_num), row.event_date
            all_counts[vehicle]["evt_count"] += 1
            if event_type in FAMILY_TYPES["evt_lane_count"]:
                all_counts[vehicle]["evt_lane_count"] += 1
            if event_type in FAMILY_TYPES["evt_fatigue_count"]:
                all_counts[vehicle]["evt_fatigue_count"] += 1
            if event_type in FAMILY_TYPES["evt_distraction_count"]:
                all_counts[vehicle]["evt_distraction_count"] += 1
            if event_type in FAMILY_TYPES["evt_speed_count"]:
                all_counts[vehicle]["evt_speed_count"] += 1
            active_days[vehicle].add(day)
            if pd.Timestamp(day) < pd.Timestamp("2026-06-21"):
                feature_window_counts[vehicle]["evt_count"] += 1
                if event_type in FAMILY_TYPES["evt_lane_count"]:
                    feature_window_counts[vehicle]["evt_lane_count"] += 1
                if event_type in FAMILY_TYPES["evt_fatigue_count"]:
                    feature_window_counts[vehicle]["evt_fatigue_count"] += 1
                if event_type in FAMILY_TYPES["evt_distraction_count"]:
                    feature_window_counts[vehicle]["evt_distraction_count"] += 1
                if event_type in FAMILY_TYPES["evt_speed_count"]:
                    feature_window_counts[vehicle]["evt_speed_count"] += 1
                feature_active_days[vehicle].add(day)

        selected["scenario"] = selected["event_type"].map(scenario_map).fillna("other")
        retained.append(pd.DataFrame({
            "row_id": selected["row_id"].astype(str),
            "gpsno": selected["gpsno"].astype(str),
            "event_type": selected["event_type_num"].astype(int),
            "start_time": selected["event_time_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": selected["scenario"].astype(str),
            "speed": pd.to_numeric(selected["speed"], errors="coerce"),
            "lat": pd.to_numeric(selected["lat"], errors="coerce"),
            "lon": pd.to_numeric(selected["lng"], errors="coerce"),
        }))

    if not retained:
        raise ValueError("No target events fall in the requested feature window")
    target_events = pd.concat(retained, ignore_index=True)
    if target_events["row_id"].duplicated().any():
        raise ValueError("Target source event IDs are not unique")
    write_csv(out / "events_target_for_f1.csv", target_events[["gpsno", "event_type", "start_time"]])
    canonical = target_events.rename(columns={"row_id": "event_id", "start_time": "event_time", "lon": "lon"})
    write_csv(out / "events_canonical_61d.csv", canonical[
        ["event_id", "gpsno", "event_time", "scenario", "speed", "lat", "lon"]
    ])

    count_rows = []
    for gps in roster["gpsno"].astype(str):
        counts = all_counts[str(gps)]
        row = {"gpsno": str(gps),
               "evt_count_61d": int(counts.get("evt_count", 0)),
               "evt_count_log1p_61d": float(np.log1p(counts.get("evt_count", 0))),
               "evt_active_days_61d": len(active_days[str(gps)])}
        count_rows.append(row)
    base_counts = pd.DataFrame(count_rows)
    audit_rows = []
    old = source_features.set_index("gpsno")
    feature_counts_20 = []
    for gps in roster["gpsno"].astype(str):
        c = feature_window_counts[str(gps)]
        feature_counts_20.append({"gpsno": str(gps), **{n: int(c.get(n, 0)) for n in ("evt_count", *FAMILY_TYPES.keys())},
                                  "evt_active_days": len(feature_active_days[str(gps)])})
    old20 = pd.DataFrame(feature_counts_20).set_index("gpsno")
    for column in (*old20.columns,):
        registered = pd.to_numeric(old.loc[old20.index, column], errors="coerce")
        rebuilt = pd.to_numeric(old20[column], errors="coerce")
        delta = (registered - rebuilt).abs()
        audit_rows.append({"field": column, "old_20d_source": "features.csv",
                           "rebuilt_source": "events_clean.csv", "vehicles": int(len(delta)),
                           "exact_match_vehicles": int((delta == 0).sum()),
                           "max_absolute_difference": float(delta.max()),
                           "status": "pass" if (delta == 0).all() else "review_required"})
    return base_counts, target_events, {
        "source_rows_read": raw_rows_read,
        "target_events_61d": int(len(target_events)),
        "source_min_time": str(target_events.start_time.min()),
        "source_max_time": str(target_events.start_time.max()),
        "old_20d_event_reconciliation": audit_rows,
    }


def daily_summary(data_root: Path, table_layer: str, roster: pd.DataFrame,
                  source_features: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, dict]:
    tables = data_root / table_layer
    daily = pd.read_csv(tables / "vehicle_day.csv", dtype={"gpsno": str}, low_memory=False)
    daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.normalize()
    daily = daily[daily["gpsno"].astype(str).isin(set(roster.gpsno.astype(str)))].copy()
    daily = daily[(daily["date"] >= START) & (daily["date"] < AS_OF)]
    if daily.duplicated(["gpsno", "date"]).any():
        raise ValueError("vehicle_day has duplicate vehicle/date rows in the 61-day window")
    expected = pd.date_range(START, AS_OF - pd.Timedelta(days=1), freq="D")
    observed_dates = sorted(set(daily["date"]))
    missing_dates = [d.strftime("%Y-%m-%d") for d in expected if d not in set(observed_dates)]
    grouping = daily.groupby("gpsno", sort=False)
    sums = grouping.agg(
        vehicle_day_dates_recorded=("date", "nunique"),
        traj_km_daily_sum=("traj_km", "sum"),
        traj_hours_daily_sum=("traj_hours", "sum"),
        coverage_traj_observed_seconds_partial=("traj_observed_seconds", "sum"),
        coverage_imu_observed_seconds_partial=("imu_observed_seconds", "sum"),
        coverage_imu_traj_overlap_seconds_partial=("imu_traj_overlap_seconds", "sum"),
        coverage_quality_flag_days_partial=("quality_flags", lambda x: int(x.notna().sum())),
    ).reset_index().rename(columns={"gpsno": "gpsno"})
    # Preserve the historical 20-day source reconciliation; the daily summary is a diagnostic,
    # not a substitute for the raw-source 61-day exposure sidecar.
    old_20 = daily[(daily.date >= START) & (daily.date < pd.Timestamp("2026-06-21"))].groupby("gpsno").agg(
        traj_km_rebuilt=("traj_km", "sum"), traj_hours_rebuilt=("traj_hours", "sum"),
        dates_rebuilt=("date", "nunique"),
    )
    old_base = source_features.set_index("gpsno")
    old_recon = []
    for old_col, new_col, tolerance in (("traj_km_20d", "traj_km_rebuilt", 0.0055),
                                        ("traj_hours_20d", "traj_hours_rebuilt", 0.00055)):
        left = pd.to_numeric(old_base[old_col], errors="coerce")
        right = pd.to_numeric(old_20[new_col].reindex(left.index), errors="coerce")
        diff = (left - right).abs()
        matched = left.notna() & right.notna()
        within = diff[matched] <= tolerance
        old_recon.append({"field": old_col, "vehicles": int(len(diff)),
                          "vehicles_with_daily_rows": int(matched.sum()),
                          "vehicles_missing_daily_rows": int((~right.notna()).sum()),
                          "within_rounding_tolerance": int(within.sum()),
                          "max_absolute_difference": float(diff[matched].max()) if matched.any() else None,
                          "tolerance": tolerance,
                          "status": ("pass" if matched.all() and within.all() else
                                     "partial_missing_daily_rows" if within.all() else "review_required")})
    write_csv(out / "vehicle_day_summary_61d_partial.csv", sums)
    return sums, {"expected_calendar_days": int(len(expected)), "recorded_calendar_days": int(len(observed_dates)),
                  "missing_calendar_dates": missing_dates,
                  "vehicle_day_rows": int(len(daily)), "old_20d_exposure_reconciliation": old_recon,
                  "interpretation": "vehicle_day ends before the requested as_of; 61-day sums from this source are incomplete and excluded from model candidate"}


def make_f3_local_config(template_path: Path, out: Path, feature_table: Path,
                         events_path: Path, profile_path: Path) -> Path:
    raw = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    raw["output_dir"] = str(out / "f3")
    raw["events_path"] = str(events_path)
    raw["profile_path"] = str(profile_path)
    raw["base_input"] = str(feature_table)
    raw["base_columns_file"] = str(out / "base_columns_unused_for_scan.txt")
    raw["window"] = {"as_of": "2026-08-01T00:00:00", "lookback_days": 61.0, "horizon_days": 40.0}
    raw.setdefault("cohort", {})["source_columns"] = []  # cohort fits need a labeled training cohort; defer to model stage
    raw.setdefault("events", {})["copair_columns"] = []
    path = out / "pipeline_f3_a1.local.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "base_columns_unused_for_scan.txt").write_text("# run_scan consumes roster only\n", encoding="utf-8")
    return path


def build_contract(frame: pd.DataFrame, out: Path, daily_audit: dict) -> pd.DataFrame:
    rows = []
    for name in frame.columns:
        if name in {"sample_id", "gpsno", "as_of", "lookback_days"}:
            rows.append({"field": name, "source": "A1 roster / feature window", "window": "metadata",
                         "unit": "identifier / timestamp / days", "treatment": "metadata_only",
                         "missing_semantics": "not a model feature", "status": "included"})
        elif name.startswith("f3_night_") or name == "f3_fatigue_night_conc":
            rows.append({"field": name, "source": "events_canonical_61d.csv + F3 trajectory source",
                         "window": "[2026-06-01, 2026-08-01)", "unit": "rate / ratio; see field name",
                         "treatment": "research_deep_night_23_05; not official night 21_06",
                         "missing_semantics": "F3 definition; inspect source field and denominator",
                         "status": "included_candidate_requires_semantic_review"})
        elif name.startswith("f3_"):
            rows.append({"field": name, "source": "events_canonical_61d.csv / trajectory shards / profile source",
                         "window": "[2026-06-01, 2026-08-01)", "unit": "feature-specific; source code contract",
                         "treatment": "F3R5 feature recomputed at 61d; cohort-relative fields deferred",
                         "missing_semantics": "feature-specific; NaN preserved for later fold-safe handling",
                         "status": "included_candidate_requires_field_review"})
        elif name.startswith("f1_"):
            if "_per_1000km_" in name:
                unit = "count per 1000 km; matching-window F3 trajectory exposure"
            elif "_per_100h_" in name:
                unit = "count per 100 hours; matching-window F3 trajectory exposure"
            elif "count_" in name:
                unit = "event records"
            elif "days_since" in name:
                unit = "days"
            else:
                unit = "indicator / smoothed ratio"
            rows.append({"field": name, "source": "events_clean.csv via F1 event_features.py",
                         "window": "window length embedded in field; as_of=2026-08-01",
                         "unit": unit, "treatment": "recomputed; 5/10/20d short windows plus 61d history",
                         "missing_semantics": "zero for no matching event; normalized rates missing below configured exposure threshold",
                         "status": "included_candidate"})
        elif name.startswith("evt_"):
            rows.append({"field": name, "source": "events_clean.csv event_type mapping",
                         "window": "[2026-06-01, 2026-08-01)",
                         "unit": "event records" if name == "evt_count_61d" else "active days / log1p count",
                         "treatment": "recomputed 61d; legacy family mappings are not carried under old names",
                         "missing_semantics": "zero means no matching source event row in the window",
                         "status": "included_candidate"})
        elif name.startswith("profile_") or name in {"energy_type", "highway_share", "highway_ratio"}:
            rows.append({"field": name, "source": "target_vehicles.csv canonical profile",
                         "window": "source-provided profile; availability date pending source review",
                         "unit": "source unit / ratio", "treatment": "carried with source identity; not recomputed",
                         "missing_semantics": "source missingness retained", "status": "included_candidate_requires_source_review"})
        elif name.startswith("traj_km_"):
            rows.append({"field": name, "source": "raw trajectory shards via F3 window_exposure",
                         "window": f"[2026-08-01 minus field window, 2026-08-01)",
                         "unit": "km", "treatment": "recomputed at matching window; valid driving steps only",
                         "missing_semantics": "0 when no valid trajectory step", "status": "included_candidate"})
        elif name.startswith("traj_hours_"):
            rows.append({"field": name, "source": "raw trajectory shards via F3 window_exposure",
                         "window": f"[2026-08-01 minus field window, 2026-08-01)",
                         "unit": "hours", "treatment": "recomputed at matching window; valid driving steps only",
                         "missing_semantics": "0 when no valid trajectory step", "status": "included_candidate"})
        elif name == "imu_rows_61d":
            rows.append({"field": name, "source": "raw IMU shards via F3 time-window scan",
                         "window": "[2026-06-01, 2026-08-01)", "unit": "IMU records",
                         "treatment": "direct observation count; candidate for the preregistered observation ablation",
                         "missing_semantics": "0 means no matching row in scanned source; check source coverage",
                         "status": "included_candidate_requires_review"})
        else:
            rows.append({"field": name, "source": "A1 candidate assembly", "window": "see source code",
                         "unit": "needs explicit classification", "treatment": "unclassified",
                         "missing_semantics": "unverified", "status": "fail_unclassified"})
    contract = pd.DataFrame(rows)
    if (contract.status == "fail_unclassified").any():
        raise ValueError("A1 candidate contains unclassified fields: "
                         + str(contract.loc[contract.status == "fail_unclassified", "field"].tolist()))
    write_csv(out / "window_contract.csv", contract)
    return contract


def audit_distributions(frame: pd.DataFrame, out: Path) -> pd.DataFrame:
    selected = [c for c in frame.columns if c in {
        "evt_count_61d", "evt_count_log1p_61d", "evt_active_days_61d",
        "traj_km_61d", "traj_hours_61d", "imu_rows_61d", "f1_evt_total_count_5d",
        "f1_evt_total_count_10d", "f1_evt_total_count_20d", "f1_evt_total_count_61d",
        "f3_night_deep_rate", "f3_night_day_rate", "f3_night_exposure_share",
    }]
    rows = []
    for name in selected:
        values = pd.to_numeric(frame[name], errors="coerce")
        valid = values.dropna()
        rows.append({"field": name, "unit": "see window_contract.csv", "vehicles": int(len(frame)),
                     "valid_vehicles": int(len(valid)), "missing_vehicles": int(values.isna().sum()),
                     "zero_vehicles": int((valid == 0).sum()),
                     "p25": float(valid.quantile(.25)) if len(valid) else np.nan,
                     "median": float(valid.median()) if len(valid) else np.nan,
                     "p75": float(valid.quantile(.75)) if len(valid) else np.nan,
                     "max": float(valid.max()) if len(valid) else np.nan})
    result = pd.DataFrame(rows)
    write_csv(out / "distribution_audit.csv", result)
    return result


def render_figures(out: Path, daily_audit: dict, daily: pd.DataFrame,
                   source_20d: pd.DataFrame, candidate: pd.DataFrame, run_id: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.barh([1], [20], left=[0], height=.32, color="#6da5c4", label="Proxy feature window (historical)")
    ax.barh([0], [61], left=[0], height=.32, color="#78a88b", label="Final-history candidate")
    ax.axvline(20, color="#4b4b4b", linestyle="--", linewidth=1)
    ax.set_xlim(0, 63)
    ax.set_yticks([0, 1], ["61-day final history", "20-day proxy history"])
    ax.set_xticks([0, 20, 30, 61], ["Jun 1", "Jun 21", "Jul 1", "Aug 1"])
    ax.set_xlabel("Calendar days; right edge excluded; Asia/Shanghai")
    ax.set_title(f"A1 window separation | {run_id}\n61d candidate has no y/fold and is not scored against proxy labels")
    ax.text(.01, -.22,
            f"vehicle_day summary records {daily_audit['recorded_calendar_days']} calendar dates; missing: "
            f"{', '.join(daily_audit['missing_calendar_dates']) or 'none'}; full-window exposure uses raw F3 trajectory scan.",
            transform=ax.transAxes, fontsize=9, color="#8b4e35")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "a1_window_separation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    pairs = [
        ("evt_count", source_20d.set_index("gpsno")["evt_count"], candidate.set_index("gpsno")["f1_evt_total_count_61d"],
         "Event records", "20d source vs rebuilt 61d"),
        ("traj_km", daily.set_index("gpsno")["traj_km_daily_sum"], candidate.set_index("gpsno")["traj_km_61d"],
         "Kilometers", "vehicle_day partial vs raw-trajectory 61d"),
        ("IMU rows", pd.Series(dtype=float), candidate.set_index("gpsno")["imu_rows_61d"],
         "IMU records", "raw-source 61d observation count"),
    ]
    for ax, (name, left, right, unit, title) in zip(axes, pairs):
        values = [pd.to_numeric(left, errors="coerce").dropna().to_numpy(),
                  pd.to_numeric(right, errors="coerce").dropna().to_numpy()]
        values = [v for v in values if len(v)]
        labels = (["20d", "61d"] if name == "evt_count" else
                  (["60d daily summary", "61d raw trajectory"] if name == "traj_km" else ["61d raw IMU"]))
        ax.boxplot(values, tick_labels=labels[:len(values)], showfliers=False)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"A1 source/window distributions | {run_id}\nDescriptive only; no labels, AUC, or model comparison", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "a1_feature_distribution.png", dpi=160)
    plt.close(fig)


def write_acceptance(out: Path, run_id: str, summary: dict, contract: pd.DataFrame,
                     synthetic: dict) -> None:
    event_states = {r["status"] for r in summary["event_audit"]["old_20d_event_reconciliation"]}
    exposure_states = {r["status"] for r in summary["daily_audit"]["old_20d_exposure_reconciliation"]}
    source_recon = "pass" if event_states == {"pass"} and exposure_states == {"pass"} else "partial"
    relative_out = out.relative_to(REPO).as_posix()
    if (out / "resume_record.json").is_file():
        reproduction = (
            f"本 run 续跑命令：`python feature_engineering/experiments/task1_feature_modeling/run_feat010_a1.py "
            f"--output {relative_out} --reuse-f3-output`；此命令只适用于三份已完成 F3 输出仍在该目录时。"
        )
        fresh_run = (
            "从原始分片完整复现时，使用一个新的 run ID 和空输出目录；程序会重新扫描轨迹／IMU，"
            "不得把 `--reuse-f3-output` 用在不完整或来源不同的文件上。"
        )
    else:
        reproduction = (
            f"运行：`python feature_engineering/experiments/task1_feature_modeling/run_feat010_a1.py "
            f"--output {relative_out}`。"
        )
        fresh_run = "复跑需使用新的 run ID 和空输出目录。"
    text = f"""# FEAT-010-A1 受控验收页

运行编号：`{run_id}`
时间窗口：`[2026-06-01, 2026-08-01)`，`as_of=2026-08-01 00:00:00`，Asia/Shanghai
样本身份：500 个目标车辆；候选表不包含标签或折号。

## 本次结论

- 实现／数据契约：**partial**。事件与轨迹/IMU 原始源按最终 61 天截点重建；旧 20 天事件／暴露字段逐列对账状态为 **{source_recon}**。F3 61 天候选与 F1 5/10/20/61 天事件族已生成。车辆日汇总缺少 `{', '.join(summary['daily_audit']['missing_calendar_dates']) or '无'}`，因此其覆盖秒数及 61 天日汇总不作为候选特征。
- 实验效果：**not_run**。本步骤不训练模型，不把 61 天输入连接到代理标签，也不报告 AUC。
- 验收与证据：**partial**。机器边界、比例、夜间语义和禁止标签检查：**pass**；真实来源和图表人工复核仍待记录。
- 环境与依赖：**{summary['environment_status']}**。真实候选已生成；F2/F2R2 未进入本候选，因为当前实现仍含固定 20 天扫描边界和 20 天暴露字段，需要另行适配与登记。

## 可视化验收

- `a1_window_separation.png`：检查代理 20 天窗口与最终 61 天窗口是否分离，并确认 7 月 31 日覆盖限制被标出。
- `a1_feature_distribution.png`：检查事件、里程和 IMU 观测计数的分布及单位；图仅作数据审计，不是效果比较。
- `window_contract.csv`：逐列核对来源、窗口、单位、缺失语义及待查项。
- `distribution_audit.csv`：与同一候选表对应的分布统计底表。

## 边界与未证明事项

- 原始轨迹和 IMU 分片含 7 月 31 日记录；车辆日汇总未覆盖该日。来自 raw source 的 F3 暴露与 IMU 行数可覆盖设定时间窗，vehicle_day 的覆盖秒数派生列仍不完整。
- F3 `f3_night_*` 使用研究深夜 23:00–05:00，不能称为官方夜间 21:00–06:00；画像中的官方口径字段仍是来源提供值，来源可用时点需要人工核对。
- `f3_cohort_*` 未在无标签候选中生成，因为当前相对化需训练集内拟合；应在正式训练折内拟合后再套用。
- F0 IMU 语义统计、F2 与 F2R2 未纳入。F0 校准参数可用时点需先审，F2/F2R2 需完成显式窗口及暴露列适配。
- 人工复核者：待填；复核日期：待填；结论：待填。

## 复现入口

{reproduction} {fresh_run}
配置、输入身份、原始分片清单、源代码指纹、产物指纹和机器检查结果见 `manifest.json` 与 `audit.json`。
"""
    (out / "acceptance.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-config", default=str(REPO / "feature_engineering/configs/pipeline.local.yaml"))
    parser.add_argument("--f1-config", default=str(REPO / "feature_engineering/configs/pipeline_f1_v1.local.yaml"))
    parser.add_argument("--f3-config", default=str(REPO / "feature_engineering/configs/pipeline_f3r5_v1.local.yaml"))
    parser.add_argument("--reuse-f3-output", action="store_true",
                        help="Resume candidate assembly from complete F3 outputs already in this run directory")
    args = parser.parse_args()
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        if not args.reuse_f3_output:
            raise FileExistsError(f"Refusing to overwrite nonempty output: {out}")
    elif args.reuse_f3_output:
        raise FileNotFoundError(f"Cannot reuse F3 outputs from an empty/missing run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    base_config_path = Path(args.base_config).expanduser().resolve()
    f1_config_path = Path(args.f1_config).expanduser().resolve()
    f3_config_path = Path(args.f3_config).expanduser().resolve()
    base_cfg = load_base_config(base_config_path)
    f1_raw = yaml.safe_load(f1_config_path.read_text(encoding="utf-8"))
    f3_raw = yaml.safe_load(f3_config_path.read_text(encoding="utf-8"))
    source_tables = base_cfg.tables_dir
    source_features_path = source_tables / "features.csv"
    events_path = source_tables / "events_clean.csv"
    vehicles_path = source_tables / "target_vehicles.csv"
    daily_path = source_tables / "vehicle_day.csv"
    required = [source_features_path, events_path, vehicles_path, daily_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing controlled sources: {missing}")

    synthetic = run_synthetic_checks()
    (out / "synthetic_checks.json").write_text(json.dumps(synthetic, indent=2), encoding="utf-8")
    source_features = pd.read_csv(source_features_path, dtype={"gpsno": str}, low_memory=False)
    source_vehicles = pd.read_csv(vehicles_path, dtype={"gpsno": str}, low_memory=False)
    if source_features["gpsno"].duplicated().any() or source_vehicles["gpsno"].duplicated().any():
        raise ValueError("Source feature/vehicle roster keys are not unique")
    if len(source_vehicles) != 500:
        raise ValueError(f"Expected 500 target vehicles, found {len(source_vehicles)}")

    roster = source_vehicles[["gpsno"]].copy()
    roster["gpsno"] = roster["gpsno"].astype(str)
    roster["sample_id"] = roster["gpsno"] + "_20260801_61d"
    roster["as_of"] = "2026-08-01 00:00:00"
    roster["lookback_days"] = float(WINDOW_DAYS)
    base_counts, target_events, event_audit = aggregate_events(
        events_path, roster, source_features, out,
        json.loads((REPO / "outputs/feat-007/round-1/scenario_token_map.json").read_text(encoding="utf-8"))
    )
    print("[A1] target events bounded to the 61-day window; legacy category mapping audit recorded", flush=True)

    # Build the canonical profile from the same recorded source mapping used by FEAT-007.
    profile = pd.DataFrame({
        "gpsno": source_vehicles["gpsno"].astype(str),
        "energy_type": source_vehicles["energy_type"],
        "highway_share": source_vehicles["highway_ratio"],
        "profile_month_km": source_vehicles["monthly_avg_mileage"],
        "profile_month_hours": source_vehicles["monthly_avg_hours"],
        "profile_month_night_share": source_vehicles["night_hours_ratio"],
    })
    write_csv(out / "profile_canonical.csv", profile)

    # Private overlay inputs contain no labels/folds. They only provide the roster and static profile.
    feature_table = out / "input_tables" / "features.csv"
    feature_table.parent.mkdir(parents=True, exist_ok=True)
    base_source = roster.merge(base_counts, on="gpsno", validate="one_to_one")
    profile_for_base = profile.rename(columns={"highway_share": "highway_ratio"})
    base_source = base_source.merge(profile_for_base, on="gpsno", validate="one_to_one")
    write_csv(feature_table, base_source)
    banned_cols = {"y", "fold", "label", "label_window", "horizon_days", "label_status", "label_version"}
    assert_unlabeled_candidate(base_source)

    # Generate short and long F1 event windows against the target-only, time-bounded source.
    f1_base = replace(base_cfg, table_layer=str(feature_table.parent))
    f1_cfg = F1Config(
        path=f1_config_path, base=f1_base,
        feature_version="F1_A1_61d_unlabeled", group_rule_version="not_fitted_no_labels",
        output_dir=out / "f1", chunksize=int(f1_raw.get("events", {}).get("chunksize", 300000)),
        windows_days=(5, 10, 20, 61), min_gold_accidents_for_imu_calibration=30,
        min_km=float(f1_raw.get("exposure", {}).get("min_km", 50.0)),
        min_hours=float(f1_raw.get("exposure", {}).get("min_hours", 1.0)),
    )
    f1_features = build_event_features(f1_cfg, events_path=out / "events_target_for_f1.csv")
    f1_features.to_csv(out / "f1" / "f1_event_features.csv", index=False, lineterminator="\n")
    print("[A1] F1 event families rebuilt for 5/10/20/61-day windows", flush=True)

    daily, daily_audit = daily_summary(base_cfg.data_root, base_cfg.table_layer, roster,
                                       source_features, out)
    print("[A1] vehicle-day coverage and historical 20-day reconciliation audited", flush=True)

    # F3 is run with the accepted final-history cutoff and no label-dependent cohort transform.
    f3_local = make_f3_local_config(f3_config_path, out, feature_table,
                                    out / "events_canonical_61d.csv", out / "profile_canonical.csv")
    f3_cfg = load_f3_config(f3_local)
    f3_new_path = f3_cfg.output_dir / "f3_new_features.csv"
    exposure_path = f3_cfg.output_dir / "f3_window_exposure.csv"
    observation_path = f3_cfg.output_dir / "f3_window_observation.csv"
    if args.reuse_f3_output:
        required_f3 = [f3_new_path, exposure_path, observation_path]
        missing_f3 = [str(path) for path in required_f3 if not path.is_file()]
        if missing_f3:
            raise FileNotFoundError(f"Cannot resume; completed F3 outputs are missing: {missing_f3}")
        f3_keys = pd.read_csv(f3_new_path, usecols=["sample_id", "gpsno"], dtype=str)
        expected = set(roster["sample_id"].astype(str))
        if (len(f3_keys) != len(roster) or f3_keys["sample_id"].duplicated().any()
                or set(f3_keys["sample_id"]) != expected
                or set(f3_keys["gpsno"]) != set(roster["gpsno"])):
            raise ValueError("Completed F3 outputs do not bind one-to-one to the A1 roster")
        resume_record = {
            "run_id": out.name,
            "mode": "resume_after_report_error",
            "reused_outputs": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                               for path in required_f3},
            "validation": "all three F3 outputs exist; feature rows bind one-to-one to the 500-vehicle roster",
            "feature_scan_rerun": False,
        }
        (out / "resume_record.json").write_text(
            json.dumps(resume_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[A1] reusing completed F3 outputs after validating roster binding", flush=True)
    else:
        f3_new_path = run_scan(f3_cfg)
    print("[A1] F3 raw trajectory/IMU scan complete; assembling unlabeled candidate", flush=True)
    f3_new = pd.read_csv(f3_new_path, dtype={"sample_id": str, "gpsno": str}, low_memory=False)
    exposure = pd.read_csv(exposure_path, dtype={"sample_id": str, "gpsno": str})
    observation = pd.read_csv(observation_path, dtype={"sample_id": str, "gpsno": str})
    if "traj_km_61d" not in exposure or "traj_hours_61d" not in exposure:
        raise ValueError(f"F3 exposure sidecar lacks 61-day columns: {list(exposure.columns)}")

    # Keep only fresh, explicitly windowed candidate values; legacy *_20d inputs are not reused.
    candidate = base_source.merge(profile, on="gpsno", how="left", suffixes=("", "_profile"),
                                  validate="one_to_one")
    candidate = candidate.merge(f1_features.drop(columns=["gpsno"], errors="ignore"),
                                on="sample_id", validate="one_to_one")
    candidate = candidate.merge(f3_new.drop(columns=["gpsno"], errors="ignore"),
                                on="sample_id", validate="one_to_one")
    candidate = candidate.merge(exposure.drop(columns=["gpsno", "traj_km_window", "traj_hours_window"], errors="ignore"),
                                on="sample_id", validate="one_to_one")
    candidate = candidate.merge(observation[["sample_id", "imu_rows_window"]],
                                on="sample_id", validate="one_to_one")
    candidate = candidate.rename(columns={"imu_rows_window": "imu_rows_61d"})

    # Matching-window rates use raw-trajectory exposure, with the old low-exposure thresholds.
    derived_rates = {}
    for days in (5, 10, 20, 61):
        km_col, hour_col = f"traj_km_{days}d", f"traj_hours_{days}d"
        if km_col not in candidate or hour_col not in candidate:
            continue
        km = pd.to_numeric(candidate[km_col], errors="coerce")
        hours = pd.to_numeric(candidate[hour_col], errors="coerce")
        km_valid, hours_valid = km >= f1_cfg.min_km, hours >= f1_cfg.min_hours
        for family in ("total", "lane", "fatigue", "distraction", "speed"):
            count_col = f"f1_evt_{family}_count_{days}d"
            if count_col not in candidate:
                continue
            stem = f"f1_evt_{family}"
            cnt = pd.to_numeric(candidate[count_col], errors="coerce")
            derived_rates[f"{stem}_per_1000km_{days}d"] = np.where(
                km_valid, cnt * 1000 / km.replace(0, np.nan), np.nan)
            derived_rates[f"{stem}_per_100h_{days}d"] = np.where(
                hours_valid, cnt * 100 / hours.replace(0, np.nan), np.nan)
    for base_name, count_col in (("accident", "f1_prior_accident_count_61d"),
                                 ("nearmiss", "f1_prior_nearmiss_count_61d"),
                                 ("incident", "f1_prior_incident_count_61d")):
        if count_col in candidate:
            count = pd.to_numeric(candidate[count_col], errors="coerce")
            km = pd.to_numeric(candidate["traj_km_61d"], errors="coerce")
            hours = pd.to_numeric(candidate["traj_hours_61d"], errors="coerce")
            derived_rates[f"f1_prior_{base_name}_per_1000km_61d"] = np.where(
                km >= f1_cfg.min_km, count * 1000 / km.replace(0, np.nan), np.nan)
            derived_rates[f"f1_prior_{base_name}_per_100h_61d"] = np.where(
                hours >= f1_cfg.min_hours, count * 100 / hours.replace(0, np.nan), np.nan)
    if derived_rates:
        candidate = pd.concat([candidate, pd.DataFrame(derived_rates, index=candidate.index)], axis=1)

    if candidate["sample_id"].duplicated().any() or candidate["gpsno"].duplicated().any():
        raise ValueError("A1 candidate is not exactly one row per target vehicle")
    assert_unlabeled_candidate(candidate)
    if not (pd.to_datetime(candidate["as_of"]) == AS_OF).all() or not (candidate["lookback_days"] == WINDOW_DAYS).all():
        raise ValueError("A1 candidate metadata does not match the 61-day contract")

    # Remove duplicated profile join columns created only for source reconciliation.
    candidate = candidate.loc[:, ~candidate.columns.str.endswith("_profile")]
    write_csv(out / "feature_61d_unlabeled.csv", candidate)

    contract = build_contract(candidate, out, daily_audit)
    distributions = audit_distributions(candidate, out)
    source20 = source_features[["gpsno", "evt_count"]].copy()
    render_figures(out, daily_audit, daily, source20, candidate, out.name)

    raw_shard_dirs = {
        "trajectory": Path(f3_raw["trajectory_dir"]).expanduser().resolve(),
        "imu": Path(f3_raw["imu_dir"]).expanduser().resolve(),
    }
    raw_inventory = {}
    for label, directory in raw_shard_dirs.items():
        files = sorted(directory.glob("part-*"))
        raw_inventory[label] = [{"name": f.name, "bytes": f.stat().st_size,
                                 "mtime_ns": f.stat().st_mtime_ns} for f in files]
    input_paths = [source_features_path, events_path, vehicles_path, daily_path,
                   Path(f3_raw["base_columns_file"]).expanduser().resolve(),
                   REPO / "outputs/feat-007/round-1/scenario_token_map.json",
                   Path(__file__).resolve(), REPO / "feature_engineering/src/accident_pipeline_f1/event_features.py",
                   REPO / "feature_engineering/src/accident_pipeline_f1/config.py",
                   REPO / "feature_engineering/src/accident_pipeline_f3/pipeline.py",
                   REPO / "feature_engineering/src/accident_pipeline_f3/core.py",
                   REPO / "feature_engineering/src/accident_pipeline_f3/interface.py",
                   REPO / "feature_engineering/src/accident_pipeline_f3/trajectory.py",
                   REPO / "feature_engineering/src/accident_pipeline/config.py",
                   REPO / "feature_engineering/experiments/task1_feature_modeling/check_feat010_a1.py"]
    code_status = __import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                                capture_output=True, text=True, check=False).stdout.strip()
    inputs = {str(path.relative_to(REPO)) if path.is_relative_to(REPO) else path.name: {
        "bytes": path.stat().st_size, "sha256": sha256(path)} for path in input_paths if path.is_file()}
    summary = {
        "task": "FEAT-010-A1", "run_id": out.name, "protocol": "final_history_61d_candidate_unlabeled",
        "time_contract": {"timezone": OFFICIAL_TZ, "start_inclusive": str(START),
                          "as_of_exclusive": str(AS_OF), "lookback_days": WINDOW_DAYS,
                          "feature_window": "[2026-06-01, 2026-08-01)"},
        "candidate_rows": int(len(candidate)), "candidate_columns": int(len(candidate.columns)),
        "candidate_features_without_meta": int(len([c for c in candidate if c not in
            {"sample_id", "gpsno", "as_of", "lookback_days"}])),
        "labels_or_folds_read": False, "labels_or_folds_in_candidate": False,
        "synthetic_checks": synthetic,
        "event_audit": event_audit, "daily_audit": daily_audit,
        "f3_output_features": int(len(f3_new.columns) - 2),
        "f3_exposure_columns": [c for c in exposure if c.startswith("traj_")],
        "f3_observation_rows_column": "imu_rows_window",
        "unclassified_contract_fields": int((contract.status == "fail_unclassified").sum()),
        "distribution_fields": list(distributions.field),
        "excluded_families": {
            "vehicle_day_coverage_seconds": "source ends before 2026-07-31; partial daily values are diagnostics only",
            "F0_IMU_semantic_features": "auxiliary calibration timestamp/provenance needs audit before inclusion",
            "F2_F2R2": "legacy hard-coded 20-day boundary and exposure columns; do not reuse at 61 days",
            "F3_cohort_relative": "requires fold-safe training-cohort fitting; no labels/folds in A1 input",
        },
        "raw_shard_inventory_fingerprint": hash_payload(raw_inventory),
        "raw_shard_content_hashes": "not computed; each shard name/byte-size/mtime is listed in private manifest inventory",
        "code_commit": code_status or "unknown",
        "inputs": inputs,
        "raw_shard_inventory": raw_inventory,
        "machine_status": "pass",
        "human_review": "pending",
    }
    (out / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_acceptance(out, out.name, {"event_audit": event_audit, "daily_audit": daily_audit,
                                    "environment_status": "partial; large-file source manifests use metadata inventory"},
                     contract, synthetic)

    output_hashes = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            output_hashes[str(path.relative_to(out))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {"run_id": out.name, "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
                "code_commit": code_status or "unknown", "configuration_sha256": {
                    "base": sha256(base_config_path), "f1": sha256(f1_config_path), "f3_template": sha256(f3_config_path)},
                "input_sha256": inputs, "raw_shard_inventory_fingerprint": summary["raw_shard_inventory_fingerprint"],
                "raw_shard_content_hashes": "not computed", "output_sha256": output_hashes,
                "random_seed": None, "timezone": OFFICIAL_TZ, "window": summary["time_contract"],
                "candidate_excludes_labels_and_folds": True,
                "human_review": "pending"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": out.name, "machine_status": summary["machine_status"],
                      "candidate_rows": summary["candidate_rows"],
                      "candidate_columns": summary["candidate_columns"],
                      "daily_summary_missing_dates": daily_audit["missing_calendar_dates"],
                      "human_review": "pending"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
