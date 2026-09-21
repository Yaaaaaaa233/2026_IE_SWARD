from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import F1Config


# Business-semantic mapping. It is deliberately independent of labels/AUC screening.
EVENT_FAMILIES: dict[str, set[int] | None] = {
    "total": None,
    "lane": {30002, 30003, 30017},
    "fatigue": {41001, 41002, 41029},
    "distraction": {41003, 41004, 41005, 41009, 41023},
    "speed": {11401, 11402, 11403, 11405, 11406},
    "collision_warning": {30000, 30005, 60292, 60294},
}
DEVICE_ISSUE_TYPES = {41006, 41021}
ACCIDENT_TYPE = 11803
NEARMISS_TYPE = 11804


def _date_ns(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series, errors="raise").to_numpy(dtype="datetime64[ns]").astype(np.int64)


def build_event_features(config: F1Config, events_path: Path | None = None) -> pd.DataFrame:
    samples = pd.read_csv(
        config.base.tables_dir / "features.csv",
        usecols=["sample_id", "gpsno", "as_of", "lookback_days"],
    ).sort_values("gpsno").reset_index(drop=True)
    if samples["gpsno"].duplicated().any():
        raise ValueError("F1_v1 currently requires one as_of window per gpsno")
    ids = samples["gpsno"].astype(np.int64).to_numpy()
    as_of_ns = _date_ns(samples["as_of"])
    n = len(samples)
    windows = tuple(sorted(set(config.windows_days)))
    counts = {
        (family, window): np.zeros(n, dtype=np.int64)
        for family in EVENT_FAMILIES for window in windows
    }
    device_issue_20d = np.zeros(n, dtype=np.int64)
    accident_count = np.zeros(n, dtype=np.int64)
    nearmiss_count = np.zeros(n, dtype=np.int64)
    last_accident_ns = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
    last_incident_ns = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
    rows_read = 0
    types_seen: Counter[int] = Counter()
    source = events_path or (config.base.tables_dir / "events_clean.csv")

    for chunk in pd.read_csv(source, usecols=["gpsno", "event_type", "start_time"], chunksize=config.chunksize):
        rows_read += len(chunk)
        gps = pd.to_numeric(chunk["gpsno"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        event_type = pd.to_numeric(chunk["event_type"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        event_ns = pd.to_datetime(chunk["start_time"], errors="coerce").to_numpy(dtype="datetime64[ns]").astype(np.int64)
        positions = np.searchsorted(ids, gps)
        safe = np.minimum(positions, n - 1)
        known = (positions < n) & (ids[safe] == gps) & (event_ns != np.iinfo(np.int64).min)
        if not known.any():
            continue
        idx = safe[known]
        et = event_type[known]
        ts = event_ns[known]
        delta_ns = as_of_ns[idx] - ts
        before_as_of = delta_ns > 0
        idx, et, ts, delta_ns = idx[before_as_of], et[before_as_of], ts[before_as_of], delta_ns[before_as_of]
        types_seen.update(et.tolist())

        for family, family_types in EVENT_FAMILIES.items():
            family_mask = np.ones(len(et), dtype=bool) if family_types is None else np.isin(et, list(family_types))
            for window in windows:
                within = family_mask & (delta_ns <= np.timedelta64(window, "D").astype("timedelta64[ns]").astype(np.int64))
                counts[(family, window)] += np.bincount(idx[within], minlength=n)

        within_20 = delta_ns <= np.timedelta64(max(windows), "D").astype("timedelta64[ns]").astype(np.int64)
        device = within_20 & np.isin(et, list(DEVICE_ISSUE_TYPES))
        device_issue_20d += np.bincount(idx[device], minlength=n)
        accident = within_20 & (et == ACCIDENT_TYPE)
        nearmiss = within_20 & (et == NEARMISS_TYPE)
        accident_count += np.bincount(idx[accident], minlength=n)
        nearmiss_count += np.bincount(idx[nearmiss], minlength=n)
        if accident.any():
            np.maximum.at(last_accident_ns, idx[accident], ts[accident])
        incident = accident | nearmiss
        if incident.any():
            np.maximum.at(last_incident_ns, idx[incident], ts[incident])

    result = samples[["sample_id", "gpsno"]].copy()
    for (family, window), value in counts.items():
        result[f"f1_evt_{family}_count_{window}d"] = value
    result["f1_evt_device_issue_count_20d"] = device_issue_20d
    result["f1_prior_accident_count_20d"] = accident_count
    result["f1_prior_nearmiss_count_20d"] = nearmiss_count
    result["f1_prior_incident_count_20d"] = accident_count + nearmiss_count
    result["f1_never_accident_20d"] = (accident_count == 0).astype(int)
    result["f1_never_incident_20d"] = ((accident_count + nearmiss_count) == 0).astype(int)

    day_ns = float(np.timedelta64(1, "D").astype("timedelta64[ns]").astype(np.int64))
    has_accident = last_accident_ns != np.iinfo(np.int64).min
    has_incident = last_incident_ns != np.iinfo(np.int64).min
    result["f1_days_since_last_accident"] = np.where(
        has_accident, (as_of_ns - last_accident_ns) / day_ns, np.nan
    )
    result["f1_days_since_last_incident"] = np.where(
        has_incident, (as_of_ns - last_incident_ns) / day_ns, np.nan
    )

    if 5 in windows and 20 in windows:
        for family in EVENT_FAMILIES:
            recent = result[f"f1_evt_{family}_count_5d"].to_numpy(dtype=float)
            baseline = result[f"f1_evt_{family}_count_20d"].to_numpy(dtype=float) / 4.0
            result[f"f1_evt_{family}_recent_5d_lift"] = (recent + 0.5) / (baseline + 0.5)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.output_dir / "f1_event_features.csv", index=False)
    gold_accident_rows = int(accident_count.sum())
    calibration_ready = gold_accident_rows >= config.min_gold_accidents_for_imu_calibration
    feasibility = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": config.feature_version,
        "events_rows_read": rows_read,
        "event_types_seen_before_as_of": {str(k): int(v) for k, v in sorted(types_seen.items())},
        "prior_accident_rows": gold_accident_rows,
        "prior_nearmiss_rows": int(nearmiss_count.sum()),
        "vehicles_with_prior_incident": int(((accident_count + nearmiss_count) > 0).sum()),
        "imu_threshold_calibration": {
            "status": "ready_for_fold_safe_calibration" if calibration_ready else "deferred_insufficient_gold_accidents",
            "minimum_gold_accidents": config.min_gold_accidents_for_imu_calibration,
            "reason": (
                "Verified pre-as_of accident timestamps meet the configured minimum."
                if calibration_ready
                else "Verified pre-as_of accident timestamps are below the configured minimum; label-window events cannot tune features."
            ),
            "minimum_recommendation": "Collect more timestamp-verified accidents or calibrate strictly inside training folds.",
        },
    }
    (config.output_dir / "imu_calibration_feasibility.json").write_text(
        json.dumps(feasibility, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
