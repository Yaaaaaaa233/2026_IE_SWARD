from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from accident_pipeline.dataset import DatasetBuilder, DatasetBundle

from .config import F1Config


EVENT_BASE_COUNTS = {
    "evt_count": "all events",
    "evt_lane_count": "existing lane/ADAS family",
    "evt_fatigue_count": "existing fatigue family",
    "evt_distraction_count": "existing distraction family",
    "evt_speed_count": "existing speeding family",
}
F1_COUNT_COLUMNS = [
    "f1_evt_collision_warning_count_20d",
    "f1_evt_device_issue_count_20d",
    "f1_prior_accident_count_20d",
    "f1_prior_nearmiss_count_20d",
    "f1_prior_incident_count_20d",
]
IMU_COUNT_COLUMNS = [
    "accident_impact_moderate_count",
    "accident_impact_severe_count",
    "accident_impact_extreme_count",
    "accident_rotation_severe_count",
    "accident_rotation_extreme_count",
    "accident_collision_candidate_count",
    "accident_rollover_candidate_count",
    "accident_any_signal_count",
]


def _normalize(frame: pd.DataFrame, count: str, denominator: str, scale: float, valid: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(valid, pd.to_numeric(frame[count], errors="coerce") * scale / frame[denominator], np.nan),
        index=frame.index,
    )


def _dictionary_entry(name: str, config: F1Config) -> dict[str, object]:
    if "per_1000km" in name:
        return {
            "name": name, "source": "derived count + features.traj_km_20d", "window": "20d",
            "normalization": "count * 1000 / traj_km_20d",
            "missing_semantics": f"missing when trajectory exposure < {config.min_km:g} km",
            "threshold_source": "F1 requirement, configurable",
        }
    if "per_100h" in name:
        return {
            "name": name, "source": "derived count + features.traj_hours_20d", "window": "20d",
            "normalization": "count * 100 / traj_hours_20d",
            "missing_semantics": f"missing when trajectory exposure < {config.min_hours:g} hour",
            "threshold_source": "F1 requirement, configurable",
        }
    if "days_since" in name:
        return {
            "name": name, "source": "v4_assessed/events_clean.csv", "window": "[as_of-20d, as_of)",
            "normalization": "none", "missing_semantics": "no matching prior event; paired with never_* flag",
            "threshold_source": "official event type 11803/11804",
        }
    if "recent_5d_lift" in name:
        return {
            "name": name, "source": "v4_assessed/events_clean.csv", "window": "5d versus 20d",
            "normalization": "(count_5d+0.5)/(count_20d/4+0.5)",
            "missing_semantics": "never missing; additive smoothing avoids division by zero",
            "threshold_source": "F1_v1 engineering definition",
        }
    if name.startswith("f1_evt_") or name.startswith("f1_prior_") or name.startswith("f1_never_"):
        return {
            "name": name, "source": "v4_assessed/events_clean.csv", "window": "encoded in column name",
            "normalization": "raw count or indicator", "missing_semantics": "zero means observed no matching event",
            "threshold_source": "official event type business mapping; no label-based screening",
        }
    if name in {"f1_low_km_exposure", "f1_low_hour_exposure"}:
        return {
            "name": name, "source": "v4_assessed/features.csv", "window": "20d",
            "normalization": "binary indicator", "missing_semantics": "1 means normalized features intentionally missing",
            "threshold_source": f"km={config.min_km:g}, hours={config.min_hours:g}",
        }
    return {
        "name": name, "source": "F0 IMU feature + trajectory exposure", "window": "20d",
        "normalization": "see column suffix", "missing_semantics": "missing for inadequate exposure",
        "threshold_source": "F0 engineering threshold; not recalibrated in F1_v1",
    }


def build_f1_interface(config: F1Config) -> DatasetBundle:
    event_path = config.output_dir / "f1_event_features.csv"
    if not event_path.exists():
        raise FileNotFoundError(f"Run build-features first: {event_path}")
    base = DatasetBuilder(config.base).build(require_imu=True)
    event_features = pd.read_csv(event_path)
    if not event_features["sample_id"].is_unique:
        raise ValueError("F1 event features contain duplicate sample_id")
    if set(event_features["sample_id"]) != set(base.frame["sample_id"]):
        raise ValueError("F1 event features do not match the F0 sample_id set")
    frame = base.frame.merge(
        event_features.drop(columns=["gpsno"], errors="ignore"), on="sample_id", how="left", validate="one_to_one"
    )
    f1_candidates = [column for column in event_features.columns if column not in {"sample_id", "gpsno"}]

    km_valid = pd.to_numeric(frame["traj_km_20d"], errors="coerce") >= config.min_km
    hour_valid = pd.to_numeric(frame["traj_hours_20d"], errors="coerce") >= config.min_hours
    frame["f1_low_km_exposure"] = (~km_valid).astype(int)
    frame["f1_low_hour_exposure"] = (~hour_valid).astype(int)
    f1_candidates += ["f1_low_km_exposure", "f1_low_hour_exposure"]

    count_columns = list(EVENT_BASE_COUNTS) + [c for c in F1_COUNT_COLUMNS if c in frame]
    for column in count_columns:
        stem = column.removesuffix("_count").removesuffix("_20d")
        km_name = f"f1_{stem.removeprefix('f1_')}_per_1000km"
        hour_name = f"f1_{stem.removeprefix('f1_')}_per_100h"
        frame[km_name] = _normalize(frame, column, "traj_km_20d", 1000.0, km_valid)
        frame[hour_name] = _normalize(frame, column, "traj_hours_20d", 100.0, hour_valid)
        f1_candidates += [km_name, hour_name]

    for column in IMU_COUNT_COLUMNS:
        if column not in frame:
            continue
        stem = column.removesuffix("_count")
        name = f"f1_{stem}_per_1000km"
        frame[name] = _normalize(frame, column, "traj_km_20d", 1000.0, km_valid)
        f1_candidates.append(name)
    frame["f1_imu_rows_per_km"] = _normalize(frame, "imu_rows_window", "traj_km_20d", 1.0, km_valid)
    f1_candidates.append("f1_imu_rows_per_km")

    duplicate_of: dict[str, str] = {}
    comparison_columns = list(base.numeric_columns)
    unique_candidates: list[str] = []
    for candidate in dict.fromkeys(f1_candidates):
        left = pd.to_numeric(frame[candidate], errors="coerce").to_numpy(dtype=float)
        for existing in comparison_columns:
            right = pd.to_numeric(frame[existing], errors="coerce").to_numpy(dtype=float)
            if np.allclose(left, right, equal_nan=True, rtol=0, atol=1e-12):
                duplicate_of[candidate] = existing
                break
        if candidate not in duplicate_of:
            comparison_columns.append(candidate)
            unique_candidates.append(candidate)
    retained = []
    for column in unique_candidates:
        if column in duplicate_of or frame[column].isna().all() or frame[column].nunique(dropna=True) <= 1:
            continue
        retained.append(column)

    feature_columns = base.feature_columns + retained
    numeric = base.numeric_columns + retained
    bundle = DatasetBundle(frame=frame, feature_columns=feature_columns, numeric_columns=numeric, categorical_columns=base.categorical_columns)

    output = config.output_dir / "model_interface"
    output.mkdir(parents=True, exist_ok=True)
    identifiers = ["sample_id", "gpsno", "as_of", "lookback_days"]
    frame["feature_version_f1"] = config.feature_version
    frame[identifiers + feature_columns + ["y", "fold", "feature_version_f1"]].to_csv(
        output / "model_input.csv", index=False
    )
    dictionary = [_dictionary_entry(name, config) for name in retained]
    pd.DataFrame(dictionary).to_csv(output / "feature_dictionary_f1.csv", index=False)
    schema = {
        "contract_version": "F1.1",
        "feature_version": config.feature_version,
        "group_rule_version": config.group_rule_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parent_contract": str(config.base.output_dir / "model_interface" / "feature_schema.json"),
        "primary_key": "sample_id",
        "time_rule": "all event features use [as_of-L, as_of); label-window events are excluded",
        "imputation_rule": "raw values and missing indicators retained; imputation belongs inside model folds",
        "row_count": len(frame),
        "feature_count": len(feature_columns),
        "f0_feature_count": len(base.feature_columns),
        "f1_retained_feature_count": len(retained),
        "f1_features": retained,
        "numeric_features": numeric,
        "categorical_features": base.categorical_columns,
        "omitted_exact_duplicates": duplicate_of,
        "f1_missing_rate": {name: round(float(frame[name].isna().mean()), 6) for name in retained},
        "contract_checks": {
            "sample_id_unique": bool(frame["sample_id"].is_unique),
            "label_complete": bool(frame["y"].notna().all()),
            "fold_complete": bool(frame["fold"].notna().all()),
            "historical_incident_vehicle_count": int((frame["f1_prior_incident_count_20d"] > 0).sum()),
        },
        "field_dictionary": dictionary,
    }
    (output / "feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle
