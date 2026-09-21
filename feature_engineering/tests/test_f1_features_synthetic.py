from __future__ import annotations

from pathlib import Path

import pandas as pd

from accident_pipeline.config import PipelineConfig
from accident_pipeline_f1.config import F1Config
from accident_pipeline_f1.event_features import build_event_features


def test_event_windows_and_label_window_exclusion(tmp_path: Path) -> None:
    root = tmp_path / "data"
    tables = root / "v4_assessed"
    tables.mkdir(parents=True)
    pd.DataFrame({
        "sample_id": ["1_20260621_20d", "2_20260621_20d"],
        "gpsno": [1, 2], "as_of": ["2026-06-21", "2026-06-21"], "lookback_days": [20, 20],
    }).to_csv(tables / "features.csv", index=False)
    pd.DataFrame([
        (1, 11803, "2026-06-20 12:00:00"),
        (1, 11804, "2026-06-12 12:00:00"),
        (1, 30002, "2026-06-19 12:00:00"),
        (1, 11803, "2026-06-21 00:00:00"),  # boundary and label-window event: excluded
        (1, 11804, "2026-07-01 00:00:00"),  # future: excluded
        (2, 41002, "2026-06-02 00:00:00"),
    ], columns=["gpsno", "event_type", "start_time"]).to_csv(tables / "events_clean.csv", index=False)
    base = PipelineConfig(workspace=tmp_path, data_root=root, output_dir=tmp_path / "base_artifacts")
    config = F1Config(
        path=tmp_path / "f1.yaml", base=base, feature_version="F1_v1",
        group_rule_version="test", output_dir=tmp_path / "f1_artifacts", chunksize=2,
        windows_days=(5, 10, 20), min_gold_accidents_for_imu_calibration=30,
        min_km=50.0, min_hours=1.0,
    )
    result = build_event_features(config)
    first = result[result.gpsno == 1].iloc[0]
    assert first.f1_prior_accident_count_20d == 1
    assert first.f1_prior_nearmiss_count_20d == 1
    assert first.f1_evt_lane_count_5d == 1
    assert first.f1_evt_total_count_20d == 3
    assert first.f1_days_since_last_accident == 0.5
    second = result[result.gpsno == 2].iloc[0]
    assert second.f1_never_incident_20d == 1
    assert pd.isna(second.f1_days_since_last_incident)
