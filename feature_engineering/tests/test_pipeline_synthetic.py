from __future__ import annotations

from pathlib import Path

import pandas as pd

from accident_pipeline.clustering import cluster_features
from accident_pipeline.config import ClusteringConfig, ImuConfig, PipelineConfig
from accident_pipeline.dataset import DatasetBuilder
from accident_pipeline.imu_features import extract_imu_features


def _fixture(tmp_path: Path) -> PipelineConfig:
    root = tmp_path / "contract"
    tables = root / "v4_assessed"
    imu_dir = root / "v1_annotated" / "imu_clean"
    tables.mkdir(parents=True)
    imu_dir.mkdir(parents=True)
    ids = list(range(101, 107))
    samples = pd.DataFrame({
        "sample_id": [f"{gps}_20260621_20d" for gps in ids],
        "gpsno": ids,
        "as_of": "2026-06-21",
        "lookback_days": 20,
        "evt_count": [1, 2, 4, 20, 22, 25],
        "traj_km_20d": [100, 120, 110, 500, 520, 550],
        "feature_version": "test",
        "source_version": "test",
    })
    samples.to_csv(tables / "features.csv", index=False)
    pd.DataFrame({
        "sample_id": samples["sample_id"], "label_window": "future", "horizon_days": 40,
        "y": [0, 0, 0, 1, 1, 1], "label_status": "complete",
        "label_version": "test", "source_version": "test",
    }).to_csv(tables / "labels.csv", index=False)
    pd.DataFrame({"gpsno": ids, "fold": [0, 1, 2, 0, 1, 2], "split_version": "test"}).to_csv(
        tables / "splits.csv", index=False
    )
    pd.DataFrame({
        "gpsno": ids, "energy_type": ["electric"] * 3 + ["diesel"] * 3,
        "monthly_avg_mileage": [100, 110, 105, 400, 420, 410], "source_version": "test",
    }).to_csv(tables / "target_vehicles.csv", index=False)
    pd.DataFrame({
        "gpsno": ids, "date": "2026-06-10", "evt_raw_count": 1, "evt_segments30": 1,
        "evt_days_flag": 1, "traj_km": 10, "traj_hours": 1,
        "traj_observed_seconds": 3600, "imu_observed_seconds": 3500,
        "imu_traj_overlap_seconds": 3400, "quality_flags": None, "source_version": "test",
    }).to_csv(tables / "vehicle_day.csv", index=False)
    pd.DataFrame({
        "gpsno": ids, "imu_rows": 3, "az_mean": 1.0, "az_std": 0.01,
        "gy_rows": 3, "gy_mean": 0.0, "gy_std": 0.1,
        "orientation_note": None, "source_version": "test",
    }).to_csv(root / "辅助_imu校准参数.csv", index=False)

    rows = []
    for offset, gps in enumerate(ids):
        for second in range(3):
            rows.append({
                "gpsno": gps, "imei": gps, "data_time": f"2026-06-10 00:00:0{second}",
                "data_date": 20260610, "ems_speed": 30.0, "gps_speed": 30.0,
                "ax": 0.05 + offset * 0.40, "ay": 0.0, "az": 1.0,
                "gx": 0.0, "gy": 0.0, "gz": float(offset * 20),
                "ems_flag": "ok", "gps_flag": "ok", "accel_flag": "ok",
                "date_flag": "ok", "window_flag": 1, "source_version": "test",
            })
        rows.append({**rows[-1], "data_time": "2026-07-01 00:00:00", "data_date": 20260701, "ax": 9.0})
    pd.DataFrame(rows).to_csv(imu_dir / "part-00000", sep="\t", index=False)
    return PipelineConfig(
        workspace=tmp_path, data_root=root, output_dir=tmp_path / "artifacts",
        imu=ImuConfig(block_size_mb=1),
        clustering=ClusteringConfig(min_clusters=2, max_clusters=3, random_state=7, n_init=5),
    )


def test_end_to_end_contract_and_clustering(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    imu = extract_imu_features(config)
    assert len(imu) == 6
    assert imu["imu_rows_window"].eq(3).all()
    assert imu.loc[imu["gpsno"] == 106, "accident_impact_severe_count"].iat[0] == 3

    builder = DatasetBuilder(config)
    bundle = builder.materialize()
    assert "y" not in bundle.feature_columns
    assert "fold" not in bundle.feature_columns
    assert "accident_sensor_risk_index" in bundle.feature_columns
    assert bundle.get_fold(0)[2].shape[0] == 2

    assignments = cluster_features(bundle, config)
    assert len(assignments) == 6
    assert assignments["cluster"].nunique() in {2, 3}
    assert (config.output_dir / "clustering" / "cluster_model.joblib").exists()
