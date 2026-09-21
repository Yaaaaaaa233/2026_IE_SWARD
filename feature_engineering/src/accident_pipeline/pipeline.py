from __future__ import annotations

from .clustering import cluster_features
from .config import PipelineConfig
from .dataset import DatasetBuilder
from .imu_features import extract_imu_features


def run_pipeline(config: PipelineConfig) -> None:
    extract_imu_features(config)
    builder = DatasetBuilder(config)
    bundle = builder.materialize()
    cluster_features(bundle, config)
