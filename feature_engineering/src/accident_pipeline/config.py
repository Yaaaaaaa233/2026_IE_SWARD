from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ImuConfig:
    workers: int = 1
    block_size_mb: int = 64
    impact_moderate_g: float = 0.35
    impact_severe_g: float = 0.75
    impact_extreme_g: float = 1.50
    lateral_instability_g: float = 0.35
    tilt_deviation_g: float = 0.35
    rotation_severe_dps: float = 45.0
    rotation_extreme_dps: float = 90.0
    moving_speed_kmh: float = 10.0


@dataclass(frozen=True)
class ClusteringConfig:
    min_clusters: int = 2
    max_clusters: int = 8
    random_state: int = 42
    n_init: int = 30
    min_trajectory_km: float = 1.0
    min_imu_rows: int = 1000


@dataclass(frozen=True)
class PipelineConfig:
    workspace: Path
    data_root: Path
    table_layer: str = "v4_assessed"
    imu_layer: str = "v1_annotated"
    output_dir: Path = Path("artifacts")
    imu: ImuConfig = field(default_factory=ImuConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)

    @property
    def tables_dir(self) -> Path:
        return self.data_root / self.table_layer

    @property
    def imu_dir(self) -> Path:
        return self.data_root / self.imu_layer / "imu_clean"


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    workspace = config_path.parent.parent
    return PipelineConfig(
        workspace=workspace,
        data_root=_resolve(workspace, raw["data_root"]),
        table_layer=raw.get("table_layer", "v4_assessed"),
        imu_layer=raw.get("imu_layer", "v1_annotated"),
        output_dir=_resolve(workspace, raw.get("output_dir", "artifacts")),
        imu=ImuConfig(**raw.get("imu", {})),
        clustering=ClusteringConfig(**raw.get("clustering", {})),
    )
