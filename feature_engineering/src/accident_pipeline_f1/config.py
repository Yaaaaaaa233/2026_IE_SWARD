from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from accident_pipeline.config import PipelineConfig, load_config


@dataclass(frozen=True)
class F1Config:
    path: Path
    base: PipelineConfig
    feature_version: str
    group_rule_version: str
    output_dir: Path
    chunksize: int
    windows_days: tuple[int, ...]
    min_gold_accidents_for_imu_calibration: int
    min_km: float
    min_hours: float


def load_f1_config(path: str | Path) -> F1Config:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    base_path = Path(raw["base_config"])
    if not base_path.is_absolute():
        base_path = source.parent / base_path
    output = Path(raw["output_dir"])
    if not output.is_absolute():
        output = source.parent / output
    return F1Config(
        path=source,
        base=load_config(base_path),
        feature_version=raw.get("feature_version", "F1_v1"),
        group_rule_version=raw.get("group_rule_version", "F1_v1_group_v1"),
        output_dir=output.resolve(),
        chunksize=int(raw.get("events", {}).get("chunksize", 300000)),
        windows_days=tuple(int(x) for x in raw.get("events", {}).get("windows_days", [5, 10, 20])),
        min_gold_accidents_for_imu_calibration=int(
            raw.get("events", {}).get("min_gold_accidents_for_imu_calibration", 30)
        ),
        min_km=float(raw.get("exposure", {}).get("min_km", 50.0)),
        min_hours=float(raw.get("exposure", {}).get("min_hours", 1.0)),
    )
