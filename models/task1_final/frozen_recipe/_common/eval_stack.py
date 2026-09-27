"""Protocol loader needed by the frozen g8 feature builders.

The transferred evaluation stack also contained exploratory model selection and
private score anchors.  Final inference only needs this data-loading subset.
The caller sets ``PROTOCOL_DIR`` to the controlled protocol directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = str(Path(__file__).resolve().parents[4])
PROTOCOL_DIR: str | None = None
BLOCKED = {
    "monthly_avg_mileage", "monthly_avg_hours", "monthly_avg_stops", "highway_ratio",
    "morning_ratio", "dusk_ratio", "night_hours_ratio", "night_mileage_ratio",
    "energy_type", "f2_prior_incident_x_night", "f2_prior_incident_x_highway",
}
META = {
    "sample_id", "gpsno", "y", "fold", "group", "as_of", "lookback_days",
    "label_window", "horizon_days", "label_version", "split_version", "label_status",
    "feature_version", "source_version",
}


def load_protocol(window: str = "20_40") -> tuple[pd.DataFrame, pd.DataFrame]:
    if PROTOCOL_DIR is None:
        raise ValueError("set PROTOCOL_DIR to the controlled protocol directory")
    files = {"20_40": "labels.csv", "40_20": "labels_40_20.csv"}
    if window not in files:
        raise ValueError(f"unknown protocol window: {window}")
    root = Path(PROTOCOL_DIR)
    labels = pd.read_csv(root / files[window], dtype={"sample_id": str})
    splits = pd.read_csv(root / "splits.csv", dtype={"gpsno": str})
    if len(labels) != len(splits) or labels.sample_id.duplicated().any() or splits.gpsno.duplicated().any():
        raise ValueError("protocol labels and vehicle splits do not match")
    return labels, splits


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in META and column not in BLOCKED]
