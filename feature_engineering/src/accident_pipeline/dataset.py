from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import PipelineConfig


IDENTIFIER_COLUMNS = {"sample_id", "gpsno", "as_of", "lookback_days"}
FORBIDDEN_FEATURE_COLUMNS = {
    "y", "fold", "label_window", "horizon_days", "label_status", "label_version",
    "split_version", "feature_version", "source_version",
}


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.feature_columns]

    @property
    def y(self) -> pd.Series:
        return self.frame["y"].astype(int)

    @property
    def folds(self) -> pd.Series:
        return self.frame["fold"].astype(int)

    def get_fold(self, validation_fold: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        validation = self.folds == validation_fold
        return self.X.loc[~validation], self.y.loc[~validation], self.X.loc[validation], self.y.loc[validation]

    def preprocessor(self) -> ColumnTransformer:
        transformers = []
        if self.numeric_columns:
            transformers.append((
                "numeric",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                self.numeric_columns,
            ))
        if self.categorical_columns:
            transformers.append((
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                self.categorical_columns,
            ))
        return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


class DatasetBuilder:
    """Build the single contract consumed by clustering and supervised models."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    @staticmethod
    def _assert_unique(table: pd.DataFrame, key: str, name: str) -> None:
        duplicates = int(table[key].duplicated().sum())
        if duplicates:
            raise ValueError(f"{name}.{key} has {duplicates} duplicate rows")

    def _vehicle_day_features(self, samples: pd.DataFrame) -> pd.DataFrame:
        daily = self._read(self.config.tables_dir / "vehicle_day.csv")
        daily["date"] = pd.to_datetime(daily["date"], errors="raise")
        windows = samples[["sample_id", "gpsno", "as_of", "lookback_days"]].copy()
        windows["as_of"] = pd.to_datetime(windows["as_of"], errors="raise")
        joined = daily.merge(windows, on="gpsno", how="inner", validate="many_to_one")
        keep = (joined["date"] < joined["as_of"]) & (
            joined["date"] >= joined["as_of"] - pd.to_timedelta(joined["lookback_days"], unit="D")
        )
        joined = joined.loc[keep]
        grouped = joined.groupby("sample_id", observed=True)
        output = grouped.agg(
            coverage_days=("date", "nunique"),
            coverage_traj_observed_seconds=("traj_observed_seconds", "sum"),
            coverage_imu_observed_seconds=("imu_observed_seconds", "sum"),
            coverage_imu_traj_overlap_seconds=("imu_traj_overlap_seconds", "sum"),
            coverage_quality_flag_days=("quality_flags", lambda x: int(x.notna().sum())),
        ).reset_index()
        output["coverage_imu_traj_overlap_ratio"] = np.divide(
            output["coverage_imu_traj_overlap_seconds"],
            output["coverage_traj_observed_seconds"],
            out=np.zeros(len(output)),
            where=output["coverage_traj_observed_seconds"].to_numpy() > 0,
        ).clip(0.0, 1.0)
        return output

    def build(self, require_imu: bool = True) -> DatasetBundle:
        tables = self.config.tables_dir
        base = self._read(tables / "features.csv")
        labels = self._read(tables / "labels.csv")
        splits = self._read(tables / "splits.csv")
        vehicles = self._read(tables / "target_vehicles.csv")
        for table, key, name in (
            (base, "sample_id", "features"), (labels, "sample_id", "labels"),
            (splits, "gpsno", "splits"), (vehicles, "gpsno", "target_vehicles"),
        ):
            self._assert_unique(table, key, name)

        expected = set(base["sample_id"])
        if set(labels["sample_id"]) != expected:
            raise ValueError("features and labels do not contain the same sample_id set")
        frame = base.merge(labels, on="sample_id", how="left", validate="one_to_one", suffixes=("", "_label"))
        frame = frame.merge(splits, on="gpsno", how="left", validate="many_to_one")
        frame = frame.merge(vehicles, on="gpsno", how="left", validate="many_to_one", suffixes=("", "_vehicle"))
        frame = frame.merge(self._vehicle_day_features(base), on="sample_id", how="left", validate="one_to_one")

        imu_path = self.config.output_dir / "imu_accident_features.csv"
        if imu_path.exists():
            imu = self._read(imu_path)
            self._assert_unique(imu, "sample_id", "imu_accident_features")
            if set(imu["sample_id"]) != expected:
                raise ValueError("IMU feature sample_id set does not match base features")
            frame = frame.merge(
                imu.drop(columns=["gpsno"], errors="ignore"), on="sample_id", how="left", validate="one_to_one"
            )
        elif require_imu:
            raise FileNotFoundError(f"Run extract-imu first: {imu_path}")

        if frame[["y", "fold"]].isna().any().any():
            raise ValueError("Model contract contains missing labels or folds")
        if not set(frame["y"].unique()).issubset({0, 1}):
            raise ValueError("Only binary y labels are supported by this contract")

        excluded = IDENTIFIER_COLUMNS | FORBIDDEN_FEATURE_COLUMNS
        excluded |= {column for column in frame if column.endswith("_version") or column.startswith("label_")}
        candidates = [column for column in frame.columns if column not in excluded]
        all_missing = [column for column in candidates if frame[column].isna().all()]
        constants = [column for column in candidates if frame[column].nunique(dropna=True) <= 1]
        feature_columns = [column for column in candidates if column not in set(all_missing + constants)]
        numeric = [column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])]
        categorical = [column for column in feature_columns if column not in numeric]
        return DatasetBundle(frame, feature_columns, numeric, categorical)

    def materialize(self, bundle: DatasetBundle | None = None) -> DatasetBundle:
        bundle = bundle or self.build()
        output = self.config.output_dir / "model_interface"
        output.mkdir(parents=True, exist_ok=True)
        columns = ["sample_id", "gpsno", "as_of", "lookback_days"] + bundle.feature_columns + ["y", "fold"]
        bundle.frame[columns].to_csv(output / "model_input.csv", index=False)
        schema = {
            "contract_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "primary_key": "sample_id",
            "entity_key": "gpsno",
            "time_rule": "all features are strictly earlier than as_of",
            "label_rule": "y and fold are metadata and never model features",
            "row_count": len(bundle.frame),
            "feature_count": len(bundle.feature_columns),
            "positive_count": int(bundle.y.sum()),
            "fold_counts": {str(k): int(v) for k, v in bundle.folds.value_counts().sort_index().items()},
            "numeric_features": bundle.numeric_columns,
            "categorical_features": bundle.categorical_columns,
            "feature_missing_rate": {
                column: round(float(bundle.frame[column].isna().mean()), 6) for column in bundle.feature_columns
            },
            "imu_samples_with_window_rows": (
                int((bundle.frame["imu_rows_window"] > 0).sum()) if "imu_rows_window" in bundle.frame else None
            ),
            "contract_checks": {
                "sample_id_unique": bool(bundle.frame["sample_id"].is_unique),
                "label_complete": bool(bundle.frame["y"].notna().all()),
                "fold_complete": bool(bundle.frame["fold"].notna().all()),
                "forbidden_columns_excluded": not bool(set(bundle.feature_columns) & FORBIDDEN_FEATURE_COLUMNS),
            },
        }
        (output / "feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
        return bundle
