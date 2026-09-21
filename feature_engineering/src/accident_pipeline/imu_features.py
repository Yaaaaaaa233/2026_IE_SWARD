from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from .config import ImuConfig, PipelineConfig


IMU_COLUMNS = ["gpsno", "data_date", "ems_speed", "gps_speed", "ax", "ay", "az", "gx", "gy", "gz"]


def _date_int(values: pd.Series) -> np.ndarray:
    return pd.to_datetime(values, errors="raise").dt.strftime("%Y%m%d").astype(np.int64).to_numpy()


def _hist_quantile(hist: np.ndarray, edges: np.ndarray, q: float) -> np.ndarray:
    result = np.full(hist.shape[0], np.nan, dtype=float)
    totals = hist.sum(axis=1)
    for row in np.flatnonzero(totals):
        target = max(1, int(np.ceil(totals[row] * q)))
        bin_index = int(np.searchsorted(np.cumsum(hist[row]), target, side="left"))
        result[row] = edges[min(bin_index + 1, len(edges) - 1)]
    return result

class _Accumulator:
    def __init__(self, samples: pd.DataFrame, calibration: pd.DataFrame, settings: ImuConfig):
        if samples["gpsno"].duplicated().any():
            raise ValueError("IMU extraction currently requires one sample window per gpsno")
        ordered = samples.sort_values("gpsno").reset_index(drop=True)
        self.samples = ordered
        self.ids = ordered["gpsno"].astype(np.int64).to_numpy()
        self.starts = _date_int(pd.to_datetime(ordered["as_of"]) - pd.to_timedelta(ordered["lookback_days"], unit="D"))
        self.ends = _date_int(ordered["as_of"])
        self.settings = settings
        n = len(ordered)

        cal = calibration.set_index("gpsno").reindex(self.ids)
        self.az_baseline = cal.get("az_mean", pd.Series(index=cal.index, dtype=float)).abs().fillna(1.0).to_numpy()
        # Acceleration is already expressed in g. Its vector norm is orientation invariant;
        # az alone is not when devices have different mounting orientations.
        self.gravity_norm = np.ones(n, dtype=float)
        self.orientation_reliable = self.az_baseline >= 0.7
        self.gy_bias = cal.get("gy_mean", pd.Series(index=cal.index, dtype=float)).fillna(0.0).to_numpy()

        self.rows = np.zeros(n, dtype=np.int64)
        self.accel_rows = np.zeros(n, dtype=np.int64)
        self.horizontal_rows = np.zeros(n, dtype=np.int64)
        self.gyro_rows = np.zeros(n, dtype=np.int64)
        self.sums = {name: np.zeros(n) for name in ("impact", "horizontal", "rotation")}
        self.sumsq = {name: np.zeros(n) for name in self.sums}
        self.maxima = {name: np.full(n, -np.inf) for name in self.sums}
        self.counts = {
            name: np.zeros(n, dtype=np.int64)
            for name in (
                "impact_moderate", "impact_severe", "impact_extreme", "lateral_instability",
                "tilt_anomaly", "rotation_severe", "rotation_extreme", "collision_candidate",
                "rollover_candidate", "any_signal",
            )
        }
        self.impact_edges = np.linspace(0.0, 3.0, 121)
        self.rotation_edges = np.linspace(0.0, 240.0, 121)
        self.impact_hist = np.zeros((n, len(self.impact_edges) - 1), dtype=np.int64)
        self.rotation_hist = np.zeros((n, len(self.rotation_edges) - 1), dtype=np.int64)
        self.rows_scanned = 0

    def merge(self, other: "_Accumulator") -> None:
        if not np.array_equal(self.ids, other.ids):
            raise ValueError("Cannot merge IMU accumulators with different sample keys")
        self.rows_scanned += other.rows_scanned
        for name in ("rows", "accel_rows", "horizontal_rows", "gyro_rows"):
            getattr(self, name)[:] += getattr(other, name)
        for name in self.sums:
            self.sums[name] += other.sums[name]
            self.sumsq[name] += other.sumsq[name]
            self.maxima[name] = np.maximum(self.maxima[name], other.maxima[name])
        for name in self.counts:
            self.counts[name] += other.counts[name]
        self.impact_hist += other.impact_hist
        self.rotation_hist += other.rotation_hist

    def _map_rows(self, gpsno: np.ndarray, dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.searchsorted(self.ids, gpsno)
        safe = np.minimum(positions, len(self.ids) - 1)
        keep = (positions < len(self.ids)) & (self.ids[safe] == gpsno)
        keep &= (dates >= self.starts[safe]) & (dates < self.ends[safe])
        return safe[keep], keep

    @staticmethod
    def _add_by_id(target: np.ndarray, ids: np.ndarray, values: np.ndarray | None = None) -> None:
        addition = np.bincount(ids, weights=values, minlength=len(target))
        target += addition.astype(target.dtype, copy=False)

    def _metric(self, name: str, ids: np.ndarray, values: np.ndarray) -> None:
        self._add_by_id(self.sums[name], ids, values)
        self._add_by_id(self.sumsq[name], ids, values * values)
        np.maximum.at(self.maxima[name], ids, values)

    @staticmethod
    def _add_hist(target: np.ndarray, ids: np.ndarray, values: np.ndarray, edges: np.ndarray) -> None:
        bins = np.searchsorted(edges, values, side="right") - 1
        bins = np.clip(bins, 0, target.shape[1] - 1)
        flat = ids * target.shape[1] + bins
        target += np.bincount(flat, minlength=target.size).reshape(target.shape)

    def update(self, columns: dict[str, np.ndarray]) -> None:
        gpsno = columns["gpsno"].astype(np.int64, copy=False)
        dates = columns["data_date"].astype(np.int64, copy=False)
        self.rows_scanned += len(gpsno)
        ids, keep = self._map_rows(gpsno, dates)
        if not keep.any():
            return
        self._add_by_id(self.rows, ids)
        values = {name: columns[name][keep].astype(float, copy=False) for name in IMU_COLUMNS[2:]}

        accel_ok = np.isfinite(values["ax"]) & np.isfinite(values["ay"]) & np.isfinite(values["az"])
        if accel_ok.any():
            aid = ids[accel_ok]
            ax, ay, az = (values[k][accel_ok] for k in ("ax", "ay", "az"))
            horizontal = np.sqrt(ax * ax + ay * ay)
            total_accel = np.sqrt(horizontal * horizontal + az * az)
            impact = np.abs(total_accel - self.gravity_norm[aid])
            tilt = np.abs(np.abs(az) - self.az_baseline[aid])
            self._add_by_id(self.accel_rows, aid)
            self._metric("impact", aid, impact)
            reliable = self.orientation_reliable[aid]
            self._add_by_id(self.horizontal_rows, aid[reliable])
            self._metric("horizontal", aid[reliable], horizontal[reliable])
            self._add_hist(self.impact_hist, aid, impact, self.impact_edges)

            moderate = impact >= self.settings.impact_moderate_g
            severe = impact >= self.settings.impact_severe_g
            extreme = impact >= self.settings.impact_extreme_g
            lateral = (horizontal >= self.settings.lateral_instability_g) & reliable
            tilted = tilt >= self.settings.tilt_deviation_g
            for name, signal in (
                ("impact_moderate", moderate), ("impact_severe", severe),
                ("impact_extreme", extreme), ("lateral_instability", lateral),
                ("tilt_anomaly", tilted),
            ):
                self._add_by_id(self.counts[name], aid[signal])

            gps_speed = values["gps_speed"][accel_ok]
            ems_speed = values["ems_speed"][accel_ok]
            speed = np.where(np.isfinite(gps_speed), gps_speed, ems_speed)
            collision = severe & np.isfinite(speed) & (speed >= self.settings.moving_speed_kmh)
            self._add_by_id(self.counts["collision_candidate"], aid[collision])

        gyro_ok = np.isfinite(values["gx"]) & np.isfinite(values["gy"]) & np.isfinite(values["gz"])
        rotation = None
        gid = None
        if gyro_ok.any():
            gid = ids[gyro_ok]
            gx, gy, gz = (values[k][gyro_ok] for k in ("gx", "gy", "gz"))
            rotation = np.sqrt(gx * gx + (gy - self.gy_bias[gid]) ** 2 + gz * gz)
            self._add_by_id(self.gyro_rows, gid)
            self._metric("rotation", gid, rotation)
            self._add_hist(self.rotation_hist, gid, rotation, self.rotation_edges)
            rot_severe = rotation >= self.settings.rotation_severe_dps
            rot_extreme = rotation >= self.settings.rotation_extreme_dps
            self._add_by_id(self.counts["rotation_severe"], gid[rot_severe])
            self._add_by_id(self.counts["rotation_extreme"], gid[rot_extreme])

        both = accel_ok & gyro_ok
        if both.any():
            bid = ids[both]
            ax, ay, az = (values[k][both] for k in ("ax", "ay", "az"))
            impact_b = np.abs(np.sqrt(ax * ax + ay * ay + az * az) - self.gravity_norm[bid])
            tilt_b = np.abs(np.abs(az) - self.az_baseline[bid])
            gx, gy, gz = (values[k][both] for k in ("gx", "gy", "gz"))
            rotation_b = np.sqrt(gx * gx + (gy - self.gy_bias[bid]) ** 2 + gz * gz)
            rollover = (tilt_b >= self.settings.tilt_deviation_g) & (rotation_b >= self.settings.rotation_severe_dps)
            any_signal = (
                (impact_b >= self.settings.impact_moderate_g)
                | ((np.sqrt(ax * ax + ay * ay) >= self.settings.lateral_instability_g) & self.orientation_reliable[bid])
                | (rotation_b >= self.settings.rotation_severe_dps)
            )
            self._add_by_id(self.counts["rollover_candidate"], bid[rollover])
            self._add_by_id(self.counts["any_signal"], bid[any_signal])

    def finish(self) -> pd.DataFrame:
        output = self.samples[["sample_id", "gpsno"]].copy()
        output["imu_rows_window"] = self.rows
        output["imu_accel_valid_ratio"] = np.divide(
            self.accel_rows, self.rows, out=np.zeros_like(self.accel_rows, dtype=float), where=self.rows > 0
        )
        output["imu_gyro_valid_ratio"] = np.divide(
            self.gyro_rows, self.rows, out=np.zeros_like(self.gyro_rows, dtype=float), where=self.rows > 0
        )
        output["imu_axis_orientation_reliable"] = self.orientation_reliable.astype(int)
        for name, denominator in (("impact", self.accel_rows), ("horizontal", self.horizontal_rows), ("rotation", self.gyro_rows)):
            mean = np.divide(self.sums[name], denominator, out=np.full(len(output), np.nan), where=denominator > 0)
            variance = np.divide(self.sumsq[name], denominator, out=np.zeros(len(output)), where=denominator > 0) - np.nan_to_num(mean) ** 2
            output[f"accident_{name}_mean"] = mean
            output[f"accident_{name}_std"] = np.where(denominator > 0, np.sqrt(np.maximum(variance, 0.0)), np.nan)
            output[f"accident_{name}_max"] = np.where(denominator > 0, self.maxima[name], np.nan)
        output["accident_impact_p95"] = _hist_quantile(self.impact_hist, self.impact_edges, 0.95)
        output["accident_impact_p99"] = _hist_quantile(self.impact_hist, self.impact_edges, 0.99)
        output["accident_rotation_p95"] = _hist_quantile(self.rotation_hist, self.rotation_edges, 0.95)
        output["accident_rotation_p99"] = _hist_quantile(self.rotation_hist, self.rotation_edges, 0.99)

        for name, count in self.counts.items():
            if name.startswith("rotation"):
                denominator = self.gyro_rows
            elif name == "lateral_instability":
                denominator = self.horizontal_rows
            else:
                denominator = self.accel_rows
            output[f"accident_{name}_count"] = count
            output[f"accident_{name}_rate_10k"] = np.divide(
                count * 10000.0, denominator, out=np.zeros(len(output)), where=denominator > 0
            )
        output["accident_sensor_risk_index"] = (
            output["accident_impact_moderate_rate_10k"]
            + 3.0 * output["accident_impact_severe_rate_10k"]
            + 8.0 * output["accident_impact_extreme_rate_10k"]
            + 2.0 * output["accident_rotation_severe_rate_10k"]
            + 6.0 * output["accident_rollover_candidate_rate_10k"]
        )
        return output


def _batch_columns(batch: pa.RecordBatch) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in IMU_COLUMNS:
        result[name] = batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
    return result


def _new_accumulator(config: PipelineConfig) -> _Accumulator:
    samples = pd.read_csv(config.tables_dir / "features.csv", usecols=["sample_id", "gpsno", "as_of", "lookback_days"])
    calibration = pd.read_csv(config.data_root / "辅助_imu校准参数.csv")
    return _Accumulator(samples, calibration, config.imu)


def _consume_file(config: PipelineConfig, path: Path, use_threads: bool) -> _Accumulator:
    accumulator = _new_accumulator(config)
    column_types = {
        "gpsno": pa.int64(), "data_date": pa.int64(),
        "ems_speed": pa.float64(), "gps_speed": pa.float64(),
        "ax": pa.float64(), "ay": pa.float64(), "az": pa.float64(),
        "gx": pa.float64(), "gy": pa.float64(), "gz": pa.float64(),
    }
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=config.imu.block_size_mb * 1024 * 1024, use_threads=use_threads),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(
            include_columns=IMU_COLUMNS,
            column_types=column_types,
            null_values=["", "\\N", "null", "NULL"],
            strings_can_be_null=True,
        ),
    )
    for batch in reader:
        accumulator.update(_batch_columns(batch))
    return accumulator


def _worker(config: PipelineConfig, path: Path) -> _Accumulator:
    return _consume_file(config, path, use_threads=False)


def extract_imu_features(
    config: PipelineConfig,
    files: Iterable[Path] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Stream IMU TSV partitions and aggregate accident-semantic features per sample window."""
    accumulator = _new_accumulator(config)
    input_files = sorted(files or config.imu_dir.glob("part-*"))
    if not input_files:
        raise FileNotFoundError(f"No IMU partitions found in {config.imu_dir}")

    workers = max(1, min(config.imu.workers, len(input_files)))
    if workers == 1:
        for number, path in enumerate(input_files, 1):
            accumulator.merge(_consume_file(config, path, use_threads=True))
            print(f"[IMU {number}/{len(input_files)}] {path.name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, config, path): path for path in input_files}
            for number, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                accumulator.merge(future.result())
                print(f"[IMU {number}/{len(input_files)}] {path.name}", flush=True)

    result = accumulator.finish()
    destination = output_path or (config.output_dir / "imu_accident_features.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_layer": config.imu_layer,
        "source_files": [str(path) for path in input_files],
        "rows_scanned": accumulator.rows_scanned,
        "rows_in_windows": int(accumulator.rows.sum()),
        "samples": len(result),
        "thresholds": asdict(config.imu),
        "window_semantics": "[as_of - lookback_days, as_of), excluding as_of day",
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
