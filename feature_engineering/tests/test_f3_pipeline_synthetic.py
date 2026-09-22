# -*- coding: utf-8 -*-
"""F3 pipeline 合成测试（FEAT-007 r4 补丁）：无表头轨迹分片（固定列序/别名列不消费/必需列校验）、
扫描缓冲特征窗预过滤（窗外行不进缓冲）、IMU 双速度源轻量扫描（\\N 缺失、窗外行排除、两车
n/均值/p95）、p95 固定直方图区间中位近似确定性、run_scan 产出 f3_ems_gps_absdiff_mean／
f3_ems_gps_absdiff_p95（imu_dir 未配置置 NaN）。

仅合成临时文件；不触及真实数据。运行：cd feature_engineering && pytest tests -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from accident_pipeline_f3.core import to_epoch_s
from accident_pipeline_f3.pipeline import (F3Config, _hist_p95_midpoint, _scan_imu_speed_pairs,
                                           _scan_trajectory_shards, load_f3_config, run_scan)

DAY_S = 86400.0
AS_OF = "2026-06-21T00:00:00"          # 特征窗右端点（标签窗起点）
W0 = "2026-06-01T00:00:00"             # 特征窗左端点（as_of−20d）
LOOKBACK_DAYS = 20.0

# 真实轨迹分片 8 列固定列序（含 dist_m/heading 非 canonical 别名列）
TRAJ_ORDER = ("gpsno", "dist_m", "run_duration_s", "data_time", "speed", "heading", "lat", "lon")
# IMU 分片 12 列固定列序
IMU_ORDER = ("gpsno", "device_sn", "data_time", "data_date", "ems_speed", "gps_speed",
             "ax", "ay", "az", "gx", "gy", "gz")


def _epoch(s: str) -> int:
    return int(to_epoch_s(pd.Series([s]))[0])


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    """写无表头 TSV 分片（真实交付格式）。"""
    path.write_text("".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")


def _cfg(root: Path, **kw) -> F3Config:
    """最小 F3Config（未用到的路径仅占位，不触及真实数据）。"""
    base = dict(path=root / "config.yaml", trajectory_dir=root / "traj",
                events_path=root / "events.csv", profile_path=root / "profile.csv",
                base_input=root / "base.csv", base_columns_file=root / "keep.txt",
                output_dir=root / "out", as_of=AS_OF,
                lookback_days=LOOKBACK_DAYS, horizon_days=30.0)
    base.update(kw)
    return F3Config(**base)


# ------------------------------------------------------------- 补丁 1：无表头轨迹分片

def test_load_f3_config_headerless_and_imu_keys(tmp_path):
    """load_f3_config 读取同名 yaml 键；imu_dir 空置 None（跳过），缺省维持带表头现状。"""
    (tmp_path / "config.yaml").write_text(f"""\
trajectory_dir: "traj"
events_path: "events.csv"
profile_path: "profile.csv"
base_input: "base.csv"
base_columns_file: "keep.txt"
output_dir: "out"
window:
  as_of: "{AS_OF}"
  lookback_days: {LOOKBACK_DAYS}
  horizon_days: 30.0
shard_delimiter: "\\t"
shard_has_header: false
trajectory_column_order: ["gpsno", "dist_m", "run_duration_s", "data_time", "speed", "heading", "lat", "lon"]
imu_dir: "imu"
imu_has_header: false
imu_column_order: ["gpsno", "device_sn", "data_time", "data_date", "ems_speed", "gps_speed", "ax", "ay", "az", "gx", "gy", "gz"]
imu_delimiter: "\\t"
""", encoding="utf-8")
    cfg = load_f3_config(tmp_path / "config.yaml")
    assert cfg.shard_has_header is False
    assert cfg.trajectory_column_order == TRAJ_ORDER     # 含 dist_m/heading 非 canonical 别名列
    assert cfg.imu_dir == tmp_path / "imu"               # 相对路径以配置目录为基准
    assert cfg.imu_has_header is False
    assert cfg.imu_column_order == IMU_ORDER
    assert cfg.imu_delimiter == "\t"
    # 缺省键：带表头现状、imu_dir 空 → None（跳过轻扫，特征按缺失处理）
    (tmp_path / "plain.yaml").write_text(f"""\
trajectory_dir: "traj"
events_path: "events.csv"
profile_path: "profile.csv"
base_input: "base.csv"
base_columns_file: "keep.txt"
output_dir: "out"
window:
  as_of: "{AS_OF}"
  lookback_days: {LOOKBACK_DAYS}
  horizon_days: 30.0
imu_dir: ""
""", encoding="utf-8")
    cfg2 = load_f3_config(tmp_path / "plain.yaml")
    assert cfg2.shard_has_header is True
    assert cfg2.trajectory_column_order == ()
    assert cfg2.imu_dir is None
    assert cfg2.imu_has_header is False
    assert cfg2.imu_column_order == ()
    assert cfg2.imu_delimiter == "\t"


def test_scan_trajectory_shards_headerless_two_vehicles(tmp_path):
    """无表头分片（两车若干行）：每车数组数值正确；非 canonical 别名列不消费；可选列缺失置 NaN。"""
    (tmp_path / "traj").mkdir()
    rows_v1 = [["V1", "0.0", "0.0", "2026-06-02 06:00:00", "10.0", "90.0", "30.1", "120.1"],
               ["V1", "0.2", "30.0", "2026-06-02 06:00:30", "11.5", "91.0", "30.2", "120.2"],
               ["V2", "0.5", "10.0", "2026-06-05 09:00:00", "20.0", "180.0", "31.0", "121.0"],
               ["V1", "0.3", "60.0", "2026-06-03 08:00:00", "12.5", "92.0", "30.3", "120.3"],
               ["V2", "0.6", "70.0", "2026-06-05 09:01:00", "21.0", "181.0", "31.1", "121.1"]]
    _write_tsv(tmp_path / "traj" / "part-000.tsv", rows_v1)     # 两车混排、跨 chunk（chunk_rows=2）
    cfg = _cfg(tmp_path, shard_has_header=False,
               trajectory_column_order=TRAJ_ORDER, chunk_rows=2)
    bufs = _scan_trajectory_shards(cfg, {"V1", "V2"})
    assert set(bufs) == {"V1", "V2"}
    b1, b2 = bufs["V1"], bufs["V2"]
    assert sorted(b1) == ["ems_speed", "gps_speed", "lat", "lon", "run_duration_s", "speed", "t"]
    assert "dist_m" not in b1 and "heading" not in b1          # 非 canonical 别名列读取后不消费
    assert b1["t"].tolist() == [_epoch("2026-06-02 06:00:00"),
                                _epoch("2026-06-02 06:00:30"),
                                _epoch("2026-06-03 08:00:00")]
    assert b1["run_duration_s"].tolist() == [0.0, 30.0, 60.0]
    assert b1["speed"].tolist() == [10.0, 11.5, 12.5]
    assert b1["lat"].tolist() == [30.1, 30.2, 30.3]
    assert b1["lon"].tolist() == [120.1, 120.2, 120.3]
    assert np.isnan(b1["ems_speed"]).all()                    # 8 列固定列序无 ems/gps 速度 → 可选列 NaN
    assert np.isnan(b1["gps_speed"]).all()
    assert b2["t"].tolist() == [_epoch("2026-06-05 09:00:00"), _epoch("2026-06-05 09:01:00")]
    assert b2["speed"].tolist() == [20.0, 21.0]
    assert b2["lat"].tolist() == [31.0, 31.1]
    assert b2["lon"].tolist() == [121.0, 121.1]
    # 带表头（缺省现状）路径维持原行为
    hdr = tmp_path / "traj2"
    hdr.mkdir()
    pd.DataFrame([{"gpsno": "V1", "data_time": "2026-06-02 06:00:00",
                   "speed": 10.0, "lat": 30.1, "lon": 120.1}]).to_csv(
        hdr / "part-000.csv", index=False)
    bufs_hdr = _scan_trajectory_shards(
        _cfg(tmp_path, trajectory_dir=hdr, shard_delimiter=","), {"V1"})
    assert bufs_hdr["V1"]["speed"].tolist() == [10.0]


def test_scan_trajectory_shards_window_prefilter(tmp_path):
    """特征窗预过滤（内存守卫）：窗外行不进缓冲，缓冲行数与窗内行数一致（边界左闭右开）。"""
    (tmp_path / "traj").mkdir()
    rows = [["V1", "0.0", "0.0", "2026-05-31 23:59:59", "99.0", "0.0", "9.0", "9.0"],    # 窗前
            ["V1", "0.0", "0.0", "2026-06-01 00:00:00", "10.0", "90.0", "30.1", "120.1"],  # 窗左边界（含）
            ["V1", "0.1", "30.0", "2026-06-10 12:00:00", "11.0", "91.0", "30.2", "120.2"],
            ["V1", "0.2", "60.0", "2026-06-20 23:59:59", "12.0", "92.0", "30.3", "120.3"],  # as_of−1s（含）
            ["V1", "0.3", "90.0", "2026-06-21 00:00:00", "98.0", "1.0", "9.1", "9.1"],     # as_of（不含）
            ["V1", "0.4", "120.0", "2026-06-25 08:00:00", "97.0", "2.0", "9.2", "9.2"],    # 标签窗
            ["V2", "0.5", "0.0", "2026-06-05 09:00:00", "20.0", "180.0", "31.0", "121.0"],
            ["V2", "0.6", "60.0", "2026-05-20 09:00:00", "96.0", "3.0", "9.3", "9.3"]]     # 窗前
    _write_tsv(tmp_path / "traj" / "part-000.tsv", rows)
    cfg = _cfg(tmp_path, shard_has_header=False,
               trajectory_column_order=TRAJ_ORDER, chunk_rows=3)
    bufs = _scan_trajectory_shards(cfg, {"V1", "V2"})
    assert bufs["V1"]["t"].tolist() == [_epoch("2026-06-01 00:00:00"),
                                        _epoch("2026-06-10 12:00:00"),
                                        _epoch("2026-06-20 23:59:59")]     # 3 行＝窗内行数
    assert len(bufs["V2"]["t"]) == 1                                        # 窗前行未进缓冲
    assert bufs["V2"]["speed"].tolist() == [20.0]
    for buf in bufs.values():                                               # 缓冲内不再含窗外行
        assert np.all(buf["t"] >= _epoch(W0))
        assert np.all(buf["t"] < _epoch(AS_OF))
        assert max(buf["speed"]) < 96.0


def test_scan_trajectory_shards_headerless_required_columns_check(tmp_path):
    """无表头分片必需列齐全校验仍生效：列序声明缺 canonical 必需列即报错。"""
    (tmp_path / "traj").mkdir()
    _write_tsv(tmp_path / "traj" / "part-000.tsv",
               [["V1", "0.0", "0.0", "2026-06-02 06:00:00", "10.0", "90.0", "120.1"]])
    order_no_lat = ("gpsno", "dist_m", "run_duration_s", "data_time", "speed", "heading", "lon")
    cfg = _cfg(tmp_path, shard_has_header=False, trajectory_column_order=order_no_lat)
    with pytest.raises(ValueError, match="缺少必需列"):
        _scan_trajectory_shards(cfg, {"V1"})


# ------------------------------------------------------------- 补丁 2：IMU 双速度源轻量扫描

def test_hist_p95_midpoint_deterministic_bins():
    """p95 固定直方图（0–50、步长 0.5）区间中位近似：确定性、越界截断、空样本置缺失。"""
    same_bin = np.full(20, 1.2)                       # 全落 [1.0,1.5) → 箱中位 1.25
    assert _hist_p95_midpoint(same_bin) == pytest.approx(1.25)
    assert _hist_p95_midpoint(same_bin) == _hist_p95_midpoint(same_bin)   # 确定性可复现
    assert _hist_p95_midpoint(np.array([80.0, 90.0])) == pytest.approx(49.75)  # >50 截断进末箱
    mixed = np.array([0.6, 1.2, 1.2, 40.0])          # ceil(0.95·4)=4 → 40.0 所在箱中位
    assert _hist_p95_midpoint(mixed) == pytest.approx(40.25)
    assert np.isnan(_hist_p95_midpoint(np.array([])))
    assert np.isnan(_hist_p95_midpoint(np.array([np.nan, np.inf])))


def _write_imu_shards(root: Path) -> None:
    """小 IMU 分片（两车 A/B＋车外 C）：含 \\N 缺失行、窗外行（窗前/as_of 起）。"""
    imu = root / "imu"
    imu.mkdir(exist_ok=True)

    def row(gps: str, t: str, ems: str, gps_sp: str) -> list[str]:
        return [gps, f"dev-{gps}", t, t[:10], ems, gps_sp, "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]

    shard0 = [row("A", "2026-06-02 06:00:00", "10.6", "10.0"),        # diff 0.6
              row("A", "2026-06-03 06:00:00", "11.2", "10.0"),        # diff 1.2
              row("A", "2026-06-04 06:00:00", "\\N", "10.0"),         # ems 缺失 → 排除
              row("B", "2026-06-10 06:00:00", "12.2", "10.0")]        # diff 2.2
    shard1 = [row("A", "2026-06-04 06:00:00", "11.2", "10.0"),        # diff 1.2
              row("A", "2026-06-05 06:00:00", "50.0", "10.0"),        # diff 40.0
              row("A", "2026-06-07 06:00:00", "30.0", "\\N"),         # gps 缺失 → 排除
              row("A", "2026-05-31 23:00:00", "60.0", "0.0"),         # 窗前 → 排除
              row("A", "2026-06-21 00:00:00", "60.0", "0.0"),         # as_of 起（标签窗）→ 排除
              row("B", "2026-06-11 06:00:00", "13.2", "10.0"),        # diff 3.2
              row("C", "2026-06-10 06:00:00", "100.0", "0.0")]        # 名册外车辆 → 不统计
    _write_tsv(imu / "part-000.tsv", shard0)
    _write_tsv(imu / "part-001.tsv", shard1)


def test_scan_imu_speed_pairs_stats_window_and_missing(tmp_path):
    """IMU 轻扫：两车 n/均值/p95 正确；\\N 与窗外行排除；名册外车辆不统计；两次运行一致。"""
    _write_imu_shards(tmp_path)
    cfg = _cfg(tmp_path, imu_dir=tmp_path / "imu", imu_column_order=IMU_ORDER, chunk_rows=3)
    stats = _scan_imu_speed_pairs(cfg, {"A", "B"})
    assert set(stats) == {"A", "B"}
    assert stats["A"]["n"] == 4                        # 8 行 A：−2 缺失 −2 窗外
    assert stats["A"]["mean"] == pytest.approx((0.6 + 1.2 + 1.2 + 40.0) / 4)
    assert stats["A"]["p95"] == pytest.approx(40.25)   # ceil(0.95·4)=4 → 40.0 箱 [40.0,40.5) 中位
    assert stats["B"]["n"] == 2
    assert stats["B"]["mean"] == pytest.approx(2.7)
    assert stats["B"]["p95"] == pytest.approx(3.25)    # 第 2 值 3.2 箱 [3.0,3.5) 中位
    assert _scan_imu_speed_pairs(cfg, {"A", "B"}) == stats          # 确定性可复现
    # imu_dir 未配置：跳过轻扫、返回空字典
    assert _scan_imu_speed_pairs(_cfg(tmp_path), {"A", "B"}) == {}


# ------------------------------------------------------------- run_scan 组装（两新列）

def _write_run_inputs(root: Path) -> None:
    """合成基座/事件/画像/无表头轨迹分片（IMU 分片由 _write_imu_shards 另写）。"""
    (root / "traj").mkdir(exist_ok=True)
    _write_tsv(root / "traj" / "part-000.tsv", [
        ["A", "0.0", "0.0", "2026-06-02 06:00:00", "10.0", "90.0", "30.1", "120.1"],
        ["A", "0.2", "30.0", "2026-06-03 06:00:30", "11.5", "91.0", "30.2", "120.2"],
        ["B", "0.5", "10.0", "2026-06-05 09:00:00", "20.0", "180.0", "31.0", "121.0"],
        ["B", "0.6", "70.0", "2026-06-05 09:01:00", "21.0", "181.0", "31.1", "121.1"]])
    pd.DataFrame([{"sample_id": "S1", "gpsno": "A", "fold": 0},
                  {"sample_id": "S2", "gpsno": "B", "fold": 1}]).to_csv(
        root / "base.csv", index=False)
    pd.DataFrame([{"event_id": "E1", "gpsno": "A", "event_time": "2026-06-05 08:00:00",
                   "scenario": "hard_brake", "speed": 30.0, "lat": 30.1, "lon": 120.1},
                  {"event_id": "E2", "gpsno": "B", "event_time": "2026-06-15 08:00:00",
                   "scenario": "turn_l", "speed": 40.0, "lat": 31.0, "lon": 121.0}]).to_csv(
        root / "events.csv", index=False)
    pd.DataFrame([{"gpsno": "A", "energy_type": "diesel", "highway_ratio": 0.3,
                   "profile_month_km": 5000.0, "profile_month_hours": 100.0,
                   "profile_month_night_share": 0.1},
                  {"gpsno": "B", "energy_type": "electric", "highway_ratio": 0.6,
                   "profile_month_km": 4000.0, "profile_month_hours": 80.0,
                   "profile_month_night_share": 0.2}]).to_csv(root / "profile.csv", index=False)


def _write_config(root: Path, imu_dir_value: str) -> None:
    (root / "config.yaml").write_text(f"""\
trajectory_dir: "traj"
events_path: "events.csv"
profile_path: "profile.csv"
base_input: "base.csv"
base_columns_file: "keep.txt"
output_dir: "out"
window:
  as_of: "{AS_OF}"
  lookback_days: {LOOKBACK_DAYS}
  horizon_days: 30.0
shard_delimiter: "\\t"
shard_has_header: false
trajectory_column_order: ["gpsno", "dist_m", "run_duration_s", "data_time", "speed", "heading", "lat", "lon"]
imu_dir: "{imu_dir_value}"
imu_has_header: false
imu_column_order: ["gpsno", "device_sn", "data_time", "data_date", "ems_speed", "gps_speed", "ax", "ay", "az", "gx", "gy", "gz"]
imu_delimiter: "\\t"
exposure:
  min_km: 0.0
  min_hours: 0.0
""", encoding="utf-8")


def test_run_scan_emits_imu_absdiff_columns(tmp_path):
    """run_scan＋imu_dir：f3_ems_gps_absdiff_mean／p95 与 IMU 轻扫统计逐车一致。"""
    _write_run_inputs(tmp_path)
    _write_imu_shards(tmp_path)
    _write_config(tmp_path, "imu")
    new_df = pd.read_csv(run_scan(load_f3_config(tmp_path / "config.yaml")))
    assert {"f3_ems_gps_absdiff_mean", "f3_ems_gps_absdiff_p95"} <= set(new_df.columns)
    got = new_df.set_index("gpsno")
    assert got.loc["A", "f3_ems_gps_absdiff_mean"] == pytest.approx((0.6 + 1.2 + 1.2 + 40.0) / 4)
    assert got.loc["A", "f3_ems_gps_absdiff_p95"] == pytest.approx(40.25)
    assert got.loc["B", "f3_ems_gps_absdiff_mean"] == pytest.approx(2.7)
    assert got.loc["B", "f3_ems_gps_absdiff_p95"] == pytest.approx(3.25)


def test_run_scan_imu_absdiff_nan_without_imu_dir(tmp_path):
    """imu_dir 未配置：两列置 NaN（可选输入缺失处理），其余特征照常产出。"""
    _write_run_inputs(tmp_path)
    _write_imu_shards(tmp_path)                       # 数据在但不配置 imu_dir → 必须跳过
    _write_config(tmp_path, "")
    new_df = pd.read_csv(run_scan(load_f3_config(tmp_path / "config.yaml")))
    assert new_df["f3_ems_gps_absdiff_mean"].isna().all()
    assert new_df["f3_ems_gps_absdiff_p95"].isna().all()
    assert "f3_traj_speed_p50" in new_df.columns      # 其余族不受可选输入影响
