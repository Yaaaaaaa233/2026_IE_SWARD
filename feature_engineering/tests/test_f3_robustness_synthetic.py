# -*- coding: utf-8 -*-
"""F3 健壮性合成测试（真实数据形态硬化）：NaN/inf 坐标、NaN 速度、NaN 事件时间/场景、
空事件列表、单点轨迹车、全 \\N 双速度源、除零——一律"置缺失不崩溃"（红线 2：缺数以缺失
表达、不删车）。重点：事件坐标部分缺失时热点特征只用有效坐标且数值正确；单点/空输入返回
NaN；全 \\N 双速度源置缺失。

仅合成数据；不触及真实数据。运行：cd feature_engineering && pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

from accident_pipeline_f3.core import (ems_gps_discrepancy, event_chains,
                                       historical_temporal_features, na_category,
                                       night_degradation, to_epoch_s)
from accident_pipeline_f3.pipeline import (F3Config, _scan_imu_speed_pairs, count_dual_track,
                                           fatigue_night_concentration, vehicle_new_features)
from accident_pipeline_f3.trajectory import (activity_radius_m, daily_rhythm_features,
                                             dbscan_hotspots, hotspot_features,
                                             route_repeat_distance_m, spatial_features,
                                             speed_shape_features, spell_features)

AS_OF = "2026-06-21T00:00:00"
IMU_ORDER = ("gpsno", "device_sn", "data_time", "data_date", "ems_speed", "gps_speed",
             "ax", "ay", "az", "gx", "gy", "gz")

t0 = int(to_epoch_s(pd.Series(["2026-06-01 08:00:00"]))[0])
ws, we = t0 - 3600, t0 + 20 * 86400              # 特征窗（we 即 as_of，标签窗起点）
AS_OF_S, LOOKBACK_S = float(we), float(we - ws)


def _epoch(s: str) -> int:
    return int(to_epoch_s(pd.Series([s]))[0])


# ------------------------------------------------------------- 事件坐标缺失（崩溃栈现场）

def test_hotspot_features_partial_nan_event_coords_uses_valid_only():
    """事件坐标部分缺失（NaN/inf）：只用有效坐标统计且数值正确，非法坐标不进 DBSCAN。"""
    rng = np.random.default_rng(2026)

    def cluster(clat: float, clon: float, n: int):
        return (clat + rng.uniform(-0.0002, 0.0002, n),
                clon + rng.uniform(-0.0002, 0.0002, n))

    la_a, lo_a = cluster(30.0, 120.0, 60)              # 簇 A
    la_b, lo_b = cluster(30.5, 120.5, 60)              # 簇 B
    la_n = 32.0 + rng.uniform(-0.05, 0.05, 30)         # 30 个孤立噪声事件
    lo_n = 122.0 + np.arange(30) * 0.01
    la = np.r_[la_a, la_b, la_n]
    lo = np.r_[lo_a, lo_b, lo_n]
    t = t0 + np.arange(150, dtype=np.int64) * 60
    bad = np.arange(120, 140)                          # 20 个噪声事件坐标缺失
    la_bad, lo_bad = la.copy(), lo.copy()
    la_bad[bad[::2]] = np.nan
    lo_bad[bad[1::2]] = np.inf
    f_bad = hotspot_features(la_bad, lo_bad, t, ws, we)
    f_ok = hotspot_features(np.delete(la, bad), np.delete(lo, bad), np.delete(t, bad), ws, we)
    assert f_bad == pytest.approx(f_ok)                # 只用有效坐标（130 事件），逐值一致
    assert f_bad["f3_traj_hotspot_n"] == 2
    assert f_bad["f3_traj_hotspot_event_share"] == pytest.approx(120 / 130)
    assert f_bad["f3_traj_samepoint_recur_n"] == 118   # Σ(簇大小-1)＝59+59（有效坐标口径）
    all_bad = hotspot_features(np.full(150, np.nan), np.full(150, np.inf), t, ws, we)
    assert all(np.isnan(v) for v in all_bad.values())  # 全非法坐标 → 全缺失，不崩溃


# ------------------------------------------------------------- 空/单点输入

def test_empty_and_single_point_inputs_return_nan_not_crash():
    """空输入/单点输入（单点轨迹车＋空事件列表）：置缺失不崩溃，可选 IMU 两列缺省 NaN。"""
    z = np.zeros(0)
    for feats in (spatial_features(z, z, z, z, z, z, ws, we),
                  spell_features(z, ws, we),
                  speed_shape_features(z, ws, we, z),
                  daily_rhythm_features(z, z, z, ws, we)):
        assert all(np.isnan(v) for v in feats.values())
    one_t = np.array([t0 + 60], dtype=np.int64)
    one = np.array([30.0])
    for feats in (spatial_features(one, one, one_t, one, one, one_t, ws, we),
                  spell_features(one_t, ws, we),
                  speed_shape_features(one_t, ws, we, np.array([50.0])),
                  daily_rhythm_features(one_t, one, one, ws, we)):
        assert all(np.isnan(v) for v in feats.values())
    feats, km, hours = vehicle_new_features(
        one_t, np.zeros(1), np.array([50.0]), one, one,
        np.array([np.nan]), np.array([np.nan]),            # 全 \N EMS/GPS（轨迹表侧）
        np.zeros(0), np.zeros(0, dtype=object), np.zeros(0), np.zeros(0), np.zeros(0),
        5000.0, 100.0, 0.1, AS_OF_S, LOOKBACK_S, 0.0, 0.0)
    assert np.isnan(feats["f3_ems_gps_absdiff_mean"])      # imu 未传 → 缺省缺失
    assert np.isnan(feats["f3_ems_gps_absdiff_p95"])
    assert np.isnan(feats["f3_ems_gps_diff_mean"])         # 全 \N 双速度源 → 缺失
    assert np.isnan(feats["f3_traj_speed_p50"])            # 单点轨迹 → 点级统计缺失
    assert np.isnan(feats["f3_event_speed_mean"])          # 空事件列表 → 缺失
    assert event_chains(np.zeros(0), np.zeros(0, dtype=object), AS_OF_S, LOOKBACK_S) == {
        "f3_chain_event_count": 0, "f3_chain_max_len": 0}  # 空事件列表 → 计数 0（非崩溃）
    assert km == 0.0 and hours == 0.0


def test_dbscan_hotspots_nan_coords_noise_and_empty():
    """dbscan_hotspots：空/单点/非法坐标不崩溃；非法坐标行标签 -1（噪声），簇数只计有效坐标。"""
    labels, n_clusters = dbscan_hotspots(np.zeros(0), np.zeros(0))
    assert labels.size == 0 and n_clusters == 0
    labels, n_clusters = dbscan_hotspots(np.array([30.0]), np.array([120.0]))
    assert labels.tolist() == [-1] and n_clusters == 0      # 单点＝噪声
    la = np.array([30.0, 30.0001, np.nan, 30.2, np.inf])
    lo = np.array([120.0, 120.0, 120.0, 120.0, 120.0])
    labels, n_clusters = dbscan_hotspots(la, lo)
    assert labels.tolist() == [0, 0, -1, -1, -1]            # 前两点成簇，非法/远离为噪声
    assert n_clusters == 1
    labels, n_clusters = dbscan_hotspots(np.array([np.nan]), np.array([np.inf]))
    assert labels.tolist() == [-1] and n_clusters == 0


# ------------------------------------------------------------- 轨迹 NaN 坐标/速度

def test_traj_nan_coord_and_speed_rows_dropped():
    """轨迹坐标 NaN/inf 行丢弃（等价于只喂有效行）；速度 NaN/inf 行只从速度统计丢弃。"""
    n = 840                                                # 7 日 × 120 点
    i = np.arange(n)
    t = t0 + (i // 120) * 86400 + (i % 120) * 60
    la = 30.0 + i * 1e-4
    lo = 120.0 + i * 1e-4
    la_bad, lo_bad = la.copy(), lo.copy()
    la_bad[0:100:2] = np.nan                               # 50 行纬度缺失
    lo_bad[1:100:2] = np.inf                               # 50 行经度非法 → 共 100 行
    ok = np.ones(n, dtype=bool)
    ok[:100] = False
    assert activity_radius_m(la_bad, lo_bad, t, ws, we) == pytest.approx(
        activity_radius_m(la[ok], lo[ok], t[ok], ws, we))
    assert route_repeat_distance_m(la_bad, lo_bad, t, ws, we) == pytest.approx(
        route_repeat_distance_m(la[ok], lo[ok], t[ok], ws, we))
    assert daily_rhythm_features(t, la_bad, lo_bad, ws, we) == pytest.approx(
        daily_rhythm_features(t[ok], la[ok], lo[ok], ws, we))
    sp_bad = 50.0 + (i % 3) * 10.0
    sp_bad[100:150] = np.nan
    sp_bad[150:200] = np.inf                               # 100 行速度非法 → 有效 740 行
    f = speed_shape_features(t, ws, we, sp_bad)
    v = sp_bad[np.isfinite(sp_bad)]
    assert f["f3_traj_speed_p50"] == pytest.approx(float(np.quantile(v, 0.5)))
    assert f["f3_traj_speed_p90"] == pytest.approx(float(np.quantile(v, 0.9)))
    assert f["f3_traj_speed_p99"] == pytest.approx(float(np.quantile(v, 0.99)))
    assert f["f3_traj_speed_cv"] == pytest.approx(float(v.std() / v.mean()))


# ------------------------------------------------------------- 事件 NaN 时间/场景

def test_event_nan_time_and_scenario_hardening():
    """事件 NaN 时间行排除、NaN/None 场景以 __na__ 表达；与"只喂有效行"逐值一致，不崩溃。"""
    t1 = t0 + 5 * 86400
    ev_t = np.array([t1, t1 + 100.0, np.nan, t1 + 200.0, t1 + 400.0, np.nan, we + 100.0])
    sc = np.array(["x", None, "y", np.nan, "x", "z", "x"], dtype=object)
    assert na_category(sc).tolist() == ["x", "__na__", "y", "__na__", "x", "z", "x"]
    chains = event_chains(ev_t, sc, AS_OF_S, LOOKBACK_S)
    assert chains["f3_chain_event_count"] == 4             # NaN 时间行与标签窗行不参与连环
    assert chains["f3_chain_max_len"] == 4
    assert not any("nan" in k for k in chains)             # 缺失场景不冒充 "nan" 类别
    keep = np.isfinite(ev_t) & (ev_t < AS_OF_S)
    assert chains == event_chains(ev_t[keep], sc[keep], AS_OF_S, LOOKBACK_S)
    assert historical_temporal_features(ev_t, AS_OF_S, LOOKBACK_S) == pytest.approx(
        historical_temporal_features(ev_t[keep], AS_OF_S, LOOKBACK_S))
    assert np.isnan(fatigue_night_concentration(            # 缺失场景不算疲劳类 → 缺失
        np.array([t1, np.nan]), np.array([None, np.nan], dtype=object), AS_OF_S, LOOKBACK_S))
    assert fatigue_night_concentration(
        np.array([_epoch("2026-06-05 15:30:00"), np.nan]),
        np.array(["fatigue_alert", np.nan], dtype=object), AS_OF_S, LOOKBACK_S
    ) == pytest.approx(1.0)                                # 有效疲劳事件（深夜）正常统计

    # night_degradation：NaN 时间/场景/窗外行清洗后逐值正确、零分母置缺失
    ex_t = np.r_[np.full(300, _epoch("2026-06-05 15:00:00")) + np.arange(300),   # 深夜 300 行
                 np.full(300, _epoch("2026-06-06 02:00:00")) + np.arange(300),   # 日间 300 行
                 np.full(50, np.nan),                                           # 时间缺失 50 行
                 (we + np.arange(40)).astype(float)].astype(float)              # 标签窗 40 行
    nd_t = np.array([_epoch("2026-06-05 15:30:00"), _epoch("2026-06-05 15:31:00"),
                     _epoch("2026-06-05 15:32:00"),                              # 深夜 3 条
                     _epoch("2026-06-06 02:30:00"), _epoch("2026-06-06 02:31:00"),  # 日间 2 条
                     np.nan, np.nan, we + 100.0], dtype=float)
    nd_sc = np.array([None, "hard_brake", np.nan,
                      "hard_brake", "hard_brake", "x", "y", "x"], dtype=object)
    nd = night_degradation(nd_t, nd_sc, ex_t, AS_OF_S, lookback_s=LOOKBACK_S)
    assert list(nd["scenario"]) == ["__merged__", "__na__", "hard_brake"]
    merged = nd[nd["scenario"] == "__merged__"].iloc[0]
    assert merged["f3_night_deep_rate"] == pytest.approx(3 / 300)
    assert merged["f3_night_day_rate"] == pytest.approx(2 / 300)
    assert merged["f3_night_degradation"] == pytest.approx(1.5)
    na_row = nd[nd["scenario"] == "__na__"].iloc[0]
    assert na_row["f3_night_deep_rate"] == pytest.approx(2 / 300)
    assert np.isnan(na_row["f3_night_degradation"])        # 日间 0 → 零分母置缺失


# ------------------------------------------------------------- 全 \N 双速度源与除零

def test_all_missing_dual_speed_and_zero_division_guards():
    """全 \\N 双速度源（含窗外有效行）置缺失；速度全 0 与零暴露的除零守卫置缺失。"""
    t = t0 + np.arange(600, dtype=np.int64)
    out = ems_gps_discrepancy(np.full(600, np.nan), np.full(600, np.nan),
                              t, AS_OF_S, LOOKBACK_S)
    assert all(np.isnan(v) for v in out.values())
    out = ems_gps_discrepancy(np.r_[np.full(600, np.nan), np.full(600, 30.0)],
                              np.r_[np.full(600, np.nan), np.full(600, 10.0)],
                              np.r_[t, (we + np.arange(600)).astype(float)],
                              AS_OF_S, LOOKBACK_S)         # 有效行全在标签窗 → 仍缺失
    assert all(np.isnan(v) for v in out.values())
    f = speed_shape_features(t, ws, we, np.zeros(600))     # 速度全 0：CV 除零置缺失
    assert np.isnan(f["f3_traj_speed_cv"])
    assert f["f3_traj_speed_p50"] == 0.0
    assert all(np.isnan(v) for v in
               count_dual_track("c", 3.0, 0.0, 0.0, 50.0, 1.0).values())   # 零暴露双轨置缺失


def test_imu_all_missing_rows_produce_no_stats(tmp_path):
    """IMU 分片全 \\N（两车）：无有效速度对 → 无统计（特征按缺失），不崩溃。"""
    (tmp_path / "imu").mkdir()
    rows = [["A", "dev-A", "2026-06-05 06:00:00", "2026-06-05", "\\N", "10.0",
             "0", "0", "0", "0", "0", "0"],
            ["A", "dev-A", "2026-06-05 06:01:00", "2026-06-05", "10.0", "\\N",
             "0", "0", "0", "0", "0", "0"],
            ["B", "dev-B", "2026-06-05 06:02:00", "2026-06-05", "\\N", "\\N",
             "0", "0", "0", "0", "0", "0"]]
    (tmp_path / "imu" / "part-000.tsv").write_text(
        "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
    cfg = F3Config(path=tmp_path / "config.yaml", trajectory_dir=tmp_path / "traj",
                   events_path=tmp_path / "events.csv", profile_path=tmp_path / "profile.csv",
                   base_input=tmp_path / "base.csv", base_columns_file=tmp_path / "keep.txt",
                   output_dir=tmp_path / "out", as_of=AS_OF,
                   lookback_days=20.0, horizon_days=30.0,
                   imu_dir=tmp_path / "imu", imu_column_order=IMU_ORDER)
    assert _scan_imu_speed_pairs(cfg, {"A", "B"}) == {}     # 全 \N → 无统计 → 特征缺失
