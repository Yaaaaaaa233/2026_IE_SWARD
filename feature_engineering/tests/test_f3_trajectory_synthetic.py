# -*- coding: utf-8 -*-
"""F3 Tier 2 轨迹族合成测试（FEAT-007 r4）：切段边界、DBSCAN 已知簇恢复、路线重复度单调性、
高速巡航占比、每日里程 CV、标签窗排除、最低样本门槛置缺失。

仅合成数据；不触及真实数据。运行：cd feature_engineering && pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

from accident_pipeline_f3.trajectory import (
    CALIBER_GAP, CALIBER_TRIP, activity_radius_m, daily_rhythm_features,
    daynight_centroid_shift_m, dbscan_hotspots, driving_spells, hotspot_features,
    route_repeat_distance_m, run_duration_field_semantics, spatial_features,
    speed_shape_features, spell_features, to_epoch_s, window_mask)


def _epoch(s: str) -> int:
    return int(to_epoch_s(pd.Series([s]))[0])


def test_epoch_s_conversion_and_window_mask_boundary():
    t = to_epoch_s(pd.Series(["2026-06-01 00:00:00", "2026-06-20 23:59:59",
                             "2026-06-21 00:00:00"]))
    start, end = int(t[0]), int(t[2])
    assert t.tolist() == [start, start + 20 * 86400 - 1, start + 20 * 86400]  # ns→秒约定
    # 特征窗 [start, end)：含窗起点、不含 as_of（标签窗起点）
    assert window_mask(t, start, end).tolist() == [True, True, False]


def test_driving_spells_gap_boundary_180min():
    t0 = _epoch("2026-06-01 06:00:00")
    ws, we = t0 - 3600, t0 + 10 * 86400
    gap_exact = t0 + np.array([0, 60, 60 + 180 * 60, 60 + 180 * 60 + 60], dtype=np.int64)
    split = driving_spells(gap_exact, ws, we)
    assert split.caliber == CALIBER_GAP
    assert split.labels.tolist() == [0, 0, 0, 0]  # 间隔恰为 180min：不切段（严格 >180min）
    gap_over = t0 + np.array([0, 60, 60 + 180 * 60 + 1, 60 + 180 * 60 + 61], dtype=np.int64)
    split2 = driving_spells(gap_over, ws, we)
    assert split2.caliber == CALIBER_GAP
    assert split2.labels.tolist() == [0, 0, 1, 1]  # 间隔 >180min：切段
    assert split2.durations_s.tolist() == [60.0, 60.0]
    # 标签窗行（>=as_of）不参与切段
    outside = np.r_[gap_over, we + 60]
    assert driving_spells(outside, ws, we).labels[-1] == -1


def test_run_duration_semantics_and_trip_spells():
    t0 = _epoch("2026-06-01 06:00:00")
    ws, we = t0 - 3600, t0 + 10 * 86400
    t = t0 + np.array([0, 60, 120, 240, 300], dtype=np.int64)
    run = np.array([0.0, 60.0, 120.0, 0.0, 60.0])
    assert run_duration_field_semantics(t, run) == CALIBER_TRIP
    split = driving_spells(t, ws, we, run_duration_s=run)
    assert split.caliber == CALIBER_TRIP
    assert split.labels.tolist() == [0, 0, 0, 1, 1]        # 字段重置直接切段（间隔仅 240s）
    assert split.durations_s.tolist() == [120.0, 60.0]     # 单行程累计口径：字段增量
    # 同一时间轴无字段：240s 间隔 <180min，不切段
    assert driving_spells(t, ws, we).labels.tolist() == [0, 0, 0, 0, 0]
    # 长间隔不重置的窗口累计量 → gap_based
    t2 = t0 + np.array([0, 60, 120, 20000, 20060], dtype=np.int64)
    run2 = np.array([0.0, 60.0, 120.0, 20000.0, 20060.0])
    assert run_duration_field_semantics(t2, run2) == CALIBER_GAP
    assert driving_spells(t2, ws, we, run_duration_s=run2).labels.tolist() == [0, 0, 0, 1, 1]
    # 回落但非"从近零重新累计" → gap_based
    assert run_duration_field_semantics(t[:4], np.array([0.0, 100.0, 60.0, 120.0])) == CALIBER_GAP


def test_spell_features_long_gt4h_night():
    # P4：本地（北京）小时＝naive+8h；naive 14:00 起＝本地 22:00→03:00（5h 段）
    t0 = _epoch("2026-06-01 14:00:00")
    ws, we = t0 - 3600, t0 + 24 * 3600
    s1 = t0 + np.arange(0, 5 * 3600 + 1, 36, dtype=np.int64)           # 本地 22:00→03:00（5h）
    s2 = t0 + 10 * 3600 + np.arange(0, 2 * 3600 + 1, 72, dtype=np.int64)  # 本地 08:00→10:00（2h）
    f = spell_features(np.concatenate([s1, s2]), ws, we)
    assert f["f3_traj_max_spell_h"] == pytest.approx(5.0)
    assert f["f3_traj_gt4h_share"] == pytest.approx(5.0 / 7.0)   # 5h 段 / (5h+2h)
    assert f["f3_traj_night_spell_h"] == pytest.approx(4.0)      # 本地深夜 23:00–03:00


def test_speed_shape_quantiles_over90_buckets_cv():
    # P4：桶按本地（北京）小时（naive+8h）；naive 23:00＝本地早 7-9 桶、naive 15:00＝本地深夜桶
    t0 = _epoch("2026-06-01 00:00:00")
    ws, we = t0 - 3600, t0 + 10 * 86400
    n_am, n_night = 400, 300
    t_am = t0 + 23 * 3600 + np.arange(n_am, dtype=np.int64)       # naive 23:00（本地 07:00，早桶）
    t_ni = t0 + 15 * 3600 + np.arange(n_night, dtype=np.int64)    # naive 15:00（本地 23:00，深夜桶）
    sp_am = np.full(n_am, 50.0)
    sp_am[:40] = 100.0
    sp_ni = np.full(n_night, 50.0)
    sp_ni[:60] = 100.0
    t = np.concatenate([t_am, t_ni])
    sp = np.concatenate([sp_am, sp_ni])
    f = speed_shape_features(t, ws, we, sp)
    assert f["f3_traj_speed_p50"] == pytest.approx(50.0)
    assert f["f3_traj_speed_p90"] == pytest.approx(100.0)
    assert f["f3_traj_speed_p99"] == pytest.approx(100.0)
    assert f["f3_traj_over90_am"] == pytest.approx(40 / 400)      # >90km/h 分时段占比
    assert f["f3_traj_over90_night"] == pytest.approx(60 / 300)
    assert f["f3_traj_speed_cv"] == pytest.approx(float(np.std(sp) / np.mean(sp)))
    # 深夜桶 299 行（<300 门槛）→ 深夜分桶缺失，早桶与全体统计不受影响
    f2 = speed_shape_features(np.concatenate([t_am, t_ni[:299]]), ws, we,
                              np.concatenate([sp_am, sp_ni[:299]]))
    assert np.isnan(f2["f3_traj_over90_night"])
    assert f2["f3_traj_over90_am"] == pytest.approx(40 / 400)
    assert f2["f3_traj_speed_p50"] == pytest.approx(50.0)


def test_cruise_share_duration_ratio():
    t0 = _epoch("2026-06-01 08:00:00")
    ws, we = t0 - 3600, t0 + 86400
    n = 1000
    t = t0 + np.arange(n, dtype=np.int64)          # 1s 间隔
    sp = np.full(n, 30.0)
    sp[:600] = 100.0                               # 前 600 点持续 >80km/h
    f = speed_shape_features(t, ws, we, sp)
    # 持续 >80 的间隔 599s ÷ 相邻有限点总间隔 999s
    assert f["f3_traj_cruise80_share"] == pytest.approx(599 / 999, abs=1e-9)


def test_dbscan_hotspots_known_clusters_and_min_sample_gate():
    rng = np.random.default_rng(2026)

    def cluster(clat: float, clon: float, n: int):
        return (clat + rng.uniform(-0.0002, 0.0002, n),
                clon + rng.uniform(-0.0002, 0.0002, n))

    la1, lo1 = cluster(30.0, 120.0, 5)             # 簇 A（<60m 内 5 点）
    la2, lo2 = cluster(30.5, 120.5, 5)             # 簇 B
    la = np.r_[la1, la2, 31.0]                     # +1 个远离噪声点
    lo = np.r_[lo1, lo2, 121.0]
    labels, n_clusters = dbscan_hotspots(la, lo)
    assert n_clusters == 2
    assert sorted(int((labels == k).sum()) for k in range(n_clusters)) == [5, 5]
    assert int((labels == -1).sum()) == 1
    t0 = _epoch("2026-06-01 08:00:00")
    ws, we = t0, t0 + 20 * 86400
    ev_t = t0 + np.arange(11, dtype=np.int64) * 60
    small = hotspot_features(la, lo, ev_t, ws, we)  # 11 事件 <100 门槛 → 置缺失
    assert all(np.isnan(v) for v in small.values())
    # 150 事件：两簇各 60 + 30 相互远离的孤立噪声
    la_a, lo_a = cluster(30.0, 120.0, 60)
    la_b, lo_b = cluster(30.5, 120.5, 60)
    la_n = 32.0 + rng.uniform(-0.05, 0.05, 30)
    lo_n = 122.0 + np.arange(30) * 0.01
    ela, elo = np.r_[la_a, la_b, la_n], np.r_[lo_a, lo_b, lo_n]
    et = t0 + np.arange(150, dtype=np.int64) * 60
    big = hotspot_features(ela, elo, et, ws, we)
    assert big["f3_traj_hotspot_n"] == 2
    assert big["f3_traj_hotspot_event_share"] == pytest.approx(120 / 150)
    assert big["f3_traj_samepoint_recur_n"] == 118   # Σ(簇大小-1)＝59+59


def test_route_repeat_monotonic_same_vs_different():
    t0 = _epoch("2026-06-01 00:00:00")
    ws, we = t0, t0 + 3 * 86400
    base_lat = np.linspace(30.0, 30.2, 200)
    base_lon = np.full(200, 120.0)

    def build(lat_offsets):
        ts, las, los = [], [], []
        for d in range(3):                          # 相邻 3 日，每日 200 点
            ts.append(t0 + d * 86400 + np.arange(200, dtype=np.int64) * 30)
            las.append(base_lat + lat_offsets[d])
            los.append(base_lon.copy())
        return np.concatenate(ts), np.concatenate(las), np.concatenate(los)

    t, la, lo = build([0.0, 0.0, 0.0])
    same = route_repeat_distance_m(la, lo, t, ws, we)      # 三日完全相同路线
    t2, la2, lo2 = build([0.0, 0.5, 1.0])
    diff = route_repeat_distance_m(la2, lo2, t2, ws, we)   # 三日完全不同路线（相距约 55km）
    assert same == pytest.approx(0.0, abs=1e-6)
    assert same < diff                          # 单调性：相同路线 < 完全不同路线
    assert diff > 1000.0


def test_activity_radius_and_daynight_centroid_shift():
    t0 = _epoch("2026-06-01 00:00:00")
    ws, we = t0, t0 + 20 * 86400
    # 活动半径：300 点在质心 +0.01°、300 点在 -0.01°（经度向）→ RMS＝单点质心距离
    t = t0 + np.arange(600, dtype=np.int64)
    la = np.full(600, 30.0)
    lo = np.where(np.arange(600) < 300, 120.01, 119.99)
    expected = 6_371_000.0 * np.radians(0.01) * np.cos(np.radians(30.0))
    assert activity_radius_m(la, lo, t, ws, we) == pytest.approx(float(expected), rel=1e-3)
    # 昼夜偏移（P4 本地小时）：日间(9-17)＝naive 01:00 起、深夜(23-5)＝naive 15:00 起
    # 日间质心 (30,120)、深夜质心 (30.1,120) → 0.1° 经线距离
    t_day = t0 + 1 * 3600 + np.arange(400, dtype=np.int64)    # naive 01:00（本地 09:00，日间桶）
    t_ni = t0 + 15 * 3600 + np.arange(400, dtype=np.int64)    # naive 15:00（本地 23:00，深夜桶）
    la2 = np.r_[np.full(400, 30.0), np.full(400, 30.1)]
    lo2 = np.full(800, 120.0)
    shift = daynight_centroid_shift_m(la2, lo2, np.concatenate([t_day, t_ni]), ws, we)
    assert shift == pytest.approx(6_371_000.0 * np.radians(0.1), rel=1e-3)


def test_daily_mileage_cv_gap_days_run_hours():
    t0 = _epoch("2026-06-01 00:00:00")
    ws, we = t0, t0 + 10 * 86400
    days = [0, 1, 2, 3, 6, 7, 8, 9]                 # day 4、5 断档
    ts, las, los, km = [], [], [], []
    for d in days:
        step_m = 100.0 * (d + 1)                    # 日里程随日变化
        step_deg = step_m / (111_194.93 * np.cos(np.radians(30.0)))
        ts.append(t0 + d * 86400 + np.arange(80, dtype=np.int64) * 60)  # 每日 80 点×60s
        las.append(np.full(80, 30.0))
        los.append(120.0 + np.arange(80) * step_deg)
        km.append(79 * step_m / 1000.0)             # 79 个有效步长
    f = daily_rhythm_features(np.concatenate(ts), np.concatenate(las),
                              np.concatenate(los), ws, we)
    expect_km = np.array(km)
    assert f["f3_traj_daily_km_cv"] == pytest.approx(
        float(expect_km.std() / expect_km.mean()), rel=1e-3)
    # P4：断档天数按本地日历日（naive+8h）——窗口边界跨入的部分本地日无数据亦计断档
    # （UTC 窗 [6-1 00:00, 6-11 00:00)＝本地 6-1 08:00~6-11 08:00，覆盖 11 个本地日）
    assert f["f3_traj_gap_days"] == 3
    assert f["f3_traj_daily_run_h"] == pytest.approx(79 * 60 / 3600.0)


def test_label_window_exclusion_all_families():
    t0 = _epoch("2026-06-01 00:00:00")
    ws, we = t0, t0 + 10 * 86400                     # we 即 as_of（标签窗起点）
    rng = np.random.default_rng(11)
    ts, las, los, sps = [], [], [], []
    for d in range(8):                               # 8 日 ×3 时段块 ×50 点
        # P4：naive 23/04/15 时＝本地 07/12/23 时（早桶/日间桶/深夜桶各一块）
        for hour in (23, 4, 15):
            n = 50
            ts.append(t0 + d * 86400 + hour * 3600 + np.arange(n, dtype=np.int64) * 30)
            las.append(30.0 + 0.001 * d + rng.uniform(-0.0001, 0.0001, n))
            los.append(120.0 + rng.uniform(-0.0001, 0.0001, n))
            sps.append(np.where(rng.uniform(0, 1, n) < 0.1, 95.0, 55.0))
    t = np.concatenate(ts)
    la = np.concatenate(las)
    lo = np.concatenate(los)
    sp = np.concatenate(sps)
    ev_la = np.r_[np.full(60, 30.02), np.full(60, 30.52), np.linspace(31.0, 31.5, 30)]
    ev_lo = np.r_[np.full(60, 120.02), np.full(60, 120.52), np.full(30, 123.0)]
    ev_t = t0 + np.arange(150, dtype=np.int64) * 3600

    def feats(t_, la_, lo_, sp_, ev_t_, ev_la_, ev_lo_):
        return {
            **spell_features(t_, ws, we),
            **speed_shape_features(t_, ws, we, sp_),
            **spatial_features(la_, lo_, t_, ev_la_, ev_lo_, ev_t_, ws, we),
            **daily_rhythm_features(t_, la_, lo_, ws, we),
        }

    before = feats(t, la, lo, sp, ev_t, ev_la, ev_lo)
    assert len(before) == 19
    assert not any(np.isnan(v) for v in before.values())   # 门槛全过，各口径均有值
    # 追加标签窗 [as_of, as_of+H) 行（极端速度/远坐标/额外热点）后特征必须逐值不变
    lab_t = we + np.arange(200, dtype=np.int64) * 60
    lab_la = np.r_[np.full(100, 33.0), np.full(100, 33.5)]
    lab_lo = np.full(200, 125.0)
    lab_sp = np.full(200, 300.0)
    lab_ev_t = we + np.arange(80, dtype=np.int64) * 60
    lab_ev_la = np.full(80, 34.0)
    lab_ev_lo = np.full(80, 126.0)
    after = feats(np.r_[t, lab_t], np.r_[la, lab_la], np.r_[lo, lab_lo], np.r_[sp, lab_sp],
                  np.r_[ev_t, lab_ev_t], np.r_[ev_la, lab_ev_la], np.r_[ev_lo, lab_ev_lo])
    assert after == before                              # 标签窗排除：全部特征零影响
    assert window_mask(np.array([ws, we], dtype=np.int64), ws, we).tolist() == [True, False]


def test_min_sample_gates_missing():
    t0 = _epoch("2026-06-01 08:00:00")
    ws, we = t0 - 3600, t0 + 10 * 86400
    # 400 行 <500 门槛：点级统计全部置缺失；事件 50 <100：热点族置缺失
    t = t0 + np.arange(400, dtype=np.int64)
    la = np.full(400, 30.0)
    lo = np.full(400, 120.0)
    sp = np.full(400, 50.0)
    assert all(np.isnan(v) for v in spell_features(t, ws, we).values())
    assert all(np.isnan(v) for v in speed_shape_features(t, ws, we, sp).values())
    assert all(np.isnan(v) for v in daily_rhythm_features(t, la, lo, ws, we).values())
    assert all(np.isnan(v) for v in
               spatial_features(la, lo, t, la[:50], lo[:50], t[:50], ws, we).values())
    # 600 行：桶内/子集样本决定分桶、昼夜、日级与日对特征的缺失
    # P4 本地小时桶：t0＝naive 08:00 → +15h＝naive 23:00（本地早 7-9 桶，400 行）、
    # +7h＝naive 15:00（本地深夜桶，200 行）
    t2 = np.r_[t0 + 15 * 3600 + np.arange(400, dtype=np.int64),
               t0 + 7 * 3600 + np.arange(200, dtype=np.int64)]
    sp2 = np.where(np.arange(600) < 300, 100.0, 50.0)
    la2 = np.full(600, 30.0)
    lo2 = np.full(600, 120.0)
    f2 = speed_shape_features(t2, ws, we, sp2)
    assert not np.isnan(f2["f3_traj_speed_p50"])
    assert not np.isnan(f2["f3_traj_over90_am"])
    assert np.isnan(f2["f3_traj_over90_night"])           # 深夜桶 200 <300
    f3 = spell_features(t2, ws, we)
    assert not np.isnan(f3["f3_traj_max_spell_h"])
    assert np.isnan(f3["f3_traj_night_spell_h"])          # 深夜桶 200 <300
    f4 = daily_rhythm_features(t2, la2, lo2, ws, we)      # 仅 1 个有效天 <min_days
    assert np.isnan(f4["f3_traj_daily_km_cv"])
    assert np.isnan(f4["f3_traj_daily_run_h"])
    assert not np.isnan(f4["f3_traj_gap_days"])           # 计数特征只受点数门槛约束
    f5 = spatial_features(la2, lo2, t2, la2[:5], lo2[:5], t2[:5], ws, we)
    assert np.isnan(f5["f3_traj_daynight_shift_m"])       # 日间桶 0 行 <300
    # P4：两块分落相邻本地日（naive 23:00→本地次日 07:00）→ 相邻日对成立，同点位→0
    assert f5["f3_traj_route_repeat_m"] == pytest.approx(0.0)
    assert not np.isnan(f5["f3_traj_activity_radius_m"])
