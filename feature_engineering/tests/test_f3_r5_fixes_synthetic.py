# -*- coding: utf-8 -*-
"""FEAT-008 r5 修复口径合成测试：P1 跳变过滤、P2 停车双切段（30min 边界）、P4 时区统一、
P5 画像折月 30.44、族配额 select_new_columns（确定性/分场景夜间排除/配额上限/轮转防挤占）。

预登记依据 docs/plans/feat-008-r5-execution.md §2（执行中不得擅改）；仅合成数据。
距离构造沿经度步进（米/度＝111194.93·cos(30°)，与 r4 既有测试同一近似口径）。
运行：cd feature_engineering && pytest tests -q
"""
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from accident_pipeline_f3.core import MONTH_DAYS, profile_deviation
from accident_pipeline_f3.interface import (FAMILY_CAP, MAX_NEW_COLUMNS, NIGHT_MERGED_ALLOW,
                                            family_of_column, is_night_scenario_variant,
                                            select_new_columns)
from accident_pipeline_f3.pipeline import window_exposure
from accident_pipeline_f3.trajectory import (CALIBER_GAP, STOP_LABEL, driving_spells,
                                             hour_of_day, jump_step_count, spell_features,
                                             to_epoch_s)

M_PER_DEG_LON = 111_194.93 * np.cos(np.radians(30.0))


def _epoch(s: str) -> int:
    return int(to_epoch_s(pd.Series([s]))[0])


def _deg_lon(m: float) -> float:
    """米 → 经度度（30°N 近似，合成数据用）。"""
    return m / M_PER_DEG_LON


# ------------------------------------------------------------- P4：时区统一

def test_hour_of_day_beijing_offset():
    t0 = _epoch("2026-06-01 00:00:00")            # naive 时间戳按 UTC 秒计
    # 本地（北京）小时 = UTC+8：00:00Z → 8 时；15:00Z → 23 时；16:00Z → 次日 0 时（深夜桶）
    assert hour_of_day(np.array([t0]))[0] == 8
    assert hour_of_day(np.array([t0 + 15 * 3600]))[0] == 23
    assert hour_of_day(np.array([t0 + 16 * 3600]))[0] == 0
    assert hour_of_day(np.array([np.nan]))[0] == -1     # 非有限时间置 -1（清洗语义）


def test_night_spell_hours_beijing_alignment():
    # 段 15:00Z→21:00Z ＝ 本地 23:00→次日 05:00：整段 6h 均在深夜桶（r4 naive 口径为 0）
    t0 = _epoch("2026-06-01 15:00:00")
    ws, we = t0 - 3600, t0 + 12 * 3600
    n = 500                                        # 过 MIN_MAIN_ROWS=500
    t = t0 + np.linspace(0, 6 * 3600, n).astype(np.int64)
    la = np.full(n, 30.0)
    lo = 120.0 + np.arange(n) * _deg_lon(300.0)    # 每点东移 300m（>50m，行驶）
    sp = np.full(n, 25.0)
    f = spell_features(t, ws, we, speed_kmh=sp, lat=la, lon=lo)
    assert f["f3_traj_max_spell_h"] == pytest.approx(6.0, abs=1e-6)
    assert f["f3_traj_night_spell_h"] == pytest.approx(6.0, abs=1e-6)


# ------------------------------------------------------------- P1：跳变过滤

def test_jump_filter_mileage_hours_and_spell_cut():
    t0 = _epoch("2026-06-01 08:00:00")
    ws, we = t0 - 3600, t0 + 3600
    # 4 个行驶点（5min 步长 3km＝10m/s）+ 1 个跳变点（5min 内东移 222km ≈ 74m/s）
    step = _deg_lon(3000.0)
    lo = 120.0 + np.array([0.0, step, 2 * step, 3 * step, 3 * step + _deg_lon(222_000.0)])
    la = np.full(5, 30.0)
    t = t0 + np.arange(5, dtype=np.int64) * 300
    sp = np.full(5, 60.0)
    km, hours = window_exposure(t, la, lo, ws, we)
    # 跳变步不计里程且不计运行时长（切段语义，随执行登记）
    assert km == pytest.approx(3 * 3000.0 / 1000.0, rel=1e-3)
    assert hours == pytest.approx(3 * 300.0 / 3600.0, rel=1e-6)
    # 质量列：窗内跳变步计数 = 1（供任务二）
    assert jump_step_count(t, la, lo, ws, we) == 1.0
    # 跳变步切断段：段 1 = 前 4 点（900s），段 2 = 跳变后单点段（时长 0）
    split = driving_spells(t, ws, we, speed_kmh=sp, lat=la, lon=lo)
    assert split.caliber == CALIBER_GAP
    assert split.labels.tolist() == [0, 0, 0, 0, 1]
    assert split.durations_s.tolist() == [900.0, 0.0]
    # 隐含速度边界（严格 >70m/s 判跳变）：5min 步 0.999×21km 不判、1.01×21km 判
    def one_step(disp_m: float) -> float:
        lo2 = 120.0 + np.array([0.0, _deg_lon(disp_m)])
        t2 = t0 + np.array([0, 300], dtype=np.int64)
        return jump_step_count(t2, np.full(2, 30.0), lo2, ws, we)
    assert one_step(70.0 * 300 * 0.999) == 0.0
    assert one_step(70.0 * 300 * 1.01) == 1.0


def test_jump_quality_column_empty_input_no_crash():
    assert jump_step_count(np.zeros(0), np.zeros(0), np.zeros(0), 0.0, 1.0) == 0.0
    assert jump_step_count(np.zeros(1), np.full(1, 30.0), np.full(1, 120.0), 0.0, 1.0) == 0.0


# ------------------------------------------------------------- P2：停车双切段

def _drive_park_drive(t0: int, park_steps: int, step_s: int = 300):
    """行驶 2 步 → 驻车 park_steps 步 → 行驶 2 步（每步 step_s 秒；返回 t/la/lo/sp）。"""
    ts, los, sps = [], [], []
    lon = 120.0
    t = t0
    for _ in range(2):                                  # 行驶段 A：每步 3km、60km/h
        ts.append(t)
        los.append(lon)
        sps.append(60.0)
        lon += _deg_lon(3000.0)
        t += step_s
    for _ in range(park_steps):                         # 驻车块：速度 0、位移 0（<50m）
        ts.append(t)
        los.append(lon)
        sps.append(0.0)
        t += step_s
    for _ in range(2):                                  # 行驶段 B：先驶离再报点（首点已位移）
        lon += _deg_lon(3000.0)
        ts.append(t)
        los.append(lon)
        sps.append(60.0)
        t += step_s
    n = len(ts)
    return (np.array(ts, dtype=np.int64), np.full(n, 30.0),
            np.array(los), np.array(sps))


def test_stop_block_cuts_spell_at_30min_boundary():
    t0 = _epoch("2026-06-01 08:00:00")
    ws, we = t0 - 3600, t0 + 12 * 3600
    # 驻车 6 步×300s：t[末驻车到达点]−t[块首出发点]＝2100−300＝1800s＝30min（边界含）→ 切段
    t, la, lo, sp = _drive_park_drive(t0, park_steps=6)
    split = driving_spells(t, ws, we, speed_kmh=sp, lat=la, lon=lo)
    assert split.labels.tolist() == [0, 0] + [STOP_LABEL] * 6 + [1, 1]
    assert split.durations_s.tolist() == [300.0, 300.0]   # 驻车时间不入任何段
    # 驻车 5 步×300s：跨度 1500s <30min → 短停计入行驶段（单段贯通、时长含短停）
    t2, la2, lo2, sp2 = _drive_park_drive(t0, park_steps=5)
    split2 = driving_spells(t2, ws, we, speed_kmh=sp2, lat=la2, lon=lo2)
    assert split2.labels.tolist() == [0] * 9
    assert split2.durations_s.tolist() == [8 * 300.0]


def test_stop_detection_by_displacement_without_speed():
    # 速度全缺失：位移 <50m 信号仍判驻车；首驻车点经 3km 移动步到达、位移信号判行驶
    # → 块自第 2 个驻车点起（7 点），首驻车点留在行驶段内（到达点语义，随执行登记）
    t0 = _epoch("2026-06-01 08:00:00")
    ws, we = t0 - 3600, t0 + 12 * 3600
    t, la, lo, _ = _drive_park_drive(t0, park_steps=8)
    split = driving_spells(t, ws, we, lat=la, lon=lo)      # 不传速度
    assert split.labels.tolist() == [0, 0, 0] + [STOP_LABEL] * 7 + [1, 1]


def test_spell_features_stop_cut_caps_max_spell():
    # 连续驾驶 5h（含端点）→ 驻车 ~1h → 驾驶 2h：max_spell=5h、gt4h 份额=5/7（驻车切断）
    t0 = _epoch("2026-06-01 01:00:00")                     # 本地 09:00（日间行驶）
    ws, we = t0 - 3600, t0 + 24 * 3600
    seg_a = t0 + np.arange(0, 5 * 3600 + 1, 30, dtype=np.int64)      # [0,18000] 共 601 点
    park = t0 + 18_000 + 300 * np.arange(1, 13, dtype=np.int64)      # 18300..21600 共 12 点
    seg_b = t0 + 21_660 + 30 * np.arange(0, 241, dtype=np.int64)     # 21660..28860 共 241 点
    t = np.concatenate([seg_a, park, seg_b])
    la = np.full(t.size, 30.0)
    lo = 120.0 + np.arange(t.size) * _deg_lon(90.0)         # 行驶 ~90m/min
    sp = np.full(t.size, 5.4)
    lo[seg_a.size:seg_a.size + park.size] = lo[seg_a.size - 1]   # 驻车点经度固定（位移 0）
    sp[seg_a.size:seg_a.size + park.size] = 0.0
    f = spell_features(t, ws, we, speed_kmh=sp, lat=la, lon=lo)
    assert f["f3_traj_max_spell_h"] == pytest.approx(5.0, abs=1e-6)
    assert f["f3_traj_gt4h_share"] == pytest.approx(5.0 / 7.0, abs=1e-6)


# ------------------------------------------------------------- P5：画像折月

def test_profile_deviation_month_scale_30_44():
    # 预登记字面公式：偏差率 =（窗内实际日均×30.44 ÷ 画像月均）− 1
    out = profile_deviation(
        np.array([1000.0]), np.array([100.0]), np.array([0.2]),
        np.array([1522.0]), np.array([152.2]), np.array([0.1]),
        lookback_days=20.0)
    assert MONTH_DAYS == pytest.approx(30.44)
    assert out["f3_profile_km_dev"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    expect = (100.0 / 20.0 * MONTH_DAYS / 152.2) - 1.0
    assert out["f3_profile_hours_dev"].iloc[0] == pytest.approx(expect, rel=1e-9)
    assert out["f3_profile_night_dev"].iloc[0] == pytest.approx(1.0, rel=1e-9)  # 0.2 vs 0.1


# ------------------------------------------------------------- 族配额选列

def _r5_candidate_columns() -> list[str]:
    """r5 真实候选列名形态（按产出序的代表性全集，覆盖全部族）。"""
    cols = [
        "f3_score_night_chain", "f3_score_fatigue",
        "f3_hist_decay_count_per_1000km", "f3_hist_decay_count_per_100h",
        "f3_hist_recurrence_gap_days", "f3_hist_burst_index",
        "f3_hist_trend_per_day", "f3_hist_x_trend",
        "f3_profile_km_dev", "f3_profile_hours_dev", "f3_profile_night_dev",
        "f3_night_deep_rate", "f3_night_day_rate", "f3_night_degradation",
        "f3_night_exposure_share",
    ]
    cols += [f"f3_night_{s}_{k}" for s in ("collision_warn", "fatigue_yawn", "near_miss")
             for k in ("deep_rate", "day_rate", "degradation")]     # 分场景变体（不入模）
    cols += [
        "f3_fatigue_night_conc",
        "f3_traj_max_spell_h", "f3_traj_gt4h_share", "f3_traj_night_spell_h",
        "f3_traj_jump_steps",
        "f3_traj_speed_p50", "f3_traj_speed_p90", "f3_traj_speed_p99",
        "f3_traj_over90_am", "f3_traj_over90_night", "f3_traj_speed_cv",
        "f3_traj_cruise80_share",
        "f3_traj_hotspot_n_per_1000km", "f3_traj_hotspot_n_per_100h",
        "f3_traj_hotspot_event_share", "f3_traj_samepoint_recur_n_per_1000km",
        "f3_traj_route_repeat_m", "f3_traj_activity_radius_m",
        "f3_traj_daynight_shift_m",
        "f3_traj_daily_km_cv", "f3_traj_gap_days_per_1000km",
        "f3_traj_gap_days_per_100h", "f3_traj_daily_run_h",
        "f3_chain_event_count_per_1000km", "f3_chain_max_len_per_1000km",
        "f3_ems_gps_diff_mean", "f3_ems_gps_diff_p95",
        "f3_ems_gps_absdiff_mean", "f3_ems_gps_absdiff_p95",
        "f3_event_speed_mean",
        "f2_prior_incident_per_1000km_cohort_z", "f2_prior_incident_per_1000km_cohort_pct",
        "f2r2_vol_wmag_std_cohort_z", "f2r2_vol_wmag_std_cohort_pct",
        "night_hours_ratio_cohort_z", "night_hours_ratio_cohort_pct",
    ]
    return cols


def test_family_of_column_mapping():
    assert family_of_column("f3_score_night_chain") == "score"
    assert family_of_column("f3_hist_burst_index") == "hist"
    assert family_of_column("f2r2_vol_wmag_std_cohort_z") == "cohort"
    assert family_of_column("f3_profile_km_dev") == "profile"
    assert family_of_column("f3_night_degradation") == "night"
    assert family_of_column("f3_traj_max_spell_h") == "spell"
    assert family_of_column("f3_traj_jump_steps") == "spell"
    assert family_of_column("f3_traj_cruise80_share") == "speed"
    assert family_of_column("f3_traj_hotspot_n_per_1000km") == "spatial"   # 剥双轨后缀归族
    assert family_of_column("f3_traj_gap_days_per_1000km") == "rhythm"
    assert family_of_column("f3_chain_max_len_per_1000km") == "chain"
    assert family_of_column("f3_ems_gps_absdiff_p95") == "ems"
    assert family_of_column("f3_event_speed_mean") == "event"
    assert family_of_column("f3_fatigue_night_conc") == "other"


def test_night_scenario_variants_excluded_from_selection():
    for c in ("f3_night_collision_warn_degradation", "f3_night_near_miss_deep_rate"):
        assert is_night_scenario_variant(c)
    for c in NIGHT_MERGED_ALLOW:
        assert not is_night_scenario_variant(c)
    sel = select_new_columns(_r5_candidate_columns())
    assert not any(is_night_scenario_variant(c) for c in sel)   # 分场景不入模
    assert not any(c.endswith("_per_100h") for c in sel)        # per_100h 不占配额


def test_family_quota_cap_and_rotation_no_starvation():
    sel = select_new_columns(_r5_candidate_columns())
    assert len(sel) == MAX_NEW_COLUMNS
    fam = Counter(family_of_column(c) for c in sel)
    assert all(v <= FAMILY_CAP for v in fam.values())           # 每族 <=6
    # r4 缺陷修复主断言：轨迹族与同群族真正入模
    for f in ("spell", "speed", "spatial", "rhythm", "cohort"):
        assert fam[f] >= 3, (f, fam)
    # 综合分恒优先（前两列即 score 族）＋轮转防挤占（候选充足的小族不为零）
    assert sel[0].startswith("f3_score_") and sel[1].startswith("f3_score_")
    for f in ("ems", "event", "other", "chain"):
        assert fam[f] >= 1, (f, fam)
    # 族内声明列序保先（spell 族首个＝产出序第一列）
    assert [c for c in sel if family_of_column(c) == "spell"][0] == "f3_traj_max_spell_h"


def test_selection_deterministic_same_input():
    cands = _r5_candidate_columns()
    assert select_new_columns(cands) == select_new_columns(list(cands))   # 同输入逐项一致


def test_selection_small_budget_and_empty():
    assert select_new_columns([]) == []
    few = ["f3_night_collision_warn_degradation", "f3_traj_speed_p50",
           "f3_score_fatigue", "x_other"]
    sel = select_new_columns(few, max_new=2)
    assert sel[0] == "f3_score_fatigue"                      # score 恒优先
    assert "f3_night_collision_warn_degradation" not in sel   # 分场景排除
    assert select_new_columns(few, max_new=10) == [
        "f3_score_fatigue", "f3_traj_speed_p50", "x_other"]
