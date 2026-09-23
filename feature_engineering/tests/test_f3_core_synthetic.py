# -*- coding: utf-8 -*-
"""F3 core 合成测试：stage-0 修正、Tier 1（历史时间结构/同群/画像/夜间退化）、Tier 3 与综合分。

仅合成数据；不触及真实数据。必测红线：特征窗边界与标签窗排除、同群折内拟合、群内 <30 台回退、
夜间双侧门槛置缺失、连环计数含 30min 边界、爆发性指数、近常数排除、衰减加权正确性。
运行：cd feature_engineering && pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

from accident_pipeline_f3.core import (CHAIN_MAX_GAP_S, COHORT_MIN_VEHICLES, DAY_S,
                                       MIN_ROWS_FULL, MIN_ROWS_SIDE, MIN_ROWS_SUB,
                                       cohort_keys, cohort_relative, composite_scores,
                                       ems_gps_discrepancy, event_chains,
                                       feature_window_mask, historical_temporal_features,
                                       label_window_mask, night_degradation,
                                       near_constant_filter, profile_deviation,
                                       rate_denominator, sort_defense, to_epoch_s)

AS_OF = 20000.0 * DAY_S          # 整数日边界，便于单日/半窗口径断言
LOOKBACK = 20.0 * DAY_S


def _assert_dict_equal(got: dict, want: dict) -> None:
    assert set(got) == set(want)
    for key, val in want.items():
        if val is None:
            assert not np.isfinite(got[key])
        else:
            assert got[key] == pytest.approx(val, abs=1e-9)


# ------------------------------------------------------------- 窗口红线

def test_feature_window_boundary_and_label_window_exclusion():
    t = np.array([
        AS_OF - 20.0 * DAY_S,        # 特征窗左边界（含）→ 计入
        AS_OF - 10.0 * DAY_S,        # 窗内 → 计入
        AS_OF - 1.0,                 # 窗内（贴近 as_of）→ 计入
        AS_OF,                       # as_of 本身（右边界不含）→ 排除
        AS_OF - 20.0 * DAY_S - 1.0,  # 窗前 → 排除
        AS_OF + 3.0 * DAY_S,         # 标签窗 [as_of, as_of+H) 内 → 排除
    ])
    got = historical_temporal_features(t, AS_OF, LOOKBACK)
    clean = historical_temporal_features(t[:3], AS_OF, LOOKBACK)
    _assert_dict_equal(got, clean)                       # 标签窗/窗外事件零影响
    assert got["f3_hist_decay_count"] == pytest.approx(4.0)  # 1+1+2（末 5 天事件权重 2）
    # 掩码口径：左闭右开、与标签窗不相交
    assert feature_window_mask(t, AS_OF, LOOKBACK).tolist() == [True, True, True, False, False, False]
    assert label_window_mask(t, AS_OF, 7.0 * DAY_S).tolist() == [False, False, False, True, False, True]
    assert not (feature_window_mask(t, AS_OF, LOOKBACK) & label_window_mask(t, AS_OF, 7.0 * DAY_S)).any()


# ------------------------------------------------------------- Tier 1 历史时间结构

def test_decay_weighted_count_recency_boundary():
    t = np.array([
        AS_OF - 4.0 * DAY_S,         # 末 5 天 → 权重 2
        AS_OF - 5.0 * DAY_S,         # 末 5 天起点（边界含）→ 权重 2
        AS_OF - 5.0 * DAY_S - 1.0,   # 边界外 1 秒 → 权重 1
        AS_OF - 19.0 * DAY_S,        # 其余 → 权重 1
    ])
    got = historical_temporal_features(t, AS_OF, LOOKBACK)
    assert got["f3_hist_decay_count"] == pytest.approx(6.0)
    # 复发间隔＝末两次事件间隔（排序后 as_of−5d 与 as_of−4d）＝1 天
    assert got["f3_hist_recurrence_gap_days"] == pytest.approx(1.0)
    # 事件不足 2 → 复发间隔缺失
    solo = historical_temporal_features(t[:1], AS_OF, LOOKBACK)
    assert not np.isfinite(solo["f3_hist_recurrence_gap_days"])


def test_burst_index_peak_over_daily_mean():
    t = np.array([19982 * DAY_S + 3600.0, 19985 * DAY_S + 3600.0, 19990 * DAY_S + 3600.0,
                  *[19998 * DAY_S + k * 3600.0 for k in range(1, 6)]])
    got = historical_temporal_features(t, AS_OF, LOOKBACK, tz_offset_s=0.0)
    # 单日峰值 5 ÷ 窗内日均 8/20=0.4 → 12.5
    assert got["f3_hist_burst_index"] == pytest.approx(12.5)
    empty = historical_temporal_features(np.array([]), AS_OF, LOOKBACK)
    assert not np.isfinite(empty["f3_hist_burst_index"])
    assert empty["f3_hist_decay_count"] == pytest.approx(0.0)


def test_history_x_trend_interaction():
    t = np.array([AS_OF - 15.0 * DAY_S, AS_OF - 14.0 * DAY_S, AS_OF - 13.0 * DAY_S,
                  AS_OF - 2.0 * DAY_S])   # 前半窗 3、后半窗 1
    got = historical_temporal_features(t, AS_OF, LOOKBACK)
    assert got["f3_hist_trend_per_day"] == pytest.approx((1 - 3) / 10.0)
    assert got["f3_hist_decay_count"] == pytest.approx(5.0)      # 1+1+1+2
    assert got["f3_hist_x_trend"] == pytest.approx(5.0 * (-0.2))


# ------------------------------------------------------------- stage 0

def test_sort_defense_groupwise_stable_time_sort():
    frame = pd.DataFrame({
        "gpsno": ["A", "B", "A", "B", "A"],
        "data_time": ["2026-06-20 05:00:00", "2026-06-20 01:00:00", "2026-06-20 03:00:00",
                      "2026-06-20 02:00:00", "2026-06-20 03:00:00"],
        "seq": [0, 1, 2, 3, 4],
    })
    out = sort_defense(frame)
    # A 组内 03:00 两行同刻（seq 2/4 稳定保序）后排 05:00；B 组内 01:00、02:00
    assert out["seq"].tolist() == [2, 4, 0, 1, 3]
    assert out["gpsno"].tolist() == ["A", "A", "A", "B", "B"]
    # to_epoch_s ns 约定与排序键一致
    assert to_epoch_s(pd.Series(["2026-06-20 00:00:00"]))[0] == 1781913600


def test_rate_denominator_verbatim_dup_after_sort():
    rng = np.random.default_rng(0)
    n = 150
    a6 = rng.normal(0.0, 1.0, (n, 6))
    t = (np.arange(n) * 10.0)
    a6[10] = a6[9]
    t[10] = t[9] + 1.0        # 逐字重复 dt=1s → 扣除
    a6[20] = a6[19]
    t[20] = t[19] + 2.0       # 逐字重复 dt=2s（边界含）→ 扣除
    a6[30] = a6[29]
    t[30] = t[29] + 3.0       # dt=3s > 2s → 不算
    a6[40] = a6[39]
    a6[40, 0] += 0.5          # dt=1s 但六轴非完全相同 → 不算
    t[40] = t[39] + 1.0
    got = rate_denominator(t, a6)
    assert got["n_dup"] == 2
    assert got["denominator"] == pytest.approx(148.0)
    assert got["n_pairs"] == 149
    assert got["dup_rate"] == pytest.approx(2 / 149)
    # 乱序输入结果一致（须排序后判定）
    order = rng.permutation(n)
    shuffled = rate_denominator(t[order], a6[order])
    assert shuffled["n_dup"] == got["n_dup"]
    assert shuffled["denominator"] == got["denominator"]
    assert shuffled["dup_rate"] == pytest.approx(got["dup_rate"])
    # 行对不足最低样本 → 重复率置缺失（率类分母不受影响）
    strict = rate_denominator(t, a6, min_pairs=10**9)
    assert not np.isfinite(strict["dup_rate"])
    assert strict["denominator"] == pytest.approx(148.0)


def test_near_constant_filter_variance_not_missingness():
    rng = np.random.default_rng(3)
    n = 200
    sparse = np.full(n, np.nan)
    sparse[:10] = np.arange(1, 11)             # 95% 缺失但取值有方差 → 不得按缺失率剔除
    frame = pd.DataFrame({
        "const": np.full(n, 5.0),
        "zero": np.zeros(n),
        "noisy": rng.normal(0.0, 1.0, n),
        "sparse": sparse,
        "tiny_var": 5.0 + 1e-9 * np.arange(n),
    })
    kept = near_constant_filter(frame)
    assert kept == ["noisy", "sparse"]          # 常数/恒 0/近常数剔除，保留列保持原列序
    assert near_constant_filter(frame) == kept  # 确定性：同输入两次一致
    assert near_constant_filter(frame, var_threshold=1e-6) == ["noisy", "sparse"]
    assert near_constant_filter(frame, var_threshold=10.0) == []


# ------------------------------------------------------------- Tier 1 同群

def test_cohort_keys_median_split_fold_fit():
    et = ["d", "d", "e", "e", "d", "d"]
    share = np.array([0.1, 0.3, 0.2, 0.4, 100.0, np.nan])
    train = np.array([True, True, True, True, False, False])
    keys, med = cohort_keys(et, share, train_mask=train)
    assert med == pytest.approx(0.25)           # 非训练行（100.0/NaN）不进中位数
    assert keys.tolist() == ["d|lo", "d|hi", "e|lo", "e|hi", "d|hi", "d|na"]
    _, med_all = cohort_keys(et, share)         # 对照：全体会被 100.0 拉高 → 折内拟合确有区别
    assert med_all == pytest.approx(0.3)


def test_cohort_relative_fold_fit_excludes_non_train():
    train_vals = np.arange(40, dtype=float)
    x = np.concatenate([train_vals, [1e6, -1e6]])      # 2 行非训练极端值
    values = pd.DataFrame({"event_rate": x})
    keys = np.array(["g"] * 42, dtype=object)
    train = np.array([True] * 40 + [False, False])
    res = cohort_relative(values, keys, train, vehicle_ids=[f"v{i}" for i in range(42)])
    mu, sd = train_vals.mean(), train_vals.std()
    assert res.loc[0, "event_rate_cohort_z"] == pytest.approx((0.0 - mu) / sd)
    assert res.loc[40, "event_rate_cohort_z"] == pytest.approx((1e6 - mu) / sd)   # 非训练行只套用训练统计量
    assert res.loc[0, "event_rate_cohort_pct"] == pytest.approx(1 / 40)           # 百分位分母=训练行 40
    assert res.loc[40, "event_rate_cohort_pct"] == pytest.approx(1.0)
    assert res.loc[41, "event_rate_cohort_pct"] == pytest.approx(0.0)
    assert (res["cohort_fallback"] == 0).all()
    z_leaky = (0.0 - x.mean()) / x.std()
    assert abs(res.loc[0, "event_rate_cohort_z"] - z_leaky) > 1.0   # 若极端值进统计量则显著不同


def test_cohort_relative_fallback_below_30_vehicles():
    x = np.concatenate([np.arange(25, dtype=float), np.arange(100, 140, dtype=float), [999.0, 999.0]])
    values = pd.DataFrame({"event_rate": x})
    keys = np.array(["s"] * 25 + ["b"] * 40 + ["s", "b"], dtype=object)
    train = np.array([True] * 65 + [False, False])
    res = cohort_relative(values, keys, train, vehicle_ids=[f"v{i}" for i in range(67)],
                          min_cohort_vehicles=COHORT_MIN_VEHICLES)
    mu_all, sd_all = x[:65].mean(), x[:65].std()
    assert res.loc[0, "cohort_fallback"] == 1      # 群 s 训练折内 25 台 < 30 → 回退全体
    assert res.loc[65, "cohort_fallback"] == 1
    assert res.loc[0, "event_rate_cohort_z"] == pytest.approx((0.0 - mu_all) / sd_all)
    assert res.loc[65, "event_rate_cohort_z"] == pytest.approx((999.0 - mu_all) / sd_all)
    assert res.loc[25, "cohort_fallback"] == 0     # 群 b 40 台 → 群内统计
    mu_b, sd_b = x[25:65].mean(), x[25:65].std()
    assert res.loc[25, "event_rate_cohort_z"] == pytest.approx((100.0 - mu_b) / sd_b)


# ------------------------------------------------------------- Tier 1 画像偏差

def test_profile_deviation_rates():
    out = profile_deviation(
        window_mileage_km=np.array([400.0, 200.0]), window_hours=np.array([40.0, 10.0]),
        window_night_share=np.array([0.3, 0.1]),
        profile_month_km=np.array([500.0, 0.0]), profile_month_hours=np.array([50.0, np.nan]),
        profile_month_night_share=np.array([0.25, 0.2]))
    # 里程/时长按窗长折月（20d→30.44d，r5 §2 P5 预登记字面公式）后对账
    scale = 30.44 / 20.0
    assert out.loc[0, "f3_profile_km_dev"] == pytest.approx((400.0 * scale - 500.0) / 500.0)
    assert out.loc[0, "f3_profile_hours_dev"] == pytest.approx((40.0 * scale - 50.0) / 50.0)
    assert out.loc[0, "f3_profile_night_dev"] == pytest.approx((0.3 - 0.25) / 0.25)
    assert out.loc[1, "f3_profile_night_dev"] == pytest.approx((0.1 - 0.2) / 0.2)
    assert not np.isfinite(out.loc[1, "f3_profile_km_dev"])      # 零基线置缺失
    assert not np.isfinite(out.loc[1, "f3_profile_hours_dev"])   # 画像缺失置缺失


# ------------------------------------------------------------- Tier 1 夜间退化度

def _exposure_rows(n: int, hour: float, day_back: float = 10.0) -> np.ndarray:
    return AS_OF - day_back * DAY_S + hour * 3600.0 + np.arange(n) * 0.001


def _night_frame(n_deep: int, n_day: int, ev_t, ev_sc, **kw) -> pd.DataFrame:
    ex_t = np.concatenate([_exposure_rows(n_deep, 2.0), _exposure_rows(n_day, 12.0)])
    return night_degradation(np.asarray(ev_t, dtype=float), ev_sc, ex_t, AS_OF,
                             tz_offset_s=0.0, **kw)


def test_night_degradation_gates_and_zero_denominator():
    ev_t = [AS_OF - 10.0 * DAY_S + 2.0 * 3600.0,
            AS_OF - 10.0 * DAY_S + 2.0 * 3600.0 + 100.0,
            AS_OF - 10.0 * DAY_S + 12.0 * 3600.0]
    ev_sc = ["fatigue", "speed", "fatigue"]
    out = _night_frame(400, 400, ev_t, ev_sc).set_index("scenario")
    assert out.loc["__merged__", "f3_night_deep_rate"] == pytest.approx(2 / 400)
    assert out.loc["__merged__", "f3_night_day_rate"] == pytest.approx(1 / 400)
    assert out.loc["__merged__", "f3_night_degradation"] == pytest.approx(2.0)
    assert out.loc["fatigue", "f3_night_degradation"] == pytest.approx(1.0)
    assert out.loc["speed", "f3_night_deep_rate"] == pytest.approx(1 / 400)
    assert not np.isfinite(out.loc["speed", "f3_night_degradation"])   # 零分母（日间率=0）置缺失
    # 合并版双侧门槛 300：深夜侧 200 行 → 深夜率与比值缺失，日间率仍可用；分场景版门槛 100 → 可用
    out2 = _night_frame(200, 400, ev_t, ev_sc).set_index("scenario")
    assert not np.isfinite(out2.loc["__merged__", "f3_night_deep_rate"])
    assert not np.isfinite(out2.loc["__merged__", "f3_night_degradation"])
    assert out2.loc["__merged__", "f3_night_day_rate"] == pytest.approx(1 / 400)
    assert out2.loc["fatigue", "f3_night_deep_rate"] == pytest.approx(1 / 200)
    assert out2.loc["fatigue", "f3_night_degradation"] == pytest.approx(2.0)
    # 子分档 100：深夜侧 80 行 → 分场景版也置缺失
    out3 = _night_frame(80, 400, ev_t, ev_sc).set_index("scenario")
    assert not np.isfinite(out3.loc["fatigue", "f3_night_deep_rate"])
    # 窗级档 500：双侧各 150（≥100 但合计 300 < 500）→ 分场景率可用、比值缺失
    out4 = _night_frame(150, 150, ev_t, ev_sc).set_index("scenario")
    assert out4.loc["fatigue", "f3_night_deep_rate"] == pytest.approx(1 / 150)
    assert not np.isfinite(out4.loc["fatigue", "f3_night_degradation"])
    assert MIN_ROWS_FULL == 500 and MIN_ROWS_SIDE == 300 and MIN_ROWS_SUB == 100
    # 暴露按里程加权时率＝事件数÷暴露和
    ex_t = np.concatenate([_exposure_rows(400, 2.0), _exposure_rows(400, 12.0)])
    amt = np.concatenate([np.full(400, 2.0), np.full(400, 1.0)])
    out5 = night_degradation(np.asarray(ev_t), ev_sc, ex_t, AS_OF, exposure_amount=amt,
                             tz_offset_s=0.0).set_index("scenario")
    assert out5.loc["__merged__", "f3_night_deep_rate"] == pytest.approx(2 / 800)


def test_night_degradation_label_window_exclusion():
    ev_t = [AS_OF - 10.0 * DAY_S + 2.0 * 3600.0,
            AS_OF - 10.0 * DAY_S + 12.0 * 3600.0,
            AS_OF + 3600.0,                                # 标签窗内 → 排除
            AS_OF - 20.0 * DAY_S - 60.0]                   # 窗前 → 排除
    ev_sc = ["fatigue", "fatigue", "fatigue", "fatigue"]
    out = _night_frame(400, 400, ev_t, ev_sc).set_index("scenario")
    clean = _night_frame(400, 400, ev_t[:2], ev_sc[:2]).set_index("scenario")
    for col in ("f3_night_deep_rate", "f3_night_day_rate", "f3_night_degradation"):
        assert out.loc["__merged__", col] == pytest.approx(clean.loc["__merged__", col])
    assert out.loc["__merged__", "f3_night_deep_rate"] == pytest.approx(1 / 400)


# ------------------------------------------------------------- Tier 3

def test_event_chains_30min_boundary_and_copairs():
    base = AS_OF - 10.0 * 3600.0
    t = np.array([base, base + CHAIN_MAX_GAP_S, base + 2 * CHAIN_MAX_GAP_S,
                  base + 3 * CHAIN_MAX_GAP_S + 1.0])
    ty = ["a", "b", "c", "a"]
    got = event_chains(t, ty, AS_OF, LOOKBACK)
    # 前三个事件间隔恰为 30min（边界含）→ 一条 3 连环；第 4 个间隔 30min+1s → 断链
    assert got["f3_chain_event_count"] == 3
    assert got["f3_chain_max_len"] == 3
    assert got["f3_copair_a__b"] == 1
    assert got["f3_copair_a__c"] == 1
    assert got["f3_copair_b__c"] == 1
    assert "f3_copair_a__a" not in got
    shuffled = event_chains(t[::-1], ty[::-1], AS_OF, LOOKBACK)   # 内部稳定排序，乱序同解
    assert shuffled["f3_chain_event_count"] == 3
    assert shuffled["f3_copair_a__b"] == 1
    # 间隔 30min+1s 不计连环 → 无连环段、无共现
    broke = event_chains(np.array([base, base + CHAIN_MAX_GAP_S + 1.0]), ["x", "x"], AS_OF, LOOKBACK)
    assert broke["f3_chain_event_count"] == 0
    assert broke["f3_chain_max_len"] == 0
    assert "f3_copair_x__x" not in broke
    # 间隔恰 30min 的同类型对共现
    pair = event_chains(np.array([base, base + CHAIN_MAX_GAP_S]), ["x", "x"], AS_OF, LOOKBACK)
    assert pair["f3_chain_event_count"] == 2
    assert pair["f3_copair_x__x"] == 1


def test_ems_gps_discrepancy_mean_p95():
    t = AS_OF - 5.0 * DAY_S + np.arange(600)
    ems = np.full(600, 10.0)
    gps = np.r_[np.full(300, 10.0), np.full(300, 12.0)]
    got = ems_gps_discrepancy(ems, gps, t, AS_OF, LOOKBACK)
    assert got["f3_ems_gps_diff_mean"] == pytest.approx(1.0)
    assert got["f3_ems_gps_diff_p95"] == pytest.approx(2.0)
    # 标签窗内行不参与统计
    t2 = np.r_[t, AS_OF + np.arange(100)]
    ems2 = np.r_[ems, np.full(100, 0.0)]
    gps2 = np.r_[gps, np.full(100, 500.0)]
    got2 = ems_gps_discrepancy(ems2, gps2, t2, AS_OF, LOOKBACK)
    assert got2["f3_ems_gps_diff_mean"] == pytest.approx(1.0)
    assert got2["f3_ems_gps_diff_p95"] == pytest.approx(2.0)
    # 有效行数不足窗级最低样本 500 → 置缺失
    small = ems_gps_discrepancy(ems[:100], gps[:100], t[:100], AS_OF, LOOKBACK)
    assert not np.isfinite(small["f3_ems_gps_diff_mean"])
    assert not np.isfinite(small["f3_ems_gps_diff_p95"])


# ------------------------------------------------------------- 综合分

def test_composite_scores_equal_weight_and_direction():
    out = composite_scores(
        night_exposure=np.array([1.0, 2.0, 3.0]), deep_night_rate=np.array([2.0, 4.0, 6.0]),
        night_degradation_ratio=np.array([3.0, 6.0, 9.0]),
        continuous_drive=np.array([6.0, np.nan, 0.0]), deep_night_continuous=np.array([3.0, 3.0, 0.0]),
        fatigue_night_concentration=np.array([0.0, 0.0, 0.0]))
    assert out["f3_score_night_chain"].tolist() == pytest.approx([2.0, 4.0, 6.0])  # 等权平均
    assert out.loc[0, "f3_score_fatigue"] == pytest.approx(3.0)     # (6+3+0)/3
    assert out.loc[1, "f3_score_fatigue"] == pytest.approx(1.5)     # 缺失项跳过，其余等权
    assert out.loc[2, "f3_score_fatigue"] == pytest.approx(0.0)
    all_nan = composite_scores(
        night_exposure=np.array([np.nan]), deep_night_rate=np.array([np.nan]),
        night_degradation_ratio=np.array([np.nan]), continuous_drive=np.array([np.nan]),
        deep_night_continuous=np.array([np.nan]), fatigue_night_concentration=np.array([np.nan]))
    assert not np.isfinite(all_nan["f3_score_night_chain"]).any()   # 全部缺失置缺失
    # 方向统一：反向组成项经 signs 翻转，分越高风险越高
    flipped = composite_scores(
        night_exposure=np.array([1.0]), deep_night_rate=np.array([2.0]),
        night_degradation_ratio=np.array([3.0]), continuous_drive=np.array([0.0]),
        deep_night_continuous=np.array([0.0]), fatigue_night_concentration=np.array([0.0]),
        night_signs=(1.0, 1.0, -1.0))
    assert flipped.loc[0, "f3_score_night_chain"] == pytest.approx(0.0)  # (1+2−3)/3
