# -*- coding: utf-8 -*-
"""F2R2 合成测试：波动统计正确性、质量统计（乱序/重复/饱和/毛刺）、方向无关阈值与触发。

仅合成数据；不触及真实数据。运行：cd feature_engineering && pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

from accident_pipeline_f2.f2_core import postprocess_events
from accident_pipeline_f2.r2_features import (FleetCal, QualAcc, VolAcc, vol_update,
                                              qual_update, derive_dirless_thresholds,
                                              filter_dirless_candidates, fleet_reference,
                                              hist_quantile, MAXW_SPEC)


def test_volatility_std_iqr_match_numpy():
    rng = np.random.default_rng(42)
    n = 200_000
    g = rng.normal(0, 2.0, (n, 3))
    a = rng.normal(0, 0.05, (n, 3))
    sp = np.full(n, 30.0)
    t = np.arange(n, dtype=float)
    hours = np.full(n, 12.0)
    vol = VolAcc()
    vol_update(vol, None, a, g, sp, t, hours, 5.0, front_epoch_s=10**18)
    wmag = np.linalg.norm(g, axis=1)
    sa = vol.sig["wmag"]
    assert abs(sa.std() - np.std(wmag)) < 0.01
    # 直方图 IQR 与 numpy IQR 差距 < 2 个 bin 宽（wmag bin 宽 1.0 dps）
    assert abs(sa.iqr() - (np.quantile(wmag, 0.75) - np.quantile(wmag, 0.25))) < 2.0
    assert "r2_vol_wmag_cv" in vol.row_stats({"amagres": 1.0}, {"amagres": 1.0})
    stats = vol.row_stats({"wmag": 0.0}, {"wmag": 0.0})
    assert np.isnan(stats["r2_vol_fwd_std"])  # R=None 时方向信号不更新 → std 缺失


def test_relative_occupancy_separates_volatility():
    rng = np.random.default_rng(7)
    cal = FleetCal()
    for scale in (0.05, 0.20):
        n = 100_000
        a = rng.normal(0, scale, (n, 3))
        amag = np.linalg.norm(a, axis=1)
        res = np.abs(amag - np.median(amag))
        cal.update("diesel", res, np.linalg.norm(rng.normal(0, 2, (n, 3)), axis=1))
    p90, p95 = fleet_reference(cal)
    low, high = VolAcc(), VolAcc()
    t = np.arange(50_000, dtype=float)
    hours = np.full(50_000, 12.0)
    sp = np.full(50_000, 30.0)
    for vol, scale in ((low, 0.03), (high, 0.30)):
        a = rng.normal(0, scale, (50_000, 3))
        vol_update(vol, None, a, np.full((50_000, 3), 0.1), sp, t, hours, 5.0, 10**18)
    s_lo = low.row_stats(p90["diesel"], p95["diesel"])
    s_hi = high.row_stats(p90["diesel"], p95["diesel"])
    assert s_hi["r2_rel_amagres_p90"] > s_lo["r2_rel_amagres_p90"]
    assert s_hi["r2_rel_amagres_p90"] > 0.10  # 高波动车远超全体 p90 占比应显著大于 10%


def test_quality_dup_rate_after_time_sort():
    rng = np.random.default_rng(0)
    n = 500
    t = np.sort(rng.integers(0, 10_000, n).astype(float))
    a6 = rng.normal(0, 1, (n, 6))
    a6[10] = a6[9]  # 相邻逐字重复（dt=1s）
    a6[100] = a6[99]
    t[10] = t[9] + 1.0
    t[100] = t[99] + 1.0
    df = pd.DataFrame({"t": t, **{f"c{i}": a6[:, i] for i in range(6)}})
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    order = np.argsort(shuffled["t"].to_numpy(), kind="stable")
    t_s = shuffled["t"].to_numpy()[order]
    a6_s = shuffled[[f"c{i}" for i in range(6)]].to_numpy()[order]
    qa = QualAcc()
    qual_update(qa, t_s, a6_s, np.full(n, 1.0))
    assert qa.n_dup == 2
    assert qa.n_pairs == n - 1


def test_saturation_const_and_desat_std():
    n = 1_000
    rng = np.random.default_rng(3)
    g = rng.normal(0, 1.0, (n, 3))
    g[5] = [500.0, 0.0, 0.0]      # 饱和行
    g[6] = [431.9, 0.0, 0.0]      # 综述定值
    g[7] = [0.0, -215.9, 0.0]     # 半量程定值
    a = rng.normal(0, 0.05, (n, 3))
    t = np.arange(n, dtype=float)
    sp = np.full(n, 30.0)
    vol, qa = VolAcc(), QualAcc()
    vol_update(vol, None, a, g, sp, t, np.full(n, 12.0), 5.0, 10**18)
    qual_update(qa, t, np.column_stack([a, g]), np.max(np.abs(g), axis=1))
    assert qa.n_sat == 2  # 500 与 431.9 两行都 >300dps
    assert qa.n_const == 2
    assert qa.max_w == 500.0
    # 去饱和统计应排除 maxw>300 的行（500 行与 431.9 行都被排除）
    wmag = np.linalg.norm(g, axis=1)
    expect = np.std(np.delete(wmag, [5, 6]))
    assert abs(vol.wmag_desat.std() - expect) < 1e-6


def test_segment_glitch_first_rows():
    n_seg, seg_len = 6, 50
    t = np.concatenate([s * 1_000.0 + np.arange(seg_len) * 1.0
                        for s in range(n_seg)])  # 6 段，段间隔 >60s
    maxw = np.full(len(t), 1.0)
    maxw[0], maxw[seg_len], maxw[2 * seg_len] = 50.0, 5.0, 80.0
    a6 = np.column_stack([np.zeros(len(t))] * 6)
    qa = QualAcc()
    qual_update(qa, t, a6, maxw)
    assert qa.seg_count == n_seg
    p99 = hist_quantile(np.bincount(np.clip(maxw.astype(int), 0, MAXW_SPEC[2] - 1),
                                    minlength=MAXW_SPEC[2]), 0.0, 350.0, 99.0)
    stats = qa.row_stats(p99)
    assert stats["r2_qa_glitch_rate"] == pytest.approx(3 / 6, abs=1e-6)  # 50/5/80 均 >p99=1.5


def test_dirless_thresholds_layered_and_events_fire():
    rng = np.random.default_rng(11)
    cal = FleetCal()
    for scale in (0.03, 0.15):
        n = 200_000
        a = rng.normal(0, scale, (n, 3))
        amag = np.linalg.norm(a, axis=1)
        cal.update("diesel" if scale < 0.1 else "electric",
                   np.abs(amag - np.median(amag)),
                   np.abs(rng.normal(0, 5 if scale < 0.1 else 25, n)))
    thr = derive_dirless_thresholds(cal, 0.10)
    assert thr["per_type"]["electric"]["long"] > thr["per_type"]["diesel"]["long"]

    cand = pd.DataFrame({
        "gpsno": ["A"] * 5, "t": [100.0, 101.0, 102.0, 103.0, 500.0],
        "res": [0.02, 0.45, 0.60, 0.05, 0.55],
        "wmag": [5.0, 5.0, 5.0, 5.0, 5.0], "speed": [40.0] * 5})
    etype = pd.Series({"A": "diesel"})
    act = filter_dirless_candidates(cand, thr, etype, 0.50, 0.35, 45.0)
    events = postprocess_events(act)
    by_scen = {}
    for e in events:
        by_scen.setdefault(e["scenario"], []).append(e)
    assert len(by_scen.get("long_anom", [])) == 2      # t=101 组 + t=500（间隔>冷却）
    assert "collision" not in by_scen or len(by_scen["collision"]) <= 2

    cand2 = pd.DataFrame({
        "gpsno": ["B"], "t": [50.0], "res": [0.70], "wmag": [60.0], "speed": [45.0]})
    act2 = filter_dirless_candidates(cand2, thr, etype, 0.50, 0.35, 45.0)
    events2 = postprocess_events(act2)
    assert {"long_anom", "turn_anom", "collision", "rollover"} <= {
        e["scenario"] for e in events2}
