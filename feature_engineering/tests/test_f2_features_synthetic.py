# -*- coding: utf-8 -*-
"""F2 合成测试：对齐恢复、阈值校准（分层/统一）、事件后处理冷却、特征窗边界排除。

仅合成数据；不触及真实数据。运行：cd feature_engineering && pytest tests -q
"""
import numpy as np
import pandas as pd

from accident_pipeline_f2.f2_core import (AlignAcc, TypeHist, acc_update,
                                          calibrate_thresholds, finalize_alignment,
                                          postprocess_events, quantile_of)
from accident_pipeline_f2.pipeline import _window_mask


def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _synthetic_device_series(rng, n=20000, fwd_scale=0.30, lat_scale=0.05):
    """车辆系合成：重力 z=1g＋噪声；前向强波动（burst 行）；横向小波动。返回设备系。"""
    burst = rng.random(n) < 0.20
    a_vehicle = np.column_stack([
        rng.normal(0, 0.08, n) + np.where(burst, rng.normal(0, fwd_scale, n), 0.0),
        rng.normal(0, lat_scale, n),
        rng.normal(1.0, 0.02, n),
    ])
    q = _random_rotation(rng)
    return q, a_vehicle @ q.T, a_vehicle


def test_alignment_recovers_known_rotation():
    rng = np.random.default_rng(42)
    q, a_device, _ = _synthetic_device_series(rng)
    acc = AlignAcc()
    dates = np.array(["20260610"] * len(a_device))
    acc_update(acc, a_device, np.full(len(a_device), 30.0), dates, 5.0)
    out = finalize_alignment(acc)
    assert out["align_pass"] is True
    assert abs(out["gravity_norm_g"] - 1.0) < 0.05
    assert out["forward_lateral_var_ratio"] >= 1.3
    r = np.array(out["R"]).reshape(3, 3)
    # 前向轴 = 车辆系 e1 在设备系中的方向（q 的第一列），符号不敏感
    assert abs(float(r[0] @ q[:, 0])) > 0.95
    assert abs(float(r[2] @ q[:, 2])) > 0.95


def test_calibration_layers_by_type_and_unifies_when_same():
    rng = np.random.default_rng(7)

    def hist_for(scale):
        th = TypeHist()
        fwd = rng.normal(0, scale, 400_000)
        th.update(fwd, np.abs(rng.normal(0, scale / 2, 400_000)),
                  np.abs(rng.normal(0, scale * 10, 400_000)))
        return th

    layered = calibrate_thresholds({"diesel": hist_for(0.08), "electric": hist_for(0.20)}, 0.10)
    assert layered["per_type"]["electric"]["accel_hi"] > layered["per_type"]["diesel"]["accel_hi"]
    assert "accel_hi" not in layered["unified"]

    same = calibrate_thresholds({"a": hist_for(0.10), "b": hist_for(0.10)}, 0.10)
    assert "accel_hi" in same["unified"]


def test_quantile_accuracy():
    th = TypeHist()
    rng = np.random.default_rng(0)
    fwd = rng.normal(0, 0.2, 1_000_000)
    th.update(fwd, np.abs(fwd), np.abs(fwd) * 20)
    q995 = quantile_of("fwd", th.hists["fwd"], 99.5)
    true_q = np.quantile(fwd, 0.995)
    assert abs(q995 - true_q) < 0.02


def test_postprocess_cooldown_merges_same_type():
    rows = pd.DataFrame({
        "gpsno": ["A"] * 4 + ["A"] * 2,
        "t": [100.0, 101.0, 102.0, 140.0, 500.0, 500.5],
        "scenario": ["hard_brake"] * 4 + ["hard_accel"] * 2,
        "value": [-0.3, -0.5, -0.4, -0.25, 0.4, 0.45],
        "speed": [40.0] * 6,
    })
    events = postprocess_events(rows)
    brakes = [e for e in events if e["scenario"] == "hard_brake"]
    accels = [e for e in events if e["scenario"] == "hard_accel"]
    assert len(brakes) == 2  # t≈101 峰值组 + t≈140（间隔 39s > 冷却 30s，独立）
    assert len(accels) == 1  # 两行同组
    assert max(abs(b["peak"]) for b in brakes) == 0.5


def test_window_mask_excludes_label_window():
    df = pd.DataFrame({"data_time": ["2026-05-31 23:59:59", "2026-06-01 00:00:00",
                                     "2026-06-20 23:59:59", "2026-06-21 00:00:00",
                                     "2026-07-15 08:00:00"]})
    m = _window_mask(df)
    assert m.tolist() == [False, True, True, False, False]
