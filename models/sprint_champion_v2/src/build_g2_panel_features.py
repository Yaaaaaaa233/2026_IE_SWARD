# -*- coding: utf-8 -*-
"""g2v1 特征工程（面板部分）：vehicle_day 纵向动态（500 车 × 20 天面板）。

信息源（正交声明）：E:/特征工程/数据筛选/契约v0.1全量/v4_assessed/vehicle_day.csv。
轨迹 44GB 的速度形态/连续驾驶/空间结构留 g2v2（流式扫描，预登记偏离登记于结果）。
窗口纪律：只用 [2026-06-01, 2026-06-21) 的日行。
构成（预登记，~32 列）：
  趋势（后10/前10 比值、近5/前15、日斜率）×(里程,时长,事件)  ~9
  波动（日 CV）×(里程,时长,事件)                              ~3
  断档（无轨迹天数、最长断档、断档占比、恢复首日里程）         ~4
  节律（周末/工作日里程比、周内里程 CV、工作日占比）           ~4
  覆盖（观测秒均值/CV、IMU-轨迹重叠比）                       ~4
  活跃（活跃天数、日均事件、事件日集中度、峰值日占比）         ~5
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
from eval_stack import load_protocol

VD = "E:/特征工程/数据筛选/契约v0.1全量/v4_assessed/vehicle_day.csv"
W_START, W_END = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-21")


def slope(s: pd.Series) -> float:
    x = np.arange(len(s), dtype=float)
    if len(s) < 2 or s.std() == 0:
        return np.nan
    return float(np.polyfit(x, s.values, 1)[0])


def build():
    labels, splits = load_protocol()
    lab = labels.assign(gpsno=labels.sample_id.str.split("_").str[0])
    tgt = lab[["sample_id", "gpsno"]].merge(splits[["gpsno", "fold"]], on="gpsno")

    vd = pd.read_csv(VD, dtype={"gpsno": str}, parse_dates=["date"])
    vd = vd[(vd.date >= W_START) & (vd.date < W_END)]
    vd = vd[vd.gpsno.isin(set(tgt.gpsno))].sort_values(["gpsno", "date"])
    assert vd.date.max() < W_END, "防泄漏：vehicle_day 越界"

    rows = []
    for gno, g in vd.groupby("gpsno"):
        n = len(g)
        d = {"gpsno": gno}
        km, hr, evt = g.traj_km.values, g.traj_hours.values, g.evt_raw_count.values.astype(float)
        h1, h2 = km[: n // 2], km[n // 2:]
        d["km_late_early"] = (h2.mean() + 1) / (h1.mean() + 1)
        k5, k15 = km[-5:], km[:-5]
        d["km_recent_lift"] = (k5.mean() + 1) / (k15.mean() + 1) if len(k15) else np.nan
        for name, arr in (("km", km), ("hr", hr), ("evt", evt)):
            d[f"{name}_slope"] = slope(pd.Series(arr))
            d[f"{name}_cv"] = arr.std() / (arr.mean() + 1e-6)
        hr2 = hr[n // 2:]
        d["hr_late_early"] = (hr2.mean() + 0.1) / (hr[n // 2].mean() if n // 2 > 0 else hr[0].mean() + 0.1) if n > 1 else np.nan
        d["hr_late_early"] = (hr[n // 2:].mean() + 0.1) / (hr[: n // 2].mean() + 0.1) if n > 1 else np.nan
        e2, e1 = evt[n // 2:], evt[: n // 2]
        d["evt_late_early"] = (e2.mean() + 0.5) / (e1.mean() + 0.5) if n > 1 else np.nan
        no_traj = (km <= 0.01)
        d["gap_days"] = int(no_traj.sum())
        max_gap = cur = 0
        for flag in no_traj:
            cur = cur + 1 if flag else 0
            max_gap = max(max_gap, cur)
        d["max_gap"] = max_gap
        d["gap_share"] = no_traj.mean()
        after_gap = [km[i] for i in range(1, n) if no_traj[i - 1] and not no_traj[i]]
        d["recover_km"] = float(np.mean(after_gap)) if after_gap else np.nan
        wd = g.date.dt.dayofweek.values < 5
        d["wk_km_ratio"] = (km[~wd].mean() + 1) / (km[wd].mean() + 1) if (~wd).any() else np.nan
        d["workday_share"] = float(wd.mean())
        week = g.date.dt.isocalendar().week.values
        wk_km = pd.Series(km).groupby(week).sum()
        d["week_km_cv"] = wk_km.std() / (wk_km.mean() + 1e-6) if len(wk_km) > 1 else np.nan
        obs = g.traj_observed_seconds.values.astype(float)
        d["obs_mean"] = obs.mean()
        d["obs_cv"] = obs.std() / (obs.mean() + 1e-6)
        ov = g.imu_traj_overlap_seconds.values.astype(float)
        d["imu_traj_overlap"] = np.nansum(ov) / (np.nansum(obs) + 1e-6)
        act = (g.evt_days_flag.values > 0)
        d["active_days"] = int(act.sum())
        d["evt_per_aday"] = evt.sum() / (act.sum() + 1)
        ed = evt[evt > 0]
        d["evt_day_conc"] = (ed.max() / (ed.mean() + 0.5)) if len(ed) else np.nan
        d["peak_day_share"] = evt.max() / (evt.sum() + 1) if evt.sum() > 0 else 0.0
        rows.append(d)

    f = pd.DataFrame(rows)
    out = tgt.merge(f, on="gpsno", how="left", validate="1:1")
    keep = [c for c in out.columns if c not in ("sample_id", "gpsno", "fold")]
    keep = [c for c in keep if pd.to_numeric(out[c], errors="coerce").std(ddof=0) > 1e-12]
    out = out[["sample_id", "gpsno"] + keep]
    print(f"g2v1(面板) 特征表: {out.shape[0]} 行 × {out.shape[1] - 2} 列")
    return out


if __name__ == "__main__":
    feat = build()
    vdir = os.path.join(HERE, "g2v1", "features")
    os.makedirs(vdir, exist_ok=True)
    feat.to_csv(os.path.join(vdir, "g2v1_panel_features.csv"), index=False, lineterminator="\n")
    print("[saved]", os.path.join(vdir, "g2v1_panel_features.csv"))
