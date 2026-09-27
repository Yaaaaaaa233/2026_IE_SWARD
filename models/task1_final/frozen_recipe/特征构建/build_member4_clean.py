# -*- coding: utf-8 -*-
"""成员4干净重建（g4v10-clean 底座）：复刻 g4v6 54 列构成，事件全部按特征窗过滤。

背景（2026-09-26 修正登记）：g4v6 及其 g6v1-v4 谱系的率特征来自未过滤全期事件计数
（含标签窗，违反红线 3；该历史版本不得用于终版）。
本脚本按同构块重建干净版本，供修正链（g4v10）与 40+20 适配检验共用。

54 列构成（与 g4v6 对齐）：
  语义率双轨 16：r_{7组}_kk/h + r_all_kk/h
  簇K4     12：cl_0..3 + cz_r_{7组+all}_kk
  簇K5z     5：cz5_r_{lane,fatigue,distract,collision,all}_kk
  能源z     3：ez_r_{lane,all,speed_risk}_kk（energy_type cohort 内 z；键不作特征，口径沿用）
  簇×险情   1：ix_cl5_days_since_risk（K5 簇内 z(days_since_risk)——重建近似，登记）
  险情结构   7：days_since_risk/risk_decay_w/risk_gap_med/risk_burst/never_risk/lift_all_5d/iv_med
  轨迹精选   6：traj_v_p90/v80_share/stop_share/seg_max_h/dv_mean/km
  FP交互     4：ix_night_x_rate/ix_night_x_lane/ix_obs_x_rate/ix_obs_x_night
              （夜间=官方 21-6 时段份额；观测=窗内观测时长 log 小时）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
from eval_stack import load_protocol

EV = None
VD = None
TV = None
TRAJ = None
W_START, W_END = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-21")
GROUPS = {
    "lane": {"30002", "30003", "30017"}, "fatigue": {"41001", "41002", "41029"},
    "distract": {"41003", "41004", "41005", "41009", "41023"},
    "speed_risk": {"11401", "11402", "11403", "11405", "11406"},
    "collision": {"30000", "30005"}, "blind": {"60292", "60294"}, "device": {"41006", "41021"},
}
NEAR_MISS, ACCIDENT = "11804", "11803"


def build(w_start: str | None = None, w_end: str | None = None, traj_csv: str | None = None):
    global W_START, W_END
    if w_start:
        W_START = pd.Timestamp(w_start)
    if w_end:
        W_END = pd.Timestamp(w_end)
    labels, splits = load_protocol()
    lab = labels.assign(gpsno=labels.sample_id.str.split("_").str[0])
    tgt = lab[["sample_id", "gpsno"]]

    ev = pd.read_csv(EV, usecols=["row_id", "gpsno", "event_type", "start_time"],
                     dtype={"gpsno": str, "event_type": str})
    ev["t"] = pd.to_datetime(ev.start_time, errors="coerce")
    ev = ev.dropna(subset=["t"])
    ev = ev[(ev.t >= W_START) & (ev.t < W_END)]
    assert (ev.t < W_END).all() and (ev.t >= W_START).all()
    ev = ev[ev.gpsno.isin(set(tgt.gpsno))]
    cnt = ev.pivot_table(index="gpsno", columns="event_type", values="row_id",
                         aggfunc="count").fillna(0.0)

    vd = pd.read_csv(VD, dtype={"gpsno": str}, parse_dates=["date"])
    vd = vd[(vd.date >= W_START) & (vd.date < W_END)]
    vd = vd[vd.gpsno.isin(set(tgt.gpsno))]
    expo = vd.groupby("gpsno").agg(traj_km=("traj_km", "sum"), traj_hours=("traj_hours", "sum"),
                                   obs_s=("traj_observed_seconds", "sum")).reset_index()
    f = tgt.merge(expo, on="gpsno", how="left")
    f["low_km"] = (f.traj_km < 50).astype(int)
    f["low_hours"] = (f.traj_hours < 1).astype(int)
    km_ok = f.traj_km.where(f.low_km == 0)
    f["obs_logh"] = np.log1p(f.obs_s / 3600.0)

    # ── 语义率双轨 ──
    for g, codes in GROUPS.items():
        cols = [c for c in codes if c in cnt.columns]
        n = f.gpsno.map(cnt[cols].sum(axis=1) if cols else pd.Series(0.0, index=cnt.index)).fillna(0)
        f[f"n_{g}"] = n
        f[f"r_{g}_kk"] = n / km_ok * 1000
        f[f"r_{g}_h"] = n / f.traj_hours.where(f.low_hours == 0) * 100
    f["n_all"] = f.gpsno.map(cnt.sum(axis=1)).fillna(0)
    f["r_all_kk"] = f.n_all / km_ok * 1000
    f["r_all_h"] = f.n_all / f.traj_hours.where(f.low_hours == 0) * 100

    # ── 聚类（K4/K5 同向量：率 rank-gauss + energy 键）──
    rate_cols = [f"r_{g}_kk" for g in GROUPS]
    X = f[rate_cols].copy()
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=rate_cols)
    X = X.apply(lambda s: pd.Series(norm.ppf(np.clip((s.rank(method="average") - .5) / len(s),
                                                   1e-6, 1 - 1e-6)), index=s.index))
    tv = pd.read_csv(TV, usecols=["gpsno", "energy_type"], dtype={"gpsno": str})
    f = f.merge(tv, on="gpsno", how="left")
    X["energy"] = f.energy_type.fillna("unknown").map({"纯电动": 0.0, "燃油": 1.0}).fillna(0.5)
    km4 = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X.values)
    km5 = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X.values)
    f["sem4"], f["sem5"] = km4, km5
    for k in range(4):
        f[f"cl_{k}"] = (f.sem4 == k).astype(int)

    def within_z(col, key):
        out = pd.Series(np.nan, index=f.index)
        for k, m in f.groupby(key).groups.items():
            v = f.loc[m, col]
            mu, sd = v.mean(), v.std(ddof=0)
            if sd > 1e-9:
                out.loc[m] = (v - mu) / sd
        return out

    for c in rate_cols + ["r_all_kk"]:
        f[f"cz_{c}"] = within_z(c, "sem4")
    for c in ("r_lane_kk", "r_fatigue_kk", "r_distract_kk", "r_collision_kk", "r_all_kk"):
        f[f"cz5_{c}"] = within_z(c, "sem5")
    for c in ("r_lane_kk", "r_all_kk", "r_speed_risk_kk"):
        f[f"ez_{c}"] = within_z(c, "energy_type")

    # ── 险情结构（过滤后事件，g1v1 同口径）──
    risk = ev[ev.event_type.isin({ACCIDENT, NEAR_MISS})].sort_values(["gpsno", "t"])
    n_risk = risk.groupby("gpsno").row_id.count()
    f["n_risk"] = f.gpsno.map(n_risk).fillna(0)
    f["never_risk"] = (f.n_risk == 0).astype(int)
    last = risk.groupby("gpsno").t.max()
    f["days_since_risk"] = (W_END - f.gpsno.map(last)).dt.days
    f.loc[f.never_risk == 1, "days_since_risk"] = np.nan
    r5 = risk[risk.t >= W_END - pd.Timedelta(days=5)].groupby("gpsno").row_id.count()
    r10 = risk[risk.t >= W_END - pd.Timedelta(days=10)].groupby("gpsno").row_id.count()
    f["n_risk_5d"] = f.gpsno.map(r5).fillna(0)
    f["n_risk_10d"] = f.gpsno.map(r10).fillna(0)
    f["risk_decay_w"] = f.n_risk - 1.0 * f.n_risk_5d - 0.5 * (f.n_risk_10d - f.n_risk_5d)
    gaps = risk.groupby("gpsno").t.diff().dt.total_seconds() / 86400
    f["risk_gap_med"] = f.gpsno.map(gaps.groupby(risk.gpsno).median())
    daily = risk.groupby(["gpsno", risk.t.dt.date]).row_id.count().reset_index(name="n")
    b = daily.groupby("gpsno").n.agg(["max", "mean"])
    f["risk_burst"] = f.gpsno.map(b["max"]) / (f.gpsno.map(b["mean"]) + 0.5)
    n_all_5d = ev[ev.t >= W_END - pd.Timedelta(days=5)].groupby("gpsno").row_id.count()
    f["n_all_5d"] = f.gpsno.map(n_all_5d).fillna(0)
    f["lift_all_5d"] = (f.n_all_5d + 0.5) / (f.n_all / max(1.0, (W_END - W_START).days / 5) + 0.5)
    evs = ev.sort_values(["gpsno", "t"])
    dt_min = evs.groupby("gpsno").t.diff().dt.total_seconds() / 60.0
    f["iv_med"] = f.gpsno.map(dt_min.groupby(evs.gpsno).median())

    # ── 簇×险情交互（重建近似：K5 簇内 z）──
    f["ix_cl5_days_since_risk"] = within_z("days_since_risk", "sem5")

    # ── 轨迹精选 ──
    traj = pd.read_csv(traj_csv or TRAJ, dtype={"gpsno": str})
    tcols = {"v_p90": "traj_v_p90", "v80_share": "traj_v80_share", "stop_share": "traj_stop_share",
             "seg_max_h": "traj_seg_max_h", "dv_mean": "traj_dv_mean", "km": "traj_km"}
    f = f.merge(traj[["gpsno"] + list(tcols)].rename(columns=tcols), on="gpsno", how="left")

    # ── FP 诊断交互 ──
    hour = ev.t.dt.hour
    night_mask = (hour >= 21) | (hour < 6)
    n_night = ev[night_mask].groupby("gpsno").row_id.count()
    f["night_share"] = f.gpsno.map(n_night).fillna(0) / (f.n_all + 1)
    f["ix_night_x_rate"] = f.night_share * f.r_all_kk.fillna(0)
    f["ix_night_x_lane"] = f.night_share * f.r_lane_kk.fillna(0)
    f["ix_obs_x_rate"] = f.obs_logh * f.r_all_kk.fillna(0)
    f["ix_obs_x_night"] = f.obs_logh * f.night_share

    meta = {"sample_id", "gpsno", "traj_km", "traj_hours", "obs_s", "obs_logh", "low_km",
            "low_hours", "n_all", "n_all_5d", "n_risk", "n_risk_5d", "n_risk_10d",
            "energy_type", "sem4", "sem5", "night_share"}
    for g in GROUPS:
        meta |= {f"n_{g}"}
    keep = [c for c in f.columns if c not in meta]
    keep = [c for c in keep if pd.to_numeric(f[c], errors="coerce").std(ddof=0) > 1e-12]
    out = f[["sample_id", "gpsno"] + keep].copy()
    out = out.loc[:, ~out.columns.duplicated()]
    print(f"成员4干净版特征: {out.shape[0]} 行 × {out.shape[1] - 2} 列 | 窗 [{W_START.date()},{W_END.date()})"
          f" | 簇规模 K4={f.sem4.value_counts().sort_index().tolist()}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="成员4干净重建（窗口可参数化）")
    ap.add_argument("--w-start", default=None)
    ap.add_argument("--w-end", default=None)
    ap.add_argument("--traj-csv", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    feat = build(w_start=a.w_start, w_end=a.w_end, traj_csv=a.traj_csv)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    feat.to_csv(a.out, index=False, lineterminator="\n")
    print("[saved]", a.out)
