# -*- coding: utf-8 -*-
"""g1v1 特征工程：事件序列组首版（纯 events_clean 深挖 + 最小公共暴露块）。

信息源（正交声明）：E:/特征工程/数据筛选/契约v0.1全量/v4_assessed/events_clean.csv（309 万行 22 类）
                  + vehicle_day.csv（仅作暴露分母与活跃日，公共块）。
窗口纪律：特征只用 [2026-06-01, 2026-06-21)，不触标签窗（防泄漏断言内置）。
构成（预登记，~100 列）：
  A 语义分组率 7 组×(计数, per_1000km)         ~16
  B 历史险情时间结构                            ~10
  C 多窗口 5/10/20 天 × 4 族 + 新近度抬升        ~18
  D 序列模式首版：类型转移 top12 对、间隔分布、昼夜时段、连环/爆发、日节律  ~45
  E 公共暴露块：km/hours/active_days/低暴露标记  ~6
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
from eval_stack import REPO, feature_columns, load_protocol

EV = "E:/特征工程/数据筛选/契约v0.1全量/v4_assessed/events_clean.csv"
VD = "E:/特征工程/数据筛选/契约v0.1全量/v4_assessed/vehicle_day.csv"
W_START, W_END = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-21")

GROUPS = {
    "lane": {"30002", "30003", "30017"},
    "fatigue": {"41001", "41002", "41029"},
    "distract": {"41003", "41004", "41005", "41009", "41023"},
    "speed_risk": {"11401", "11402", "11403", "11405", "11406"},
    "collision": {"30000", "30005"},
    "blind": {"60292", "60294"},
    "device": {"41006", "41021"},
}
NEAR_MISS, ACCIDENT = "11804", "11803"


def build():
    labels, splits = load_protocol()
    # 标签表只有 sample_id（形如 95201803_20260621_20d），gpsno 从其前缀解析
    labels = labels.assign(gpsno=labels.sample_id.str.split("_").str[0])
    tgt = labels[["sample_id", "gpsno"]].merge(splits[["gpsno", "fold"]], on="gpsno")

    ev = pd.read_csv(EV, usecols=["row_id", "gpsno", "event_type", "start_time", "speed"],
                     dtype={"gpsno": str, "event_type": str})
    ev["t"] = pd.to_datetime(ev.start_time, errors="coerce")
    ev = ev.dropna(subset=["t"])
    ev = ev[(ev.t >= W_START) & (ev.t < W_END)].copy()          # 特征窗
    ev["gpsno"] = ev.gpsno.astype(str)
    ev = ev[ev.gpsno.isin(set(tgt.gpsno))]

    vd = pd.read_csv(VD, dtype={"gpsno": str}, parse_dates=["date"])
    vd = vd[(vd.date >= W_START) & (vd.date < W_END)]
    vd = vd[vd.gpsno.isin(set(tgt.gpsno))]
    expo = vd.groupby("gpsno").agg(traj_km=("traj_km", "sum"),
                                   traj_hours=("traj_hours", "sum"),
                                   active_days=("evt_days_flag", "sum")).reset_index()
    f = tgt.merge(expo, on="gpsno", how="left")
    f["low_km"] = (f.traj_km < 50).astype(int)
    f["low_hours"] = (f.traj_hours < 1).astype(int)

    # ── A 语义分组率 ──
    cnt = ev.pivot_table(index="gpsno", columns="event_type", values="row_id",
                         aggfunc="count").fillna(0.0)
    for g, codes in GROUPS.items():
        cols = [c for c in codes if c in cnt.columns]
        f[f"n_{g}"] = f.gpsno.map(cnt[cols].sum(axis=1) if cols else pd.Series(0.0, index=cnt.index))
        km = f.traj_km.where(f.low_km == 0)
        f[f"r_{g}_per_kk"] = f[f"n_{g}"] / km * 1000
    f["n_all"] = f.gpsno.map(cnt.sum(axis=1))
    f["r_all_per_kk"] = f.n_all / f.traj_km.where(f.low_km == 0) * 1000

    # ── B 历史险情结构 ──
    risk = ev[ev.event_type.isin({ACCIDENT, NEAR_MISS})]
    rc = risk.groupby("gpsno").agg(n_risk=("row_id", "count")).reset_index()
    acc = risk[risk.event_type == ACCIDENT].groupby("gpsno").row_id.count().rename("n_acc").reset_index()
    nm = risk[risk.event_type == NEAR_MISS].groupby("gpsno").row_id.count().rename("n_nm").reset_index()
    f = f.merge(rc, on="gpsno", how="left").merge(acc, on="gpsno", how="left").merge(nm, on="gpsno", how="left")
    for c in ("n_risk", "n_acc", "n_nm"):
        f[c] = f[c].fillna(0)
    f["never_risk"] = (f.n_risk == 0).astype(int)
    last = risk.groupby("gpsno").t.max()
    f["days_since_risk"] = (W_END - f.gpsno.map(last)).dt.days
    f.loc[f.never_risk == 1, "days_since_risk"] = np.nan
    r5 = risk[risk.t >= W_END - pd.Timedelta(days=5)].groupby("gpsno").row_id.count()
    r10 = risk[risk.t >= W_END - pd.Timedelta(days=10)].groupby("gpsno").row_id.count()
    f["n_risk_5d"] = f.gpsno.map(r5).fillna(0)
    f["n_risk_10d"] = f.gpsno.map(r10).fillna(0)
    f["risk_decay_w"] = f.n_risk - 1.0 * f.n_risk_5d - 0.5 * (f.n_risk_10d - f.n_risk_5d)  # 早段权重1 晚段2
    gaps = risk.sort_values("t").groupby("gpsno").t.diff().dt.total_seconds() / 86400
    f["risk_gap_med"] = f.gpsno.map(gaps.groupby(risk.gpsno).median())
    daily_risk = risk.groupby(["gpsno", risk.t.dt.date]).row_id.count().reset_index(name="n")
    burst = daily_risk.groupby("gpsno").n.agg(["max", "mean"])
    f["risk_burst"] = f.gpsno.map(burst["max"]) / (f.gpsno.map(burst["mean"]) + 0.5)

    # ── C 多窗口 + 新近度 ──
    f["n_all_5d"] = f.gpsno.map(ev[ev.t >= W_END - pd.Timedelta(days=5)].groupby("gpsno").row_id.count()).fillna(0)
    f["n_all_10d"] = f.gpsno.map(ev[ev.t >= W_END - pd.Timedelta(days=10)].groupby("gpsno").row_id.count()).fillna(0)
    for g in ("all", "lane", "fatigue", "distract"):
        codes = None if g == "all" else GROUPS[g]
        sub = ev if g == "all" else ev[ev.event_type.isin(codes)]
        for d in (5, 10):
            c = sub[sub.t >= W_END - pd.Timedelta(days=d)].groupby("gpsno").row_id.count()
            f[f"n_{g}_{d}d"] = f.gpsno.map(c).fillna(0)
        n20 = f["n_all"] if g == "all" else f[f"n_{g}"]
        n5 = f["n_all_5d"] if g == "all" else f[f"n_{g}_5d"]
        f[f"lift_{g}_5d"] = (n5 + 0.5) / (n20 / 4 + 0.5)
    f["lift_all_5d"] = (f.n_all_5d + 0.5) / (f.n_all / 4 + 0.5)

    # ── D 序列模式 ──
    ev = ev.sort_values(["gpsno", "t"])
    ev["prev_type"] = ev.groupby("gpsno").event_type.shift(1)
    ev["dt_min"] = ev.groupby("gpsno").t.diff().dt.total_seconds() / 60.0
    # 转移对 top12（全队频次选择，无标签参与；登记为转导性选择）
    tr = ev.dropna(subset=["prev_type"])
    top_pairs = (tr.groupby(["prev_type", "event_type"]).row_id.count().sort_values(ascending=False)
                 .head(12).index.tolist())
    for a, b in top_pairs:
        col = f"tr_{a}_{b}"
        f[col] = f.gpsno.map(tr[(tr.prev_type == a) & (tr.event_type == b)]
                             .groupby("gpsno").row_id.count()).fillna(0)
    # 间隔分布
    gstat = ev.groupby("gpsno").dt_min.agg(iv_med="median", iv_p25=lambda s: s.quantile(.25),
                                           iv_p75=lambda s: s.quantile(.75)).reset_index()
    f = f.merge(gstat[["gpsno", "iv_med", "iv_p25", "iv_p75"]], on="gpsno", how="left")
    # 昼夜（官方时段）
    ev["hour"] = ev.t.dt.hour
    night = ev[(ev.hour >= 21) | (ev.hour < 6)]
    morning = ev[(ev.hour >= 6) & (ev.hour < 9)]
    dusk = ev[(ev.hour >= 18) & (ev.hour < 21)]
    f["night_share"] = f.gpsno.map(night.groupby("gpsno").row_id.count()).fillna(0) / (f.n_all + 1)
    f["night_n"] = f.gpsno.map(night.groupby("gpsno").row_id.count()).fillna(0)
    f["morning_share"] = f.gpsno.map(morning.groupby("gpsno").row_id.count()).fillna(0) / (f.n_all + 1)
    f["dusk_share"] = f.gpsno.map(dusk.groupby("gpsno").row_id.count()).fillna(0) / (f.n_all + 1)
    # 连环（≤30min）
    ev["chain_new"] = ~((ev.dt_min <= 30) & ev.dt_min.notna())
    ev["chain_id"] = ev.groupby("gpsno").chain_new.cumsum()
    ch = ev.groupby(["gpsno", "chain_id"]).size().reset_index(name="len")
    chs = ch.groupby("gpsno").agg(chain_n=("len", "count"), chain_max=("len", "max"))
    f["chain_n"] = f.gpsno.map(chs.chain_n).fillna(0)
    f["chain_max"] = f.gpsno.map(chs.chain_max).fillna(0)
    f["chain_share"] = f.chain_n / (f.n_all + 1)
    # 日节律
    daily = ev.groupby(["gpsno", ev.t.dt.date]).row_id.count().reset_index(name="n")
    ds = daily.groupby("gpsno").n.agg(day_cv=lambda s: s.std() / (s.mean() + 1e-6),
                                      day_max="max", day_mean="mean")
    f["evt_day_cv"] = f.gpsno.map(ds.day_cv)
    f["evt_day_max"] = f.gpsno.map(ds.day_max).fillna(0)
    f["evt_day_mean"] = f.gpsno.map(ds.day_mean).fillna(0)
    f["burst_all"] = f.evt_day_max / (f.evt_day_mean + 0.5)

    # 防泄漏断言与收尾
    assert (ev.t < W_END).all() and (ev.t >= W_START).all()
    meta = {"sample_id", "gpsno", "traj_km", "traj_hours", "active_days", "fold",
            "low_km", "low_hours", "n_all", "n_risk", "n_risk_5d", "n_risk_10d",
            "n_all_5d", "n_all_10d", "evt_day_max", "evt_day_mean", "night_n",
            "n_acc", "n_nm"}
    keep = [c for c in f.columns if c not in meta]
    out = f[["sample_id", "gpsno"] + keep].copy()
    out = out.loc[:, ~out.columns.duplicated()]
    keep = [c for c in out.columns if c not in ("sample_id", "gpsno")]
    keep = [c for c in keep if pd.to_numeric(out[c], errors="coerce").std(ddof=0) > 1e-12]
    out = out[["sample_id", "gpsno"] + keep]
    print(f"g1v1 特征表: {out.shape[0]} 行 × {out.shape[1] - 2} 列", flush=True)
    return out, f


if __name__ == "__main__":
    feat, full = build()
    vdir = os.path.join(HERE, "g1v1", "features")
    os.makedirs(vdir, exist_ok=True)
    feat.to_csv(os.path.join(vdir, "g1v1_features.csv"), index=False, lineterminator="\n")
    print("[saved]", os.path.join(vdir, "g1v1_features.csv"))
