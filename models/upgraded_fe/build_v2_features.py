# -*- coding: utf-8 -*-
"""升级版特征工程 v2 构建：F1 底座 → 诚实收缩 + 四层金字塔增强。

五项努力（2026-09-23 方案落地）：
 1 IMU 诚实收缩：删除恒零/污染列（impact moderate/severe/extreme、collision/rollover
   candidate、lateral_instability、rotation_max 共 13 列）；新增 imu_axis_tilt_deg
   （由校准表 az_mean 反解真实安装倾角，L0 可信度层）。
 2 暴露—行为分层：全列登记金字塔层级（L0 可信度 / L1 暴露 / L2 行为率 / L3 条件强度 / L4 动态）。
 3 安静车强度分位：同暴露十分位组内的行为强度百分位（f8_ 前缀）。
 4 事件语义增强：事件时车速严重度（压线 speed 分位、高速压线计数、高速事件占比）+
   时段结构（夜间/清晨占比、深夜疲劳占比）+ 多样性（f7_ 前缀）——全部窗口内、新信息。
 5 数据可信度升为一等公民：L0 层显式命名。
输出：outputs/upgraded_features.csv（500×N）、outputs/feature_dictionary_v2.csv、清单 json。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def _load_local_config() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_local.json")
    if not os.path.exists(path):
        raise RuntimeError("复制 models/upgraded_fe/config_local.template.json 为 config_local.json 并填写 feature_root。")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["feature_root"]


E = _load_local_config()
OUT = os.environ.get("UPGRADED_FE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
os.makedirs(OUT, exist_ok=True)
AS_OF = pd.Timestamp("2026-06-21")

BROKEN_IMU = [
    "accident_impact_moderate_count", "accident_impact_moderate_rate_10k",
    "accident_impact_severe_count", "accident_impact_severe_rate_10k",
    "accident_impact_extreme_count", "accident_impact_extreme_rate_10k",
    "accident_collision_candidate_count", "accident_collision_candidate_rate_10k",
    "accident_rollover_candidate_count", "accident_rollover_candidate_rate_10k",
    "accident_lateral_instability_count", "accident_lateral_instability_rate_10k",
    "accident_rotation_max",
]

LANE_TYPES = {30002, 30003, 30017}
FATIGUE_TYPES = {41001, 41002, 41029}


def layer_of(col: str) -> str:
    if col.startswith(("coverage_", "imu_accel", "imu_gyro", "imu_axis", "imu_rows", "f2_tilt")):
        return "L0可信度"
    if col.startswith(("traj_", "monthly_")) or col in ("highway_ratio", "morning_ratio",
                                                        "dusk_ratio", "night_hours_ratio",
                                                        "night_mileage_ratio"):
        return "L1暴露"
    if col.startswith("f8_"):
        return "L3条件强度"
    if col.startswith("f7_"):
        return "L2行为率" if ("per" in col or "ratio" in col or "diversity" in col or "since" in col) else "L3条件强度"
    if any(k in col for k in ("_5d", "_10d", "recency", "prior", "never", "days_since", "lift")):
        return "L4动态"
    if col.startswith("evt_"):
        return "L1暴露" if col in ("evt_count", "evt_count_log1p", "evt_active_days") else "L2行为率"
    if col.startswith("accident_"):
        return "L2行为率"
    if col == "energy_type" or col.startswith("f9_"):
        return "L2行为率"
    return "L2行为率"


def build() -> dict:
    # ---- 底座：F1 ----
    mi = pd.read_csv(f"{E}/artifacts_f1_v1/model_interface/model_input.csv")
    meta = ["sample_id", "gpsno", "as_of", "lookback_days", "y", "fold"]
    X = mi.drop(columns=[c for c in mi.columns if c in meta or c in META_EXTRAS_LOCAL()])
    X.insert(0, "gpsno", mi.gpsno.values)
    X.insert(0, "sample_id", mi.sample_id.values)

    # ---- 1) IMU 诚实收缩 ----
    dropped = [c for c in BROKEN_IMU if c in X.columns]
    X = X.drop(columns=dropped)

    # 轴向倾角（L0）：|az_mean| → 夹角
    cal = pd.read_csv(f"{E}/数据筛选/契约v0.1全量/辅助_imu校准参数.csv")
    cal["f2_tilt_deg"] = np.degrees(np.arccos(cal.az_mean.abs().clip(0, 1)))
    X = X.merge(cal[["gpsno", "f2_tilt_deg"]], on="gpsno", how="left")

    # ---- 4) 事件语义增强（f7_） ----
    ev = pd.read_csv(f"{E}/数据筛选/契约v0.1全量/v4_assessed/events_clean.csv",
                     usecols=["gpsno", "event_type", "start_time", "speed"],
                     parse_dates=["start_time"])
    ev = ev[(ev.start_time >= "2026-06-01") & (ev.start_time < AS_OF)]
    ev["hour"] = ev.start_time.dt.hour
    ev["is_night"] = ev.hour.isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    ev["is_morning_peak"] = ev.hour.isin([7, 8, 9]).astype(int)
    ev["spd"] = ev.speed  # 清洗线已将非法速度置空（A6-A8），此处仅取非空非负
    ev = ev[ev.spd.notna() & (ev.spd >= 0)]

    lane = ev[ev.event_type.isin(LANE_TYPES)]
    agg_lane = lane.groupby("gpsno").spd.agg(lane_spd_p50="median", lane_spd_p90=lambda s: s.quantile(0.9))
    hi_lane = lane[lane.spd >= 60].groupby("gpsno").size().rename("f7_lane_highspd_count")
    spd_p90 = ev.groupby("gpsno").spd.quantile(0.9).rename("f7_evt_spd_p90")
    hi_ratio = (ev.assign(hi=(ev.spd >= 60).astype(int)).groupby("gpsno").hi.mean()
                .rename("f7_highspd_evt_ratio"))
    night_ratio = ev.groupby("gpsno").is_night.mean().rename("f7_night_evt_ratio")
    mp_ratio = ev.groupby("gpsno").is_morning_peak.mean().rename("f7_morning_evt_ratio")
    fat = ev[ev.event_type.isin(FATIGUE_TYPES)]
    night_fat = fat[fat.is_night == 1].groupby("gpsno").size()
    fat_all = fat.groupby("gpsno").size()
    night_fat_share = (night_fat / fat_all).rename("f7_night_fatigue_share")
    diversity = ev.groupby("gpsno").event_type.nunique().rename("f7_evt_type_diversity")
    last_evt = ev.groupby("gpsno").start_time.max()
    f7s = [agg_lane, hi_lane, spd_p90, hi_ratio, night_ratio, mp_ratio, night_fat_share,
           diversity, last_evt]
    F7 = pd.DataFrame(index=pd.Index(mi.gpsno.values, name="gpsno"))
    for s_ in f7s:
        F7 = F7.join(s_)
    F7["f7_days_since_any_evt"] = (AS_OF - F7.pop("start_time")).dt.days
    F7 = F7.reset_index()
    for c in F7.columns:
        if c != "gpsno" and F7[c].isna().all():
            F7 = F7.drop(columns=c)
    X = X.merge(F7, on="gpsno", how="left")

    # ---- 3) 安静车强度分位（f8_）：同暴露十分位组内百分位 ----
    km_bin = pd.qcut(X.traj_km_20d, 10, labels=False, duplicates="drop")
    for name, rate in [("f8_lane_rate_pctl_in_exp", X.evt_lane_count / X.traj_km_20d.replace(0, np.nan)),
                       ("f8_fatigue_rate_pctl_in_exp", X.evt_fatigue_count / X.traj_km_20d.replace(0, np.nan)),
                       ("f8_evt_per_active_day_pctl", X.evt_count / X.evt_active_days.replace(0, np.nan))]:
        grp_rank = pd.Series(rate).groupby(km_bin).rank(pct=True)
        X[name] = grp_rank.values
    X = X.replace([np.inf, -np.inf], np.nan)

    # ---- 分层字典 ----
    dict_rows = []
    for c in X.columns:
        if c in ("sample_id", "gpsno"):
            continue
        dict_rows.append({"feature": c, "layer": layer_of(c),
                          "source": ("F1底座" if c in mi.columns else
                                     "IMU校准表" if c == "f2_tilt_deg" else
                                     "事件明细v4" if c.startswith("f7_") else
                                     "模型线v2" if c.startswith("f8_") else "?"),
                          "missing_rate": round(float(X[c].isna().mean()), 4)})
    dictionary = pd.DataFrame(dict_rows)
    dictionary.to_csv(os.path.join(OUT, "feature_dictionary_v2.csv"), index=False, encoding="utf-8-sig")
    X.to_csv(os.path.join(OUT, "upgraded_features.csv"), index=False)

    manifest = {"n_features": X.shape[1] - 2,
                "dropped_broken_imu": dropped,
                "new_f7": [c for c in F7.columns if c != "gpsno"],
                "new_f8": ["f8_lane_rate_pctl_in_exp", "f8_fatigue_rate_pctl_in_exp",
                            "f8_evt_per_active_day_pctl"],
                "layer_counts": dictionary.layer.value_counts().to_dict()}
    with open(os.path.join(OUT, "upgraded_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def META_EXTRAS_LOCAL():
    return {"feature_version", "source_version", "split_version", "label_version",
            "label_window", "horizon_days", "label_status", "group_rule_version",
            "feature_version_f1", "feature_version_f0"}


if __name__ == "__main__":
    m = build()
    print(json.dumps(m, ensure_ascii=False, indent=1))
