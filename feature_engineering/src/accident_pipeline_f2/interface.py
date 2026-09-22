# -*- coding: utf-8 -*-
"""F2 interface：组装 F2_v1 model_input。

合并：F0 基底（含暴露与画像）＋F1 多窗口族与 prior 计数（FEAT-003 裁定并入）＋
F2 对齐元数据＋六场景聚合（计数 / rate_10k / per_1000km / per_100h 双轨，低暴露置缺失）。
对齐失败车：六场景特征置缺失（f2_align_pass=0）。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import F2Config
from .f2_core import SCENARIOS

MULTIWINDOW_RE = re.compile(
    r"^f1_evt_(total|lane|fatigue|distraction|speed)_(count_(5|10)d|recent_5d_lift)$")
META = {"sample_id", "gpsno", "y", "fold", "feature_version", "feature_version_f1",
        "feature_version_f2", "as_of", "window_start", "window_end",
        "lookback_days", "horizon_days", "label_window", "label_status", "label_version",
        "split_version", "source_version"}


def build_f2_interface(cfg: F2Config) -> Path:
    f0 = pd.read_csv(cfg.f0_model_input, encoding="utf-8-sig", dtype={"gpsno": str})
    f1 = pd.read_csv(cfg.f1_model_input, encoding="utf-8-sig", dtype={"gpsno": str})
    al = pd.read_csv(cfg.output_dir / "f2_alignment.csv", dtype={"gpsno": str})
    ev = pd.read_csv(cfg.output_dir / "f2_events.csv", dtype={"gpsno": str})

    multi_cols = [c for c in f1.columns if MULTIWINDOW_RE.match(c)]
    prior_cols = ["f1_prior_incident_count_20d"]
    base = f0.merge(f1[["sample_id"] + multi_cols + prior_cols], on="sample_id",
                    how="left", validate="one_to_one")
    base = base.merge(al[["gpsno", "align_pass", "forward_lateral_var_ratio"]], on="gpsno",
                      how="left", validate="one_to_one")
    base["f2_align_pass"] = base["align_pass"].fillna(False).astype(int)
    base["f2_forward_lateral_var_ratio"] = base["forward_lateral_var_ratio"]

    km = base["traj_km_20d"]
    hours = base["traj_hours_20d"]
    imu_rows = base.get("imu_rows_window", pd.Series(np.nan, index=base.index))
    km_ok = km >= cfg.min_km
    hours_ok = hours >= cfg.min_hours
    base["f2_low_km_exposure"] = (~km_ok).astype(int)
    base["f2_low_hour_exposure"] = (~hours_ok).astype(int)

    counts = ev.pivot_table(index="gpsno", columns="scenario", values="event_time",
                            aggfunc="count").reindex(base["gpsno"].to_numpy())
    counts.index = base.index
    align_ok = base["f2_align_pass"] == 1
    for scen in SCENARIOS:
        if scen in counts.columns:
            cnt = counts[scen].fillna(0.0)
        else:
            cnt = pd.Series(0.0, index=base.index)
        cnt = cnt.where(align_ok, np.nan)
        base[f"f2_{scen}_count_20d"] = cnt
        base[f"f2_{scen}_rate_10k"] = np.where(
            align_ok & (imu_rows > 0), cnt * 1e4 / imu_rows.replace(0, np.nan), np.nan)
        base[f"f2_{scen}_per_1000km"] = np.where(km_ok, cnt * 1000 / km.replace(0, np.nan), np.nan)
        base[f"f2_{scen}_per_100h"] = np.where(hours_ok, cnt * 100 / hours.replace(0, np.nan), np.nan)
    scen_cols = [f"f2_{s}_count_20d" for s in SCENARIOS]
    base["f2_any_scenario_count"] = base[scen_cols].sum(axis=1, min_count=1)

    prior = pd.to_numeric(base["f1_prior_incident_count_20d"], errors="coerce").fillna(0.0)
    base["f2_prior_incident_per_1000km"] = np.where(km_ok, prior * 1000 / km.replace(0, np.nan), np.nan)
    for ratio, name in (("night_hours_ratio", "night"), ("highway_ratio", "highway")):
        if ratio in base.columns:
            base[f"f2_prior_incident_x_{name}"] = prior * pd.to_numeric(
                base[ratio], errors="coerce").fillna(0.0)

    base["feature_version_f2"] = "F2_v1"
    feature_cols = [c for c in base.columns if c not in META and c not in ("align_pass",
                                                                           "forward_lateral_var_ratio")]
    out_dir = cfg.output_dir / "model_interface"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "model_input.csv"
    base.to_csv(out, index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": "F2_v1",
        "rows": int(len(base)), "feature_count": len(feature_cols),
        "scenarios": list(SCENARIOS), "multiwindow_cols": multi_cols,
        "align_pass_count": int(align_ok.sum()),
        "events_total": int(len(ev)),
        "f0_model_input": str(cfg.f0_model_input), "f1_model_input": str(cfg.f1_model_input),
    }
    (out_dir / "f2_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"OK F2 model_input：{len(base)} 行 × {len(feature_cols)} 特征 -> {out}", flush=True)
    return out
