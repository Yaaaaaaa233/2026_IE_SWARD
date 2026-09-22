# -*- coding: utf-8 -*-
"""F2R2 管线编排：一遍扫描（波动+质量+方向无关校准/候选）＋ interface 组装（FEAT-005）。

复用 r1 产物（对齐矩阵、六场景事件表、F2_v1 model_input），不重跑 r1 三遍。
模块级时间约定（r1 缺陷类教训）：epoch 秒一律先 astype("datetime64[ns]") 再 astype(int64)//1e9。
分片内按车辆分组、组内按时间稳定排序后才做相邻行统计（综述：块内乱序）。
amagres 的每车中位数取"分片内该组"中位数（分片局部中心化，与 r1 base_vert 同口径）。
产物写入 F2R2 输出目录（本地受控）；饱和核查结论为扫描第一输出。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .f2_core import postprocess_events
from .interface import META
from .pipeline import COLS_BC, F2Config, _energy_map, _read_shard_cols, _shards, _speed, _window_mask
from .r2_features import (LOOSE_RES_G, LOOSE_WMAG_DPS, MAXW_SPEC, FleetCal, QualAcc, VolAcc,
                          derive_dirless_thresholds, filter_dirless_candidates,
                          fleet_reference, hist_quantile, qual_update, vol_update)

FRONT_DATE = "2026-06-11"  # 特征窗前/后半 10 天分界
R2_VERSION = "F2R2_v1"
DIRLESS_SCENARIOS = ("long_anom", "turn_anom", "collision", "rollover")


@dataclass(frozen=True)
class F2R2Config(F2Config):
    r1_output_dir: Path = Path(".")


def load_f2r2_config(path: str | Path) -> F2R2Config:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    base_path = Path(raw["base_config"])
    if not base_path.is_absolute():
        base_path = source.parent / base_path
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    ev = raw.get("events", {})
    r1_dir = Path(raw["r1_output_dir"])
    if not r1_dir.is_absolute():
        r1_dir = source.parent / r1_dir
    return F2R2Config(
        path=source,
        data_root=Path(base["data_root"]),
        imu_dir=Path(base["data_root"]) / base.get("imu_layer", "v1_annotated") / "imu_clean",
        output_dir=Path(raw["output_dir"]),
        f0_model_input=Path(raw["f0_model_input"]),
        f1_model_input=r1_dir / "model_interface" / "model_input.csv",
        drive_kmh=float(ev.get("driving_speed_kmh", 5.0)),
        moving_kmh=float(ev.get("moving_speed_kmh", 10.0)),
        unify_rel_tol=float(raw.get("unify_rel_tol", 0.10)),
        collision_g=float(ev.get("collision_g", 0.50)),
        tilt_g=float(ev.get("tilt_g", 0.35)),
        rollover_dps=float(ev.get("rollover_dps", 45.0)),
        min_km=float(raw.get("exposure", {}).get("min_km", 50.0)),
        min_hours=float(raw.get("exposure", {}).get("min_hours", 1.0)),
        r1_output_dir=r1_dir.resolve(),
    )


def _epoch_s(series: pd.Series) -> np.ndarray:
    return (pd.to_datetime(series, errors="coerce").astype("datetime64[ns]")
            .astype("int64").to_numpy() // 10**9)


# ------------------------------------------------------------- 一遍扫描

def run_r2_scan(cfg: F2R2Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    R = _r2_R_map(cfg)
    etype = _energy_map(cfg)
    front_epoch = pd.Timestamp(FRONT_DATE).value // 10**9
    vol: dict[str, VolAcc] = {}
    qual: dict[str, QualAcc] = {}
    cal = FleetCal()
    cands: list[pd.DataFrame] = []
    maxw_hist_global = np.zeros(MAXW_SPEC[2], dtype=np.int64)
    total_rows = 0
    shards = _shards(cfg)
    for i, shard in enumerate(shards):
        df = _read_shard_cols(shard, COLS_BC)
        df = df[_window_mask(df)]
        if not len(df):
            print(f"[r2 {i+1}/{len(shards)}] window rows 0", flush=True)
            continue
        total_rows += len(df)
        a = np.column_stack([pd.to_numeric(df[c], errors="coerce") for c in ("ax", "ay", "az")])
        g = np.column_stack([pd.to_numeric(df[c], errors="coerce") for c in ("gx", "gy", "gz")])
        sp = _speed(df)
        t = _epoch_s(df["data_time"])
        hours = pd.to_datetime(df["data_time"], errors="coerce").dt.hour.to_numpy(dtype=float)
        a6 = np.column_stack([a, g])
        maxw = np.max(np.abs(g), axis=1)
        fin_ag = np.isfinite(a).all(axis=1) & np.isfinite(g).all(axis=1)
        order = np.argsort(df["gpsno"].to_numpy(), kind="stable")
        gps_arr = df["gpsno"].to_numpy()
        starts = np.where(np.r_[True, gps_arr[order][1:] != gps_arr[order][:-1]])[0]
        ends = np.r_[starts[1:], len(order)]
        shard_cand = []
        for s, e in zip(starts, ends):
            idx = order[s:e]
            o2 = idx[np.argsort(t[idx], kind="stable")]  # 组内时间稳定排序（乱序防御）
            gps = gps_arr[o2[0]]
            Rm = R.get(gps)
            va = vol.setdefault(gps, VolAcc())
            qa = qual.setdefault(gps, QualAcc())
            m_fin = fin_ag[o2]
            if m_fin.any():
                av, gv, spv, tv, hv = (a[o2][m_fin], g[o2][m_fin], sp[o2][m_fin],
                                       t[o2][m_fin], hours[o2][m_fin])
                vol_update(va, Rm, av, gv, spv, tv, hv, cfg.drive_kmh, front_epoch)
                drv = np.abs(spv) >= cfg.drive_kmh
                if drv.any():
                    res = np.abs(np.linalg.norm(av, axis=1)
                                 - np.median(np.linalg.norm(av[drv], axis=1)))
                    cal.update(str(etype.get(gps, "unknown")), res[drv],
                               np.linalg.norm(gv, axis=1)[drv])
            qual_update(qa, t[o2], a6[o2], maxw[o2])
            maxw_hist_global += qa.maxw_hist
            qa.maxw_hist = np.zeros_like(qa.maxw_hist)
            mv = (np.abs(sp[o2]) >= cfg.moving_kmh) & m_fin
            if mv.any():
                amag_g = np.linalg.norm(a[o2], axis=1)
                drv2 = (np.abs(sp[o2]) >= cfg.drive_kmh) & m_fin
                res_full = np.abs(amag_g - np.median(amag_g[drv2]))
                wmag_full = np.linalg.norm(g[o2], axis=1)
                sel = mv & ((res_full >= LOOSE_RES_G) | (wmag_full >= LOOSE_WMAG_DPS))
                if sel.any():
                    shard_cand.append(pd.DataFrame({
                        "gpsno": gps, "t": t[o2][sel], "res": res_full[sel],
                        "wmag": wmag_full[sel], "speed": sp[o2][sel]}))
        if shard_cand:
            cands.append(pd.concat(shard_cand, ignore_index=True))
        print(f"[r2 {i+1}/{len(shards)}] window_rows={len(df)} cand={sum(len(c) for c in shard_cand)}",
              flush=True)

    # ---- 第一输出：饱和矛盾核查（先于一切特征结论）
    maxw_p99 = hist_quantile(maxw_hist_global, MAXW_SPEC[0], MAXW_SPEC[1], 99.0)
    sat_rows = int(sum(q.n_sat for q in qual.values()))
    const_rows = int(sum(q.n_const for q in qual.values()))
    sat_vehicles = sorted(g for g, q in qual.items() if q.n_sat > 0)
    sat_check = {
        "scope": "feature_window [2026-06-01, 2026-06-21), imu_clean 全车队全量行",
        "rows_scanned": int(total_rows),
        "sat_dps": 300.0,
        "sat_rows": sat_rows,
        "sat_rate": sat_rows / max(total_rows, 1),
        "sat_vehicles": sat_vehicles,
        "n_sat_vehicles": len(sat_vehicles),
        "const_rows_431_9_or_215_9": const_rows,
        "fleet_maxw_dps": max((q.max_w for q in qual.values()), default=0.0),
        "maxw_p99_dps": round(maxw_p99, 2),
        "verdict": "窗内无饱和" if sat_rows == 0 else f"窗内存在饱和行 {sat_rows} 行",
    }
    (cfg.output_dir / "r2_sat_check.json").write_text(
        json.dumps(sat_check, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[r2] 饱和核查（第一输出）：{sat_check['verdict']}；"
          f"max|ω|={sat_check['fleet_maxw_dps']:.1f}dps，定值行={const_rows}", flush=True)

    # ---- 全体阈值与每车特征
    thr = derive_dirless_thresholds(cal, cfg.unify_rel_tol)
    p90, p95 = fleet_reference(cal)
    thr["fleet_p90"] = p90
    thr["fleet_p95"] = p95
    thr["maxw_p99_dps"] = round(maxw_p99, 2)
    thr["n_rows_by_type"] = cal.n_by_type
    (cfg.output_dir / "r2_thresholds.json").write_text(
        json.dumps(thr, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for gps in sorted(vol):
        va, qa = vol[gps], qual.get(gps, QualAcc())
        et = str(etype.get(gps, "unknown"))
        p90m = p90.get(et) or p90["__global__"]
        p95m = p95.get(et) or p95["__global__"]
        rows.append({"gpsno": gps, **va.row_stats(p90m, p95m), **qa.row_stats(maxw_p99)})
    feat = pd.DataFrame(rows)
    feat.to_csv(cfg.output_dir / "r2_vehicle_features.csv", index=False)
    print(f"[r2] 每车特征 {feat.shape} -> r2_vehicle_features.csv", flush=True)

    # ---- 方向无关事件
    cand = pd.concat(cands, ignore_index=True) if cands else pd.DataFrame(
        columns=["gpsno", "t", "res", "wmag", "speed"])
    act = filter_dirless_candidates(cand, thr, etype, cfg.collision_g, cfg.tilt_g,
                                    cfg.rollover_dps)
    events = postprocess_events(act) if len(act) else []
    ev = pd.DataFrame(events)
    if len(ev):
        ev["event_time"] = pd.to_datetime(ev["t_event"], unit="s")
        ev["energy_type"] = ev["gpsno"].map(etype).fillna("unknown")
        ev = ev[["gpsno", "event_time", "scenario", "duration_s", "peak", "speed_kmh",
                 "energy_type"]]
    else:
        ev = pd.DataFrame(columns=["gpsno", "event_time", "scenario", "duration_s", "peak",
                                   "speed_kmh", "energy_type"])
    ev.to_csv(cfg.output_dir / "r2_dirless_events.csv", index=False)
    print(f"[r2] 方向无关事件 {len(ev)} 条", flush=True)
    if len(ev):
        print(ev["scenario"].value_counts().to_string(), flush=True)


def _r2_R_map(cfg: F2R2Config) -> dict:
    al = pd.read_csv(cfg.r1_output_dir / "f2_alignment.csv", dtype={"gpsno": str})
    out = {}
    for r in al.itertuples(index=False):
        if getattr(r, "align_pass", False) and isinstance(r.R, str):
            mat = np.array(json.loads(r.R), dtype=float).reshape(3, 3)
            out[r.gpsno] = mat
    return out


# ------------------------------------------------------------- interface

def build_r2_interface(cfg: F2R2Config) -> Path:
    base = pd.read_csv(cfg.r1_output_dir / "model_interface" / "model_input.csv",
                       encoding="utf-8-sig", dtype={"gpsno": str})
    feat = pd.read_csv(cfg.output_dir / "r2_vehicle_features.csv", dtype={"gpsno": str})
    feat = feat.rename(columns={c: f"f2{c}" for c in feat.columns if c.startswith("r2_")})
    base = base.merge(feat, on="gpsno", how="left", validate="one_to_one")

    rows_ok = pd.to_numeric(base.get("imu_rows_window"), errors="coerce").fillna(0) > 0
    km = pd.to_numeric(base["traj_km_20d"], errors="coerce")
    hours = pd.to_numeric(base["traj_hours_20d"], errors="coerce")
    km_ok = km >= cfg.min_km
    hours_ok = hours >= cfg.min_hours

    ev = pd.read_csv(cfg.output_dir / "r2_dirless_events.csv", dtype={"gpsno": str})
    counts = ev.pivot_table(index="gpsno", columns="scenario", values="event_time",
                            aggfunc="count") if len(ev) else pd.DataFrame()
    for scen in DIRLESS_SCENARIOS:
        cnt = counts[scen] if scen in getattr(counts, "columns", []) else pd.Series(dtype=float)
        cnt = cnt.reindex(base["gpsno"].to_numpy()).fillna(0.0)
        cnt.index = base.index
        cnt = cnt.where(rows_ok, np.nan)
        base[f"f2r2_dirless_{scen}_count_20d"] = cnt
        base[f"f2r2_dirless_{scen}_per_1000km"] = np.where(
            rows_ok & km_ok, cnt * 1000 / km.replace(0, np.nan), np.nan)
        base[f"f2r2_dirless_{scen}_per_100h"] = np.where(
            rows_ok & hours_ok, cnt * 100 / hours.replace(0, np.nan), np.nan)

    ev1 = pd.read_csv(cfg.r1_output_dir / "f2_events.csv", dtype={"gpsno": str})
    hr = pd.to_datetime(ev1["event_time"], errors="coerce").dt.hour
    ev1 = ev1[["gpsno", "scenario"]].assign(is_am=(hr >= 7) & (hr < 9),
                                            is_night=(hr >= 23) | (hr < 5))
    align_ok = pd.to_numeric(base["f2_align_pass"], errors="coerce").fillna(0) == 1
    piv_am = ev1[ev1["is_am"]].pivot_table(index="gpsno", columns="scenario",
                                           values="is_am", aggfunc="count")
    piv_ni = ev1[ev1["is_night"]].pivot_table(index="gpsno", columns="scenario",
                                              values="is_night", aggfunc="count")
    for scen in ("hard_accel", "hard_brake", "turn_l", "sway_s", "collision", "rollover"):
        am = (piv_am[scen] if scen in getattr(piv_am, "columns", [])
              else pd.Series(dtype=float)).reindex(base["gpsno"].to_numpy()).fillna(0.0)
        ni = (piv_ni[scen] if scen in getattr(piv_ni, "columns", [])
              else pd.Series(dtype=float)).reindex(base["gpsno"].to_numpy()).fillna(0.0)
        am.index = ni.index = base.index
        base[f"f2r2_evt_{scen}_am"] = am.where(align_ok, np.nan)
        base[f"f2r2_evt_{scen}_night"] = ni.where(align_ok, np.nan)
    am_all = base[[f"f2r2_evt_{s}_am" for s in ("hard_accel", "hard_brake", "turn_l",
                                                "sway_s", "collision", "rollover")]].sum(
        axis=1, min_count=1)
    ni_all = base[[f"f2r2_evt_{s}_night" for s in ("hard_accel", "hard_brake", "turn_l",
                                                   "sway_s", "collision", "rollover")]].sum(
        axis=1, min_count=1)
    base["f2r2_evt_am_count"] = am_all
    base["f2r2_evt_night_count"] = ni_all
    base["f2r2_evt_am_per_1000km"] = np.where(align_ok & km_ok,
                                              am_all * 1000 / km.replace(0, np.nan), np.nan)
    base["f2r2_evt_night_per_1000km"] = np.where(align_ok & km_ok,
                                                 ni_all * 1000 / km.replace(0, np.nan), np.nan)

    base["feature_version_f2r2"] = R2_VERSION
    meta_all = META | {"feature_version_f2r2", "align_pass", "forward_lateral_var_ratio"}
    feature_cols = [c for c in base.columns if c not in meta_all]
    out_dir = cfg.output_dir / "model_interface"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "model_input.csv"
    base.to_csv(out, index=False)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": R2_VERSION, "rows": int(len(base)),
        "feature_count": len(feature_cols),
        "r2_new_features": len([c for c in feature_cols if c.startswith("f2r2_")]),
        "dirless_events": int(len(ev)), "r1_events": int(len(ev1)),
        "parent": "F2_v1",
    }
    (out_dir / "f2r2_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"OK F2R2 model_input：{len(base)} 行 × {len(feature_cols)} 特征 -> {out}", flush=True)
    return out
