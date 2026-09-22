# -*- coding: utf-8 -*-
"""F2 管线编排：三遍扫描（对齐→校准→提取）＋ interface 组装。

运行阶段（cli）：align / calibrate / extract / interface / run-all。
产物全部写入 F2 输出目录（本地受控，不入公开仓库）；特征工程冻结目录只读。
防泄漏：三遍扫描均限制在特征窗 [WIN_LO, WIN_HI)，无标签参与。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .f2_core import (AlignAcc, TypeHist, acc_update, calibrate_thresholds,
                      finalize_alignment, postprocess_events)

WIN_LO, WIN_HI = "2026-06-01", "2026-06-21"
COLS_A = ["gpsno", "data_time", "data_date", "ems_speed", "gps_speed", "ax", "ay", "az"]
COLS_BC = COLS_A + ["gx", "gy", "gz"]


@dataclass(frozen=True)
class F2Config:
    path: Path
    data_root: Path
    imu_dir: Path
    output_dir: Path
    f0_model_input: Path
    f1_model_input: Path
    drive_kmh: float = 5.0
    moving_kmh: float = 10.0
    min_km: float = 50.0
    min_hours: float = 1.0
    unify_rel_tol: float = 0.10
    collision_g: float = 0.50
    tilt_g: float = 0.35
    rollover_dps: float = 45.0


def load_f2_config(path: str | Path) -> F2Config:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    base_path = Path(raw["base_config"])
    if not base_path.is_absolute():
        base_path = source.parent / base_path
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    ev = raw.get("events", {})
    return F2Config(
        path=source,
        data_root=Path(base["data_root"]),
        imu_dir=Path(base["data_root"]) / base.get("imu_layer", "v1_annotated") / "imu_clean",
        output_dir=Path(raw["output_dir"]),
        f0_model_input=Path(raw["f0_model_input"]),
        f1_model_input=Path(raw["f1_model_input"]),
        drive_kmh=float(ev.get("driving_speed_kmh", 5.0)),
        moving_kmh=float(ev.get("moving_speed_kmh", 10.0)),
        min_km=float(raw.get("exposure", {}).get("min_km", 50.0)),
        min_hours=float(raw.get("exposure", {}).get("min_hours", 1.0)),
        unify_rel_tol=float(raw.get("unify_rel_tol", 0.10)),
        collision_g=float(ev.get("collision_g", 0.50)),
        tilt_g=float(ev.get("tilt_g", 0.35)),
        rollover_dps=float(ev.get("rollover_dps", 45.0)),
    )


def _read_shard_cols(path: Path, cols: list[str]):
    import pyarrow as pa
    import pyarrow.csv as pcsv

    return pcsv.read_csv(
        str(path),
        parse_options=pcsv.ParseOptions(delimiter="\t"),
        convert_options=pcsv.ConvertOptions(include_columns=cols,
                                            column_types={"gpsno": pa.string()}),
        read_options=pcsv.ReadOptions(block_size=1 << 26),
    ).to_pandas()


def _window_mask(df: pd.DataFrame) -> np.ndarray:
    t = df["data_time"].astype(str)
    return ((t >= WIN_LO) & (t < WIN_HI)).to_numpy()


def _speed(df: pd.DataFrame) -> np.ndarray:
    sp = pd.to_numeric(df["gps_speed"], errors="coerce")
    return sp.fillna(pd.to_numeric(df["ems_speed"], errors="coerce")).to_numpy(dtype=float)


def _shards(cfg: F2Config) -> list[Path]:
    return sorted(cfg.imu_dir.glob("part-*"))


# ---------------------------------------------------------------- pass A：对齐

def run_align(cfg: F2Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    acc: dict[str, AlignAcc] = {}
    shards = _shards(cfg)
    for i, shard in enumerate(shards):
        df = _read_shard_cols(shard, COLS_A)
        df = df[_window_mask(df)]
        if not len(df):
            print(f"[align {i+1}/{len(shards)}] window rows 0", flush=True)
            continue
        a = np.column_stack([pd.to_numeric(df[c], errors="coerce") for c in ("ax", "ay", "az")])
        sub = np.isfinite(a).all(axis=1)
        df, a, sp = df[sub], a[sub], _speed(df[sub])
        for gps, grp_df, grp_a, grp_sp in _per_vehicle(df, a, sp):
            acc_update(acc.setdefault(gps, AlignAcc()), grp_a, grp_sp,
                       grp_df["data_date"].astype(str).to_numpy(), cfg.drive_kmh)
        print(f"[align {i+1}/{len(shards)}] vehicles={len(acc)}", flush=True)
    rows = [{"gpsno": g, **finalize_alignment(acc[g])} for g in sorted(acc)]
    out = cfg.output_dir / "f2_alignment.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    n_pass = int(pd.DataFrame(rows)["align_pass"].sum()) if rows else 0
    print(f"OK 对齐完成：{len(rows)} 辆，通过 {n_pass} -> {out}", flush=True)


def _per_vehicle(df: pd.DataFrame, a: np.ndarray, sp: np.ndarray):
    order = np.argsort(df["gpsno"].to_numpy(), kind="stable")
    gps_sorted = df["gpsno"].to_numpy()[order]
    starts = np.where(np.r_[True, gps_sorted[1:] != gps_sorted[:-1]])[0]
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        yield gps_sorted[s], df.iloc[idx], a[idx], sp[idx]


# ---------------------------------------------------------------- pass B：分位数校准

def _R_map(cfg: F2Config) -> dict:
    al = pd.read_csv(cfg.output_dir / "f2_alignment.csv", dtype={"gpsno": str})
    out = {}
    for r in al.itertuples(index=False):
        if getattr(r, "align_pass", False) and isinstance(r.R, str):
            mat = np.array(json.loads(r.R), dtype=float).reshape(3, 3)
            out[r.gpsno] = mat
    return out


def run_calibrate(cfg: F2Config) -> None:
    etype = _energy_map(cfg)
    R = _R_map(cfg)
    per_type: dict[str, TypeHist] = {}
    shards = _shards(cfg)
    for i, shard in enumerate(shards):
        df = _read_shard_cols(shard, COLS_BC)
        df = df[_window_mask(df)]
        df = df[df["gpsno"].isin(R)]
        if not len(df):
            print(f"[calib {i+1}/{len(shards)}] 0", flush=True)
            continue
        a = np.column_stack([pd.to_numeric(df[c], errors="coerce") for c in ("ax", "ay", "az")])
        g = np.column_stack([pd.to_numeric(df[c], errors="coerce") for c in ("gx", "gy", "gz")])
        sp = _speed(df)
        ok = np.isfinite(a).all(axis=1) & np.isfinite(g).all(axis=1) & (np.abs(sp) >= cfg.drive_kmh)
        df, a, g, sp = df[ok].reset_index(drop=True), a[ok], g[ok], sp[ok]
        fwd = np.full(len(df), np.nan)
        lat = np.full(len(df), np.nan)
        gyroz = np.full(len(df), np.nan)
        for gps, grp_df, grp_a, grp_sp in _per_vehicle(df, a, sp):
            Rm = R[gps]
            gidx = grp_df.index.to_numpy()
            gsub = g[gidx]
            fwd[gidx] = grp_a @ Rm[0]
            lat[gidx] = grp_a @ Rm[1]
            gyroz[gidx] = gsub @ Rm[2]
        m = np.isfinite(fwd) & np.isfinite(lat) & np.isfinite(gyroz)
        for et in df["gpsno"].map(etype).fillna("unknown")[m].unique():
            sel = m & (df["gpsno"].map(etype).fillna("unknown") == et).to_numpy()
            per_type.setdefault(et, TypeHist()).update(fwd[sel], np.abs(lat[sel]), np.abs(gyroz[sel]))
        print(f"[calib {i+1}/{len(shards)}] types={list(per_type)}", flush=True)
    cal = calibrate_thresholds(per_type, cfg.unify_rel_tol)
    cal["n_rows_by_type"] = {k: int(v.n) for k, v in per_type.items()}
    out = cfg.output_dir / "f2_thresholds.json"
    out.write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")
    qrows = []
    for et, th in per_type.items():
        from .f2_core import HIST_SPECS, quantile_of
        for key in HIST_SPECS:
            qrows.append({"energy_type": et, "signal": key,
                          "p50": round(quantile_of(key, th.hists[key], 50), 4),
                          "p95": round(quantile_of(key, th.hists[key], 95), 4),
                          "p99": round(quantile_of(key, th.hists[key], 99), 4),
                          "p995": round(quantile_of(key, th.hists[key], 99.5), 4)})
    pd.DataFrame(qrows).to_csv(cfg.output_dir / "f2_signal_quantiles.csv", index=False)
    print(f"OK 校准完成 -> {out}", flush=True)


def _energy_map(cfg: F2Config) -> pd.Series:
    f0 = pd.read_csv(cfg.f0_model_input, encoding="utf-8-sig", dtype={"gpsno": str})
    return f0.set_index("gpsno")["energy_type"]


# ---------------------------------------------------------------- pass C：事件提取

def _thr_for(etype: str, cal: dict) -> dict:
    per = cal.get("per_type", {})
    uni = cal.get("unified", {})
    if etype in per:
        d = dict(per[etype])
    else:
        d = next(iter(per.values())) if per else {"accel_hi": 0.2, "accel_lo": -0.2,
                                                  "gyro": 20.0, "lat": 0.2}
    for k, v in uni.items():
        d[k] = v
    return d


def run_extract(cfg: F2Config) -> None:
    etype = _energy_map(cfg)
    R = _R_map(cfg)
    cal = json.loads((cfg.output_dir / "f2_thresholds.json").read_text(encoding="utf-8"))
    cand = []
    shards = _shards(cfg)
    for i, shard in enumerate(shards):
        df = _read_shard_cols(shard, COLS_BC)
        df = df[_window_mask(df)]
        df = df[df["gpsno"].isin(R)]
        if not len(df):
            continue
        a = np.column_stack([pd.to_numeric(df[c], errors="coerce") for c in ("ax", "ay", "az")])
        g = np.column_stack([pd.to_numeric(df[c], errors="coerce") for c in ("gx", "gy", "gz")])
        sp = _speed(df)
        t = (pd.to_datetime(df["data_time"], errors="coerce").astype("datetime64[ns]")
             .astype("int64").to_numpy() // 10**9)
        ok = np.isfinite(a).all(axis=1) & np.isfinite(g).all(axis=1) & np.isfinite(t)
        df, a, g, sp, t = df[ok].reset_index(drop=True), a[ok], g[ok], sp[ok], t[ok]
        rows = []
        for gps, grp_df, grp_a, grp_sp in _per_vehicle(df, a, sp):
            Rm = R[gps]
            gidx = grp_df.index.to_numpy()
            gsub = g[gidx]
            fwd = grp_a @ Rm[0]
            lat = grp_a @ Rm[1]
            vert = grp_a @ Rm[2]
            gyroz = gsub @ Rm[2]
            thr = _thr_for(str(etype.get(gps, "unknown")), cal)
            base_vert = np.nanmedian(np.abs(vert))
            m_move = grp_sp >= cfg.moving_kmh
            conds = {
                "hard_accel": m_move & (fwd > thr["accel_hi"]),
                "hard_brake": m_move & (fwd < thr["accel_lo"]),
                "turn_l": m_move & (np.abs(gyroz) >= thr["gyro"])
                          & (np.abs(lat) >= thr["lat"]) & (lat * gyroz > 0),
                "sway_s": m_move & (np.abs(gyroz) >= thr["gyro"])
                          & (np.abs(lat) >= thr["lat"]) & (lat * gyroz < 0),
                "collision": m_move & ((fwd <= -cfg.collision_g)
                                       | (np.abs(lat) >= cfg.collision_g)),
                "rollover": m_move & (np.abs(np.abs(vert) - base_vert) >= cfg.tilt_g)
                            & (np.abs(gyroz) >= cfg.rollover_dps),
            }
            vals = {"hard_accel": fwd, "hard_brake": fwd, "turn_l": lat, "sway_s": lat,
                    "collision": fwd, "rollover": gyroz}
            tsub = t[gidx]
            for scen, m in conds.items():
                if m.any():
                    rows.append(pd.DataFrame({
                        "gpsno": gps, "t": tsub[m], "scenario": scen,
                        "value": vals[scen][m], "speed": grp_sp[m]}))
        if rows:
            cand.append(pd.concat(rows, ignore_index=True))
        print(f"[extract {i+1}/{len(shards)}] candidates={sum(len(r) for r in rows)}", flush=True)
    if not cand:
        raise RuntimeError("无任何候选事件")
    events = postprocess_events(pd.concat(cand, ignore_index=True))
    ev = pd.DataFrame(events)
    ev["event_time"] = pd.to_datetime(ev["t_event"], unit="s")
    ev["energy_type"] = ev["gpsno"].map(etype).fillna("unknown")
    ev = ev[["gpsno", "event_time", "scenario", "duration_s", "peak", "speed_kmh", "energy_type"]]
    ev.to_csv(cfg.output_dir / "f2_events.csv", index=False)
    print(f"OK 事件提取：{len(ev)} 条 -> {cfg.output_dir / 'f2_events.csv'}", flush=True)
    print(ev["scenario"].value_counts().to_string(), flush=True)
