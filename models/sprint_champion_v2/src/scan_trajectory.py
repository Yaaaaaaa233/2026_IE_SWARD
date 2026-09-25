# -*- coding: utf-8 -*-
"""轨迹表流式扫描（G2/G3 共用底座）：44GB 一遍带出（全向量化）。

产出（写入 E:/特征工程/冲刺_任务1/trajectory/，只新建文件）：
  traj_day.csv     每车每日：点数、里程、运行时、速度直方图分位、变速强度、停车占比、连续驾驶段
  traj_vehicle.csv 每车窗内聚合
窗口纪律：trigger_time ∈ [2026-06-01, 2026-06-21)。
连续驾驶口径：同车相邻点间隔 ≤30min 为同一驾驶段；块/日边界切段（近似，登记）。
注意：同一车日跨块时以块为单位聚合后再合并（和/均值按 n_pts 加权重建）。
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_stack import load_protocol

TRAJ_DIR = "E:/特征工程/数据筛选/契约v0.1全量/v1_annotated/trajectory_clean"
OUT_DIR = "E:/特征工程/冲刺_任务1/trajectory"
W_START, W_END = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-21")
SPEED_EDGES = np.arange(0, 130, 5.0)
SEG_GAP_MIN = 30.0
CHUNK = 4_000_000


def process_chunk(chunk: pd.DataFrame, targets: set) -> pd.DataFrame:
    chunk = chunk[chunk.gpsno.isin(targets)].copy()
    if not len(chunk):
        return pd.DataFrame()
    chunk = chunk.sort_values(["gpsno", "t"], kind="mergesort")
    gap_min = chunk.groupby("gpsno").t.diff().dt.total_seconds() / 60.0
    same_v = chunk.gpsno.eq(chunk.gpsno.shift())
    new_seg = (~same_v) | (gap_min > SEG_GAP_MIN) | gap_min.isna()
    chunk["seg_id"] = new_seg.cumsum()
    chunk["date"] = chunk.t.dt.date
    chunk["dv"] = chunk.groupby("gpsno").speed_kmh.diff().abs()
    chunk["vbin"] = np.digitize(chunk.speed_kmh.values, SPEED_EDGES)

    # 连续驾驶段（段级聚合 → 车日级）
    seg = chunk.groupby(["gpsno", "date", "seg_id"], sort=False).t.agg(["min", "max"])
    seg["span_h"] = (seg["max"] - seg["min"]).dt.total_seconds() / 3600.0
    seg_day = seg.groupby(["gpsno", "date"]).agg(seg_max_h=("span_h", "max"),
                                                 seg_over4h=("span_h", lambda s: float((s > 4).sum())))

    # 速度直方图（车日×bin 计数）
    hb = chunk.groupby(["gpsno", "date", "vbin"], sort=False).size().rename("c").reset_index()
    hist = hb.pivot_table(index=["gpsno", "date"], columns="vbin", values="c", fill_value=0)

    def q(cols: np.ndarray, weights_row: np.ndarray, qval: float):
        pass

    quant_rows = []
    centers = np.append(SPEED_EDGES[:-1] + 2.5, SPEED_EDGES[-1] + 5)
    hist_m = hist.values.astype(float)
    for i, (gno, d) in enumerate(hist.index):
        w = hist_m[i]
        tot = w.sum()
        if tot == 0:
            quant_rows.append((gno, d, np.nan, np.nan, np.nan))
            continue
        csum = np.cumsum(w)
        p50 = centers[int(np.searchsorted(csum, 0.50 * tot))]
        p90 = centers[int(np.searchsorted(csum, 0.90 * tot))]
        p99 = centers[int(np.searchsorted(csum, 0.99 * tot))]
        quant_rows.append((gno, d, p50, p90, p99))
    qq = pd.DataFrame(quant_rows, columns=["gpsno", "date", "v_p50", "v_p90", "v_p99"])

    base = chunk.groupby(["gpsno", "date"], sort=False).agg(
        n_pts=("speed_kmh", "size"), km=("distance_cm", "sum"), run_h=("run_time_s", "sum"),
        v_mean=("speed_kmh", "mean"), v_std=("speed_kmh", "std"),
        stop_share=("speed_kmh", lambda s: float((s < 5).mean())),
        v80_share=("speed_kmh", lambda s: float((s > 80).mean())),
        dv_mean=("dv", "mean"),
    ).reset_index()
    base["km"] = base.km / 100000.0
    base["run_h"] = base.run_h / 3600.0
    base = base.merge(seg_day.reset_index(), on=["gpsno", "date"], how="left")
    base = base.merge(qq, on=["gpsno", "date"], how="left")
    return base


def merge_dup_days(day: pd.DataFrame) -> pd.DataFrame:
    """同一车日跨块产生的重复行合并：可加量求和、均值按 n_pts 加权、极大取 max。"""
    if not day.duplicated(["gpsno", "date"]).any():
        return day
    w = day.n_pts.clip(lower=1)
    for c in ("v_p50", "v_p90", "v_p99", "v_mean", "dv_mean", "stop_share", "v80_share"):
        day[c] = day[c] * w
    agg = day.groupby(["gpsno", "date"], as_index=False).agg(
        n_pts=("n_pts", "sum"), km=("km", "sum"), run_h=("run_h", "sum"),
        v_mean=("v_mean", "sum"), v_p50=("v_p50", "sum"), v_p90=("v_p90", "sum"),
        v_p99=("v_p99", "sum"), stop_share=("stop_share", "sum"), v80_share=("v80_share", "sum"),
        dv_mean=("dv_mean", "sum"), seg_max_h=("seg_max_h", "max"), seg_over4h=("seg_over4h", "sum"),
        v_std_sum=("v_std", lambda s: float(np.nansum(s))))
    tw = agg.n_pts.clip(lower=1)
    for c in ("v_mean", "v_p50", "v_p90", "v_p99", "stop_share", "v80_share", "dv_mean"):
        agg[c] = agg[c] / tw
    agg = agg.rename(columns={"v_std_sum": "v_std"})
    return agg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    labels, splits = load_protocol()
    targets = set(labels.sample_id.str.split("_").str[0])
    shards = sorted(glob.glob(os.path.join(TRAJ_DIR, "part-*")))
    print(f"shards: {len(shards)}", flush=True)
    parts = []
    t0 = time.time()
    for si, shard in enumerate(shards):
        for chunk in pd.read_csv(shard, chunksize=CHUNK, sep="\t",
                                 usecols=["gpsno", "trigger_time", "speed_kmh",
                                          "distance_cm", "run_time_s"],
                                 dtype={"gpsno": str}):
            chunk["t"] = pd.to_datetime(chunk.trigger_time, errors="coerce")
            chunk = chunk.dropna(subset=["t"])
            chunk = chunk[(chunk.t >= W_START) & (chunk.t < W_END)]
            if not len(chunk):
                continue
            r = process_chunk(chunk, targets)
            if len(r):
                parts.append(r)
        print(f"  shard {si + 1}/{len(shards)} ({time.time() - t0:.0f}s, parts={len(parts)})",
              flush=True)
        if parts and sum(p.shape[0] for p in parts) > 3_000_000:
            parts = [merge_dup_days(pd.concat(parts, ignore_index=True))]
    day = merge_dup_days(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame()
    day.to_csv(os.path.join(OUT_DIR, "traj_day.csv"), index=False, lineterminator="\n")
    print(f"[saved] traj_day.csv {day.shape}", flush=True)

    agg = day.groupby("gpsno").agg(
        v_p50=("v_p50", "mean"), v_p90=("v_p90", "mean"), v_p99=("v_p99", "mean"),
        v_mean=("v_mean", "mean"), v_daycv=("v_mean", lambda s: s.std() / (s.mean() + 1e-6)),
        stop_share=("stop_share", "mean"), v80_share=("v80_share", "mean"),
        dv_mean=("dv_mean", "mean"), seg_max_h=("seg_max_h", "max"),
        seg_over4h=("seg_over4h", "sum"), km=("km", "sum"), run_h=("run_h", "sum"),
        n_days=("date", "nunique"),
    ).reset_index()
    agg["v_cv"] = agg.v_mean / (agg.v_mean + 1e-6)  # 占位修正于消费端
    agg.to_csv(os.path.join(OUT_DIR, "traj_vehicle.csv"), index=False, lineterminator="\n")
    print(f"[saved] traj_vehicle.csv {agg.shape}", flush=True)


if __name__ == "__main__":
    main()
