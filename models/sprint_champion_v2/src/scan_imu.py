# -*- coding: utf-8 -*-
"""IMU 表流式扫描（98GB 一遍带出，全向量化）：方向无关模长统计 + 设备健康。

产出（E:/特征工程/冲刺_任务1/imu/imu_vehicle.csv）：
  impact=|sqrt(ax²+ay²+az²)-1|、horizontal=sqrt(ax²+ay²)、rotation=sqrt(gx²+gy²+gz²)
  每车窗内直方图分位（p50/p90/p99/max 近似）、超阈率、|ems-gps| 速度差分位、行数
窗口纪律：data_time ∈ [2026-06-01, 2026-06-21)。
轴向不可靠设备口径：F0 的可靠标志在特征侧按车 join，本扫描只产方向无关量。
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

IMU_DIR = "E:/特征工程/数据筛选/契约v0.1全量/v1_annotated/imu_clean"
OUT_DIR = "E:/特征工程/冲刺_任务1/imu"
W_START, W_END = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-21")
IMPACT_EDGES = np.array([0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.8, 1.2, 2.0, 3.0])
ROT_EDGES = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.6, 2.5, 4.0])
VDIFF_EDGES = np.array([0.5, 1, 2, 3, 5, 8, 12, 18, 25, 35])
CHUNK = 4_000_000


def q_from_hist(csum: np.ndarray, total: float, edges: np.ndarray, q: float, below_frac: float) -> float:
    # below_frac: 无穷小区间占比（值<edges[0]），先行跳过
    idx = int(np.searchsorted(csum, below_frac + q * (total - below_frac)))
    return float(edges[min(idx, len(edges) - 1)])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    labels, splits = load_protocol()
    targets = set(labels.sample_id.str.split("_").str[0])
    shards = sorted(glob.glob(os.path.join(IMU_DIR, "part-*")))
    print(f"shards: {len(shards)}", flush=True)

    acc_i = {}   # gpsno -> impact hist counts
    acc_r = {}   # gpsno -> rotation hist counts
    acc_v = {}   # gpsno -> vdiff hist counts
    acc_h = {}   # gpsno -> horizontal stats (sum, sumsq, n)
    n_rows = {}
    t0 = time.time()
    for si, shard in enumerate(shards):
        for chunk in pd.read_csv(shard, chunksize=CHUNK, sep="\t",
                                 usecols=["gpsno", "data_time", "ems_speed", "gps_speed",
                                          "ax", "ay", "az", "gx", "gy", "gz"],
                                 dtype={"gpsno": str}):
            chunk = chunk[chunk.gpsno.isin(targets)]
            chunk = chunk.dropna(subset=["data_time"])
            t = pd.to_datetime(chunk.data_time, errors="coerce")
            m = (t >= W_START) & (t < W_END)
            chunk = chunk[m]
            if not len(chunk):
                continue
            a = chunk[["ax", "ay", "az"]].to_numpy(float)
            g = chunk[["gx", "gy", "gz"]].to_numpy(float)
            impact = np.abs(np.sqrt(np.nansum(a * a, axis=1)) - 1.0)
            rot = np.sqrt(np.nansum(g * g, axis=1))
            horiz = np.sqrt(np.nansum(a[:, :2] * a[:, :2], axis=1))
            vdiff = (chunk.ems_speed - chunk.gps_speed).abs().to_numpy(float)
            gi, ri, vi = (np.digitize(np.nan_to_num(x, nan=0.0), e) for x, e in
                          ((impact, IMPACT_EDGES), (rot, ROT_EDGES), (vdiff, VDIFF_EDGES)))
            for gno, sel in chunk.groupby("gpsno").indices.items():
                ii = np.bincount(gi[sel], minlength=len(IMPACT_EDGES) + 2)
                rr = np.bincount(ri[sel], minlength=len(ROT_EDGES) + 2)
                vv = np.bincount(vi[sel], minlength=len(VDIFF_EDGES) + 2)
                acc_i[gno] = acc_i.get(gno, 0) + ii
                acc_r[gno] = acc_r.get(gno, 0) + rr
                acc_v[gno] = acc_v.get(gno, 0) + vv
                h = horiz[sel]
                h = h[np.isfinite(h)]
                s, s2, n = acc_h.get(gno, (0.0, 0.0, 0))
                acc_h[gno] = (s + h.sum(), s2 + (h ** 2).sum(), n + len(h))
                n_rows[gno] = n_rows.get(gno, 0) + len(sel)
        print(f"  shard {si + 1}/{len(shards)} ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for gno in sorted(acc_i):
        d = {"gpsno": gno, "imu_rows": n_rows[gno]}
        hi, hr, hv = acc_i[gno], acc_r[gno], acc_v[gno]
        ti, tr, tv = hi.sum(), hr.sum(), hv.sum()
        bi, br, bv = hi[0], hr[0], hv[0]   # below-first-edge counts
        ci, cr, cv = np.cumsum(hi), np.cumsum(hr), np.cumsum(hv)
        for q, name in ((0.5, "p50"), (0.9, "p90"), (0.99, "p99")):
            d[f"impact_{name}"] = q_from_hist(ci, ti, IMPACT_EDGES, q, bi)
            d[f"rot_{name}"] = q_from_hist(cr, tr, ROT_EDGES, q, br)
            d[f"vdiff_{name}"] = q_from_hist(cv, tv, VDIFF_EDGES, q, bv)
        d["impact_over_0.12"] = hi[3:].sum() / ti
        d["impact_over_0.25"] = hi[5:].sum() / ti
        d["rot_over_0.2"] = hr[3:].sum() / tr
        d["vdiff_over_5"] = hv[4:].sum() / tv
        s, s2, n = acc_h[gno]
        d["horiz_mean"] = s / n
        d["horiz_std"] = float(np.sqrt(max(s2 / n - (s / n) ** 2, 0)))
        rows.append(d)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "imu_vehicle.csv"), index=False, lineterminator="\n")
    print(f"[saved] imu_vehicle.csv {out.shape}", flush=True)


if __name__ == "__main__":
    main()
