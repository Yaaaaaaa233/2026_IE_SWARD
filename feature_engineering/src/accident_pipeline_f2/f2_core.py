# -*- coding: utf-8 -*-
"""F2 核心纯函数：对齐充分统计量、分位数校准、事件后处理。

设计依据 docs/plans/feat-004-r1-execution.md（分布先行校准、预登记分位数法+物理护栏、
防泄漏：仅特征窗信号、无标签参与）。本模块不做文件 IO。
坐标系约定：车辆系 = [前向, 横向, 垂直(重力)]，a_vehicle = R @ a_raw，R 行向量依次为
前向/横向/重力轴在原始设备系中的单位向量。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9


# ---------------------------------------------------------------- 对齐（pass A 充分统计量）

@dataclass
class AlignAcc:
    """单辆车对齐充分统计量（窗内、有限值行）。"""

    n: int = 0
    sum_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_drive: int = 0
    sum_a_drive: np.ndarray = field(default_factory=lambda: np.zeros(3))
    m3_drive: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    days: dict = field(default_factory=dict)  # date -> [n, sum_a(3), m3(3,3)]（行驶行）


def acc_update(acc: AlignAcc, a: np.ndarray, speed: np.ndarray,
               dates: np.ndarray, drive_kmh: float) -> None:
    ok = np.isfinite(a).all(axis=1)
    ao, sp, dt = a[ok], speed[ok], dates[ok]
    acc.n += len(ao)
    acc.sum_a += ao.sum(axis=0)
    drv = np.abs(sp) >= drive_kmh
    ad = ao[drv]
    acc.n_drive += len(ad)
    if len(ad):
        acc.sum_a_drive += ad.sum(axis=0)
        acc.m3_drive += ad.T @ ad
        for d in np.unique(dt[drv]):
            m = dt[drv] == d
            st = acc.days.setdefault(d, [0, np.zeros(3), np.zeros((3, 3))])
            st[0] += int(m.sum())
            st[1] += ad[m].sum(axis=0)
            st[2] += ad[m].T @ ad[m]


def _basis(g: np.ndarray) -> np.ndarray:
    """以 g 为 z 轴的正交基 [e1, e2, g]（列向量）。"""
    ref = np.array([1.0, 0.0, 0.0]) if abs(g[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, g)
    e1 /= max(np.linalg.norm(e1), EPS)
    e2 = np.cross(g, e1)
    return np.column_stack([e1, e2, g])


def finalize_alignment(acc: AlignAcc, var_min: float = 1.3, cos_min: float = 0.8,
                       g_lo: float = 0.85, g_hi: float = 1.15) -> dict:
    """由充分统计量得 R 与质量指标（判据沿用探针 D）。"""
    if acc.n == 0 or acc.n_drive < 500:
        return {"align_pass": False, "reason": "insufficient_rows",
                "n": acc.n, "n_drive": acc.n_drive}
    g_norm = float(np.linalg.norm(acc.sum_a) / acc.n)
    g = acc.sum_a / max(np.linalg.norm(acc.sum_a), EPS)
    B = _basis(g)  # 3x2 水平基 + g
    B2 = B[:, :2]
    mean_d = acc.sum_a_drive / acc.n_drive
    cov3 = acc.m3_drive / acc.n_drive - np.outer(mean_d, mean_d)
    cov2 = B2.T @ cov3 @ B2
    cov2 = (cov2 + cov2.T) / 2
    evals, evecs = np.linalg.eigh(cov2)
    lam1, lam2 = float(evals[-1]), float(evals[-2])
    ratio = lam1 / max(lam2, EPS)
    forward = B2 @ evecs[:, -1]
    lateral = np.cross(g, forward)
    R = np.vstack([forward, lateral, g])
    cos_list = []
    for d, (n_d, s_d, m_d) in acc.days.items():
        if n_d >= 1000:
            mu = s_d / n_d
            c2 = B2.T @ (m_d / n_d - np.outer(mu, mu)) @ B2
            _, ev = np.linalg.eigh((c2 + c2.T) / 2)
            cos_list.append(abs(float(ev[:, -1] @ evecs[:, -1])))
    daily_cos = float(np.mean(cos_list)) if cos_list else None
    ok = g_lo <= abs(g_norm) <= g_hi and ratio >= var_min and (daily_cos is None or daily_cos >= cos_min)
    return {"align_pass": bool(ok), "gravity_norm_g": round(g_norm, 4),
            "forward_lateral_var_ratio": round(ratio, 4),
            "daily_pc1_mean_abs_cos": None if daily_cos is None else round(daily_cos, 4),
            "n": acc.n, "n_drive": acc.n_drive,
            "R": [round(float(x), 6) for x in R.ravel()], "reason": ""}


# ---------------------------------------------------------------- 校准（pass B 直方图 → 阈值）

HIST_SPECS = {
    "fwd": (-3.0, 3.0, 600),
    "lat": (0.0, 3.0, 300),
    "gyro_z": (0.0, 200.0, 400),
}


@dataclass
class TypeHist:
    hists: dict = field(default_factory=lambda: {k: np.zeros(n, dtype=np.int64)
                                                 for k, (_, _, n) in HIST_SPECS.items()})
    n: int = 0

    def update(self, fwd: np.ndarray, lat: np.ndarray, gyroz: np.ndarray) -> None:
        self.n += len(fwd)
        for key, arr in (("fwd", fwd), ("lat", lat), ("gyro_z", gyroz)):
            lo, hi, nb = HIST_SPECS[key]
            idx = np.clip(((arr - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
            self.hists[key] += np.bincount(idx, minlength=nb)


def quantile_of(key: str, hist: np.ndarray, q: float) -> float:
    lo, hi, nb = HIST_SPECS[key]
    cum = np.cumsum(hist)
    total = cum[-1]
    if total == 0:
        return float("nan")
    target = q / 100.0 * total
    idx = int(np.searchsorted(cum, target))
    width = (hi - lo) / nb
    return lo + (idx + 0.5) * width


def calibrate_thresholds(per_type: dict[str, TypeHist], unify_rel_tol: float = 0.10) -> dict:
    """预登记分位数法＋物理护栏；类型间差异 < tol 时统一为全局阈值。"""
    raw = {}
    for etype, th in per_type.items():
        raw[etype] = {
            "accel_hi": np.clip(quantile_of("fwd", th.hists["fwd"], 99.5), 0.10, 0.50),
            "accel_lo": np.clip(quantile_of("fwd", th.hists["fwd"], 0.5), -0.50, -0.10),
            "gyro": np.clip(quantile_of("gyro_z", th.hists["gyro_z"], 99.0), 10.0, 60.0),
            "lat": np.clip(quantile_of("lat", th.hists["lat"], 99.0), 0.10, 0.40),
        }
    out = {"per_type": {k: {m: round(float(v), 4) for m, v in d.items()} for k, d in raw.items()},
           "unified": {}, "unify_log": []}
    for m in ("accel_hi", "accel_lo", "gyro", "lat"):
        vals = np.array([raw[e][m] for e in raw])
        if len(vals) == 0:
            continue
        rel = (vals.max() - vals.min()) / max(abs(vals.mean()), EPS)
        if rel < unify_rel_tol:
            out["unified"][m] = round(float(vals.mean()), 4)
            out["unify_log"].append(f"{m}: 类型间相对差异 {rel:.3f} < {unify_rel_tol} → 统一阈值")
        else:
            out["unify_log"].append(f"{m}: 类型间相对差异 {rel:.3f} ≥ {unify_rel_tol} → 分层阈值")
    return out


# ---------------------------------------------------------------- 事件后处理（pass C 候选 → 事件表）

SCENARIOS = ("hard_accel", "hard_brake", "turn_l", "sway_s", "collision", "rollover")


def postprocess_events(df) -> list[dict]:
    """对护栏级激活行做事件分组。df 列：gpsno, t(epoch s), scenario, value, speed。
    规则：同型事件按时间排序，行间隔 ≤ group_gap_s 归同组；组内取峰值行；
    同型事件起点间隔 < cooldown_s 则合并（保留峰值更高者）。
    """
    events: list[dict] = []
    group_gap, cooldown = 2.0, 30.0
    for (gpsno, scen), g in df.groupby(["gpsno", "scenario"], sort=False):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        val = np.abs(g["value"].to_numpy())
        sp = g["speed"].to_numpy()
        starts = np.where(np.r_[True, np.diff(t) > group_gap])[0]
        ends = np.r_[starts[1:], len(t)]
        for s, e in zip(starts, ends):
            seg = slice(s, e)
            peak = s + int(np.argmax(val[seg]))
            events.append({"gpsno": gpsno, "scenario": scen,
                           "t_event": float(t[peak]), "duration_s": float(t[e - 1] - t[s]),
                           "peak": float(g["value"].to_numpy()[peak]),
                           "speed_kmh": float(sp[peak])})
    # 同型冷却合并（每键事件列表，与最近一条比较）
    final: dict = {}
    ordered = []
    for ev in sorted(events, key=lambda x: (x["gpsno"], x["scenario"], x["t_event"])):
        key = (ev["gpsno"], ev["scenario"])
        prev = final.get(key)
        if prev and ev["t_event"] - prev["t_event"] < cooldown:
            if abs(ev["peak"]) > abs(prev["peak"]):
                final[key] = ev
                ordered[-1] = ev
        else:
            final[key] = ev
            ordered.append(ev)
    return ordered
