# -*- coding: utf-8 -*-
"""F2R2 核心纯函数：波动统计、设备质量统计、方向无关候选与阈值（FEAT-005）。

设计依据 docs/plans/feat-005-r2-execution.md §3（波动/健康分/方向无关三族）。
预登记定义要点：
- 波动统计一律在行驶行（|speed| >= drive_kmh）、组内时间排序后计算；
- 相对波动＝车辆行驶样本超过同能源类型全体 p90/p95 的占比（“本车 p90 占比”按定义恒为
  10%/5% 无信息，故修正为全体分位参照，理由随轮次登记）；
- 变异系数只对正值合模长信号（amagres/wmag）定义，零中心方向信号均值≈0 时 CV 无意义；
- 逐字重复＝时间排序后相邻两行六轴完全相等且间隔 <=2s 的行对占比；
- 段首毛刺＝以间隔 >60s 切段后，段首行 max|gyro| 超过全体 p99 的段占比；
- 饱和行＝max(|gx|,|gy|,|gz|) > sat_dps；去饱和统计在全部扫描中同步计算，无论核查结论。
本模块不做文件 IO。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-9
SAT_DPS = 300.0
SAT_CONSTS = (431.9, 215.9)  # 综述样例级饱和/半量程定值（全量核查对象）
SAT_CONST_TOL = 0.05
SEG_GAP_S = 60.0
DUP_MAX_GAP_S = 2.0
MIN_DRIVE_ROWS = 500      # 波动族最低行驶样本（与对齐判据一致）
MIN_SEG_ROWS = 300        # 时段分波动最低样本
MIN_HALF_ROWS = 100       # 趋势前后半窗最低样本
MIN_DUP_PAIRS = 100       # 重复率最低分母
MIN_SEGMENTS = 5          # 毛刺率最低段数
LOOSE_RES_G = 0.05        # 方向无关候选的宽松预筛界（阈值下护栏 0.10 之上留余量）
LOOSE_WMAG_DPS = 15.0     # 同上（阈值下护栏 10）

# 每车直方图规格（IQR/相对占用率用；区间覆盖 r1 校准同一量级）
R2_HIST_SPECS = {
    "fwd": (-3.0, 3.0, 600),
    "lat": (-3.0, 3.0, 600),
    "vert": (-3.0, 3.0, 600),
    "gyroz": (-200.0, 200.0, 400),
    "amagres": (0.0, 2.0, 400),
    "wmag": (0.0, 350.0, 350),
}
MAXW_SPEC = (0.0, 350.0, 350)  # max|gyro| 全体行（毛刺 p99 与饱和核查）
# 分能源类型校准只用这两个合模长信号（方向无关族的阈值来源）
CAL_SIGNALS = ("amagres", "wmag")


def hist_quantile(hist: np.ndarray, lo: float, hi: float, q: float) -> float:
    cum = np.cumsum(hist)
    total = cum[-1]
    if total == 0:
        return float("nan")
    target = q / 100.0 * total
    idx = int(np.searchsorted(cum, target))
    nb = len(hist)
    return lo + (min(idx, nb - 1) + 0.5) * (hi - lo) / nb


def _spec(key: str) -> tuple[float, float, int]:
    return R2_HIST_SPECS.get(key, MAXW_SPEC)


@dataclass
class SignalAcc:
    """单信号充分统计量：sum/sumsq 给 std，直方图给 IQR 与相对占用率。"""

    key: str
    n: int = 0
    s: float = 0.0
    ss: float = 0.0
    hist: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.hist is None:
            self.hist = np.zeros(_spec(self.key)[2], dtype=np.int64)

    @property
    def lo(self) -> float:
        return _spec(self.key)[0]

    @property
    def hi(self) -> float:
        return _spec(self.key)[1]

    def update(self, x: np.ndarray) -> None:
        x = x[np.isfinite(x)]
        if not len(x):
            return
        self.n += len(x)
        self.s += float(x.sum())
        self.ss += float((x * x).sum())
        nb = len(self.hist)
        idx = np.clip(((x - self.lo) / (self.hi - self.lo) * nb).astype(int), 0, nb - 1)
        self.hist += np.bincount(idx, minlength=nb)

    def mean(self) -> float:
        return self.s / self.n if self.n else float("nan")

    def std(self) -> float:
        if self.n < 2:
            return float("nan")
        var = self.ss / self.n - (self.s / self.n) ** 2
        return float(np.sqrt(max(var, 0.0)))

    def quantile(self, q: float) -> float:
        return hist_quantile(self.hist, self.lo, self.hi, q)

    def iqr(self) -> float:
        return self.quantile(75.0) - self.quantile(25.0)

    def cv(self) -> float:
        m = abs(self.mean())
        return self.std() / m if self.n and m > EPS else float("nan")

    def occupancy_above(self, thr) -> float:
        if self.n == 0 or thr is None or not np.isfinite(thr):
            return float("nan")
        nb = len(self.hist)
        k = int(np.clip((thr - self.lo) / (self.hi - self.lo) * nb, 0, nb))
        return float(self.hist[k:].sum() / self.n)


def _new_signal(key: str) -> SignalAcc:
    return SignalAcc(key=key)


@dataclass
class VolAcc:
    """单辆车波动族充分统计量（行驶行）。"""

    sig: dict = field(default_factory=lambda: {k: _new_signal(k) for k in R2_HIST_SPECS})
    wmag_desat: SignalAcc = field(default_factory=lambda: _new_signal("wmagdesat"))
    front: dict = field(default_factory=lambda: {k: _new_signal(k) for k in CAL_SIGNALS})
    back: dict = field(default_factory=lambda: {k: _new_signal(k) for k in CAL_SIGNALS})
    am: dict = field(default_factory=lambda: {k: _new_signal(k) for k in ("amagres", "fwd")})
    night: dict = field(default_factory=lambda: {k: _new_signal(k) for k in ("amagres", "fwd")})

    def row_stats(self, fleet_p90: dict, fleet_p95: dict) -> dict:
        """汇总为本轮预登记的波动特征（行驶行不足 MIN_DRIVE_ROWS 置缺失）。"""
        out: dict = {}
        enough = self.sig["amagres"].n >= MIN_DRIVE_ROWS
        for key in R2_HIST_SPECS:
            sa = self.sig[key]
            out[f"r2_vol_{key}_std"] = sa.std()
            out[f"r2_vol_{key}_iqr"] = sa.iqr()
            if key in CAL_SIGNALS:
                # 相对占用率仅合模长信号（全车队可比）；方向类只对对齐车有意义且与 std 共线
                p90, p95 = fleet_p90.get(key), fleet_p95.get(key)
                out[f"r2_rel_{key}_p90"] = sa.occupancy_above(p90)
                out[f"r2_rel_{key}_p95"] = sa.occupancy_above(p95)
                out[f"r2_vol_{key}_cv"] = sa.cv()
            if not enough:
                cols = [f"r2_vol_{key}_std", f"r2_vol_{key}_iqr"]
                if key in CAL_SIGNALS:
                    cols += [f"r2_rel_{key}_p90", f"r2_rel_{key}_p95", f"r2_vol_{key}_cv"]
                for c in cols:
                    out[c] = float("nan")
        out["r2_vol_wmagdesat_std"] = (self.wmag_desat.std()
                                       if self.wmag_desat.n >= MIN_DRIVE_ROWS else float("nan"))
        for key in CAL_SIGNALS:
            f, b = self.front[key], self.back[key]
            out[f"r2_vol_{key}_fbratio"] = (f.std() / max(b.std(), EPS)
                                            if f.n >= MIN_HALF_ROWS and b.n >= MIN_HALF_ROWS
                                            and np.isfinite(b.std()) and b.std() > EPS
                                            else float("nan"))
        for tag, seg in (("am", self.am), ("night", self.night)):
            for key, sa in seg.items():
                out[f"r2_vol_{key}_std_{tag}"] = sa.std() if sa.n >= MIN_SEG_ROWS else float("nan")
        return out


@dataclass
class QualAcc:
    """单辆车设备质量充分统计量（全部有限行，时间排序后）。"""

    n_rows: int = 0
    n_pairs: int = 0
    n_dup: int = 0
    n_sat: int = 0
    n_const: int = 0
    max_w: float = 0.0
    seg_count: int = 0
    seg_first: list = field(default_factory=list)
    maxw_hist: np.ndarray = field(default_factory=lambda: np.zeros(MAXW_SPEC[2], dtype=np.int64))

    def row_stats(self, maxw_p99: float) -> dict:
        dup = self.n_dup / self.n_pairs if self.n_pairs >= MIN_DUP_PAIRS else float("nan")
        glitch = (sum(1 for v in self.seg_first if v > maxw_p99) / self.seg_count
                  if self.seg_count >= MIN_SEGMENTS else float("nan"))
        return {
            "r2_qa_maxw": self.max_w if self.n_rows else float("nan"),
            "r2_qa_sat_count": float(self.n_sat),
            "r2_qa_sat_rate": self.n_sat / self.n_rows if self.n_rows else float("nan"),
            "r2_qa_const_count": float(self.n_const),
            "r2_qa_dup_rate": dup,
            "r2_qa_glitch_rate": glitch,
            "r2_qa_segments": float(self.seg_count),
        }


def vol_update(vol: VolAcc, R: np.ndarray | None, a: np.ndarray, g: np.ndarray,
               sp: np.ndarray, t_epoch: np.ndarray, hours: np.ndarray,
               drive_kmh: float, front_epoch_s: float) -> None:
    """波动族行级更新。a/g 为有限值行（已过滤），t/hours 与行对齐；需已按时间排序。"""
    amag = np.linalg.norm(a, axis=1)
    wmag = np.linalg.norm(g, axis=1)
    drv = np.abs(sp) >= drive_kmh
    if not drv.any():
        return
    med = float(np.median(amag[drv]))
    res = np.abs(amag - med)
    if R is not None:
        fwd = a @ R[0]
        lat = a @ R[1]
        vert = a @ R[2]
        gyroz = g @ R[2]
    else:
        fwd = lat = vert = gyroz = None
    for key, arr in (("amagres", res), ("wmag", wmag)):
        vol.sig[key].update(arr[drv])
    if fwd is not None:
        for key, arr in (("fwd", fwd), ("lat", lat), ("vert", vert), ("gyroz", gyroz)):
            vol.sig[key].update(arr[drv])
    sat = np.max(np.abs(g), axis=1)
    vol.wmag_desat.update(wmag[drv & (sat <= SAT_DPS)])
    front = t_epoch < front_epoch_s
    for key, arr in (("amagres", res), ("wmag", wmag)):
        vol.front[key].update(arr[drv & front])
        vol.back[key].update(arr[drv & ~front])
    is_am = (hours >= 7) & (hours < 9)
    is_night = (hours >= 23) | (hours < 5)
    for key, arr in (("amagres", res), ("fwd", fwd)):
        if arr is None:
            continue
        vol.am[key].update(arr[drv & is_am])
        vol.night[key].update(arr[drv & is_night])


def qual_update(qa: QualAcc, t_epoch: np.ndarray, a6: np.ndarray, maxw: np.ndarray) -> None:
    """质量族行级更新。输入为全体行（未滤六轴有限），a6=[ax..az,gx..gz]，需已按时间排序。"""
    fin_t = np.isfinite(t_epoch)
    fin6 = np.isfinite(a6).all(axis=1)
    ok = fin_t & fin6
    t, a6o, mw = t_epoch[ok], a6[ok], maxw[ok]
    qa.n_rows += len(t)
    if len(t) >= 2:
        dt = np.diff(t)
        same = (a6o[1:] == a6o[:-1]).all(axis=1)
        qa.n_pairs += int(len(dt))
        qa.n_dup += int((same & (dt <= DUP_MAX_GAP_S)).sum())
        gap = np.r_[np.inf, dt] > SEG_GAP_S
        qa.seg_count += int(gap.sum())
        qa.seg_first.extend(mw[gap].tolist())
    qa.n_sat += int((mw > SAT_DPS).sum())
    qa.n_const += int(sum((np.abs(mw - c) <= SAT_CONST_TOL).sum() for c in SAT_CONSTS))
    if len(mw):
        qa.max_w = max(qa.max_w, float(mw.max()))
        nb = MAXW_SPEC[2]
        idx = np.clip(((mw - MAXW_SPEC[0]) / (MAXW_SPEC[1] - MAXW_SPEC[0]) * nb).astype(int),
                      0, nb - 1)
        qa.maxw_hist += np.bincount(idx, minlength=nb)


@dataclass
class FleetCal:
    """分能源类型合模长直方图（方向无关阈值与相对占用率的全体参照）。"""

    per_type: dict = field(default_factory=dict)
    n_by_type: dict = field(default_factory=dict)

    def update(self, etype: str, res: np.ndarray, wmag: np.ndarray) -> None:
        d = self.per_type.setdefault(etype, {k: np.zeros(R2_HIST_SPECS[k][2], dtype=np.int64)
                                             for k in CAL_SIGNALS})
        self.n_by_type[etype] = self.n_by_type.get(etype, 0) + len(res)
        for key, arr in (("amagres", res), ("wmag", wmag)):
            lo, hi, nb = R2_HIST_SPECS[key]
            idx = np.clip(((arr - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
            d[key] += np.bincount(idx, minlength=nb)

    def quantiles(self) -> dict:
        out = {}
        for et, d in self.per_type.items():
            out[et] = {}
            for key in CAL_SIGNALS:
                lo, hi, _ = R2_HIST_SPECS[key]
                out[et][key] = {qname: hist_quantile(d[key], lo, hi, q)
                                for qname, q in (("p90", 90), ("p95", 95),
                                                 ("p99", 99), ("p995", 99.5))}
        return out


def fleet_reference(cal: FleetCal) -> tuple[dict, dict]:
    """相对占用率的全体参照（分能源类型 p90/p95，未知类型回退全局）。"""
    qs = cal.quantiles()
    glob = {}
    for key in CAL_SIGNALS:
        lo, hi, _ = R2_HIST_SPECS[key]
        merged = np.zeros(R2_HIST_SPECS[key][2], dtype=np.int64)
        for d in cal.per_type.values():
            merged = merged + d[key]
        glob[key] = {qname: hist_quantile(merged, lo, hi, q)
                     for qname, q in (("p90", 90), ("p95", 95))}
    p90 = {et: {k: v["p90"] for k, v in d.items()} for et, d in qs.items()}
    p95 = {et: {k: v["p95"] for k, v in d.items()} for et, d in qs.items()}
    p90["__global__"] = {k: v["p90"] for k, v in glob.items()}
    p95["__global__"] = {k: v["p95"] for k, v in glob.items()}
    return p90, p95


def derive_dirless_thresholds(cal: FleetCal, unify_rel_tol: float = 0.10) -> dict:
    """方向无关触发阈值：合模长残差 p99.5（护栏 [0.10,0.50] g）、|ω| p99（护栏 [10,60] dps）；
    类型间相对差异 < tol 时统一（与 r1 校准规则同构）。"""
    raw = {}
    for et, q in cal.quantiles().items():
        raw[et] = {
            "long": float(np.clip(q["amagres"]["p995"], 0.10, 0.50)),
            "turn": float(np.clip(q["wmag"]["p99"], 10.0, 60.0)),
        }
    out = {"per_type": {k: {m: round(v, 4) for m, v in d.items()} for k, d in raw.items()},
           "unified": {}, "unify_log": []}
    for m in ("long", "turn"):
        vals = np.array([raw[e][m] for e in raw]) if raw else np.array([])
        if len(vals) == 0:
            continue
        rel = (vals.max() - vals.min()) / max(abs(vals.mean()), EPS)
        if rel < unify_rel_tol:
            out["unified"][m] = round(float(vals.mean()), 4)
            out["unify_log"].append(f"{m}: 类型间相对差异 {rel:.3f} < {unify_rel_tol} → 统一阈值")
        else:
            out["unify_log"].append(f"{m}: 类型间相对差异 {rel:.3f} ≥ {unify_rel_tol} → 分层阈值")
    return out


def filter_dirless_candidates(cand: pd.DataFrame, thr: dict, etype_map, collision_g: float,
                              tilt_g: float, rollover_dps: float) -> pd.DataFrame:
    """候选行 → 方向无关场景激活行（postprocess_events 输入格式）。

    cand 列：gpsno, t, res, wmag（已限行驶行与宽松预筛）。
    场景：long_anom（急加速/急刹合并，合模长残差）、turn_anom（急转/猛打合并，|ω|）、
    collision（物理残差 >=collision_g）、rollover（残差 >=tilt_g 且 |ω| >=rollover_dps）。
    """
    et = cand["gpsno"].map(etype_map).fillna("unknown")
    per = thr.get("per_type", {})
    uni = thr.get("unified", {})

    def _thr_for(e: str, key: str, default: float) -> float:
        d = per.get(e, next(iter(per.values())) if per else {})
        return float(uni.get(key, d.get(key, default)))

    long_t = et.map(lambda e: _thr_for(e, "long", 0.20)).to_numpy()
    turn_t = et.map(lambda e: _thr_for(e, "turn", 20.0)).to_numpy()
    res = cand["res"].to_numpy()
    wmag = cand["wmag"].to_numpy()
    conds = {
        "long_anom": res >= long_t,
        "turn_anom": wmag >= turn_t,
        "collision": res >= collision_g,
        "rollover": (res >= tilt_g) & (wmag >= rollover_dps),
    }
    vals = {"long_anom": res, "turn_anom": wmag, "collision": res, "rollover": wmag}
    rows = []
    for scen, m in conds.items():
        if m.any():
            rows.append(pd.DataFrame({
                "gpsno": cand["gpsno"].to_numpy()[m], "t": cand["t"].to_numpy()[m],
                "scenario": scen, "value": vals[scen][m],
                "speed": cand["speed"].to_numpy()[m]}))
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["gpsno", "t", "scenario", "value", "speed"])
