# -*- coding: utf-8 -*-
"""F3 Tier 2 轨迹族纯函数：连续驾驶切段、速度形态、空间结构、日节律。

FEAT-007 r4 建立本模块；FEAT-008 r5 按 docs/plans/feat-008-r5-execution.md §2 预登记
修复（执行中不得擅改）：
- P1 坐标跳变过滤：相邻点隐含速度 >70 m/s 判坐标跳变——该步不计里程/运行时长并切断段；
  产出质量列 f3_traj_jump_steps（跳变步计数，供任务二，无样本门槛）；
- P2 停车双切段：连续驻车步（速度=0 或位移 <50m）自出发点至末驻车到达点跨越 >=30min
  （边界含）判驻车块切段（块内点不属任何段），间隔 >180min 与跳变步亦切段；
  段内 <30min 短停计入行驶时长；
- P4 时区统一：小时/日界一律本地（北京时间 UTC+8）口径，与 core.hour_of_day 一致。

设计依据 docs/plans/feat-007-r4-execution.md §3.0 预登记定义（原口径继承）：
- 特征窗 [as_of-lookback, as_of)：窗口边界一律显式传参（红线 6），window_end_s 即 as_of，
  各特征函数内部按半开区间 [window_start_s, window_end_s) 过滤，一切统计不触
  [as_of, as_of+H) 标签窗（合成测试断言）；
- 速度形态：速度 p50/p90/p99、分时段超速占比（>90km/h，早 7-9/深夜 23-5 分桶）、速度 CV、
  高速巡航占比（相邻两点均 >80km/h 的累计时长占比；道路类型代理，禁地图匹配）；
- 空间结构：仅自包含统计，禁任何外部地图/路网/POI 数据；事件坐标 DBSCAN（Haversine，
  eps 200m，min_samples 2，sklearn）→ 热点数、热点事件占比、同点位复发次数；路线重复度＝
  相邻日路径点集平均最近邻距离；活动半径＝相对质心 RMS 距离；昼夜活动区偏移＝
  日间(9-17)/深夜(23-5)点集质心距离；
- 日节律：每日里程 CV、断档天数、日均运行时长；
- 最低样本门槛沿用 r2（500/300/100，不足置缺失）：500＝窗内轨迹有效点（全体点级统计）、
  300＝时段桶/昼夜点集、100＝事件坐标与相邻日子集；日级统计另需有效天数 >=min_days
  （随实现登记，见 daily_rhythm_features）；
- 时间戳转秒统一 astype("datetime64[ns]") -> astype(int64) // 1e9（r1 缺陷类教训）；
- 原始高频序列不直接入模，入模均为统计量与阈值事件计数（红线 4）。本模块不做文件 IO；
- 统一清洗（红线 2：缺数以缺失表达、不删车）：所有消费 lat/lon/t/speed 的函数入口丢弃
  非有限值行（finite_row_mask/window_mask 收口），空输入/单点输入返回 NaN 特征不崩溃，
  非法坐标不进 DBSCAN（视为噪声）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

EPS = 1e-9
EARTH_R_M = 6_371_000.0          # 地球平均半径（米），Haversine 用
SPELL_GAP_MIN = 180.0            # §3.0 切段阈值：相邻点间隔 >180 min 切段
SPELL_GAP_S = SPELL_GAP_MIN * 60.0
LONG_SPELL_H = 4.0               # >4h 连续驾驶占比阈值
OVER_SPEED_KMH = 90.0            # 分时段超速阈值
CRUISE_KMH = 80.0                # 高速巡航阈值（道路类型代理）
AM_LO, AM_HI = 7, 9              # 早桶 [7,9)
DAY_LO, DAY_HI = 9, 17           # 日间桶 [9,17)（与 §3.0 夜间退化度日间口径一致）
NIGHT_LO, NIGHT_HI = 23, 5       # 深夜桶 [23,24)∪[0,5)
# ---- r5 预登记修复口径（docs/plans/feat-008-r5-execution.md §2，执行中不得擅改）----
TZ_OFFSET_S = 8.0 * 3600.0       # P4：小时/日界统一北京时间 UTC+8（与 core.hour_of_day 一致）
JUMP_SPEED_MPS = 70.0            # P1：隐含速度 >70 m/s（≈252km/h）判坐标跳变
STOP_DISP_M = 50.0               # P2：步位移 <50m 视为驻车信号
STOP_SPEED_KMH = 0.0             # P2：速度=0 判驻车信号（预登记字面口径）
STOP_SPAN_S = 30.0 * 60.0        # P2：驻车块持续 >=30min（边界含）切段
STOP_LABEL = -2                  # P2：>=30min 驻车块行标签（不属任何连续驾驶段；窗外仍为 -1）
DBSCAN_EPS_M = 200.0             # §3.0 空间热点：eps 200m
DBSCAN_MIN_SAMPLES = 2           # §3.0 空间热点：min_samples 2
MIN_MAIN_ROWS = 500              # 最低样本门槛（r2 沿用）：窗内轨迹有效点
MIN_BUCKET_ROWS = 300            # 最低样本门槛（r2 沿用）：时段桶/昼夜点集
MIN_SUB_ROWS = 100               # 最低样本门槛（r2 沿用）：事件坐标/相邻日子集
MIN_DAYS = 5                     # 日级统计最低有效天数（§3.0 未覆盖处的实现口径）
CALIBER_TRIP = "trip_cumulative"  # 口径标志：运行时长字段为单行程累计，按字段重置切段
CALIBER_GAP = "gap_180min"        # 口径标志：按相邻点间隔 >180 min 切段
RUN_RESET_TOL_S = 1e-6           # 字段重置判定容差（秒）
RUN_RESET_START_FRAC = 0.20      # 重置后"从近零重新累计"判定比例
NEAR_ZERO_S = 1e-6               # 近零累计判定绝对容差（秒）


def to_epoch_s(times) -> np.ndarray:
    """时间戳 → epoch 秒：astype("datetime64[ns]") -> astype(int64) // 1e9（r1 缺陷类教训）。"""
    return (pd.to_datetime(times, errors="coerce").astype("datetime64[ns]")
            .astype("int64").to_numpy() // 10**9)


def hour_of_day(t_epoch: np.ndarray, tz_offset_s: float = TZ_OFFSET_S) -> np.ndarray:
    """epoch 秒 → 本地小时（0-23）；P4：默认北京时间 UTC+8，与 core.hour_of_day 口径统一；
    非有限时间置 -1（清洗语义）。"""
    t = np.asarray(t_epoch, dtype=float)
    safe = np.where(np.isfinite(t), t + float(tz_offset_s), 0.0)
    hours = (safe.astype(np.int64) // 3600 % 24).astype(int)
    return np.where(np.isfinite(t), hours, -1)


def window_mask(t_epoch: np.ndarray, window_start_s: float, window_end_s: float) -> np.ndarray:
    """特征窗半开区间 [window_start_s, window_end_s) 掩码。

    window_end_s 即 as_of（标签窗起点）：as_of 及其后的标签窗行一律排除，
    边界含 start 不含 end；非有限时间（NaN/inf，含 NaT 解析失败行）一律排除（统一清洗）。
    """
    t = np.asarray(t_epoch, dtype=float)
    return np.isfinite(t) & (t >= window_start_s) & (t < window_end_s)


def finite_row_mask(*cols) -> np.ndarray:
    """统一清洗（红线 2）：丢弃任一列非有限值（NaN/inf）的行，返回逐行保留掩码。

    缺数以缺失表达、不删车；空输入返回空掩码。消费 lat/lon/speed 的统计函数入口统一用本
    掩码取"有效行"（顺序保留），空/单点有效输入由各函数门槛置缺失、不崩溃。
    """
    mask: np.ndarray | None = None
    for c in cols:
        keep = np.isfinite(np.asarray(c, dtype=float).ravel())
        mask = keep if mask is None else (mask & keep)
    return np.zeros(0, dtype=bool) if mask is None else mask


def haversine_m(lat1, lon1, lat2, lon2):
    """Haversine 球面距离（米），支持广播；仅自包含坐标统计，不引入任何外部地图数据。"""
    p1 = np.radians(np.asarray(lat1, dtype=float))
    p2 = np.radians(np.asarray(lat2, dtype=float))
    dphi = p2 - p1
    dlam = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = (np.sin(dphi / 2.0) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2)
    return 2.0 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _is_am(hours: np.ndarray) -> np.ndarray:
    return (hours >= AM_LO) & (hours < AM_HI)


def _is_day(hours: np.ndarray) -> np.ndarray:
    return (hours >= DAY_LO) & (hours < DAY_HI)


def _is_night(hours: np.ndarray) -> np.ndarray:
    return (hours >= NIGHT_LO) | (hours < NIGHT_HI)


def _window_sorted(t_epoch: np.ndarray, window_start_s: float, window_end_s: float,
                   *cols) -> tuple:
    """窗内行（标签窗排除、非有限时间排除）按 data_time 稳定排序后返回 (t, *cols)（stage 0 排序纪律）。"""
    t = np.asarray(t_epoch, dtype=float).ravel()
    idx = np.flatnonzero(window_mask(t, window_start_s, window_end_s))
    idx = idx[np.argsort(t[idx], kind="stable")]
    out = [t[idx].astype(np.int64)]
    for c in cols:
        out.append(np.asarray(c).ravel()[idx])
    return tuple(out)


def _step_geometry(t: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                   speed_kmh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """相邻步几何（已排序窗内行）：(dt, 位移 m, 跳变步, 驻车步)——P1/P2 预登记口径。

    跳变步＝dt>0 且位移 > JUMP_SPEED_MPS·dt（隐含速度 >70 m/s，坐标跳变）；
    驻车步＝到达点驻车：到达点速度有限且 =0，或步位移有限且 <STOP_DISP_M（50m）
    （到达点语义：驶离步出发点速度=0 不误判驻车，随执行登记）；
    位移需两端坐标有限；速度/坐标缺失时对应信号不参与判定（缺数不冒充，红线 2）。
    """
    n = t.size
    if n < 2:
        z = np.zeros(0)
        return z, z.copy(), z.astype(bool), z.astype(bool)
    dt = (t[1:] - t[:-1]).astype(float)
    la = np.asarray(lat, dtype=float)
    lo = np.asarray(lon, dtype=float)
    disp = np.full(n - 1, np.nan)
    both = (np.isfinite(la[:-1]) & np.isfinite(la[1:])
            & np.isfinite(lo[:-1]) & np.isfinite(lo[1:]))
    if both.any():
        disp[both] = np.asarray(haversine_m(la[:-1][both], lo[:-1][both],
                                            la[1:][both], lo[1:][both]), dtype=float)
    jump = (dt > 0) & np.isfinite(disp) & (disp > JUMP_SPEED_MPS * dt)
    sp = np.asarray(speed_kmh, dtype=float)
    sp_arr_zero = np.isfinite(sp[1:]) & (sp[1:] <= STOP_SPEED_KMH)
    stationary = ~jump & (sp_arr_zero | (np.isfinite(disp) & (disp < STOP_DISP_M)))
    return dt, disp, jump, stationary


def valid_drive_steps(t: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                      max_gap_s: float = SPELL_GAP_S) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """有效行驶步（P1 里程/时长统一口径）：返回 (dt, 位移 m, ok 掩码)。

    ok 排除间隔 >max_gap_s 与跳变步（隐含速度 >70 m/s）；跳变步不计里程且不计运行时长
    （切段语义下该步不属任何连续段，随执行登记）；位移需两端坐标有限。
    """
    no_speed = np.full(t.size, np.nan)
    dt, disp, jump, _ = _step_geometry(t, lat, lon, no_speed)
    ok = (dt > 0) & (dt <= max_gap_s) & np.isfinite(disp) & ~jump
    return dt, disp, ok


def jump_step_count(t_epoch: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                    window_start_s: float, window_end_s: float) -> float:
    """P1 质量列 f3_traj_jump_steps：窗内坐标跳变步计数（隐含速度 >70 m/s）。

    无样本门槛（计数供任务二暴露族治理；空/单点输入返回 0 不崩溃，统一清洗同窗口口径）。
    """
    t, la, lo = _window_sorted(t_epoch, window_start_s, window_end_s, lat, lon)
    if t.size < 2:
        return 0.0
    _, _, jump, _ = _step_geometry(t, np.asarray(la, dtype=float),
                                   np.asarray(lo, dtype=float), np.full(t.size, np.nan))
    return float(int(jump.sum()))


def _night_overlap_s(t0: float, t1: float, tz_offset_s: float = TZ_OFFSET_S) -> float:
    """[t0, t1] 与深夜桶（本地 [23,24)∪[0,5) 时）重叠秒数（P4：本地=epoch+tz_offset_s）。"""
    if t1 <= t0:
        return 0.0
    total = 0.0
    a0, a1 = t0 + tz_offset_s, t1 + tz_offset_s
    d0 = int(np.floor(a0 / 86400.0))
    d1 = int(np.floor(a1 / 86400.0))
    for d in range(d0, d1 + 1):
        base = d * 86400.0
        seg_lo = max(a0, base)
        seg_hi = min(a1, base + 86400.0)
        for lo_h, hi_h in ((NIGHT_LO, 24), (0, NIGHT_HI)):
            n_lo = max(seg_lo, base + lo_h * 3600.0)
            n_hi = min(seg_hi, base + hi_h * 3600.0)
            if n_hi > n_lo:
                total += n_hi - n_lo
    return total


# ------------------------------------------------------------- 连续驾驶切段（§3.0）

def run_duration_field_semantics(t_epoch: np.ndarray, run_duration_s: np.ndarray,
                                 gap_limit_s: float = SPELL_GAP_S) -> str:
    """判定运行时长字段语义（§3.0 连续驾驶切段前置判定，返回口径标志）。

    CALIBER_TRIP（单行程累计）需同时满足：
    1) 字段全部有限且 n >= 2；
    2) 存在字段重置（相邻行回落 >RUN_RESET_TOL_S）；
    3) 每次重置后从近零重新累计：新段起点值 <= RUN_RESET_START_FRAC*段末值 + NEAR_ZERO_S
       （首段豁免——窗内可能自行程中段开始）；
    4) 每个长间隔（>gap_limit_s）都伴随字段重置（长休息后重新起算）。
    其余（含无重置的窗口累计量）一律 CALIBER_GAP：按相邻点间隔 >gap_limit_s 切段。
    统一清洗：非有限时间行丢弃后再判定；字段含非有限值或清洗后 n < 2 → CALIBER_GAP。
    """
    t = np.asarray(t_epoch, dtype=float).ravel()
    r = np.asarray(run_duration_s, dtype=float).ravel()
    if r.size != t.size:
        return CALIBER_GAP
    keep = np.isfinite(t)
    t, r = t[keep], r[keep]
    n = t.size
    if n < 2 or not np.isfinite(r).all():
        return CALIBER_GAP
    order = np.argsort(t, kind="stable")
    t, r = t[order], r[order]
    d = np.diff(r)
    reset = d < -RUN_RESET_TOL_S
    if not reset.any():
        return CALIBER_GAP
    seg_start = np.r_[0, np.flatnonzero(reset) + 1]
    seg_end = np.r_[seg_start[1:] - 1, n - 1]
    for s, e in zip(seg_start[1:], seg_end[1:]):
        if r[s] > RUN_RESET_START_FRAC * r[e] + NEAR_ZERO_S:
            return CALIBER_GAP
    if not bool(reset[np.diff(t) > gap_limit_s].all()):
        return CALIBER_GAP
    return CALIBER_TRIP


@dataclass
class SpellSplit:
    """连续驾驶切段结果：逐行段标签（窗外为 -1）、段起止行号、段时长（秒）与口径标志。"""

    labels: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    durations_s: np.ndarray
    caliber: str


def driving_spells(t_epoch: np.ndarray, window_start_s: float, window_end_s: float,
                   run_duration_s: np.ndarray | None = None,
                   speed_kmh: np.ndarray | None = None,
                   lat: np.ndarray | None = None, lon: np.ndarray | None = None,
                   gap_limit_s: float = SPELL_GAP_S) -> SpellSplit:
    """连续驾驶切段（r5 §2 P2 双切段口径）。

    切段规则（预登记，执行中不得擅改）：
    - 驻车块：连续驻车步（速度=0 或位移 <50m）自驻车出发点（末个行驶点）至末个驻车
      到达点跨越 >=30min（边界含）→ 驻车到达点标 STOP_LABEL（-2，不属任何段），段在
      块前结束、块后重开；不足 30min 的短停点与时长计入所在段（段内 <30min 短停计入行驶）；
    - 相邻点间隔 >gap_limit_s（默认 180min，严格大于）切段；
    - 坐标跳变步（隐含速度 >70 m/s，P1）切段；
    - run_duration_field_semantics 判为单行程累计（trip_cumulative）时字段重置亦切段。
    速度与坐标均未提供时驻车/跳变检测无信号、退化到间隔/字段口径（随运行登记）。
    段时长：trip_cumulative 取字段窗内增量，否则取段首末时刻差（驻车块时间不计入任何段）。
    标签窗外行与非有限时间行不参与切段（标签一律 -1）。
    """
    t_all = np.asarray(t_epoch, dtype=float).ravel()
    labels = np.full(t_all.size, -1, dtype=int)
    idx = np.flatnonzero(window_mask(t_all, window_start_s, window_end_s))
    idx = idx[np.argsort(t_all[idx], kind="stable")]
    t = t_all[idx]
    r = (np.asarray(run_duration_s, dtype=float).ravel()[idx]
         if run_duration_s is not None else None)
    n = t.size
    if n == 0:
        return SpellSplit(labels=labels, starts=np.zeros(0, dtype=int),
                          ends=np.zeros(0, dtype=int), durations_s=np.zeros(0),
                          caliber=CALIBER_GAP)
    sp = (np.asarray(speed_kmh, dtype=float).ravel()[idx]
          if speed_kmh is not None else np.full(n, np.nan))
    la = (np.asarray(lat, dtype=float).ravel()[idx]
          if lat is not None else np.full(n, np.nan))
    lo = (np.asarray(lon, dtype=float).ravel()[idx]
          if lon is not None else np.full(n, np.nan))
    caliber = (CALIBER_TRIP
               if r is not None and run_duration_field_semantics(t, r, gap_limit_s) == CALIBER_TRIP
               else CALIBER_GAP)
    new_seg = np.zeros(n, dtype=bool)
    new_seg[0] = True
    stop_point = np.zeros(n, dtype=bool)
    if n >= 2:
        dt, _, jump, stationary = _step_geometry(t, la, lo, sp)
        new_seg[1:] |= (dt > gap_limit_s) | jump
        if caliber == CALIBER_TRIP:
            new_seg[1:] |= r[1:] < r[:-1] - RUN_RESET_TOL_S
        i, m = 0, stationary.size
        while i < m:                     # 驻车块扫描：span = t[末驻车到达点] − t[块首出发点]
            if stationary[i]:
                j = i
                while j + 1 < m and stationary[j + 1]:
                    j += 1
                if t[j + 1] - t[i] >= STOP_SPAN_S:
                    stop_point[i + 1:j + 2] = True
                    if j + 2 < n:
                        new_seg[j + 2] = True
                i = j + 1
            else:
                i += 1
    seg_id = np.cumsum(new_seg) - 1
    labels[idx] = np.where(stop_point, STOP_LABEL, seg_id)
    starts_l: list[int] = []
    ends_l: list[int] = []
    k = 0
    while k < n:                          # 连续驾驶段＝同段内相邻非驻车点的极大连续段
        if stop_point[k]:
            k += 1
            continue
        s = k
        while k + 1 < n and not stop_point[k + 1] and seg_id[k + 1] == seg_id[s]:
            k += 1
        starts_l.append(s)
        ends_l.append(k)
        k += 1
    starts = np.asarray(starts_l, dtype=int)
    ends = np.asarray(ends_l, dtype=int)
    if caliber == CALIBER_TRIP and starts.size:
        durations = r[ends] - r[starts]
    elif starts.size:
        durations = (t[ends] - t[starts]).astype(float)
    else:
        durations = np.zeros(0)
    return SpellSplit(labels=labels, starts=idx[starts], ends=idx[ends],
                      durations_s=durations, caliber=caliber)


def spell_features(t_epoch: np.ndarray, window_start_s: float, window_end_s: float,
                   run_duration_s: np.ndarray | None = None,
                   speed_kmh: np.ndarray | None = None,
                   lat: np.ndarray | None = None, lon: np.ndarray | None = None) -> dict[str, float]:
    """连续驾驶切段统计（§3.0 疲劳结构；r5 §2 P1/P2 修复口径）。

    - f3_traj_max_spell_h：最长连续驾驶段时长（小时；≥30min 驻车/断档/跳变均切段）；
    - f3_traj_gt4h_share：>4h 段累计时长占全部连续驾驶累计时长之比；
    - f3_traj_night_spell_h：各段时长落在深夜 23-5（本地，P4 +8h）的累计小时数。
    门槛（不足置缺失）：窗内有效点 <MIN_MAIN_ROWS 全部缺失；深夜项另需深夜桶点数
    >=MIN_BUCKET_ROWS。切段口径见 driving_spells（随 SpellSplit.caliber 登记）；
    speed/lat/lon 提供驻车与跳变检测信号（缺省退化到间隔/字段口径）。
    统一清洗：非有限时间行不参与统计（window_mask 收口）。
    """
    out = {"f3_traj_max_spell_h": float("nan"),
           "f3_traj_gt4h_share": float("nan"),
           "f3_traj_night_spell_h": float("nan")}
    t_all = np.asarray(t_epoch, dtype=float).ravel()
    mask = window_mask(t_all, window_start_s, window_end_s)
    if int(mask.sum()) < MIN_MAIN_ROWS:
        return out
    split = driving_spells(t_all, window_start_s, window_end_s, run_duration_s,
                           speed_kmh, lat, lon)
    dur = split.durations_s
    if dur.size:
        out["f3_traj_max_spell_h"] = float(dur.max() / 3600.0)
        total = float(dur.sum())
        if total > EPS:
            out["f3_traj_gt4h_share"] = float(dur[dur > LONG_SPELL_H * 3600.0].sum() / total)
    hours = hour_of_day(t_all)
    if int((mask & _is_night(hours)).sum()) >= MIN_BUCKET_ROWS:
        night_s = sum(_night_overlap_s(float(t_all[s]), float(t_all[e]))
                      for s, e in zip(split.starts, split.ends))
        out["f3_traj_night_spell_h"] = float(night_s / 3600.0)
    return out


# ------------------------------------------------------------- 速度形态（§3.0）

def speed_shape_features(t_epoch: np.ndarray, window_start_s: float, window_end_s: float,
                         speed_kmh: np.ndarray) -> dict[str, float]:
    """速度形态（§3.0）：分位数、分时段超速、速度 CV、高速巡航占比。

    - f3_traj_speed_p50/p90/p99：窗内有限速度分位数（km/h）；
    - f3_traj_over90_am / f3_traj_over90_night：>90km/h 行占比（早 7-9 / 深夜 23-5 分桶）；
    - f3_traj_speed_cv：速度变异系数（std/mean，ddof=0）；
    - f3_traj_cruise80_share：高速巡航占比＝相邻两点均 >80km/h 的间隔累计时长
      （持续 >80km/h）占相邻有限点总累计时长之比（道路类型代理，禁地图）。
    门槛（不足置缺失）：全体需窗内有限点 >=MIN_MAIN_ROWS；分时段项另需桶内有限点
    >=MIN_BUCKET_ROWS。统一清洗：非有限速度行只从速度统计中丢弃（巡航占比按相邻有效点对
    计），非有限时间行由 window_mask 排除；空/单点输入返回 NaN 不崩溃。
    """
    keys = ("f3_traj_speed_p50", "f3_traj_speed_p90", "f3_traj_speed_p99",
            "f3_traj_over90_am", "f3_traj_over90_night",
            "f3_traj_speed_cv", "f3_traj_cruise80_share")
    out = {k: float("nan") for k in keys}
    t, sp = _window_sorted(t_epoch, window_start_s, window_end_s, speed_kmh)
    sp = np.asarray(sp, dtype=float)
    fin = np.isfinite(sp)
    if int(fin.sum()) < MIN_MAIN_ROWS:
        return out
    v = sp[fin]
    q50, q90, q99 = np.quantile(v, [0.5, 0.9, 0.99])
    out["f3_traj_speed_p50"] = float(q50)
    out["f3_traj_speed_p90"] = float(q90)
    out["f3_traj_speed_p99"] = float(q99)
    mean = float(v.mean())
    if abs(mean) > EPS:
        out["f3_traj_speed_cv"] = float(v.std() / mean)
    if t.size >= 2:
        both = fin[:-1] & fin[1:]
        dt = (np.diff(t)).astype(float)
        total = float(dt[both].sum())
        if total > EPS:
            cruise = both & (sp[:-1] > CRUISE_KMH) & (sp[1:] > CRUISE_KMH)
            out["f3_traj_cruise80_share"] = float(dt[cruise].sum() / total)
    hours = hour_of_day(t)
    for name, bucket in (("f3_traj_over90_am", _is_am(hours)),
                         ("f3_traj_over90_night", _is_night(hours))):
        sel = bucket & fin
        if int(sel.sum()) >= MIN_BUCKET_ROWS:
            out[name] = float(np.mean(sp[sel] > OVER_SPEED_KMH))
    return out


# ------------------------------------------------------------- 空间结构（§3.0）

def dbscan_hotspots(event_lat: np.ndarray, event_lon: np.ndarray,
                    eps_m: float = DBSCAN_EPS_M,
                    min_samples: int = DBSCAN_MIN_SAMPLES) -> tuple[np.ndarray, int]:
    """事件坐标 DBSCAN 热点（Haversine，eps 200m，min_samples 2，sklearn）。

    统一清洗：非有限坐标行不进聚类、标签置 -1（噪声），簇数只计有效坐标簇；
    空输入/全非法坐标返回（逐点 -1 标签，0 簇），单点输入为噪声不崩溃。
    返回（逐点簇标签，噪声为 -1；簇数）。仅自包含坐标统计，禁地图匹配/路网/POI 外部数据。
    """
    la = np.asarray(event_lat, dtype=float).ravel()
    lo = np.asarray(event_lon, dtype=float).ravel()
    n = la.size
    if n == 0:
        return np.zeros(0, dtype=int), 0
    if lo.size != n:
        raise ValueError("event_lat 与 event_lon 行数不一致")
    valid = finite_row_mask(la, lo)
    labels = np.full(n, -1, dtype=int)
    if not valid.any():
        return labels, 0
    coords = np.radians(np.column_stack([la[valid], lo[valid]]))
    labels[valid] = DBSCAN(eps=eps_m / EARTH_R_M, min_samples=min_samples,
                           metric="haversine").fit(coords).labels_
    n_clusters = int(len(set(labels[valid].tolist()) - {-1}))
    return labels.astype(int), n_clusters


def hotspot_features(event_lat: np.ndarray, event_lon: np.ndarray,
                     event_t_epoch: np.ndarray, window_start_s: float,
                     window_end_s: float) -> dict[str, float]:
    """空间热点统计（§3.0）：热点数、热点事件占比、同点位复发次数（事件坐标输入）。

    - f3_traj_hotspot_n：DBSCAN 簇（热点）数；
    - f3_traj_hotspot_event_share：落入热点（非噪声）事件占比；
    - f3_traj_samepoint_recur_n：同点位复发次数＝Σ(簇大小-1)（各热点超出首次的事件数）。
    统一清洗：非有限坐标行丢弃（只统计有效坐标事件），有效坐标不足 MIN_SUB_ROWS 置缺失。
    门槛（不足置缺失）：窗内有效事件坐标 >=MIN_SUB_ROWS。
    """
    keys = ("f3_traj_hotspot_n", "f3_traj_hotspot_event_share", "f3_traj_samepoint_recur_n")
    out = {k: float("nan") for k in keys}
    _, la, lo = _window_sorted(event_t_epoch, window_start_s, window_end_s,
                               event_lat, event_lon)
    keep = finite_row_mask(la, lo)                       # 非法坐标行丢弃（不进 DBSCAN）
    la, lo = np.asarray(la)[keep], np.asarray(lo)[keep]
    n = la.size
    if n < MIN_SUB_ROWS:
        return out
    labels, n_clusters = dbscan_hotspots(la, lo)
    clustered = labels >= 0
    out["f3_traj_hotspot_n"] = float(n_clusters)
    out["f3_traj_hotspot_event_share"] = float(clustered.mean())
    out["f3_traj_samepoint_recur_n"] = float(int(clustered.sum()) - n_clusters)
    return out


def route_repeat_distance_m(lat: np.ndarray, lon: np.ndarray, t_epoch: np.ndarray,
                            window_start_s: float, window_end_s: float,
                            min_day_rows: int = MIN_SUB_ROWS) -> float:
    """路线重复度＝相邻日（日历相邻且均 >=min_day_rows 点）路径点集平均最近邻距离（米）。

    日对距离取双向（A→B 与 B→A）最近邻距离的平均，再对日对取均值；越小越重复。
    日分组按本地日历日（epoch+8h，P4 与 core 一致）。
    统一清洗：非有限坐标行丢弃（finite_row_mask）。门槛（不足置缺失）：清洗后窗内有效点
    >=MIN_MAIN_ROWS 且至少一个有效日对；空输入/单点输入返回 NaN 不崩溃。
    """
    t, la, lo = _window_sorted(t_epoch, window_start_s, window_end_s, lat, lon)
    keep = finite_row_mask(la, lo)
    t, la, lo = t[keep], np.asarray(la, dtype=float)[keep], np.asarray(lo, dtype=float)[keep]
    if t.size < MIN_MAIN_ROWS:
        return float("nan")
    day = (t + TZ_OFFSET_S) // 86400
    unique_days, starts = np.unique(day, return_index=True)
    ends = np.r_[starts[1:], day.size]
    # _window_sorted 返回按时间排序的行，因此同一本地日的点连续。
    # 只为达到门槛的日子建树，并让相邻日对复用同一棵树。
    points_by_day = {}
    trees_by_day = {}
    for d, start, end in zip(unique_days, starts, ends):
        if end - start < min_day_rows:
            continue
        points = np.radians(np.column_stack([la[start:end], lo[start:end]]))
        points_by_day[int(d)] = points
        trees_by_day[int(d)] = BallTree(points, metric="haversine")
    pair_vals: list[float] = []
    for d in sorted(points_by_day):
        if d + 1 not in points_by_day:
            continue
        a, b = points_by_day[d], points_by_day[d + 1]
        nn_ab = trees_by_day[d].query(b, k=1, return_distance=True)[0][:, 0]
        nn_ba = trees_by_day[d + 1].query(a, k=1, return_distance=True)[0][:, 0]
        pair_vals.append(float(np.mean(np.concatenate([nn_ab, nn_ba])) * EARTH_R_M))
    return float(np.mean(pair_vals)) if pair_vals else float("nan")


def activity_radius_m(lat: np.ndarray, lon: np.ndarray, t_epoch: np.ndarray,
                      window_start_s: float, window_end_s: float) -> float:
    """活动半径＝坐标点相对质心的 RMS 距离（米）。

    统一清洗：非有限坐标行丢弃。门槛：清洗后窗内有效点 >=MIN_MAIN_ROWS（空/单点置缺失）。
    """
    t, la, lo = _window_sorted(t_epoch, window_start_s, window_end_s, lat, lon)
    keep = finite_row_mask(la, lo)
    la, lo = np.asarray(la, dtype=float)[keep], np.asarray(lo, dtype=float)[keep]
    if t[keep].size < MIN_MAIN_ROWS:
        return float("nan")
    d = haversine_m(la, lo, float(la.mean()), float(lo.mean()))
    return float(np.sqrt(np.mean(np.asarray(d, dtype=float) ** 2)))


def daynight_centroid_shift_m(lat: np.ndarray, lon: np.ndarray, t_epoch: np.ndarray,
                              window_start_s: float, window_end_s: float) -> float:
    """昼夜活动区偏移＝日间(9-17)/深夜(23-5)点集质心距离（米）。

    统一清洗：非有限坐标行丢弃。门槛（不足置缺失）：清洗后窗内有效点 >=MIN_MAIN_ROWS，
    且昼夜两侧点集各 >=MIN_BUCKET_ROWS；空/单点输入返回 NaN 不崩溃。
    """
    t, la, lo = _window_sorted(t_epoch, window_start_s, window_end_s, lat, lon)
    keep = finite_row_mask(la, lo)
    t = t[keep]
    la, lo = np.asarray(la, dtype=float)[keep], np.asarray(lo, dtype=float)[keep]
    if t.size < MIN_MAIN_ROWS:
        return float("nan")
    hours = hour_of_day(t)
    m_day, m_night = _is_day(hours), _is_night(hours)
    if int(m_day.sum()) < MIN_BUCKET_ROWS or int(m_night.sum()) < MIN_BUCKET_ROWS:
        return float("nan")
    return float(haversine_m(la[m_day].mean(), lo[m_day].mean(),
                             la[m_night].mean(), lo[m_night].mean()))


def spatial_features(lat: np.ndarray, lon: np.ndarray, t_epoch: np.ndarray,
                     event_lat: np.ndarray, event_lon: np.ndarray,
                     event_t_epoch: np.ndarray, window_start_s: float,
                     window_end_s: float) -> dict[str, float]:
    """空间结构族（§3.0）：热点复发、路线重复度、活动半径、昼夜活动区偏移（禁外部地图数据）。

    轨迹点统计与事件坐标热点统计合并输出；各口径与门槛见 hotspot_features /
    route_repeat_distance_m / activity_radius_m / daynight_centroid_shift_m。
    统一清洗在各子函数入口收口（非有限坐标/时间行丢弃），空/单点输入整族返回 NaN 不崩溃。
    """
    out = hotspot_features(event_lat, event_lon, event_t_epoch, window_start_s, window_end_s)
    out["f3_traj_route_repeat_m"] = route_repeat_distance_m(
        lat, lon, t_epoch, window_start_s, window_end_s)
    out["f3_traj_activity_radius_m"] = activity_radius_m(
        lat, lon, t_epoch, window_start_s, window_end_s)
    out["f3_traj_daynight_shift_m"] = daynight_centroid_shift_m(
        lat, lon, t_epoch, window_start_s, window_end_s)
    return out


# ------------------------------------------------------------- 日节律（§3.0）

def daily_rhythm_features(t_epoch: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                          window_start_s: float, window_end_s: float,
                          min_days: int = MIN_DAYS) -> dict[str, float]:
    """日节律（§3.0）：每日里程 CV、断档天数、日均运行时长（r5 §2 P1/P4 口径）。

    - 日里程/日运行时长：窗内有效行驶步（valid_drive_steps：跳变步不计、间隔 <=180 min、
      坐标有限）的位移与间隔，按前点本地日历日（epoch+8h，P4 与 core 一致）累计
      （跨日/长间隔对不计）；
    - f3_traj_daily_km_cv：有效天日里程变异系数（std/mean，ddof=0）；
    - f3_traj_gap_days：窗内无数据本地日历天数（断档天数；窗口边界跨入的部分本地日
      无数据亦计断档，随执行登记）；
    - f3_traj_daily_run_h：有效天日均运行时长（小时）。
    门槛（不足置缺失）：清洗后窗内有效点 >=MIN_MAIN_ROWS；CV/日均另需有效天数 >=min_days
    （日级样本下限，随实现登记）；断档天数为计数，仅受点数门槛约束。
    统一清洗：非有限坐标行丢弃（finite_row_mask），空/单点输入返回 NaN。
    """
    keys = ("f3_traj_daily_km_cv", "f3_traj_gap_days", "f3_traj_daily_run_h")
    out = {k: float("nan") for k in keys}
    t, la, lo = _window_sorted(t_epoch, window_start_s, window_end_s, lat, lon)
    keep = finite_row_mask(la, lo)
    t = t[keep]
    la = np.asarray(la, dtype=float)[keep]
    lo = np.asarray(lo, dtype=float)[keep]
    n = t.size
    if n < MIN_MAIN_ROWS:
        return out
    day = (t + TZ_OFFSET_S) // 86400
    active = np.unique(day)
    d_lo = int((window_start_s + TZ_OFFSET_S) // 86400)
    d_hi = int((window_end_s - 1 + TZ_OFFSET_S) // 86400)
    out["f3_traj_gap_days"] = float(max(d_hi - d_lo + 1 - active.size, 0))
    n_active = int(active.size)
    if n < 2:
        return out
    dt, disp, ok = valid_drive_steps(t, la, lo)
    pair_day = day[:-1][ok]
    slot = np.searchsorted(active, pair_day)  # 有效天（含无步长天）内定位
    km_sum = np.bincount(slot, weights=disp[ok] / 1000.0, minlength=n_active)[:n_active]
    run_sum = np.bincount(slot, weights=dt[ok], minlength=n_active)[:n_active]
    if n_active >= min_days:
        mean_km = float(km_sum.mean())
        if mean_km > EPS:
            out["f3_traj_daily_km_cv"] = float(km_sum.std() / mean_km)
        out["f3_traj_daily_run_h"] = float(run_sum.mean() / 3600.0)
    return out
