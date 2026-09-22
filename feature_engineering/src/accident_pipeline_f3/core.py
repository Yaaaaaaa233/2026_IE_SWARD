# -*- coding: utf-8 -*-
"""F3 核心纯函数：stage-0 修正工具、Tier 1、Tier 3 与综合分（FEAT-007 r4）。

设计依据 docs/plans/feat-007-r4-execution.md §3.0 预登记定义（执行中不得擅改）：
- 窗口与红线：特征窗 [as_of−lookback, as_of)（左闭右开）；一切统计不触标签窗 [as_of, as_of+H)，
  各特征函数内部统一经 feature_window_mask 收口（红线 6 显式参数、禁止硬编码窗口）；
- 时间戳转秒统一 astype("datetime64[ns]") → astype(int64)//1e9（r1 缺陷教训）；
- 排序防御 sort_defense：分片内按车辆分组、组内按 data_time 稳定排序，相邻行统计前必须先排序；
- 逐字重复＝排序后相邻两行六轴完全相同且 dt<=2s；率类分母 rate_denominator 去逐字重复计数并产出重复率；
- 近常数排除 near_constant_filter：按方差阈值排除（不得按缺失率判定）；
- 衰减加权计数：末 5 天权重 2 其余 1；复发间隔＝末两次事件间隔；爆发性＝单日密度峰值÷窗内日均；
- 历史×趋势交互＝衰减加权计数 ×（后半窗日均−前半窗日均）（趋势口径与 r2 前后半窗一致）；
- 同群相对化：群键＝能源类型×（高速里程占比车队中位二分）；群内 z-score／百分位的均值、方差等
  统计量仅用 fold_train_mask=True 行拟合（红线 3）；群内 <30 台回退全体统计并置回退标志；
- 画像偏差 profile_deviation：窗内里程/时长/夜间占比相对画像月均的偏差率（里程/时长按窗长折月对账）；
- 夜间退化度 night_degradation＝深夜(23–5)事件率÷日间(9–17)事件率（六场景合并与分场景两版）；
  双侧最低样本门槛沿用 r2 500/300/100 三档：窗级合计 500、合并版逐侧 300、分场景版逐侧 100；
  任一门槛不足置缺失，零分母（日间率 <=EPS 或缺失）置缺失；
- 事件结构 event_chains：相邻事件间隔 <=30min 计连环；产出连环事件数、最大连环长、同连环段内类型对共现计数；
- 双速度源 ems_gps_discrepancy：|EMS−GPS| 窗内均值／p95；
- 综合分 composite_scores：固定等权合成、不做数据驱动调权；夜间风险链分＝夜间暴露＋深夜事件率＋夜间退化度，
  疲劳结构分＝连续驾驶（调用方传入）＋深夜连续＋疲劳报警夜间集中度；方向统一为分越高风险越高。

本模块不做文件 IO，依赖仅 numpy/pandas。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

EPS = 1e-9
DAY_S = 86400.0
DEFAULT_LOOKBACK_S = 20.0 * DAY_S     # §3.0 特征窗 20 天（显式配置覆盖，默认即预登记值）
DEFAULT_LOOKBACK_DAYS = 20.0
HORIZON_NOTE = "标签窗 [as_of, as_of+H) 由调用方显式配置 horizon，特征统计只用 [as_of−lookback, as_of)"
TZ_OFFSET_S = 8.0 * 3600.0            # 业务本地时间（北京时间）口径
DUP_MAX_GAP_S = 2.0                   # 逐字重复判定 dt 上限（含）
CHAIN_MAX_GAP_S = 30.0 * 60.0         # 连环判定 dt 上限 30min（含）
RECENCY_DAYS = 5.0                    # 衰减加权“末 5 天”
RECENCY_WEIGHT = 2.0                  # 末 5 天权重，其余权重 1
MIN_ROWS_FULL = 500                   # 沿用 r2 MIN_DRIVE_ROWS（窗级最低样本）
MIN_ROWS_SIDE = 300                   # 沿用 r2 MIN_SEG_ROWS（分段级/逐侧最低样本）
MIN_ROWS_SUB = 100                    # 沿用 r2 MIN_HALF_ROWS（子分级最低样本）
COHORT_MIN_VEHICLES = 30              # 群内最低车辆台数，不足回退全体
NEAR_CONST_VAR = 1e-12                # 近常数排除的方差阈值（低于等于阈值剔除）
MONTH_DAYS = 30.0                     # 画像月均折算口径
NIGHT_HOUR_RANGES = ((23.0, 24.0), (0.0, 5.0))   # 深夜 23–5（半开区间 [23,24)∪[0,5)）
DAY_HOUR_RANGE = (9.0, 17.0)                     # 日间 9–17（半开区间 [9,17)）


# ------------------------------------------------------------- 时间与窗口

def to_epoch_s(series: pd.Series) -> np.ndarray:
    """时间戳转 epoch 秒（统一 ns 约定，r1 缺陷教训）：astype("datetime64[ns]") → astype(int64)//1e9。

    无法解析的时间为 NaT 哨兵（int64 最小值整除 1e9），须配合 valid_time_mask 判定。
    """
    return (pd.to_datetime(series, errors="coerce").astype("datetime64[ns]")
            .astype("int64").to_numpy() // 10**9)


def valid_time_mask(series: pd.Series) -> np.ndarray:
    """可解析时间掩码（to_epoch_s 的 NaT 哨兵配套）。"""
    return pd.to_datetime(series, errors="coerce").notna().to_numpy()


def hour_of_day(t_epoch: np.ndarray, tz_offset_s: float = TZ_OFFSET_S) -> np.ndarray:
    """epoch 秒 → 本地小时（0–23），tz_offset_s 默认北京时间 UTC+8。"""
    t = np.asarray(t_epoch, dtype=np.int64)
    return (((t + int(tz_offset_s)) // 3600) % 24).astype(int)


def feature_window_mask(t_epoch: np.ndarray, as_of_s: float,
                        lookback_s: float = DEFAULT_LOOKBACK_S) -> np.ndarray:
    """特征窗掩码 [as_of−lookback, as_of)（左闭右开）。

    >= as_of 的行（含标签窗 [as_of, as_of+H)）一律排除——红线：一切特征统计不触标签窗。
    """
    t = np.asarray(t_epoch, dtype=float)
    return (t >= as_of_s - lookback_s) & (t < as_of_s)


def label_window_mask(t_epoch: np.ndarray, as_of_s: float, horizon_s: float) -> np.ndarray:
    """标签窗掩码 [as_of, as_of+horizon)（左闭右开）。

    仅用于红线校验（合成测试断言特征统计未命中标签窗任何行），特征函数不使用该掩码取数。
    """
    t = np.asarray(t_epoch, dtype=float)
    return (t >= as_of_s) & (t < as_of_s + horizon_s)


# ------------------------------------------------------------- stage 0 工具

def sort_defense(frame: pd.DataFrame, vehicle_col: str = "gpsno",
                 time_col: str = "data_time") -> pd.DataFrame:
    """stage-0 排序防御：分片内按车辆分组、组内按 data_time 稳定排序（乱序修复）。

    相邻行统计（差分/滑窗/逐字重复）前必须先过本函数。返回按 (车辆, 时间) 排序的新
    DataFrame（index 重置）；同车辆同时间的行按原始先后保留（稳定排序）；无法解析的时间行
    排到该车组末尾。
    """
    if vehicle_col not in frame.columns or time_col not in frame.columns:
        raise KeyError(f"缺少排序列：{vehicle_col!r} 或 {time_col!r}")
    n = len(frame)
    if n == 0:
        return frame.copy().reset_index(drop=True)
    codes, _ = pd.factorize(frame[vehicle_col], use_na_sentinel=False)
    t = to_epoch_s(frame[time_col])
    valid = valid_time_mask(frame[time_col])
    sort_t = np.where(valid, t, np.iinfo(np.int64).max // 2)
    origin = np.arange(n)
    order = np.lexsort((origin, sort_t, codes))  # 车辆为主键、时间为次键、原序保稳定
    return frame.iloc[order].reset_index(drop=True)


def rate_denominator(t_epoch: np.ndarray, six_axes: np.ndarray,
                     dt_max_s: float = DUP_MAX_GAP_S,
                     group: np.ndarray | None = None,
                     min_pairs: int = MIN_ROWS_SUB) -> dict:
    """stage-0 率类分母去逐字重复计数（须排序后判定，本函数内部稳定排序）。

    逐字重复＝同一车辆内相邻两行六轴（ax..gz）完全相同且 0 <= dt <= dt_max_s（默认 2s，边界含）；
    每个重复行从率类分母中扣除。返回：denominator（去重复后的率类分母）、n_rows、n_dup、
    n_pairs（相邻行对数）、dup_rate（n_dup/n_pairs，行对不足 min_pairs 置缺失，重复率列并行入模）。
    group 缺省视为单一分片（单车）；多车分片须传车辆列，跨车辆不判重复。
    """
    t = np.asarray(t_epoch, dtype=float)
    a6 = np.asarray(six_axes, dtype=float)
    if a6.ndim != 2 or a6.shape[1] != 6:
        raise ValueError("six_axes 须为 (n, 6) 六轴矩阵 [ax,ay,az,gx,gy,gz]")
    n = len(t)
    if len(a6) != n:
        raise ValueError("t_epoch 与 six_axes 行数不一致")
    if group is None:
        codes = np.zeros(n, dtype=int)
    else:
        codes, _ = pd.factorize(np.asarray(group), use_na_sentinel=False)
    if n < 2:
        return {"denominator": float(n), "n_rows": int(n), "n_dup": 0,
                "n_pairs": max(n - 1, 0), "dup_rate": float("nan")}
    origin = np.arange(n)
    order = np.lexsort((origin, t, codes))
    ts, a6s, cs = t[order], a6[order], codes[order]
    same_group = cs[1:] == cs[:-1]
    same_axes = (a6s[1:] == a6s[:-1]).all(axis=1)
    dt = ts[1:] - ts[:-1]
    dup = same_group & same_axes & (dt >= 0.0) & (dt <= dt_max_s)
    n_pairs = int(same_group.sum())
    n_dup = int(dup.sum())
    dup_rate = (n_dup / n_pairs) if n_pairs >= min_pairs else float("nan")
    return {"denominator": float(n - n_dup), "n_rows": int(n), "n_dup": n_dup,
            "n_pairs": n_pairs, "dup_rate": float(dup_rate)}


def near_constant_filter(frame: pd.DataFrame, var_threshold: float = NEAR_CONST_VAR,
                         columns: Sequence[str] | None = None) -> list[str]:
    """stage-0 近常数排除：按方差阈值排除近常数列，返回保留列清单（保持原列序，确定性）。

    §3.0：接口层按方差阈值排除（不得按缺失率判定）；恒 0／常数列不得混入 model_input。
    方差在有限非缺失值上按总体方差（ddof=0）计算；有效值不足 2 个视作无信息列剔除。
    columns 缺省取全部数值列；保留判据为 var > var_threshold。
    """
    if columns is None:
        cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    else:
        cols = list(columns)
    kept: list[str] = []
    for col in cols:
        x = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 2:
            continue
        if float(x.var()) > var_threshold:  # ddof=0 总体方差
            kept.append(col)
    return kept


# ------------------------------------------------------------- Tier 1

def historical_temporal_features(t_epoch: np.ndarray, as_of_s: float,
                                 lookback_s: float = DEFAULT_LOOKBACK_S,
                                 tz_offset_s: float = TZ_OFFSET_S) -> dict:
    """Tier1 历史险情时间结构（单车事件时刻序列，特征窗内统计）。

    - f3_hist_decay_count：衰减加权计数——末 5 天（as_of−5d 起，边界含）权重 2，其余权重 1；
    - f3_hist_recurrence_gap_days：复发间隔——末两次事件间隔（天），事件不足 2 置缺失；
    - f3_hist_burst_index：爆发性指数——单日密度峰值 ÷ 窗内日均（日均＝窗内事件数÷窗天数，
      单日按本地日历日聚合）；窗内无事件置缺失；
    - f3_hist_trend_per_day：趋势——后半窗日均−前半窗日均（事件/天，前后半窗各 lookback/2）；
    - f3_hist_x_trend：历史×趋势交互——衰减加权计数 × f3_hist_trend_per_day。
    仅取 [as_of−lookback, as_of) 的事件（标签窗排除由 feature_window_mask 收口）。
    """
    t = np.asarray(t_epoch, dtype=float)
    ts = np.sort(t[feature_window_mask(t, as_of_s, lookback_s)])
    n = len(ts)
    window_days = lookback_s / DAY_S
    recent_cut = as_of_s - RECENCY_DAYS * DAY_S
    weights = np.where(ts >= recent_cut, RECENCY_WEIGHT, 1.0)
    decay = float(weights.sum())
    if n >= 2:
        gap_days = float((ts[-1] - ts[-2]) / DAY_S)
    else:
        gap_days = float("nan")
    if n == 0:
        burst = float("nan")
    else:
        day_idx = np.floor((ts + tz_offset_s) / DAY_S).astype(np.int64)
        _, counts = np.unique(day_idx, return_counts=True)
        mean_daily = n / window_days
        burst = float(counts.max() / mean_daily) if mean_daily > EPS else float("nan")
    half_s = lookback_s / 2.0
    front = int(((ts >= as_of_s - lookback_s) & (ts < as_of_s - half_s)).sum())
    back = int(((ts >= as_of_s - half_s) & (ts < as_of_s)).sum())
    trend = float((back - front) / (half_s / DAY_S))
    return {
        "f3_hist_decay_count": decay,
        "f3_hist_recurrence_gap_days": gap_days,
        "f3_hist_burst_index": burst,
        "f3_hist_trend_per_day": trend,
        "f3_hist_x_trend": float(decay * trend),
    }


def cohort_keys(energy_type: pd.Series | np.ndarray, highway_share: np.ndarray,
                train_mask: np.ndarray | None = None,
                split_median: float | None = None) -> tuple[np.ndarray, float]:
    """同群群键：能源类型 ×（高速里程占比车队中位二分）。

    中位数按红线 3 仅在 train_mask=True 行上拟合（缺省用全体行）；调用方也可传入既定
    split_median 复用同一口径。二分：占比 > 中位数 → "hi"，否则 "lo"；占比缺失或中位数
    不可用 → "na"。返回 (群键数组 "能源|分支", 使用的中位数)。
    """
    et = pd.Series(energy_type, dtype="object").fillna("unknown").astype(str).to_numpy()
    share = np.asarray(highway_share, dtype=float)
    fit = np.ones(len(share), dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    if split_median is None:
        cand = share[fit & np.isfinite(share)]
        med = float(np.median(cand)) if len(cand) else float("nan")
    else:
        med = float(split_median)
    if np.isfinite(med):
        branch = np.where(~np.isfinite(share), "na", np.where(share > med, "hi", "lo"))
    else:
        branch = np.full(len(share), "na", dtype=object)
    keys = np.array([f"{e}|{b}" for e, b in zip(et, branch.tolist())], dtype=object)
    return keys, med


def cohort_relative(values: pd.DataFrame, cohort_key: np.ndarray,
                    fold_train_mask: np.ndarray,
                    vehicle_ids: pd.Series | np.ndarray | None = None,
                    min_cohort_vehicles: int = COHORT_MIN_VEHICLES) -> pd.DataFrame:
    """同群相对化：群内 z-score 与群内百分位（仅对既有强特征列相对化）。

    均值／方差／百分位统计量仅用 fold_train_mask=True 行拟合（红线 3，非训练行不进统计量），
    再套用到全部行。群内（训练折内）不足 min_cohort_vehicles 台车辆时回退全体统计量并置
    cohort_fallback=1（车辆台数＝训练折内该群的车辆去重计数，vehicle_ids 缺省按行即车）。
    z=(x−μ)/σ，σ<=EPS 时 z 置缺失；百分位＝训练拟合样本中取值 <=x 的占比；x 缺失则两列均缺失。
    输出列：{col}_cohort_z、{col}_cohort_pct、cohort_fallback（0/1）。
    """
    key = np.asarray(cohort_key).astype(str)
    train = np.asarray(fold_train_mask, dtype=bool)
    n = len(key)
    if len(train) != n:
        raise ValueError("cohort_key 与 fold_train_mask 行数不一致")
    veh = np.arange(n) if vehicle_ids is None else np.asarray(vehicle_ids)
    train_key, train_veh = key[train], veh[train]
    group_size = {k: len(set(train_veh[train_key == k].tolist()))
                  for k in pd.unique(train_key)}
    fallback = {k: (group_size.get(k, 0) < min_cohort_vehicles) for k in pd.unique(key)}
    out = pd.DataFrame(index=values.index)
    for col in values.columns:
        x = pd.to_numeric(values[col], errors="coerce").to_numpy(dtype=float)
        fit_all = train & np.isfinite(x)
        stats = {}
        for k in pd.unique(key):
            m = fit_all & (key == k)
            stats[k] = (float(x[m].mean()) if m.any() else float("nan"),
                        float(x[m].std()) if m.any() else float("nan"),
                        x[m])
        mu_all = float(x[fit_all].mean()) if fit_all.any() else float("nan")
        sd_all = float(x[fit_all].std()) if fit_all.any() else float("nan")
        pool_all = x[fit_all]
        z = np.full(n, np.nan)
        pct = np.full(n, np.nan)
        for i in range(n):
            if not np.isfinite(x[i]):
                continue
            mu, sd, pool = stats[key[i]]
            if fallback[key[i]]:
                mu, sd, pool = mu_all, sd_all, pool_all
            if np.isfinite(mu) and np.isfinite(sd) and sd > EPS:
                z[i] = (x[i] - mu) / sd
            if len(pool):
                pct[i] = float((pool <= x[i]).mean())
        out[f"{col}_cohort_z"] = z
        out[f"{col}_cohort_pct"] = pct
    out["cohort_fallback"] = np.array([int(fallback[k]) for k in key], dtype=int)
    return out


def profile_deviation(window_mileage_km: np.ndarray, window_hours: np.ndarray,
                      window_night_share: np.ndarray,
                      profile_month_km: np.ndarray, profile_month_hours: np.ndarray,
                      profile_month_night_share: np.ndarray,
                      lookback_days: float = DEFAULT_LOOKBACK_DAYS,
                      month_days: float = MONTH_DAYS) -> pd.DataFrame:
    """画像基线偏差：窗内里程/时长/夜间占比相对画像月均的偏差率（预期／实际对账）。

    偏差率＝(窗内值−画像月均)/|画像月均|；里程与时长为窗口总量，按月均口径折月
    （× month_days/lookback_days）后对账；夜间占比为率、直接对账。画像月均为 0 或缺失、
    窗内值缺失时置缺失（零分母置缺失）。输出列：f3_profile_km_dev、f3_profile_hours_dev、
    f3_profile_night_dev。
    """
    scale = month_days / lookback_days

    def _dev(actual: np.ndarray, base: np.ndarray) -> np.ndarray:
        a = np.asarray(actual, dtype=float)
        b = np.asarray(base, dtype=float)
        out = np.full(len(a), np.nan)
        ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
        out[ok] = (a[ok] - b[ok]) / np.abs(b[ok])
        return out

    return pd.DataFrame({
        "f3_profile_km_dev": _dev(np.asarray(window_mileage_km, dtype=float) * scale, profile_month_km),
        "f3_profile_hours_dev": _dev(np.asarray(window_hours, dtype=float) * scale, profile_month_hours),
        "f3_profile_night_dev": _dev(window_night_share, profile_month_night_share),
    })


def night_degradation(event_t_epoch: np.ndarray,
                      event_scenario: Sequence[str] | np.ndarray | None,
                      exposure_t_epoch: np.ndarray, as_of_s: float,
                      exposure_amount: np.ndarray | None = None,
                      lookback_s: float = DEFAULT_LOOKBACK_S,
                      tz_offset_s: float = TZ_OFFSET_S,
                      min_rows_total: int = MIN_ROWS_FULL,
                      min_rows_side: int = MIN_ROWS_SIDE,
                      min_rows_sub: int = MIN_ROWS_SUB) -> pd.DataFrame:
    """夜间退化度族：深夜(23–5)事件率 ÷ 日间(9–17)事件率（六场景合并与分场景两版）。

    率＝该时段事件数 ÷ 该时段暴露和（exposure_amount 缺省每行计 1，单位由调用方决定）；
    事件与暴露行均只取特征窗 [as_of−lookback, as_of)（标签窗排除）。输出长表，每行一个版本
    （scenario="__merged__" 为六场景合并版，其余为分场景版），列：
    f3_night_deep_rate（深夜事件率）、f3_night_day_rate（日间事件率）、
    f3_night_degradation（比值）。

    双侧最低样本门槛沿用 r2 500/300/100 三档（按作用域取档）：
    - 率类：各自时段暴露行数 ≥ 逐侧档位门槛（合并版 min_rows_side=300、分场景版 min_rows_sub=100）；
    - 比值：双侧行数均过逐侧档位门槛，且双侧行数合计 ≥ min_rows_total=500（窗级档）；
    任一门槛不足置缺失；零分母（日间率 <=EPS 或缺失）置缺失。
    """
    ev_t = np.asarray(event_t_epoch, dtype=float)
    ev_m = feature_window_mask(ev_t, as_of_s, lookback_s)
    ev_t = ev_t[ev_m]
    if event_scenario is None:
        ev_sc = np.array(["__merged__"] * len(ev_t), dtype=object)
    else:
        ev_sc = np.asarray(event_scenario, dtype=object)[ev_m]
    ex_t = np.asarray(exposure_t_epoch, dtype=float)
    ex_m = feature_window_mask(ex_t, as_of_s, lookback_s)
    ex_t = ex_t[ex_m]
    if exposure_amount is None:
        ex_amt = np.ones(len(ex_t), dtype=float)
    else:
        ex_amt = np.asarray(exposure_amount, dtype=float)[ex_m]

    def _band_hours(t: np.ndarray) -> np.ndarray:
        return hour_of_day(t, tz_offset_s).astype(float)

    ev_h, ex_h = _band_hours(ev_t), _band_hours(ex_t)

    def _in_night(h: np.ndarray) -> np.ndarray:
        return ((h >= NIGHT_HOUR_RANGES[0][0]) | (h < NIGHT_HOUR_RANGES[1][1]))

    def _in_day(h: np.ndarray) -> np.ndarray:
        return (h >= DAY_HOUR_RANGE[0]) & (h < DAY_HOUR_RANGE[1])

    deep_rows, day_rows = int(_in_night(ex_h).sum()), int(_in_day(ex_h).sum())
    deep_exp = float(ex_amt[_in_night(ex_h)].sum())
    day_exp = float(ex_amt[_in_day(ex_h)].sum())
    total_rows = deep_rows + day_rows

    scenarios = ["__merged__"] + sorted({str(s) for s in ev_sc} - {"__merged__"})
    rows = []
    for scen in scenarios:
        sel = np.ones(len(ev_t), dtype=bool) if scen == "__merged__" else (ev_sc.astype(str) == scen)
        n_deep = int((sel & _in_night(ev_h)).sum())
        n_day = int((sel & _in_day(ev_h)).sum())
        side_gate = min_rows_side if scen == "__merged__" else min_rows_sub
        deep_ok = (deep_rows >= side_gate) and (deep_exp > EPS)
        day_ok = (day_rows >= side_gate) and (day_exp > EPS)
        deep_rate = (n_deep / deep_exp) if deep_ok else float("nan")
        day_rate = (n_day / day_exp) if day_ok else float("nan")
        ratio_ok = (deep_ok and day_ok and total_rows >= min_rows_total
                    and np.isfinite(day_rate) and day_rate > EPS)
        degr = (deep_rate / day_rate) if ratio_ok else float("nan")
        rows.append({"scenario": scen,
                     "f3_night_deep_rate": deep_rate,
                     "f3_night_day_rate": day_rate,
                     "f3_night_degradation": degr})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- Tier 3

def event_chains(t_epoch: np.ndarray, event_type: Sequence[str] | np.ndarray,
                 as_of_s: float, lookback_s: float = DEFAULT_LOOKBACK_S,
                 max_gap_s: float = CHAIN_MAX_GAP_S) -> dict:
    """Tier3 事件连环／共现（纯事件表，单车事件序列，特征窗内统计）。

    相邻事件间隔 <=max_gap_s（30min，边界含）计连环；连环段＝时间排序后间隔均 <=30min 的
    极大连续段，段内事件数 >=2 才算连环。产出：
    - f3_chain_event_count：连环事件数（所有连环段内事件总数）；
    - f3_chain_max_len：最大连环长（最长连环段内事件数，无连环为 0）；
    - f3_copair_{a}__{b}：同连环段内类型对共现计数（段内全部事件位置对 i<j，类型名排序后
      拼接，同类型对形如 f3_copair_x__x）。
    """
    t = np.asarray(t_epoch, dtype=float)
    ty = np.asarray(event_type).astype(str)
    m = feature_window_mask(t, as_of_s, lookback_s)
    t, ty = t[m], ty[m]
    order = np.argsort(t, kind="stable")
    t, ty = t[order], ty[order]
    n = len(t)
    if n == 0:
        return {"f3_chain_event_count": 0, "f3_chain_max_len": 0}
    breaks = np.r_[True, np.diff(t) > max_gap_s]
    seg_id = np.cumsum(breaks) - 1
    out: dict = {"f3_chain_event_count": 0, "f3_chain_max_len": 0}
    for seg in range(seg_id[-1] + 1):
        idx = np.flatnonzero(seg_id == seg)
        if len(idx) < 2:
            continue
        out["f3_chain_event_count"] += int(len(idx))
        out["f3_chain_max_len"] = max(out["f3_chain_max_len"], int(len(idx)))
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = sorted((ty[idx[i]], ty[idx[j]]))
                key = f"f3_copair_{a}__{b}"
                out[key] = out.get(key, 0) + 1
    return out


def ems_gps_discrepancy(ems_speed: np.ndarray, gps_speed: np.ndarray, t_epoch: np.ndarray,
                        as_of_s: float, lookback_s: float = DEFAULT_LOOKBACK_S,
                        min_rows: int = MIN_ROWS_FULL) -> dict:
    """Tier3 双速度源一致性：|EMS−GPS| 窗内均值／p95。

    只取特征窗 [as_of−lookback, as_of) 内两源均有限的行（标签窗排除）；有效行数 < min_rows
    （窗级最低样本 500）置缺失。p95 为线性插值分位数。输出：f3_ems_gps_diff_mean、
    f3_ems_gps_diff_p95。
    """
    t = np.asarray(t_epoch, dtype=float)
    e = np.asarray(ems_speed, dtype=float)
    g = np.asarray(gps_speed, dtype=float)
    m = feature_window_mask(t, as_of_s, lookback_s) & np.isfinite(e) & np.isfinite(g)
    diff = np.abs(e[m] - g[m])
    if len(diff) < min_rows:
        return {"f3_ems_gps_diff_mean": float("nan"), "f3_ems_gps_diff_p95": float("nan")}
    return {"f3_ems_gps_diff_mean": float(diff.mean()),
            "f3_ems_gps_diff_p95": float(np.quantile(diff, 0.95))}


# ------------------------------------------------------------- 综合分

def _equal_weight(parts: Sequence[np.ndarray], signs: Sequence[float]) -> np.ndarray:
    """方向统一后等权算术平均；缺失项跳过（全部缺失置缺失），不做数据驱动调权。"""
    stack = np.vstack([np.asarray(p, dtype=float) * float(s)
                       for p, s in zip(parts, signs)])
    valid = np.isfinite(stack)
    cnt = valid.sum(axis=0)
    total = np.where(valid, stack, 0.0).sum(axis=0)
    return np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)


def composite_scores(night_exposure: np.ndarray, deep_night_rate: np.ndarray,
                     night_degradation_ratio: np.ndarray,
                     continuous_drive: np.ndarray, deep_night_continuous: np.ndarray,
                     fatigue_night_concentration: np.ndarray,
                     night_signs: Sequence[float] = (1.0, 1.0, 1.0),
                     fatigue_signs: Sequence[float] = (1.0, 1.0, 1.0)) -> pd.DataFrame:
    """综合分：固定等权合成、不做数据驱动调权，方向统一为分越高风险越高。

    - f3_score_night_chain（夜间风险链分）＝夜间暴露＋深夜事件率＋夜间退化度 等权平均；
    - f3_score_fatigue（疲劳结构分）＝连续驾驶（由调用方传入数组）＋深夜连续驾驶＋
      疲劳报警夜间集中度 等权平均。
    各组成项按预登记语义即“越高风险越高”；反向指标经 signs 传负号显式翻转（方向统一）。
    组成项缺失时对其余可用项等权平均，全部缺失置缺失。组成列在特征字典登记。
    """
    night = _equal_weight([night_exposure, deep_night_rate, night_degradation_ratio], night_signs)
    fatigue = _equal_weight([continuous_drive, deep_night_continuous,
                             fatigue_night_concentration], fatigue_signs)
    return pd.DataFrame({"f3_score_night_chain": night, "f3_score_fatigue": fatigue})
