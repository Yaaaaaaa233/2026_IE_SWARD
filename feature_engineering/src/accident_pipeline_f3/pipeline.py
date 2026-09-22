# -*- coding: utf-8 -*-
"""F3 管线编排：一遍扫描（分片读取→聚合）＋ interface 组装（FEAT-007 r4）。

运行阶段（cli）：scan / interface / run-all。设计依据
docs/plans/feat-007-r4-execution.md §3.0 预登记定义（执行中不得擅改）：
- 真实数据路径一律来自本地配置（configs/pipeline_f3_v1.example.yaml 为占位模板，不写本机路径）；
- 窗口三元组 as_of／lookback／horizon 显式配置、禁止硬编码（红线 6），特征窗
  [as_of−lookback, as_of)，一切统计不触 [as_of, as_of+H)（core.feature_window_mask 收口）；
- 一遍扫描：轨迹分片分块流式读取、按车聚合缓冲（参照 IMU 大表设计），Tier 2 全部轨迹特征
  一次带出；事件表／画像表／基座表小表整读；聚合阶段完成 Tier 1（历史时间结构／画像偏差／
  夜间退化）、Tier 3（事件连环／双速度源／事件时速）、同群相对化（统计量折内拟合，红线 3）、
  综合分（固定等权、不做数据驱动调权）与计数归一双轨（§3.0），产出 f3_new_features.csv
  后交 interface.build_f3_interface 组装 F3_v1 model_input；
- 时间戳转秒统一 astype("datetime64[ns]") → astype(int64)//1e9（r1 缺陷教训）；
- 输入分片 schema（canonical 列名，轨迹交付表 `运行时长` 等源列名经 column_map 映射）：
  轨迹分片 part-*：gpsno, data_time, speed, lat, lon［必需］；run_duration_s, ems_speed,
  gps_speed［可选，缺失置 NaN］；
  事件表：event_id, gpsno, event_time, scenario, speed（交付表 `speed` 列）, lat, lon；
  画像表：gpsno, energy_type, highway_share, profile_month_km, profile_month_hours,
  profile_month_night_share；
  基座表：sample_id, gpsno, y, fold＋lean 基座来源特征列（同群源列等）。
本模块依赖仅 numpy/pandas + PyYAML（仓库既有配置解析依赖）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .core import (COHORT_MIN_VEHICLES, composite_scores, cohort_keys, cohort_relative,
                   ems_gps_discrepancy, event_chains, historical_temporal_features,
                   hour_of_day, night_degradation, profile_deviation, to_epoch_s)
from .interface import (DUAL_SUFFIX_H, DUAL_SUFFIX_KM, NEW_FEATURES_FILE, build_f3_interface,
                        is_count_column)
from .trajectory import (NIGHT_HI, NIGHT_LO, SPELL_GAP_S, daily_rhythm_features,
                         haversine_m, spatial_features, speed_shape_features, spell_features)

FATIGUE_KEYWORD = "fatigue"   # 疲劳类场景判定（scenario 名含该词，疲劳报警夜间集中度口径）
DAY_S = 86400.0
EPS = 1e-9

TRAJ_REQUIRED_COLS = ("gpsno", "data_time", "speed", "lat", "lon")
TRAJ_OPTIONAL_COLS = ("run_duration_s", "ems_speed", "gps_speed")
TRAJ_COLS = TRAJ_REQUIRED_COLS + TRAJ_OPTIONAL_COLS


@dataclass(frozen=True)
class F3Config:
    """F3 本地配置（真实数据路径来自本地配置文件，example.yaml 仅占位模板）。

    as_of／lookback_days／horizon_days 显式配置、无默认值（§3.0 红线 6 禁止硬编码窗口）。
    """

    path: Path
    trajectory_dir: Path
    events_path: Path
    profile_path: Path
    base_input: Path
    base_columns_file: Path
    output_dir: Path
    as_of: str
    lookback_days: float
    horizon_days: float
    shard_delimiter: str = "\t"
    chunk_rows: int = 200_000
    min_km: float = 50.0            # 计数归一低暴露置缺失（沿用 F1 规则）
    min_hours: float = 1.0
    cohort_energy_column: str = "energy_type"
    cohort_highway_column: str = "highway_ratio"
    cohort_source_columns: tuple = ()
    copair_columns: tuple = ()
    column_map: dict = field(default_factory=dict)

    @property
    def as_of_s(self) -> float:
        """as_of（标签窗起点）→ epoch 秒（ns 约定）。"""
        return float(to_epoch_s(pd.Series([self.as_of]))[0])

    @property
    def lookback_s(self) -> float:
        return float(self.lookback_days) * DAY_S

    @property
    def horizon_s(self) -> float:
        return float(self.horizon_days) * DAY_S


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def load_f3_config(path: str | Path) -> F3Config:
    """加载 F3 本地配置（yaml）；相对路径以配置文件目录为基准解析。

    窗口三元组 window.as_of／lookback_days／horizon_days 必填（红线 6）；
    cohort.source_columns 为同群相对化的既有强特征清单（占位示例见 example.yaml）；
    events.copair_columns 为预登记显著共现组合（§3.0：预登记显著组合入特征，缺省空清单）。
    """
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    base = source.parent
    win = raw.get("window", {})
    for key in ("as_of", "lookback_days", "horizon_days"):
        if key not in win:
            raise ValueError(f"窗口配置缺失 window.{key}（红线 6：as_of/lookback/horizon 显式配置）")
    cohort = raw.get("cohort", {})
    events = raw.get("events", {})
    exposure = raw.get("exposure", {})
    traj_cols = raw.get("trajectory_columns", {})
    return F3Config(
        path=source,
        trajectory_dir=_resolve(base, raw["trajectory_dir"]),
        events_path=_resolve(base, raw["events_path"]),
        profile_path=_resolve(base, raw["profile_path"]),
        base_input=_resolve(base, raw["base_input"]),
        base_columns_file=_resolve(base, raw["base_columns_file"]),
        output_dir=_resolve(base, raw["output_dir"]),
        as_of=str(win["as_of"]),
        lookback_days=float(win["lookback_days"]),
        horizon_days=float(win["horizon_days"]),
        shard_delimiter=str(raw.get("shard_delimiter", "\t")),
        chunk_rows=int(raw.get("chunk_rows", 200_000)),
        min_km=float(exposure.get("min_km", 50.0)),
        min_hours=float(exposure.get("min_hours", 1.0)),
        cohort_energy_column=str(cohort.get("energy_type_column", "energy_type")),
        cohort_highway_column=str(cohort.get("highway_share_column", "highway_ratio")),
        cohort_source_columns=tuple(cohort.get("source_columns", ())),
        copair_columns=tuple(events.get("copair_columns", ())),
        column_map={str(k): str(v) for k, v in traj_cols.items()},
    )


# ------------------------------------------------------------- 一遍扫描（分片读取）

def _rename_canonical(chunk: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """按 column_map（canonical → 源列名）把分片列名映射为 canonical（缺省同名）。"""
    if not column_map:
        return chunk
    return chunk.rename(columns={src: dst for dst, src in column_map.items()})


def _scan_trajectory_shards(cfg: F3Config,
                            vehicles: set[str]) -> dict[str, dict[str, np.ndarray]]:
    """轨迹分片一遍流式扫描：分块读取 part-*，按车聚合缓冲（组内保持到达序）。

    Tier 2 的全部轨迹特征由聚合阶段对每车缓冲一次带出（§3.3 一遍扫描）。
    返回 {gpsno: {canonical 列名: 数组}}；t 列为 epoch 秒（ns 约定转换）。
    """
    buffers: dict[str, dict[str, list]] = {}
    for shard in sorted(cfg.trajectory_dir.glob("part-*")):
        reader = pd.read_csv(shard, sep=cfg.shard_delimiter, chunksize=cfg.chunk_rows,
                             dtype={"gpsno": str})
        for chunk in reader:
            chunk = _rename_canonical(chunk, cfg.column_map)
            missing = [c for c in TRAJ_REQUIRED_COLS if c not in chunk.columns]
            if missing:
                raise ValueError(f"轨迹分片缺少必需列 {missing}：{shard}")
            chunk = chunk[chunk["gpsno"].astype(str).isin(vehicles)]
            if not len(chunk):
                continue
            n = len(chunk)
            data: dict[str, np.ndarray] = {"t": to_epoch_s(chunk["data_time"])}
            for c in TRAJ_OPTIONAL_COLS + ("speed", "lat", "lon"):
                data[c] = (pd.to_numeric(chunk[c], errors="coerce").to_numpy(dtype=float)
                           if c in chunk.columns else np.full(n, np.nan))
            gps = chunk["gpsno"].astype(str).to_numpy()
            for g in dict.fromkeys(gps.tolist()):
                sel = gps == g
                buf = buffers.setdefault(g, {c: [] for c in data})
                for c, arr in data.items():
                    buf[c].append(arr[sel])
    return {g: {c: np.concatenate(v) for c, v in d.items()} for g, d in buffers.items()}


# ------------------------------------------------------------- 单车特征（Tier 1/2/3）

def window_exposure(t_epoch: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                    window_start_s: float, window_end_s: float) -> tuple[float, float]:
    """窗内暴露（§3.0 计数归一分母）：里程 km 与运行时长 h。

    相邻有效点（坐标有限、0 < dt <= 180 min）的 Haversine 步长与间隔累计
    （与 trajectory.daily_rhythm 同口径）；窗内不足 2 个有效点对时相应值为 0。
    """
    t = np.asarray(t_epoch, dtype=float)
    la = np.asarray(lat, dtype=float)
    lo = np.asarray(lon, dtype=float)
    m = (t >= window_start_s) & (t < window_end_s) & np.isfinite(la) & np.isfinite(lo)
    t, la, lo = t[m], la[m], lo[m]
    order = np.argsort(t, kind="stable")
    t, la, lo = t[order], la[order], lo[order]
    if t.size < 2:
        return 0.0, 0.0
    dt = np.diff(t)
    ok = (dt > 0) & (dt <= SPELL_GAP_S)
    step_m = np.asarray(haversine_m(la[:-1], lo[:-1], la[1:], lo[1:]), dtype=float)
    km = float(step_m[ok].sum() / 1000.0)
    hours = float(dt[ok].sum() / 3600.0)
    return km, hours


def night_exposure_share(t_epoch: np.ndarray, window_start_s: float, window_end_s: float) -> float:
    """深夜暴露占比＝窗内轨迹点中深夜（23–5）点占比（§3.2 夜间风险链分的“夜间暴露”组成）。

    口径随实现登记：以轨迹点计数占比度量夜间暴露；窗内无轨迹点置缺失。
    """
    t = np.asarray(t_epoch, dtype=float)
    m = (t >= window_start_s) & (t < window_end_s)
    if not m.any():
        return float("nan")
    h = hour_of_day(t[m])
    return float(((h >= NIGHT_LO) | (h < NIGHT_HI)).mean())


def fatigue_night_concentration(event_t_epoch: np.ndarray, event_scenario,
                                as_of_s: float, lookback_s: float) -> float:
    """疲劳报警夜间集中度＝窗内疲劳类（scenario 名含 "fatigue"）事件中深夜（23–5）占比。

    口径随实现登记（§3.2 综合分组成项）：疲劳类事件为 0 置缺失；标签窗排除同其余特征。
    """
    t = np.asarray(event_t_epoch, dtype=float)
    sc = np.asarray(event_scenario).astype(str)
    m = (t >= as_of_s - lookback_s) & (t < as_of_s)
    t, sc = t[m], sc[m]
    fat = np.array([FATIGUE_KEYWORD in s for s in sc], dtype=bool)
    if not fat.any():
        return float("nan")
    h = hour_of_day(t[fat])
    return float(((h >= NIGHT_LO) | (h < NIGHT_HI)).mean())


def event_speed_mean(event_t_epoch: np.ndarray, event_speed: np.ndarray,
                     as_of_s: float, lookback_s: float) -> float:
    """事件时速（§3.0 双速度源：事件发生时车速，交付表 `speed` 列）。

    每车一行聚合口径随实现登记：窗内事件车速（有限值）均值；窗内无事件置缺失。
    """
    t = np.asarray(event_t_epoch, dtype=float)
    sp = np.asarray(event_speed, dtype=float)
    m = (t >= as_of_s - lookback_s) & (t < as_of_s) & np.isfinite(sp)
    return float(sp[m].mean()) if m.any() else float("nan")


def count_dual_track(stem: str, value: float, km: float, hours: float,
                     min_km: float, min_hours: float) -> dict:
    """计数归一（§3.0）：计数类 per_1000km／per_100h 双轨，低暴露置缺失（沿用 F1 规则）。"""
    ok_v = np.isfinite(value)
    per_km = (value * 1000.0 / km) if ok_v and km >= min_km and km > EPS else float("nan")
    per_h = (value * 100.0 / hours) if ok_v and hours >= min_hours and hours > EPS else float("nan")
    return {stem + DUAL_SUFFIX_KM: per_km, stem + DUAL_SUFFIX_H: per_h}


def vehicle_new_features(t_epoch: np.ndarray, run_duration_s: np.ndarray, speed: np.ndarray,
                         lat: np.ndarray, lon: np.ndarray,
                         ems_speed: np.ndarray, gps_speed: np.ndarray,
                         ev_t: np.ndarray, ev_scenario: np.ndarray, ev_speed: np.ndarray,
                         ev_lat: np.ndarray, ev_lon: np.ndarray,
                         profile_month_km: float, profile_month_hours: float,
                         profile_month_night_share: float,
                         as_of_s: float, lookback_s: float,
                         min_km: float, min_hours: float,
                         copair_columns=()) -> tuple[dict, float, float]:
    """单车 Tier 1/2/3 新增特征（一遍扫描聚合阶段调用，纯函数）。

    返回 (特征字典, 窗内里程 km, 窗内运行时长 h)。键序即声明列序：Tier1 历史险情时间结构 →
    画像基线偏差 → 夜间退化度族 → Tier2 轨迹族（疲劳结构/速度形态/空间结构/日节律）→
    Tier3 事件连环/共现/双速度源/事件时速。计数类按 §3.0 双轨产出
    （{stem}_per_1000km／{stem}_per_100h，低暴露置缺失），原始计数不入表（原始计数收敛）；
    共现列仅取预登记组合（copair_columns，§3.0 预登记显著组合入特征）。
    """
    w0, w1 = as_of_s - lookback_s, as_of_s
    km, hours = window_exposure(t_epoch, lat, lon, w0, w1)
    lookback_days = lookback_s / DAY_S
    raw: dict = {}
    # ---- Tier 1：历史险情时间结构
    hist = historical_temporal_features(ev_t, as_of_s, lookback_s)
    for key in ("f3_hist_decay_count", "f3_hist_recurrence_gap_days", "f3_hist_burst_index",
                "f3_hist_trend_per_day", "f3_hist_x_trend"):
        raw[key] = hist[key]
    # ---- Tier 1：画像基线偏差（预期/实际对账）
    night_share = night_exposure_share(t_epoch, w0, w1)
    prof = profile_deviation(np.array([km]), np.array([hours]), np.array([night_share]),
                             np.array([profile_month_km]), np.array([profile_month_hours]),
                             np.array([profile_month_night_share]),
                             lookback_days=lookback_days)
    for c in prof.columns:
        raw[c] = float(prof[c].iloc[0])
    # ---- Tier 1：夜间退化度族（六场景合并＋分场景）
    nd = night_degradation(ev_t, ev_scenario, t_epoch, as_of_s, lookback_s=lookback_s)
    for _, r in nd.iterrows():
        tag = "night" if r["scenario"] == "__merged__" else f"night_{r['scenario']}"
        raw[f"f3_{tag}_deep_rate"] = r["f3_night_deep_rate"]
        raw[f"f3_{tag}_day_rate"] = r["f3_night_day_rate"]
        raw[f"f3_{tag}_degradation"] = r["f3_night_degradation"]
    raw["f3_night_exposure_share"] = night_share
    raw["f3_fatigue_night_conc"] = fatigue_night_concentration(ev_t, ev_scenario,
                                                               as_of_s, lookback_s)
    # ---- Tier 2：轨迹族一次带出（疲劳结构/速度形态/空间结构/日节律）
    raw.update(spell_features(t_epoch, w0, w1, run_duration_s))
    raw.update(speed_shape_features(t_epoch, w0, w1, speed))
    raw.update(spatial_features(lat, lon, t_epoch, ev_lat, ev_lon, ev_t, w0, w1))
    raw.update(daily_rhythm_features(t_epoch, lat, lon, w0, w1))
    # ---- Tier 3：事件连环/共现/双速度源/事件时速
    chains = event_chains(ev_t, ev_scenario, as_of_s, lookback_s)
    raw["f3_chain_event_count"] = chains.pop("f3_chain_event_count")
    raw["f3_chain_max_len"] = chains.pop("f3_chain_max_len")
    for col in copair_columns:
        raw[col] = chains.get(col, 0.0)
    raw.update(ems_gps_discrepancy(ems_speed, gps_speed, t_epoch, as_of_s, lookback_s))
    raw["f3_event_speed_mean"] = event_speed_mean(ev_t, ev_speed, as_of_s, lookback_s)
    # ---- 计数归一双轨（§3.0：计数类一律双轨，入基座只留 per_1000km 版由收尾通则处理）
    out: dict = {}
    for key, value in raw.items():
        if is_count_column(key):
            out.update(count_dual_track(key, value, km, hours, min_km, min_hours))
        else:
            out[key] = value
    return out, km, hours


# ------------------------------------------------------------- 车队聚合（同群/综合分）

def cohort_relativize_fleet(values: pd.DataFrame, energy_type, highway_share, fold,
                            vehicle_ids=None,
                            min_cohort_vehicles: int = COHORT_MIN_VEHICLES) -> pd.DataFrame:
    """同群相对化（§3.0/§3.2，仅对既有强特征列）：群内 z-score／百分位，统计量折内拟合（红线 3）。

    群键＝能源类型×（高速里程占比车队中位二分）；对每个折 f 仅用 fold != f 的行拟合群内
    均值/方差/百分位与中位二分（OOF 防泄漏，缺失 fold 按 "__na__" 单列处理），套用到折 f 的行；
    群内不足 min_cohort_vehicles 台回退全体统计并置 cohort_fallback=1。
    输出列：{col}_cohort_z、{col}_cohort_pct、cohort_fallback（回退诊断标志，非特征列）。
    """
    folds = pd.Series(fold).fillna("__na__").astype(str).to_numpy()
    cols: list[str] = []
    for c in values.columns:
        cols += [f"{c}_cohort_z", f"{c}_cohort_pct"]
    cols += ["cohort_fallback"]  # 列序与 core.cohort_relative 输出一致
    res = pd.DataFrame(np.nan, index=values.index, columns=cols)
    for f in sorted(set(folds.tolist())):
        train = folds != f
        keys, _ = cohort_keys(energy_type, highway_share, train_mask=train)
        part = cohort_relative(values, keys, train, vehicle_ids=vehicle_ids,
                               min_cohort_vehicles=min_cohort_vehicles)
        sel = np.flatnonzero(folds == f)
        res.iloc[sel, :] = part.iloc[sel, :].to_numpy()
    return res


# ------------------------------------------------------------- 编排（scan / interface / run-all）

def _empty_arrays(n: int = 0) -> dict[str, np.ndarray]:
    return {c: np.full(n, np.nan) for c in TRAJ_COLS[2:]} | {"t": np.zeros(n)}


def run_scan(cfg: F3Config) -> Path:
    """一遍扫描编排（分片读取→聚合）→ f3_new_features.csv（每车一行，键 sample_id/gpsno）。

    轨迹分片一遍流式读取按车缓冲；事件/画像/基座小表整读；聚合阶段按车调用
    vehicle_new_features（Tier 1/2/3），再做同群相对化（折内拟合）与综合分合成
    （固定等权、不做数据驱动调权）。
    """
    base = pd.read_csv(cfg.base_input, encoding="utf-8-sig", dtype={"gpsno": str})
    profile = pd.read_csv(cfg.profile_path, encoding="utf-8-sig", dtype={"gpsno": str})
    events = pd.read_csv(cfg.events_path, encoding="utf-8-sig", dtype={"gpsno": str})
    roster = base[["sample_id", "gpsno"]].copy()
    roster["gpsno"] = roster["gpsno"].astype(str)
    vehicles = set(roster["gpsno"])

    buffers = _scan_trajectory_shards(cfg, vehicles)
    ev_groups = {g: sub for g, sub in events.groupby(events["gpsno"].astype(str))}
    prof_idx = profile.assign(gpsno=profile["gpsno"].astype(str)).set_index("gpsno")

    as_of_s, lookback_s = cfg.as_of_s, cfg.lookback_s
    rows = []
    for rec in roster.itertuples(index=False):
        gps = str(rec.gpsno)
        buf = buffers.get(gps)
        if buf is None:
            buf = _empty_arrays()
        ev = ev_groups.get(gps)
        if ev is None:
            ev_t = np.zeros(0)
            ev_sc = np.zeros(0, dtype=object)
            ev_sp = np.zeros(0)
            ev_la = np.zeros(0)
            ev_lo = np.zeros(0)
        else:
            ev_t = to_epoch_s(ev["event_time"])
            ev_sc = ev["scenario"].astype(str).to_numpy()
            ev_sp = pd.to_numeric(ev["speed"], errors="coerce").to_numpy(dtype=float)
            ev_la = pd.to_numeric(ev["lat"], errors="coerce").to_numpy(dtype=float)
            ev_lo = pd.to_numeric(ev["lon"], errors="coerce").to_numpy(dtype=float)
        if gps in prof_idx.index:
            prof = prof_idx.loc[gps]
            if isinstance(prof, pd.DataFrame):
                prof = prof.iloc[0]
            p_km = float(pd.to_numeric(prof.get("profile_month_km"), errors="coerce"))
            p_hours = float(pd.to_numeric(prof.get("profile_month_hours"), errors="coerce"))
            p_night = float(pd.to_numeric(prof.get("profile_month_night_share"),
                                          errors="coerce"))
        else:
            p_km = p_hours = p_night = float("nan")
        feats, km, hours = vehicle_new_features(
            buf["t"], buf["run_duration_s"], buf["speed"], buf["lat"], buf["lon"],
            buf["ems_speed"], buf["gps_speed"],
            ev_t, ev_sc, ev_sp, ev_la, ev_lo,
            p_km, p_hours, p_night, as_of_s, lookback_s,
            cfg.min_km, cfg.min_hours, copair_columns=cfg.copair_columns)
        rows.append({"sample_id": rec.sample_id, "gpsno": gps, **feats})

    new_df = pd.DataFrame(rows)
    # 同群相对化（既有强特征，统计量折内拟合）→ 新增列
    if cfg.cohort_source_columns:
        src_cols = [c for c in cfg.cohort_source_columns if c in base.columns]
        missing_src = [c for c in cfg.cohort_source_columns if c not in base.columns]
        if missing_src:
            raise ValueError(f"同群相对化源列不在基座表：{missing_src}")
        keyed = base.set_index("sample_id").loc[new_df["sample_id"]]
        energy = keyed[cfg.cohort_energy_column].to_numpy() \
            if cfg.cohort_energy_column in keyed.columns else np.full(len(keyed), "unknown")
        highway = pd.to_numeric(keyed[cfg.cohort_highway_column], errors="coerce").to_numpy() \
            if cfg.cohort_highway_column in keyed.columns else np.full(len(keyed), np.nan)
        values = keyed[src_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        rel = cohort_relativize_fleet(values, energy, highway,
                                      keyed["fold"].to_numpy(),
                                      vehicle_ids=new_df["gpsno"].to_numpy())
        rel = rel.reset_index(drop=True)
        for c in rel.columns:
            new_df[c] = rel[c].to_numpy()
    # 综合分（§3.0：方向统一后固定等权合成，不做数据驱动调权；组成列在特征字典登记）
    scores = composite_scores(
        new_df["f3_night_exposure_share"], new_df["f3_night_deep_rate"],
        new_df["f3_night_degradation"], new_df["f3_traj_max_spell_h"],
        new_df["f3_traj_night_spell_h"], new_df["f3_fatigue_night_conc"])
    new_df["f3_score_night_chain"] = scores["f3_score_night_chain"].to_numpy()
    new_df["f3_score_fatigue"] = scores["f3_score_fatigue"].to_numpy()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / NEW_FEATURES_FILE
    new_df.to_csv(out, index=False)
    print(f"OK F3 新增特征：{len(new_df)} 行 × {len(new_df.columns) - 2} 列 -> {out}", flush=True)
    return out


def run_interface(cfg: F3Config) -> Path:
    """interface 组装（lean 基座＋新增保留列 → F3_v1 model_input＋剔除清单＋manifest）。"""
    return build_f3_interface(cfg)


def run_all(cfg: F3Config) -> Path:
    """一遍扫描（分片读取→聚合）→ interface 组装（全流程）。"""
    run_scan(cfg)
    return run_interface(cfg)
