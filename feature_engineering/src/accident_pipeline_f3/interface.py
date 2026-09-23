# -*- coding: utf-8 -*-
"""F3 interface：组装 F3R5_v1 model_input（lean 基座＋新增保留列）。

FEAT-007 r4 建立；FEAT-008 r5 按 docs/plans/feat-008-r5-execution.md §2 预登记改
select_new_columns 为族配额制（每族 <=6、分场景夜间不入模、综合分恒优先、总量 <=40、
族间轮转防挤占——修复 r4 夜间分场景 30 列吃光预算/轨迹 0 列入模的构成缺陷）。

设计依据 docs/plans/feat-007-r4-execution.md §3.0 预登记定义（原口径继承）：
- 基座预登记：候选基座＝FEAT-006 候选 B（lean）保留清单；load_base_columns 从配置声明的
  列清单文件读取（FEAT-006 保留/剔除清单在受控目录不在仓库，
  configs/pipeline_f3_v1.example.yaml 留 base_columns_file 占位），清单列序即声明列序；
- 新增列＝Tier 1/2/3 产出，上限 40（r5 §2 族配额制）：select_new_columns 按族配额
  （每族 <=6、族内声明列序、score 恒优先、族间按预登记族序轮转），确定性；
- 计数归一（§3.0）：计数类 per_1000km／per_100h 双轨由 pipeline 产出，入基座时同场景只保留
  per_1000km 版——收尾通则统一处理（finalize_columns 第一步剔除 per_100h 版）；
- 近常数排除（§3.0）：接口层按方差阈值排除（不按缺失率），恒 0／常数列不得混入 model_input
  （复用 core.near_constant_filter，方差阈值 core.NEAR_CONST_VAR）；
- 收尾通则（§3.0，FEAT-006 §1 规则 7 沿用）：|Spearman ρ|>0.95 成对去后者（按声明列序保先者，
  固定列序确定性）、近零方差剔除；剔除清单 JSON 登记（同输入两次运行逐字节一致）；
- manifest：行数／列数／代码提交号／输入指纹（后者 resolve_input_fingerprint 留占位函数），
  并登记基座保留／新增保留清单，支撑 L1 契约“总列数＝lean 基座＋新增保留列可对账”；
- 契约断言 assert_contract（L1 #4）：sample_id 唯一、每车一行、y/fold/窗口元数据不得进特征列、
  事件表主键唯一且事件时刻落在特征窗内、总列数＝基座保留＋新增保留可对账。
纯函数层不做文件 IO；build_f3_interface 承担输入读取与产物写出。依赖仅 numpy/pandas。
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from .core import NEAR_CONST_VAR, near_constant_filter, to_epoch_s, valid_time_mask

if TYPE_CHECKING:  # pragma: no cover - 仅类型标注，避免与 pipeline 循环导入
    from .pipeline import F3Config

FEATURE_VERSION = "F3R5_v1"
RHO_THRESHOLD = 0.95          # §3.0 收尾通则：|Spearman ρ|>0.95 成对去后者（边界不含）
MAX_NEW_COLUMNS = 40          # §3.0 新增列上限（r5 §2 族配额制）
COMPOSITE_PREFIX = "f3_score_"
COPAIR_PREFIX = "f3_copair_"
DUAL_SUFFIX_KM = "_per_1000km"
DUAL_SUFFIX_H = "_per_100h"

# ---- r5 §2 族配额构成规则（预登记，修复 r4 的 40 列预算挤占缺陷）--------------------
FAMILY_CAP = 6                # 每族上限 6 列
FAMILY_ORDER = ("score", "hist", "cohort", "profile", "night", "spell", "speed",
                "spatial", "rhythm", "chain", "ems", "event", "other")
# f3_night_* 只留合并版（含暴露占比）；分场景变体不入模（预登记 §2 白名单）
NIGHT_MERGED_ALLOW = frozenset({
    "f3_night_deep_rate", "f3_night_day_rate",
    "f3_night_degradation", "f3_night_exposure_share",
})
# 族→前缀精确映射（预登记"随执行登记"：只对名字、不改规则；干名先剥计数双轨后缀）
SPELL_STEMS = frozenset({
    "f3_traj_max_spell_h", "f3_traj_gt4h_share", "f3_traj_night_spell_h",
    "f3_traj_jump_steps",
})
SPEED_PREFIXES = ("f3_traj_speed_", "f3_traj_over90_", "f3_traj_cruise80_")
SPATIAL_PREFIXES = ("f3_traj_hotspot_", "f3_traj_samepoint_")
SPATIAL_STEMS = frozenset({
    "f3_traj_route_repeat_m", "f3_traj_activity_radius_m", "f3_traj_daynight_shift_m",
})
RHYTHM_STEMS = frozenset({
    "f3_traj_daily_km_cv", "f3_traj_gap_days", "f3_traj_daily_run_h",
})

# 计数类列干（§3.0 计数归一“一律”口径）：事件/热点/复发/断档等离散发生次数统计量
COUNT_STEMS = frozenset({
    "f3_hist_decay_count",        # 衰减加权计数
    "f3_chain_event_count",       # 连环事件数
    "f3_chain_max_len",           # 最大连环长
    "f3_traj_hotspot_n",          # 热点数
    "f3_traj_samepoint_recur_n",  # 同点位复发次数
    "f3_traj_gap_days",           # 断档天数
})

# 元数据列（y/fold/窗口元数据等不得进特征列；cohort_fallback 为回退诊断标志，非风险特征）
META_ORDER = (
    "sample_id", "gpsno", "y", "fold",
    "as_of", "window_start", "window_end", "lookback_days", "horizon_days",
    "label_window", "label_status", "label_version", "split_version", "source_version",
    "feature_version", "feature_version_f1", "feature_version_f2", "feature_version_f3",
    "energy_type", "highway_share", "cohort_fallback",
)
META_COLUMNS = frozenset(META_ORDER)

MODEL_INTERFACE_DIR = "model_interface"
MODEL_INPUT_FILE = "model_input.csv"
NEW_FEATURES_FILE = "f3_new_features.csv"
DROPPED_FILE = "f3_dropped_columns.json"
MANIFEST_FILE = "f3_manifest.json"


# ------------------------------------------------------------- 清单与列名规则

def load_base_columns(path: str | Path) -> list[str]:
    """读取配置声明的 lean 基座保留清单（FEAT-006 候选 B，受控目录文件）。

    支持两种声明格式（确定性，均保持文件内列序）：
    - JSON 数组：["col_a", "col_b", ...]；
    - 纯文本：一行一列名，空行与以 # 开头的注释行忽略。
    返回去重后的列名清单（保持首次出现顺序）；文件缺失/为空/含重复列名时报错
    （清单必须确定，剔除清单与对账才可逐字节复现）。
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"基座保留清单为空：{p}")
    if text.lstrip().startswith("["):
        cols = json.loads(text)
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError(f"基座保留清单 JSON 格式错误（须为字符串数组）：{p}")
    else:
        cols = [ln.strip() for ln in text.splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c in seen:
            raise ValueError(f"基座保留清单含重复列名：{c!r}（{p}）")
        seen.add(c)
        out.append(c)
    if not out:
        raise ValueError(f"基座保留清单为空：{p}")
    return out


def count_stem(column: str) -> str:
    """剥离计数双轨后缀，返回计数列干（非双轨列返回原名）。"""
    for suf in (DUAL_SUFFIX_H, DUAL_SUFFIX_KM):
        if column.endswith(suf):
            return column[: -len(suf)]
    return column


def is_count_column(column: str) -> bool:
    """计数类判定（§3.0 计数归一/原始计数收敛共用口径）：列干在 COUNT_STEMS 或 f3_copair_ 前缀。"""
    stem = count_stem(column)
    return stem.startswith(COPAIR_PREFIX) or stem in COUNT_STEMS


def is_night_scenario_variant(column: str) -> bool:
    """f3_night_* 分场景变体判定（r5 §2：分场景夜间不入模，只留合并版白名单）。"""
    return column.startswith("f3_night_") and column not in NIGHT_MERGED_ALLOW


def family_of_column(column: str) -> str:
    """列 → 族（r5 §2 预登记映射，随执行登记；先剥计数双轨后缀取干名再归类）。

    score←f3_score_；hist←f3_hist_；cohort←干名以 _cohort_z/_cohort_pct 结尾；
    profile←f3_profile_；night←f3_night_（分场景变体已在候选期排除）；
    spell←连续驾驶四列（含 P1 质量列 f3_traj_jump_steps）；speed←f3_traj_speed_/
    over90_/cruise80_；spatial←hotspot_/samepoint_ 前缀与路线重复/活动半径/昼夜偏移；
    rhythm←日里程 CV/断档天数/日均运行时长；chain←f3_chain_/f3_copair_；ems←f3_ems_；
    event←f3_event_；其余（如 f3_fatigue_night_conc）→ other。
    """
    stem = count_stem(column)
    if stem.startswith(COMPOSITE_PREFIX):
        return "score"
    if stem.startswith("f3_hist_"):
        return "hist"
    if stem.endswith("_cohort_z") or stem.endswith("_cohort_pct"):
        return "cohort"
    if stem.startswith("f3_profile_"):
        return "profile"
    if stem.startswith("f3_night_"):
        return "night"
    if stem in SPELL_STEMS:
        return "spell"
    if stem.startswith(SPEED_PREFIXES):
        return "speed"
    if stem.startswith(SPATIAL_PREFIXES) or stem in SPATIAL_STEMS:
        return "spatial"
    if stem in RHYTHM_STEMS:
        return "rhythm"
    if stem.startswith("f3_chain_") or stem.startswith(COPAIR_PREFIX):
        return "chain"
    if stem.startswith("f3_ems_"):
        return "ems"
    if stem.startswith("f3_event_"):
        return "event"
    return "other"


def select_new_columns(candidate_columns: Sequence[str],
                       max_new: int = MAX_NEW_COLUMNS) -> list[str]:
    """新增列选取（r5 §2 族配额制，预登记；修复 r4 的 40 列预算挤占缺陷）。

    规则（执行中不得擅改）：
    1. f3_night_* 分场景变体不入模（只留合并版白名单 NIGHT_MERGED_ALLOW）；
    2. 计数双轨 per_100h 版不占配额（收尾通则必剔，候选期即排除）；
    3. 按族配额选取：每族上限 FAMILY_CAP=6 列、族内按声明列序保先、总量 <=max_new=40；
       综合分（score 族）恒优先＝整族最先入选，其余族按 FAMILY_ORDER（预登记列举序）
       轮转取列——任一族不挤占其他族为零（正对 r4 缺陷：夜间分场景 30 列吃光预算、
       轨迹 0 列入模）；
    4. 返回顺序即新增列声明列序（score 先行、余族轮转获取序），同输入两次调用逐项一致。
    """
    seen: set[str] = set()
    buckets: dict[str, list[str]] = {f: [] for f in FAMILY_ORDER}
    for c in candidate_columns:
        if c in seen or c in META_COLUMNS:
            continue
        seen.add(c)
        if is_night_scenario_variant(c):
            continue
        if c.endswith(DUAL_SUFFIX_H):
            continue                     # per_100h 双轨版不占族配额（收尾通则必剔）
        buckets[family_of_column(c)].append(c)
    selected: list[str] = buckets["score"][:min(FAMILY_CAP, max_new)]   # score 恒优先
    ptr = {f: 0 for f in FAMILY_ORDER}
    rest = tuple(f for f in FAMILY_ORDER if f != "score")
    while len(selected) < max_new:
        took = False
        for f in rest:                   # 轮转：score 之外按预登记族序
            if len(selected) >= max_new:
                break
            i, cap = ptr[f], min(FAMILY_CAP, len(buckets[f]))
            if i < cap:
                selected.append(buckets[f][i])
                ptr[f] = i + 1
                took = True
        if not took:
            break
    return selected


# ------------------------------------------------------------- 收尾通则（§3.0）

def _spearman_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """声明列的 Spearman 相关矩阵（秩相关，成对完整观测；常量列已先行剔除）。"""
    sub = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    return sub.corr(method="spearman")


def finalize_columns(frame: pd.DataFrame, declared: Sequence[str],
                     rho_threshold: float = RHO_THRESHOLD,
                     var_threshold: float = NEAR_CONST_VAR) -> tuple[list[str], list[dict]]:
    """执行收尾通则（§3.0）：双轨去 per_100h → 近零方差剔除 → |Spearman ρ|>0.95 成对去后者。

    1. 计数双轨（§3.0）：同场景声明 per_1000km 版时，per_100h 版一律剔除（入基座只留
       per_1000km 版，FEAT-006 教训；无论 twin 后续是否存活）；
    2. 近零方差（§3.0）：按方差阈值排除（不按缺失率），复用 core.near_constant_filter；
    3. |Spearman ρ|>rho_threshold 成对去后者：按声明列序保先者（固定列序确定性），
       候选与已保留列比较，命中即去后者并登记配对方与 ρ。
    返回 (kept, dropped)：kept 保持声明列序；dropped 为剔除清单条目列表，确定性顺序
    （双轨 → 近零方差 → Spearman，各阶段内按声明列序），每条形如
    {column, stage, reason, partner?, rho?}。同输入两次调用逐字节一致（可 JSON 序列化复现）。
    """
    cols = list(dict.fromkeys(declared))
    dropped: list[dict] = []
    # 1. 计数双轨：同场景只保留 per_1000km 版
    remaining: list[str] = []
    declared_set = set(cols)
    for c in cols:
        if c.endswith(DUAL_SUFFIX_H) and (c[: -len(DUAL_SUFFIX_H)] + DUAL_SUFFIX_KM) in declared_set:
            dropped.append({"column": c, "stage": "dual_track",
                            "reason": "同场景只保留 per_1000km 版（§3.0 计数归一）",
                            "partner": c[: -len(DUAL_SUFFIX_H)] + DUAL_SUFFIX_KM})
        else:
            remaining.append(c)
    # 2. 近零方差剔除（方差阈值，不按缺失率）
    kept_var = set(near_constant_filter(frame, var_threshold=var_threshold, columns=remaining))
    kept_step2: list[str] = []
    for c in remaining:
        if c in kept_var:
            kept_step2.append(c)
        else:
            dropped.append({"column": c, "stage": "near_zero_variance",
                            "reason": f"近零方差/常数列（方差阈值 {var_threshold:g}）"})
    # 3. |Spearman ρ|>0.95 成对去后者（按声明列序保先者）
    if kept_step2:
        corr = _spearman_matrix(frame, kept_step2)
    else:
        corr = pd.DataFrame()
    kept: list[str] = []
    for c in kept_step2:
        partner = None
        rho = float("nan")
        for k in kept:
            v = float(corr.loc[c, k]) if len(corr) else float("nan")
            if np.isfinite(v) and abs(v) > rho_threshold:
                partner, rho = k, v
                break
        if partner is None:
            kept.append(c)
        else:
            dropped.append({"column": c, "stage": "spearman_rho",
                            "reason": f"|Spearman ρ|>{rho_threshold:g} 成对去后者（§3.0 收尾通则）",
                            "partner": partner, "rho": round(rho, 6)})
    return kept, dropped


# ------------------------------------------------------------- 契约断言（L1 #4）

def assert_contract(model_input: pd.DataFrame, base_kept: Sequence[str],
                    new_kept: Sequence[str], events: pd.DataFrame | None = None,
                    as_of_s: float | None = None, lookback_s: float | None = None,
                    sample_col: str = "sample_id", vehicle_col: str = "gpsno",
                    event_id_col: str = "event_id",
                    event_time_col: str = "event_time") -> None:
    """契约断言（FEAT-007 §5 L1 #4；违反抛 ValueError）：

    1. sample_id 唯一且非缺失；
    2. 每车一行（gpsno 唯一）；
    3. y/fold/窗口元数据不得进特征列（base_kept ∪ new_kept 与 META_COLUMNS 交集须为空）；
    4. 事件表主键唯一且事件时刻落在特征窗 [as_of−lookback, as_of) 内（events 非空时；
       窗口边界显式传参，红线 6）；
    5. 总列数对账：model_input 特征列（剔除元数据列）== base_kept + new_kept（顺序一致）。
    """
    # 1. sample_id 唯一
    if sample_col not in model_input.columns:
        raise ValueError(f"契约违反：缺少 {sample_col} 列")
    sid = model_input[sample_col]
    if sid.isna().any():
        raise ValueError("契约违反：sample_id 缺失")
    if sid.duplicated().any():
        raise ValueError("契约违反：sample_id 不唯一")
    # 2. 每车一行
    if vehicle_col not in model_input.columns:
        raise ValueError(f"契约违反：缺少 {vehicle_col} 列")
    if model_input[vehicle_col].isna().any():
        raise ValueError(f"契约违反：{vehicle_col} 缺失")
    if model_input[vehicle_col].duplicated().any():
        raise ValueError(f"契约违反：每车一行不成立（{vehicle_col} 重复）")
    # 3. y/fold/窗口元数据不得进特征列
    declared_features = list(base_kept) + list(new_kept)
    leaked = sorted(set(declared_features) & META_COLUMNS)
    if leaked:
        raise ValueError(f"契约违反：元数据混入特征列：{leaked}")
    # 4. 事件表主键唯一且时刻落在特征窗内
    if events is not None and len(events):
        if event_id_col in events.columns and events[event_id_col].duplicated().any():
            raise ValueError("契约违反：事件表主键不唯一")
        if as_of_s is None or lookback_s is None:
            raise ValueError("契约违反：事件窗校验需显式 as_of/lookback（红线 6）")
        valid = valid_time_mask(events[event_time_col])
        t = to_epoch_s(events[event_time_col])
        ok = valid & (t >= as_of_s - lookback_s) & (t < as_of_s)
        if not bool(np.all(ok)):
            raise ValueError("契约违反：事件时刻不在特征窗 [as_of−lookback, as_of) 内")
    # 5. 总列数对账：总列数＝基座保留＋新增保留
    actual = [c for c in model_input.columns if c not in META_COLUMNS]
    if actual != declared_features:
        raise ValueError(
            "契约违反：总列数与基座保留＋新增保留不可对账"
            f"（model_input 特征 {len(actual)} 列 ≠ 基座保留 {len(list(base_kept))} ＋"
            f" 新增保留 {len(list(new_kept))}）")


# ------------------------------------------------------------- 版本登记辅助

def resolve_code_commit(repo_root: str | Path | None = None) -> str:
    """代码提交号：git rev-parse HEAD；不可用时返回 "unknown"（占位）。"""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root or "."),
                             capture_output=True, text=True, timeout=10)
        commit = out.stdout.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def resolve_input_fingerprint(cfg: "F3Config") -> str:
    """输入指纹：占位函数（FEAT-007 §5 L1 #3 版本登记三件套之一）。

    真实数据指纹算法与落库位置由受控执行补全；公开仓库不落数据派生值，暂统一返回 "PENDING"。
    """
    return "PENDING"


# ------------------------------------------------------------- 组装

def build_f3_interface(cfg: "F3Config") -> Path:
    """组装 F3_v1 model_input（lean 基座＋新增保留列）并写出剔除清单 JSON 与 manifest。

    流程（§3.0）：读 lean 基座保留清单 → 基座列（缺失列登记剔除清单）＋新增列（上限 40、
    综合分优先）→ 收尾通则（双轨去 per_100h／近零方差／|Spearman ρ|>0.95 去后者）→
    契约断言（sample_id 唯一／每车一行／元数据不入特征／事件窗／总列数对账）→ 产物写出：
    model_interface/model_input.csv、f3_dropped_columns.json（逐字节确定性）、f3_manifest.json
    （行数/列数/代码提交号/输入指纹占位）。
    """
    sample_col, vehicle_col = "sample_id", "gpsno"
    base_raw = pd.read_csv(cfg.base_input, encoding="utf-8-sig", dtype={vehicle_col: str})
    new = pd.read_csv(cfg.output_dir / NEW_FEATURES_FILE, encoding="utf-8-sig",
                      dtype={vehicle_col: str})
    events = pd.read_csv(cfg.events_path, encoding="utf-8-sig", dtype={vehicle_col: str})

    declared_base = load_base_columns(cfg.base_columns_file)
    missing = [c for c in declared_base if c not in base_raw.columns]
    base_present = [c for c in declared_base if c in base_raw.columns]
    candidates = [c for c in new.columns if c not in META_COLUMNS]
    selected = select_new_columns(candidates)
    declared = base_present + selected

    keep_cols = [sample_col] + [c for c in selected]
    merged = base_raw.merge(new[keep_cols], on=sample_col, how="left",
                            validate="one_to_one", suffixes=("", "__f3new"))
    kept, dropped_steps = finalize_columns(merged, declared)
    dropped_all = [{"column": c, "stage": "missing_in_base_input",
                    "reason": "声明列不在基座输入表"} for c in missing] + dropped_steps
    base_kept_set = set(base_present)
    base_kept = [c for c in kept if c in base_kept_set]
    new_kept = [c for c in kept if c not in base_kept_set]

    meta_out = [c for c in META_ORDER if c in merged.columns and c != "feature_version_f3"]
    model_input = merged[meta_out + kept].copy()
    model_input.insert(len(meta_out), "feature_version_f3", FEATURE_VERSION)

    assert_contract(model_input, base_kept, new_kept, events=events,
                    as_of_s=cfg.as_of_s, lookback_s=cfg.lookback_s,
                    sample_col=sample_col, vehicle_col=vehicle_col)

    out_dir = cfg.output_dir / MODEL_INTERFACE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model_input.to_csv(out_dir / MODEL_INPUT_FILE, index=False)

    dropped_payload = {
        "feature_version": FEATURE_VERSION,
        "rules": {"rho_threshold": RHO_THRESHOLD, "var_threshold": NEAR_CONST_VAR,
                  "max_new_columns": MAX_NEW_COLUMNS, "dual_track_keep": DUAL_SUFFIX_KM,
                  "family_quota": {"cap": FAMILY_CAP, "order": list(FAMILY_ORDER),
                                   "night_merged_allow": sorted(NIGHT_MERGED_ALLOW)}},
        "declared_columns": declared,
        "kept_base": base_kept,
        "kept_new": new_kept,
        "dropped": dropped_all,
    }
    (out_dir / DROPPED_FILE).write_text(
        json.dumps(dropped_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": FEATURE_VERSION,
        "rows": int(len(model_input)),
        "feature_count": len(base_kept) + len(new_kept),
        "base_declared_count": len(declared_base),
        "base_kept_count": len(base_kept),
        "new_candidate_count": len(candidates),
        "new_kept_count": len(new_kept),
        "base_kept": base_kept,
        "new_kept": new_kept,
        "new_family_counts": {f: sum(1 for c in new_kept if family_of_column(c) == f)
                              for f in FAMILY_ORDER},
        "window": {"as_of": cfg.as_of, "lookback_days": cfg.lookback_days,
                   "horizon_days": cfg.horizon_days},
        "code_commit": resolve_code_commit(),
        "input_fingerprint": resolve_input_fingerprint(cfg),
        "inputs": {"base_input": str(cfg.base_input),
                   "base_columns_file": str(cfg.base_columns_file),
                   "events_path": str(cfg.events_path),
                   "new_features": str(cfg.output_dir / NEW_FEATURES_FILE)},
        "dropped_file": DROPPED_FILE,
    }
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return out_dir / MODEL_INPUT_FILE
