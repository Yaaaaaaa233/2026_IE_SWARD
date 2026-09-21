# -*- coding: utf-8 -*-
"""评估指标库：严格按 EVAL-001 v0.1（MOE/MOP）实现。

要点：
- AUC/AP/Brier/LogLoss 公式与 sklearn 口径一致；单一类别/非法输入显式报错或记 undefined。
- Top-K 固定名额政策：K=ceil(q·N)，同分按 sample_id 字典序，报告等分数量（EVAL-001 §3.3）。
- 配对 bootstrap：车辆级、成对抽新旧预测、2000 次、固定种子、95% 百分位区间（§5.4）；
  无效重采样（无双类）记录次数与原因，不静默补值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from common import PREREG


# ---------------------------------------------------------------- 硬检查
def hard_check_predictions(sample_ids, y, p) -> list[str]:
    """EVAL-001 §6.1 先通过硬检查；返回失败原因列表（空=通过）。"""
    errors = []
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)
    ids = np.asarray(sample_ids)
    if len(p) != len(y) or len(ids) != len(y):
        errors.append("长度不一致")
    if np.isnan(p).any() or np.isinf(p).any():
        errors.append("概率含缺失/非有限值")
    if ((p < 0) | (p > 1)).any():
        errors.append("概率越界 [0,1]")
    if pd.Series(ids).duplicated().any():
        errors.append("sample_id 重复")
    if not set(np.unique(y)) <= {0, 1}:
        errors.append("标签非 0/1")
    return errors


# ---------------------------------------------------------------- 指标
def auc_or_none(y, s) -> float | None:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None  # EVAL-001 §3.1：单一类别无定义，不补值
    return float(roc_auc_score(y, s))


def ap_or_none(y, s) -> float | None:
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or y.sum() == 0:
        return None
    return float(average_precision_score(y, s))


def brier(y, p) -> float:
    return float(brier_score_loss(np.asarray(y), np.asarray(p, dtype=float)))


def logloss(y, p, eps: float | None = None) -> float:
    eps = eps if eps is not None else PREREG["EPS"]
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def topk_table(y, p, sample_ids, q_levels=None) -> pd.DataFrame:
    """固定管理名额政策：降序取前 K，同分按 sample_id 字典序；报告等分数与 TP_K。"""
    q_levels = q_levels or PREREG["Q_LEVELS"]
    df = pd.DataFrame({"sid": sample_ids, "y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    n, npos = len(df), int(df.y.sum())
    rows = []
    for q in q_levels:
        k = int(np.ceil(q * n))
        df_sorted = df.sort_values(["p", "sid"], ascending=[False, True], kind="mergesort")
        top = df_sorted.head(k)
        tp = int(top.y.sum())
        tie_at_cut = int((top.p == top.p.iloc[-1]).sum()) if k > 0 else 0
        rows.append({
            "q": q, "K": k, "K/N": round(k / n, 3), "TP_K": tp,
            "Recall@K": (tp / npos) if npos > 0 else None,
            "Precision@K": (tp / k) if k > 0 else None,
            "Lift@K": ((tp / k) / (npos / n)) if (npos > 0 and k > 0) else None,
            "ties_at_cut": tie_at_cut,
        })
    return pd.DataFrame(rows)


def calibration_bins(y, p, n_bins: int = 10) -> pd.DataFrame:
    """等宽分箱校准曲线（预登记 10 箱；EVAL-001 §3.4）。"""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.cut(df.p, bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    g = df.groupby("bin", observed=True).agg(
        n=("y", "size"), pred_mean=("p", "mean"), true_rate=("y", "mean"))
    return g.reset_index()


def all_metrics(y, p, sample_ids) -> dict:
    """MOE/MOP 汇总（不含 bootstrap）。"""
    errs = hard_check_predictions(sample_ids, y, p)
    tk = topk_table(y, p, sample_ids)
    k100 = tk[tk.K == PREREG["K_QUOTA"]]
    return {
        "hard_check_errors": errs,
        "N": len(y), "N_pos": int(np.sum(y)),
        "AUC": auc_or_none(y, p),
        "AP": ap_or_none(y, p),
        "Recall@100": float(k100["Recall@K"].iloc[0]) if len(k100) else None,
        "Precision@100": float(k100["Precision@K"].iloc[0]) if len(k100) else None,
        "Lift@100": float(k100["Lift@K"].iloc[0]) if len(k100) else None,
        "Brier": brier(y, p),
        "LogLoss": logloss(y, p),
    }


# ---------------------------------------------------------------- 配对 bootstrap
def paired_bootstrap_auc(y, p_new, p_base, sample_ids,
                         n_resamples=None, seed=None, ci=None) -> dict:
    """车辆级配对 bootstrap：ΔAUC = AUC_new − AUC_base 的 95% 百分位区间。

    一次抽样同时携带该车的新旧预测；同车绑定，不拆散。无效重采样（无双类）计数并记录。
    """
    n_resamples = n_resamples or PREREG["BOOT_N"]
    seed = PREREG["BOOT_SEED"] if seed is None else seed
    ci = PREREG["CI"] if ci is None else ci
    y = np.asarray(y); p_new = np.asarray(p_new, float); p_base = np.asarray(p_base, float)
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    point = auc_or_none(y, p_new) - auc_or_none(y, p_base)
    deltas, invalid = [], 0
    for _ in range(n_resamples):
        ps = rng.choice(pos_idx, len(pos_idx), replace=True)
        ng = rng.choice(neg_idx, len(neg_idx), replace=True)
        yy = np.r_[np.ones(len(ps)), np.zeros(len(ng))]
        a1 = roc_auc_score(yy, np.r_[p_new[ps], p_new[ng]])
        a2 = roc_auc_score(yy, np.r_[p_base[ps], p_base[ng]])
        deltas.append(a1 - a2)
    deltas = np.array(deltas)
    lo, hi = np.quantile(deltas, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "delta_auc_point": float(point),
        "ci_low": float(lo), "ci_high": float(hi),
        "p_improve": float((deltas > 0).mean()),
        "n_resamples": n_resamples, "invalid_resamples": int(invalid),
        "seed": seed,
    }


def paired_bootstrap_topk(y, p_new, p_base, sample_ids, K=None,
                          n_resamples=None, seed=None) -> dict:
    """Top-K 配对：每次重采样集合上重新排序、按固定名额政策重选 K（EVAL-001 §5.4）。"""
    K = K or PREREG["K_QUOTA"]
    n_resamples = n_resamples or PREREG["BOOT_N"]
    seed = PREREG["BOOT_SEED"] if seed is None else seed
    y = np.asarray(y); p_new = np.asarray(p_new, float); p_base = np.asarray(p_base, float)
    ids = np.asarray(sample_ids)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    deltas = []
    for _ in range(n_resamples):
        take = rng.choice(idx, len(idx), replace=True)
        yy, ii = y[take], ids[take]
        if len(np.unique(yy)) < 2:
            continue
        k = max(1, int(np.ceil(0.2 * len(yy))))
        order_n = np.lexsort((ii, -p_new[take]))
        order_b = np.lexsort((ii, -p_base[take]))
        npos = yy.sum()
        tp_n = yy[order_n[:k]].sum(); tp_b = yy[order_b[:k]].sum()
        if npos > 0:
            deltas.append(tp_n / npos - tp_b / npos)
    deltas = np.array(deltas)
    return {
        "delta_recall_point_median": float(np.median(deltas)),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "n_valid": len(deltas),
    }
