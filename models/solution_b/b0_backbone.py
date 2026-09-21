# -*- coding: utf-8 -*-
"""B0 全局骨架：RF + EBM 双成员，冻结五折折外（OOF），rank 平均融合。

嵌套结构（防泄漏核心）：
- 外层 fold k：验证集从不参与任何拟合；
- 对每个外层折 k，在训练部分（folds≠k）内部再做五折，产生**仅属于该折**的
  inner-OOF 概率矩阵 inner_rf[k]、inner_ebm[k]（形状 [n]，验证折位置为 NaN）。
- B1 校准器拟合材料 = 第 k 折的 inner-OOF（训练部分），应用对象 = 第 k 折验证集。
  每个外层折的校准器互相独立，任何验证标签都不进入拟合。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from common import OUT_DIR, Bundle, load_bundle, save_manifest

SEED = 42


def make_rf(seed: int) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])


def make_ebm(seed: int) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("ebm", ExplainableBoostingClassifier(
            random_state=seed, max_bins=64, interactions=10)),
    ])


def run_b0(bundle: Bundle, seed: int = SEED, tag: str = "B0"):
    X, y, fold = bundle.X.values, bundle.y, bundle.fold
    n = len(y)
    p_rf = np.zeros(n); p_ebm = np.zeros(n)
    inner_rf = np.full((5, n), np.nan)   # [外层折k, 样本i]：k折训练部分的内部折外概率
    inner_ebm = np.full((5, n), np.nan)

    for k in range(5):
        tr, va = fold != k, fold == k
        tr_idx = np.where(tr)[0]
        inner_folds = [j for j in range(5) if j != k]  # 训练部分内实际存在的折
        for j in inner_folds:
            g_itr = tr_idx[fold[tr_idx] != j]
            g_iva = tr_idx[fold[tr_idx] == j]
            m = make_rf(seed); m.fit(X[g_itr], y[g_itr])
            inner_rf[k, g_iva] = m.predict_proba(X[g_iva])[:, 1]
            m = make_ebm(seed); m.fit(X[g_itr], y[g_itr])
            inner_ebm[k, g_iva] = m.predict_proba(X[g_iva])[:, 1]
        m = make_rf(seed); m.fit(X[tr], y[tr])
        p_rf[va] = m.predict_proba(X[va])[:, 1]
        m = make_ebm(seed); m.fit(X[tr], y[tr])
        p_ebm[va] = m.predict_proba(X[va])[:, 1]

    p_b0 = (pd.Series(p_rf).rank() + pd.Series(p_ebm).rank()).values

    out = pd.DataFrame({
        "sample_id": bundle.sample_id, "gpsno": bundle.gpsno,
        "y": y, "fold": fold, "group": bundle.group,
        "p_rf": p_rf, "p_ebm": p_ebm, "p_b0_rank": p_b0,
    })
    path = os.path.join(OUT_DIR, f"oof_predictions_{tag}.csv")
    out.to_csv(path, index=False)
    np.savez(os.path.join(OUT_DIR, f"inner_oof_{tag}.npz"),
             inner_rf=inner_rf, inner_ebm=inner_ebm,
             outer_rf=p_rf, outer_ebm=p_ebm)

    save_manifest(tag.lower(), {
        "seed": seed,
        "members": ["RandomForest(500,leaf5,sqrt,balanced)", "EBM(64bins,10inter)"],
        "fusion": "rank-average on OOF probabilities",
        "nested": ("outer frozen 5-fold; per-outer-fold inner 5-fold inside its train part; "
                   "calibration material for fold k = inner-OOF of train part only"),
        "outputs": [os.path.basename(path), f"inner_oof_{tag}.npz"],
    })
    return out, inner_rf, inner_ebm


if __name__ == "__main__":
    b = load_bundle()
    out, inner_rf, inner_ebm = run_b0(b)
    from metrics import all_metrics
    print("== B0 成员与融合（OOF） ==")
    for col, name, is_prob in [("p_rf", "RF 单模", True), ("p_ebm", "EBM 单模", True),
                               ("p_b0_rank", "B0 rank融合", False)]:
        if is_prob:
            mm = all_metrics(out.y, out[col], out.sample_id)
            print(f"{name:>10}: AUC={mm['AUC']:.4f}  AP={mm['AP']:.4f}  R@100={mm['Recall@100']:.3f}  "
                  f"Brier={mm['Brier']:.4f}  硬检查={mm['hard_check_errors'] or 'PASS'}")
        else:
            # rank 分数仅作排序用；概率尺度指标待 B1 校准后报告
            from metrics import topk_table, auc_or_none, ap_or_none
            a, ap_ = auc_or_none(out.y, out[col]), ap_or_none(out.y, out[col])
            r100 = topk_table(out.y, out[col], out.sample_id)
            r100 = r100[r100.K == 100]["Recall@K"].iloc[0]
            print(f"{name:>10}: AUC={a:.4f}  AP={ap_:.4f}  R@100={r100:.3f}  （rank 分，概率指标见 B1）")
    # 每个外层折的 inner 材料恰好覆盖其训练部分（400 样本），验证折为 NaN
    for k in range(5):
        cov = (~np.isnan(inner_rf[k])).sum()
        assert cov == (b.fold != k).sum(), f"fold{k} inner 覆盖 {cov} ≠ 训练部分 {(b.fold != k).sum()}"
        assert np.isnan(inner_rf[k][b.fold == k]).all(), f"fold{k} inner 泄漏到验证折"
    print("inner-OOF 结构检查：PASS（每折恰覆盖训练部分，验证折 NaN）")
