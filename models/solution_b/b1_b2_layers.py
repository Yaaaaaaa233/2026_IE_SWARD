# -*- coding: utf-8 -*-
"""B1 分群校准 + B2 收缩残差。

B1（每个外层折独立执行，材料=该折训练部分的 inner-OOF）：
- 融合分数 s = mean(logit(p_rf), logit(p_ebm))（尺度与样本量无关，inner/outer 一致）；
- 全局校准器：Platt（1D 逻辑回归）拟合 s→y（训练部分全部样本）；
- low 群：群内独立 Platt；high 群：logit 层收缩混合 w·群内 + (1−w)·全局，
  w = n_high/(n_high+n0)，n0=80（先验强度）；insufficient 群：纯全局。
- 输出概率 = sigmoid(z)。跨群排序因此改变 → AUC/R@K 可与 B0 比较。

B2（仅 low/high 群；insufficient 强制 λ=0）：
- 残差模型 = L2 逻辑回归(C=0.1)，特征 = 全部模型特征 + s（B0 分数列），
  群内拟合于训练部分（s 取该折 inner-OOF 材料，样本自身标签从未参与其 s 的生成）；
- λ 在 {0,0.1,0.2,0.3,0.5} 网格上按训练部分内部群内 CV AUC 选择（只用训练部分标签）；
- 最终 logit：z = (1−λ)·z_B1 + λ·logit(p_resid)。λ=0 时恒等于 B1。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from common import OUT_DIR, Bundle, save_manifest
from metrics import auc_or_none

N0_PRIOR = 80          # high 群校准收缩先验强度
LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]


def fusion_score(p_rf, p_ebm):
    """B0 融合分数：成员 logit 均值（单调、尺度与样本量无关）。"""
    eps = 1e-6
    z1 = np.log(np.clip(p_rf, eps, 1 - eps) / (1 - np.clip(p_rf, eps, 1 - eps)))
    z2 = np.log(np.clip(p_ebm, eps, 1 - eps) / (1 - np.clip(p_ebm, eps, 1 - eps)))
    return (z1 + z2) / 2.0


def platt_fit(s, y):
    lr = LogisticRegression(C=1e6, max_iter=2000)  # 无正则 Platt
    lr.fit(np.asarray(s).reshape(-1, 1), y)
    return lr


def platt_apply(lr, s):
    return lr.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]


def logit_np(p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def make_residual(seed=42):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=0.1, max_iter=3000, random_state=seed)),
    ])


def run_b1_b2(bundle: Bundle, inner_rf, inner_ebm, outer_rf, outer_ebm):
    X, y, fold, grp = bundle.X.values, bundle.y, bundle.fold, bundle.group
    n = len(y)
    s_outer = fusion_score(outer_rf, outer_ebm)
    s_inner = fusion_score(inner_rf, inner_ebm)  # [5,n]，NaN=验证折位置

    p_b1 = np.zeros(n)
    p_b2 = np.zeros(n)
    lam_chosen = {g: [] for g in ("low", "high")}
    trace = []

    for k in range(5):
        tr, va = fold != k, fold == k
        tr_idx, va_idx = np.where(tr)[0], np.where(va)[0]
        mat = ~np.isnan(s_inner[k])             # 该折训练部分
        assert (mat == tr).all(), f"fold{k} inner 材料与训练部分不一致"

        s_tr, y_tr, g_tr = s_inner[k][tr], y[tr], grp[tr]
        s_va = s_outer[va]

        # ---- 全局 Platt（材料：训练部分 inner 分数） ----
        g_glob = platt_fit(s_tr, y_tr)
        z_glob_tr = logit_np(platt_apply(g_glob, s_tr))
        z_glob_va = logit_np(platt_apply(g_glob, s_va))

        # ---- 群校准参数 ----
        z_va_final = z_glob_va.copy()           # insufficient 默认纯全局
        z_tr_final = z_glob_tr.copy()
        group_fits = {}
        for g in ("low", "high"):
            m_g = g_tr == g
            if m_g.sum() < 30 or y_tr[m_g].sum() < 5:
                group_fits[g] = {"mode": "global", "n": int(m_g.sum())}
                continue
            pl_g = platt_fit(s_tr[m_g], y_tr[m_g])
            z_g_tr = logit_np(platt_apply(pl_g, s_tr[m_g]))
            z_g_va = logit_np(platt_apply(pl_g, s_va[grp[va] == g]))
            if g == "low":
                w = 1.0                          # 低风险群样本充足，独立校准
            else:
                w = m_g.sum() / (m_g.sum() + N0_PRIOR)  # 高风险群收缩
            z_tr_final[m_g] = w * z_g_tr + (1 - w) * z_glob_tr[m_g]
            sel = grp[va] == g
            z_va_final[sel] = w * z_g_va + (1 - w) * z_glob_va[sel]
            group_fits[g] = {"mode": "shrunk" if g == "high" else "group",
                             "w": round(float(w), 3), "n": int(m_g.sum())}

        p_b1[va_idx] = 1 / (1 + np.exp(-z_va_final))

        # ---- B2 残差（low/high；λ 网格在训练部分内部选） ----
        z_b2_va = z_va_final.copy()
        for g in ("low", "high"):
            idx_g_tr = tr_idx[g_tr == g]
            idx_g_va = va_idx[grp[va] == g]
            if len(idx_g_tr) < 40 or y[idx_g_tr].sum() < 8:
                lam_chosen[g].append(0.0)
                continue
            Xg = np.column_stack([X[idx_g_tr], s_inner[k][idx_g_tr]])
            yg = y[idx_g_tr]
            cv = cross_val_predict(make_residual(), Xg, yg, cv=5,
                                   method="predict_proba")[:, 1]
            z_res_tr = logit_np(cv)
            z_base_tr = z_tr_final[g_tr == g]
            best_lam, best_auc = 0.0, auc_or_none(yg, 1 / (1 + np.exp(-z_base_tr)))
            for lam in LAMBDA_GRID:
                if lam == 0:
                    continue
                zb = (1 - lam) * z_base_tr + lam * z_res_tr
                a = auc_or_none(yg, 1 / (1 + np.exp(-zb)))
                if a is not None and best_auc is not None and a > best_auc + 1e-4:
                    best_auc, best_lam = a, lam
            lam_chosen[g].append(float(best_lam))
            if best_lam > 0:
                rm = make_residual().fit(Xg, yg)
                Xg_va = np.column_stack([X[idx_g_va], s_outer[idx_g_va]])
                p_res_va = rm.predict_proba(Xg_va)[:, 1]
                z_b2_va[grp[va] == g] = (
                    (1 - best_lam) * z_va_final[grp[va] == g]
                    + best_lam * logit_np(p_res_va))
        p_b2[va_idx] = 1 / (1 + np.exp(-z_b2_va))
        trace.append({"fold": k, "group_fits": group_fits,
                      "lambda_low": lam_chosen["low"][-1],
                      "lambda_high": lam_chosen["high"][-1]})

    out = pd.DataFrame({
        "sample_id": bundle.sample_id, "gpsno": bundle.gpsno,
        "y": y, "fold": fold, "group": grp,
        "p_rf": outer_rf, "p_ebm": outer_ebm, "s_b0": s_outer,
        "p_b1": p_b1, "p_b2": p_b2,
    })
    path = os.path.join(OUT_DIR, "oof_predictions_B1B2.csv")
    out.to_csv(path, index=False)
    save_manifest("b1b2", {
        "n0_prior": N0_PRIOR, "lambda_grid": LAMBDA_GRID,
        "residual": "L2 LogisticRegression(C=0.1) on X+s, per group, inner-material fit",
        "lambda_trace": trace,
        "outputs": [os.path.basename(path)],
    })
    return out, trace


if __name__ == "__main__":
    from common import load_bundle
    from b0_backbone import run_b0
    from metrics import all_metrics, topk_table

    b = load_bundle()
    b0out, inner_rf, inner_ebm = run_b0(b)
    out, trace = run_b1_b2(b, inner_rf, inner_ebm,
                           b0out.p_rf.values, b0out.p_ebm.values)
    print("== B1/B2 阶梯（OOF） ==")
    for col, name in [("p_rf", "B0-RF"), ("p_ebm", "B0-EBM"),
                      ("p_b1", "B1 分群校准"), ("p_b2", "B2 收缩残差")]:
        mm = all_metrics(out.y, out[col], out.sample_id)
        print(f"{name:>10}: AUC={mm['AUC']:.4f}  AP={mm['AP']:.4f}  R@100={mm['Recall@100']:.3f}  "
              f"Brier={mm['Brier']:.4f}  LogLoss={mm['LogLoss']:.4f}")
    print("\nλ 轨迹：")
    for t in trace:
        print(f"  fold{t['fold']}: low={t['lambda_low']} high={t['lambda_high']} "
              f"fits={t['group_fits']}")
    print("\n分群分层（B1）：")
    for g, sub in out.groupby("group"):
        a = auc_or_none(sub.y, sub.p_b1)
        print(f"  {g:>13}: n={len(sub):3d} pos={sub.y.sum():3d} ({sub.y.mean():.1%}) AUC={a}")
