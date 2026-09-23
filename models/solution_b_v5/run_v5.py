# -*- coding: utf-8 -*-
"""方案B V5 一键运行：F1 × 五成员嵌套折外 × 分群校准 → 三类输出（受控）。

运行：
    cd models/solution_b_v5
    cp config_local.template.json config_local.json   # 填 feature_root
    python run_v5.py

输出（OUT_DIR，默认本目录 outputs/，已被 .gitignore 排除）：
    oof_predictions_v5.csv   逐车折外概率（评估/裁决复算用）
    model_score.csv          评分线消费：gpsno, prob, rank_pct, calib_flag, group_id
    forecast_result.csv      官方三列 gpsno, accident, risk_prob（固定名额前100）
    manifest.json            输入指纹与版本

防泄漏：嵌套折外；校准与 n0 只在训练折内拟合选择；同种子确定性复现。
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "solution_b"))
from metrics import all_metrics, paired_bootstrap_auc  # noqa: E402

from common_v5 import (OUT_DIR, META_EXTRAS, load_f1, logit, save_manifest,  # noqa: E402
                       paths, VERSIONS)

SEED = 42
MEMBERS = ["ebm2026", "rf7", "ebm7", "ebm_i0_7", "ebm_i0_2026"]
N0_GRID = [40, 80, 160, 10**9]


def make_rf(s):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                                  max_features="sqrt", class_weight="balanced",
                                                  random_state=s, n_jobs=-1))])


def make_ebm(s, interactions):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", ExplainableBoostingClassifier(random_state=s, max_bins=64,
                                                         interactions=interactions))])


def mk(name):
    if name.startswith("rf"):
        return make_rf(int(name[2:]))
    if name.startswith("ebm_i0_"):
        return make_ebm(int(name.replace("ebm_i0_", "")), 0)
    if name.startswith("ebm"):
        return make_ebm(int(name[3:]), 10)
    raise ValueError(name)


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - y) ** 2))


def platt(s, y):
    lr = LogisticRegression(C=1e6, max_iter=2000)
    lr.fit(np.asarray(s).reshape(-1, 1), y)
    return lr


def apply_platt(lr, s):
    return lr.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]


def cal_group(s_fit, y_fit, g_fit, s_apply, g_apply, n0):
    """全局 Platt + 分群收缩（low 独立 / high w=n/(n+n0) / insufficient 纯全局）。"""
    z = logit(apply_platt(platt(s_fit, y_fit), s_apply))
    for g in ("low", "high"):
        mg = g_fit == g
        if mg.sum() < 30 or y_fit[mg].sum() < 5:
            continue
        z_g = logit(apply_platt(platt(s_fit[mg], y_fit[mg]), s_apply[g_apply == g]))
        w = 1.0 if g == "low" else mg.sum() / (mg.sum() + n0)
        sel = g_apply == g
        z[sel] = w * z_g + (1 - w) * z[sel]
    return z


def main():
    df = load_f1()
    meta = ["sample_id", "gpsno", "as_of", "lookback_days", "y", "fold", "cluster",
            "cluster_risk_rank", "group"]
    feats = [c for c in df.columns if c not in meta and c not in META_EXTRAS]
    X = df[feats].copy()
    if "energy_type" in X.columns:
        X["energy_type"] = X["energy_type"].fillna(X.energy_type.mode()[0])
        X = pd.get_dummies(X, columns=["energy_type"], drop_first=True)
    X = X.astype(float)
    y, fold, grp, sid = df.y.values, df.fold.values, df.group.values, df.sample_id.values
    n = len(y)
    print(f"[V5] F1 特征 {X.shape[1]} 列，成员 {MEMBERS}", flush=True)

    # ---- 嵌套折外 ----
    outer = {m: np.zeros(n) for m in MEMBERS}
    inner = {m: np.full((5, n), np.nan) for m in MEMBERS}
    for k in range(5):
        tr, va = fold != k, fold == k
        tr_idx = np.where(tr)[0]
        for j in [f for f in range(5) if f != k]:
            g_itr, g_iva = tr_idx[fold[tr_idx] != j], tr_idx[fold[tr_idx] == j]
            for nm in MEMBERS:
                m = mk(nm); m.fit(X.iloc[g_itr], y[g_itr])
                inner[nm][k, g_iva] = m.predict_proba(X.iloc[g_iva])[:, 1]
        for nm in MEMBERS:
            m = mk(nm); m.fit(X.iloc[tr], y[tr])
            outer[nm][va] = m.predict_proba(X.iloc[va])[:, 1]
        print(f"  fold{k} done", flush=True)

    s_out = np.mean([logit(outer[nm]) for nm in MEMBERS], axis=0)
    inner_avg = {nm: np.nanmean(logit(np.where(np.isnan(inner[nm]), np.nan, inner[nm])), axis=0)
                 for nm in MEMBERS}
    s_in = np.mean([inner_avg[nm] for nm in MEMBERS], axis=0)

    # ---- 分群校准（n0 折内按 Brier 选） ----
    p = np.zeros(n)
    for k in range(5):
        tr, va = fold != k, fold == k
        rng = np.random.default_rng(42 + k)
        half = rng.random(tr.sum()) < 0.5
        tr_idx = np.where(tr)[0]
        f_m = np.zeros(n, bool); f_m[tr_idx[~half]] = True
        e_m = np.zeros(n, bool); e_m[tr_idx[half]] = True
        n0b = min(N0_GRID, key=lambda n0: brier(
            y[e_m], 1 / (1 + np.exp(-cal_group(s_in[f_m], y[f_m], grp[f_m],
                                               s_in[e_m], grp[e_m], n0)))))
        z = cal_group(s_in[tr], y[tr], grp[tr], s_out[va], grp[va], n0b)
        p[va] = 1 / (1 + np.exp(-z))
        print(f"  fold{k} n0={n0b}", flush=True)

    # ---- 指标与 ADR-0004 判据（若有基线 OOF 在受控 outputs） ----
    mm = all_metrics(y, p, sid)
    print(f"[V5] AUC={mm['AUC']:.4f} AP={mm['AP']:.4f} R@100={mm['Recall@100']:.3f} "
          f"Brier={mm['Brier']:.4f}（数字存受控 outputs，不入公开仓库）", flush=True)
    bl_path = os.path.join(os.path.dirname(HERE), "outputs", "task1_baselines",
                           "task1_baseline_oof_predictions.csv")
    if os.path.exists(bl_path):
        bl = pd.read_csv(bl_path)
        m = pd.DataFrame({"sample_id": sid, "p": p}).merge(bl, on="sample_id")
        b = paired_bootstrap_auc(y, p, m.p_random_forest.values, sid)
        gate = b["ci_low"] > 0 and b["delta_auc_point"] > 0
        print(f"[ADR-0004] vs 冻结基线RF: ΔAUC={b['delta_auc_point']:+.4f} "
              f"[{b['ci_low']:+.4f},{b['ci_high']:+.4f}] → {'过门' if gate else '未过门'}", flush=True)

    # ---- 输出 ----
    pd.DataFrame({"sample_id": sid, "gpsno": df.gpsno.values, "y": y, "fold": fold,
                  "group": grp, "p_v5": p}).to_csv(
        os.path.join(OUT_DIR, "oof_predictions_v5.csv"), index=False)
    out = pd.DataFrame({"gpsno": df.gpsno.values, "prob": p})
    K = int(np.ceil(0.2 * n))
    order = out.sort_values("prob", ascending=False, kind="mergesort")
    flagged = set(order.head(K).gpsno)
    out["accident"] = out.gpsno.isin(flagged).astype(int)
    out["rank_pct"] = out.prob.rank(pct=True, method="average")
    out["calib_flag"] = grp
    out[["gpsno", "accident", "prob"]].to_csv(
        os.path.join(OUT_DIR, "forecast_result.csv"), index=False, float_format="%.6f")
    out[["gpsno", "prob", "rank_pct", "calib_flag"]].assign(
        group_id=grp, model=VERSIONS["model_version"],
        feature_version=VERSIONS["feature_version"]).to_csv(
        os.path.join(OUT_DIR, "model_score.csv"), index=False, float_format="%.6f")
    save_manifest({"seed": SEED, "members": MEMBERS,
                   "metrics_controlled": {k2: round(v, 4) for k2, v in mm.items()
                                          if isinstance(v, float)}})
    print(f"[V5] 输出已写入 {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
