# -*- coding: utf-8 -*-
"""改进B实验（模型侧三项，批准后执行）：
 B-5 多种子成员包：RF/EBM/LGBM单调 各3种子，logit 均值融合
 B-6 单调约束 LightGBM 第三成员（暴露与危险信号单调递增、高速占比递减）
 B-7 高风险群校准收缩先验 n0 网格 {40,80,160}，训练部分内部3split选择
纪律：嵌套折外与 v1 相同；对照 v1-B1 与 RF 单模，配对 bootstrap 判优（预登记规则同 PREREG）。
"""
import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from common import PREREG, load_bundle
from metrics import all_metrics, paired_bootstrap_auc, paired_bootstrap_topk, topk_table

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
SEEDS = [42, 7, 2026]
N0_GRID = [40, 80, 160]
LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]

# 单调方向表（特征名前缀/全名 → +1/-1/0）；未列出的为 0（不约束）
MONO = {
    "evt_count": 1, "evt_count_log1p": 1, "evt_lane_count": 1, "evt_fatigue_count": 1,
    "evt_distraction_count": 1, "evt_speed_count": 1, "evt_active_days": 1,
    "traj_km_20d": 1, "traj_hours_20d": 1, "monthly_avg_mileage": 1, "monthly_avg_hours": 1,
    "highway_ratio": -1, "night_hours_ratio": 1, "night_mileage_ratio": 1,
    "imu_rows_window": 1,
}


def mono_constraints(cols):
    cons = []
    for c in cols:
        if c.startswith("accident_"):
            cons.append(1)          # IMU 语义风险量单调递增
        else:
            cons.append(MONO.get(c, 0))
    return cons


def make_rf(s):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                                  max_features="sqrt", class_weight="balanced",
                                                  random_state=s, n_jobs=-1))])


def make_ebm(s):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", ExplainableBoostingClassifier(random_state=s, max_bins=64,
                                                         interactions=10))])


def make_lgbm_factory(cols):
    cons = mono_constraints(cols)

    def factory(seed):
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", lgb.LGBMClassifier(
                             n_estimators=300, learning_rate=0.05, num_leaves=15,
                             min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, random_state=seed, verbose=-1,
                             monotone_constraints=cons))])
    return factory


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def run(bundle, members):
    """members: dict name->maker(seed)->estimator。返回外层OOF与每折inner材料。"""
    X, y, fold = bundle.X, bundle.y, bundle.fold
    n = len(y)
    outer = {m: np.zeros(n) for m in members}
    inner = {m: np.full((5, n), np.nan) for m in members}
    for k in range(5):
        tr, va = fold != k, fold == k
        tr_idx = np.where(tr)[0]
        for j in [f for f in range(5) if f != k]:
            g_itr = tr_idx[fold[tr_idx] != j]
            g_iva = tr_idx[fold[tr_idx] == j]
            for name, mk in members.items():
                m = mk()
                m.fit(X.iloc[g_itr], y[g_itr])
                inner[name][k, g_iva] = m.predict_proba(X.iloc[g_iva])[:, 1]
        for name, mk in members.items():
            m = mk()
            m.fit(X.iloc[tr], y[tr])
            outer[name][va] = m.predict_proba(X.iloc[va])[:, 1]
    return outer, inner


def fusion_score(preds):
    zs = [logit(p) for p in preds]
    return np.mean(zs, axis=0)


def platt(s, y):
    lr = LogisticRegression(C=1e6, max_iter=2000)
    lr.fit(np.asarray(s).reshape(-1, 1), y)
    return lr


def apply_platt(lr, s):
    return lr.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]


def b1_v2(bundle, s_outer, s_inner, n0_mode="grid"):
    """分群校准 v2：low 独立 / high 收缩(n0 可选) / insufficient 全局。"""
    y, fold, grp = bundle.y, bundle.fold, bundle.group
    n = len(y)
    p = np.zeros(n)
    chosen = []
    for k in range(5):
        tr, va = fold != k, fold == k
        s_tr, y_tr, g_tr = s_inner[k][tr], y[tr], grp[tr]
        s_va = s_outer[va]
        g_glob = platt(s_tr, y_tr)
        z_glob_tr = logit(apply_platt(g_glob, s_tr))
        z_glob_va = logit(apply_platt(g_glob, s_va))

        # n0 选择：训练部分内部 3 split（fold%3），评估跨群重排后的 AUC
        if n0_mode == "grid":
            scores = {}
            sp = fold % 3
            for n0 in N0_GRID:
                aucs = []
                for h in range(3):
                    fit_m = tr & (sp != h)
                    eva_m = tr & (sp == h)
                    if eva_m.sum() == 0 or len(np.unique(y[eva_m])) < 2:
                        continue
                    gg = platt(s_inner[k][fit_m], y[fit_m])
                    z_ev_glob = logit(apply_platt(gg, s_inner[k][eva_m]))
                    z_ev = z_ev_glob.copy()
                    for g in ("low", "high"):
                        mg = grp[fit_m] == g
                        if mg.sum() < 30 or y[fit_m][mg].sum() < 5:
                            continue
                        pl = platt(s_inner[k][fit_m][mg], y[fit_m][mg])
                        z_g = logit(apply_platt(pl, s_inner[k][eva_m][grp[eva_m] == g]))
                        w = (mg.sum() / (mg.sum() + n0)) if g == "high" else 1.0
                        sel = grp[eva_m] == g
                        z_ev[sel] = w * z_g + (1 - w) * z_ev_glob[sel]
                    yy = y[eva_m]
                    if len(np.unique(yy)) > 1:
                        aucs.append(roc_auc_score(yy, 1 / (1 + np.exp(-z_ev))))
                scores[n0] = np.mean(aucs) if aucs else 0.5
            best = max(N0_GRID, key=lambda x: (scores[x], -x))  # 平局取更大 n0（更保守）
            chosen.append(best)
        else:
            best = 80
            chosen.append(best)

        z_tr = z_glob_tr.copy()
        z_va = z_glob_va.copy()
        for g in ("low", "high"):
            mg = g_tr == g
            if mg.sum() < 30 or y_tr[mg].sum() < 5:
                continue
            pl = platt(s_tr[mg], y_tr[mg])
            z_g_tr = logit(apply_platt(pl, s_tr[mg]))
            z_g_va = logit(apply_platt(pl, s_va[grp[va] == g]))
            w = 1.0 if g == "low" else mg.sum() / (mg.sum() + best)
            z_tr[mg] = w * z_g_tr + (1 - w) * z_glob_tr[mg]
            sel = grp[va] == g
            z_va[sel] = w * z_g_va + (1 - w) * z_glob_va[sel]
        p[va] = 1 / (1 + np.exp(-z_va))
    return p, chosen


def main():
    b = load_bundle()
    print("成员库 v2：RF×3 + EBM×3 + LGBM单调×3 = 9 成员", flush=True)
    members = {}
    lgb_factory = make_lgbm_factory(list(b.X.columns))
    for s in SEEDS:
        members[f"rf{s}"] = (lambda seed=s: make_rf(seed))
        members[f"ebm{s}"] = (lambda seed=s: make_ebm(seed))
        members[f"lgb{s}"] = (lambda seed=s: lgb_factory(seed))
    outer, inner = run(b, members)

    results = {}
    # v2-A：6成员（不含LGBM）  v2-B：9成员（含LGBM）
    names6 = [f"rf{s}" for s in SEEDS] + [f"ebm{s}" for s in SEEDS]
    names9 = names6 + [f"lgb{s}" for s in SEEDS]
    for tag, names in [("v2_6成员", names6), ("v2_9成员", names9)]:
        s_out = fusion_score([outer[m] for m in names])
        s_in = np.mean([logit(inner[m]) for m in names], axis=0)
        s_in = 1 / (1 + np.exp(-s_in))  # inner 层先回到概率，B1 内部再取 logit 校准
        p_b1, n0_chosen = b1_v2(b, s_out, s_in, n0_mode="grid")
        key = f"B0{tag}→B1"
        results[key] = p_b1
        mm = all_metrics(b.y, p_b1, b.sample_id)
        print(f"{key}: AUC={mm['AUC']:.4f} AP={mm['AP']:.4f} R@100={mm['Recall@100']:.3f} "
              f"Brier={mm['Brier']:.4f} LogLoss={mm['LogLoss']:.4f} n0={n0_chosen}", flush=True)
        results[key + "_n0"] = n0_chosen

    # B0 层排序质量（不带校准）
    for tag, names in [("v2_6成员", names6), ("v2_9成员", names9)]:
        s_out = fusion_score([outer[m] for m in names])
        a = roc_auc_score(b.y, s_out)
        r = topk_table(b.y, s_out, b.sample_id)
        r100 = float(r[r.K == 100]["Recall@K"].iloc[0])
        print(f"B0{tag} 融合分: AUC={a:.4f} R@100={r100:.3f}", flush=True)

    # 对照：v1-B1 与 RF单模
    v1 = pd.read_csv(os.path.join(OUT, "oof_predictions_B1B2.csv"))
    rf_v1 = v1.p_rf.values
    b1_v1 = v1.p_b1.values
    y, sid = b.y, b.sample_id

    print("\n== 配对 bootstrap 判优（预登记：ΔAUC≥0.005 或 (ΔAUC>0 且 ΔBrier≥0.005)，R@100 退化≤0.02）==", flush=True)
    best_key, best_p, best_score = None, None, -1
    for key in [k for k in results if k.endswith("→B1")]:
        p_new = results[key]
        mm_new = all_metrics(y, p_new, sid)
        mm_old = all_metrics(y, b1_v1, sid)
        ba = paired_bootstrap_auc(y, p_new, b1_v1, sid)
        br = paired_bootstrap_topk(y, p_new, b1_v1, sid)
        d_brier = mm_old["Brier"] - mm_new["Brier"]
        d_r100 = mm_new["Recall@100"] - mm_old["Recall@100"]
        adopt = ((ba["delta_auc_point"] >= PREREG["DELTA_AUC_MIN"]) or
                 (ba["delta_auc_point"] > 0 and d_brier >= 0.005)) and d_r100 >= -PREREG["R100_TOLERANCE"]
        print(f"{key} vs v1-B1: ΔAUC={ba['delta_auc_point']:+.4f} [{ba['ci_low']:+.4f},{ba['ci_high']:+.4f}] "
              f"p改善={ba['p_improve']:.2f} | ΔR@100={d_r100:+.3f} ΔBrier={d_brier:+.4f} → {'采纳' if adopt else '不采纳'}", flush=True)
        if adopt and ba["delta_auc_point"] > best_score:
            best_key, best_p, best_score = key, p_new, ba["delta_auc_point"]

    # 也与 RF 单模对照（强度参照）
    if best_p is not None:
        ba = paired_bootstrap_auc(y, best_p, rf_v1, sid)
        print(f"{best_key} vs RF单模: ΔAUC={ba['delta_auc_point']:+.4f} [{ba['ci_low']:+.4f},{ba['ci_high']:+.4f}]", flush=True)

    # 分群分层（最优者）
    if best_p is not None:
        df = pd.DataFrame({"group": b.group, "y": y, "p": best_p})
        print("\n分群分层：", flush=True)
        for g, sub in df.groupby("group"):
            print(f"  {g:>13}: n={len(sub)} pos={sub.y.sum()} AUC={roc_auc_score(sub.y, sub.p):.4f}", flush=True)
        np.save(os.path.join(OUT, "improve_b_best_oof.npy"), best_p)
        with open(os.path.join(OUT, "improve_b_results.json"), "w", encoding="utf-8") as f:
            json.dump({"best_key": best_key, "n0_chosen": results.get(best_key + "_n0")},
                      f, ensure_ascii=False)

    # 单成员对照：LGBM单调 单模表现
    lgb_oof = fusion_score([outer[f"lgb{s}"] for s in SEEDS])
    mm = all_metrics(y, 1 / (1 + np.exp(-lgb_oof)), sid)
    print(f"\n参照 LGBM单调×3平均: AUC={mm['AUC']:.4f} R@100={mm['Recall@100']:.3f} Brier={mm['Brier']:.4f}", flush=True)
    ebm_oof = fusion_score([outer[f"ebm{s}"] for s in SEEDS])
    mm = all_metrics(y, 1 / (1 + np.exp(-ebm_oof)), sid)
    print(f"参照 EBM×3平均:     AUC={mm['AUC']:.4f} R@100={mm['Recall@100']:.3f} Brier={mm['Brier']:.4f}", flush=True)
    rf_oof = fusion_score([outer[f"rf{s}"] for s in SEEDS])
    mm = all_metrics(y, 1 / (1 + np.exp(-rf_oof)), sid)
    print(f"参照 RF×3平均:      AUC={mm['AUC']:.4f} R@100={mm['Recall@100']:.3f} Brier={mm['Brier']:.4f}", flush=True)


if __name__ == "__main__":
    main()
