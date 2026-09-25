# -*- coding: utf-8 -*-
"""冲刺公共评估栈：协议 v2 标签/折分装载、统一指标、双锚配对、叶安式严格嵌套选型。

一切 gXvY 版本共用本模块，保证组间/版本间可比。
协议：label_v2_record_count_20260923 / split_v2_record_strat5_seed42（54 正类）；
特征窗 [2026-06-01, 2026-06-21)，标签窗 [2026-06-21, 2026-07-31)；
时间可见性排除集（对齐叶安 v2 排除清单）：8 动态画像列 + energy_type + 画像派生。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SPRINT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(SPRINT, ".."))
sys.path.insert(0, os.path.join(REPO, "models", "solution_b"))
from metrics import all_metrics, paired_bootstrap_auc, paired_bootstrap_topk  # noqa: E402

PROTOCOL_DIR = os.path.join(SPRINT, "_common", "protocol")
BLOCKED = {
    "monthly_avg_mileage", "monthly_avg_hours", "monthly_avg_stops", "highway_ratio",
    "morning_ratio", "dusk_ratio", "night_hours_ratio", "night_mileage_ratio",
    "energy_type", "f2_prior_incident_x_night", "f2_prior_incident_x_highway",
}
META = {"sample_id", "gpsno", "y", "fold", "group", "as_of", "lookback_days",
        "label_window", "horizon_days", "label_version", "split_version", "label_status",
        "feature_version", "source_version"}

# 双锚（预登记引用值，不随版本变）
ANCHOR_RF_F0V2 = 0.8130        # 我线 MODEL-007 复跑（受控）
ANCHOR_YE_F3 = 0.8450          # 叶安 FEAT-009 F3 嵌套（受控引用，未逐车对账）


def load_protocol() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(os.path.join(PROTOCOL_DIR, "labels.csv"),
                         dtype={"sample_id": str, "gpsno": str})
    splits = pd.read_csv(os.path.join(PROTOCOL_DIR, "splits.csv"), dtype={"gpsno": str})
    assert len(labels) == 500 and len(splits) == 500
    assert int(labels.y.sum()) == 54, "正类数与协议 v2 不符"
    assert labels.label_version.iloc[0] == "label_v2_record_count_20260923"
    return labels, splits


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in META and c not in BLOCKED]
    return cols


def evaluate(tag: str, y, p, sid) -> dict:
    m = all_metrics(y, p, sid)
    return {"tag": tag, "AUC": round(m["AUC"], 4), "AP": round(m["AP"], 4),
            "R100": round(m["Recall@100"], 4), "Brier": round(m["Brier"], 4),
            "LogLoss": round(m["LogLoss"], 4)}


def pair(y, sid, p_new, p_ref, label) -> dict:
    b = paired_bootstrap_auc(y, p_new, p_ref, sid)
    t = paired_bootstrap_topk(y, p_new, p_ref, sid)
    return {"vs": label, "delta_auc": round(b["delta_auc_point"], 4),
            "ci": [round(b["ci_low"], 4), round(b["ci_high"], 4)],
            "p_improve": round(b["p_improve"], 2),
            "dR100_med": round(t["delta_recall_point_median"], 3)}


def save_version(group: str, ver: str, rows: list[dict], pairs: list[dict],
                 oof: pd.DataFrame, extra: dict | None = None) -> str:
    """版本落盘三件套：指标 JSON + OOF CSV + 台账行。"""
    gdir = os.path.join(SPRINT, group)
    vdir = os.path.join(gdir, ver)
    os.makedirs(os.path.join(vdir, "results"), exist_ok=True)
    oof.to_csv(os.path.join(vdir, "results", "oof.csv"), index=False, lineterminator="\n")
    rec = {"group": group, "version": ver, "rows": rows, "pairs": pairs,
           "extra": extra or {}, "protocol": "label_v2_record_count_20260923"}
    with open(os.path.join(vdir, "results", "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=2)
    best = max(rows, key=lambda r: r["AUC"])
    line = (f"{group}/{ver},{best['tag']},{best['AUC']},{best['AP']},{best['R100']},"
            f"{best['Brier']},{len(rows)}\n")
    ledger = os.path.join(SPRINT, "_common", "ledger.csv")
    if not os.path.exists(ledger):
        with open(ledger, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("version,best_tag,AUC,AP,R100,Brier,n_rows\n")
    with open(ledger, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
    print(f"[saved] {vdir}")
    return vdir


# ────────────── 叶安式严格嵌套选型（FEAT-009 modeling.py 的忠实复刻） ──────────────

CANDIDATES = (
    {"name": "EBM-A", "kind": "ebm", "interactions": 0, "max_bins": 64, "min_samples_leaf": 10},
    {"name": "EBM-B", "kind": "ebm", "interactions": 5, "max_bins": 64, "min_samples_leaf": 10},
    {"name": "LightGBM-A", "kind": "lgbm", "num_leaves": 4, "min_data_in_leaf": 30},
    {"name": "LightGBM-B", "kind": "lgbm", "num_leaves": 8, "min_data_in_leaf": 40},
)
EBM_C = {"name": "EBM-C", "kind": "ebm", "interactions": 0, "max_bins": 32, "min_samples_leaf": 20}


def make_candidate(spec: dict, seed: int = 42):
    if spec["kind"] == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(
            interactions=spec["interactions"], max_bins=spec["max_bins"],
            min_samples_leaf=spec["min_samples_leaf"], random_state=seed, n_jobs=1)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        num_leaves=spec["num_leaves"], min_child_samples=spec["min_data_in_leaf"],
        learning_rate=0.05, n_estimators=200, reg_lambda=1.0, colsample_bytree=0.8,
        objective="binary", random_state=seed, n_jobs=1, verbosity=-1,
        deterministic=True, force_col_wise=True)


def fold_prune(x_tr: pd.DataFrame, rho: float = 0.95) -> list[str]:
    """训练折内 |Spearman ρ|>rho 成对去后者 + 近零方差剔除（叶安收尾通则的折内版）。"""
    keep: list[str] = []
    for c in x_tr.columns:
        s = x_tr[c]
        if s.std(ddof=0) < 1e-12:
            continue
        ok = True
        for k in keep:
            if abs(s.corr(x_tr[k])) > rho:
                ok = False
                break
        if ok:
            keep.append(c)
    return keep


def nested_select(x: pd.DataFrame, y: np.ndarray, fold: np.ndarray,
                  candidates=CANDIDATES, seed: int = 42, prune: bool = True):
    """外 5 冻结折 × 内 3 折（seed42）选型。返回 (oof, diagnostics)。"""
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    n = len(y)
    oof = np.full(n, np.nan)
    diags = []
    for k in sorted(np.unique(fold)):
        tr_idx, va_idx = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores: dict[str, list[float]] = {s["name"]: [] for s in candidates}
        for itr_rel, iva_rel in inner.split(np.zeros(len(tr_idx)), y[tr_idx]):
            itr, iva = tr_idx[itr_rel], tr_idx[iva_rel]
            x_itr, x_iva = x.iloc[itr], x.iloc[iva]
            cols = fold_prune(x_itr) if prune else list(x.columns)
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
            a = imp.fit_transform(x_itr[cols]).astype(float)
            b = imp.transform(x_iva[cols]).astype(float)
            for spec in candidates:
                m = make_candidate(spec, seed)
                m.fit(a, y[itr])
                scores[spec["name"]].append(float(roc_auc_score(y[iva], m.predict_proba(b)[:, 1])))
        means = {k2: float(np.mean(v)) for k2, v in scores.items()}
        selected = max(means, key=means.get)
        cols = fold_prune(x.iloc[tr_idx]) if prune else list(x.columns)
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        a = imp.fit_transform(x.iloc[tr_idx][cols]).astype(float)
        b = imp.transform(x.iloc[va_idx][cols]).astype(float)
        spec = next(s for s in candidates if s["name"] == selected)
        m = make_candidate(spec, seed)
        m.fit(a, y[tr_idx])
        oof[va_idx] = m.predict_proba(b)[:, 1]
        diags.append({"outer_fold": int(k), "selected": selected,
                      "inner_auc": {k2: round(v, 4) for k2, v in means.items()},
                      "n_cols": len(cols)})
        print(f"  fold{k}: {selected} inner={ {k2: round(v,3) for k2,v in means.items()} } cols={len(cols)}",
              flush=True)
    assert np.isfinite(oof).all()
    return oof, diags


def fixed_model_oof(x: pd.DataFrame, y: np.ndarray, fold: np.ndarray, spec: dict,
                    seed: int = 42, prune: bool = True):
    """固定配置的五折 OOF（不选型）。"""
    from sklearn.impute import SimpleImputer
    n = len(y)
    oof = np.full(n, np.nan)
    for k in sorted(np.unique(fold)):
        tr, va = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        cols = fold_prune(x.iloc[tr]) if prune else list(x.columns)
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        a = imp.fit_transform(x.iloc[tr][cols]).astype(float)
        b = imp.transform(x.iloc[va][cols]).astype(float)
        m = make_candidate(spec, seed)
        m.fit(a, y[tr])
        oof[va] = m.predict_proba(b)[:, 1]
    assert np.isfinite(oof).all()
    return oof


SMOOTH_CANDIDATES = tuple(
    {"name": f"EBM-b{b}-l{l}", "kind": "ebm", "interactions": 0, "max_bins": b, "min_samples_leaf": l}
    for b, l in ((32, 15), (32, 20), (48, 10), (48, 15), (64, 10), (64, 15), (64, 20))
)
