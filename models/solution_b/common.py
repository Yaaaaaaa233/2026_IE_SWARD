# -*- coding: utf-8 -*-
"""方案B 共用层：数据装载（只读受控数据源）、冻结折分、防泄漏配置与版本清单。

边界声明：
- 特征工程目录全程只读；本目录产物只写入 models/solution_b/outputs 与 report（git 忽略）。
- 折分使用契约 splits 表（split_v0.1_strat5），绝不重新划分。
- 一切学习型处理（插补/标准化/校准/残差模型/λ）只在训练折内拟合。
- 机器路径不写死：复制 config_local.template.json 为 config_local.json 后填写本机路径
  （config_local.json 已被 .gitignore 排除，不入公开仓库）。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_local_config() -> dict:
    path = os.path.join(HERE, "config_local.json")
    if not os.path.exists(path):
        raise RuntimeError(
            "缺少本机路径配置：请复制 models/solution_b/config_local.template.json "
            "为 config_local.json 并填写 feature_root（特征工程目录）。该文件不入库。")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def paths() -> dict:
    cfg = _load_local_config()
    root = cfg["feature_root"]
    return {
        "model_input": f"{root}/artifacts/model_interface/model_input.csv",
        "schema": f"{root}/artifacts/model_interface/feature_schema.json",
        "cluster": f"{root}/artifacts/clustering/cluster_assignments.csv",
        "splits": f"{root}/数据筛选/契约v0.1全量/v4_assessed/splits.csv",
        "target_vehicles": f"{root}/数据筛选/契约v0.1全量/v4_assessed/target_vehicles.csv",
    }


OUT_DIR = os.path.join(HERE, "outputs")
REPORT_DIR = os.path.join(HERE, "report")
for _d in (OUT_DIR, REPORT_DIR):
    os.makedirs(_d, exist_ok=True)

VERSIONS = {
    "source_version": "2.0",
    "feature_version": "F0_v1(特征线 2026-09-20 交付，正式登记待 F0 冻结)",
    "split_version": "split_v0.1_strat5",
    "group_rule_version": "cluster_v0(k=2+insufficient,特征线 2026-09-20 交付)",
    "eval_protocol": "EVAL-001 v0.1(Proposed)",
}

META_COLS = ["sample_id", "gpsno", "as_of", "lookback_days", "y", "fold"]

# 预登记决策参数（EVAL-001 §6.2：查看结果前登记）
PREREG = {
    "K_QUOTA": 100,               # Recall@K 固定名额：500×20%
    "Q_LEVELS": [0.10, 0.20],
    "BOOT_N": 2000,               # 配对 bootstrap 次数
    "BOOT_SEED": 42,
    "CI": 0.95,
    "JITTER_SEEDS": [42, 7, 2026],  # 种子抖动估计
    "DELTA_AUC_MIN": 0.005,       # 采纳更高层所需最小 AUC 增益
    "R100_TOLERANCE": 0.02,       # 采纳时允许的 Recall@100 最大退化
    "BRIER_TOLERANCE": 0.005,     # 采纳时允许的 Brier 最大退化
    "EPS": 1e-15,                 # LogLoss 截断（EVAL-001 §3.4）
}


# ---------------------------------------------------------------- 数据装载
@dataclass
class Bundle:
    """一次装载、多处消费；携带全部元数据与群组信息。"""
    X: pd.DataFrame            # 仅特征（energy_type 已独热）
    y: np.ndarray
    fold: np.ndarray
    sample_id: np.ndarray
    gpsno: np.ndarray
    group: np.ndarray          # 'low' / 'high' / 'insufficient'
    raw: pd.DataFrame          # 原始 model_input（含元数据与原始特征列）
    feature_cols: list = field(default_factory=list)


def load_bundle() -> Bundle:
    p = paths()
    mi = pd.read_csv(p["model_input"])
    cl = pd.read_csv(p["cluster"])
    sp = pd.read_csv(p["splits"])
    tv = pd.read_csv(p["target_vehicles"])

    # ---- 完整性断言（失败即抛错，不静默） ----
    assert mi.sample_id.is_unique, "model_input sample_id 不唯一"
    assert set(mi.sample_id) == set(cl.sample_id), "model_input 与聚类表样本集不一致"
    m = mi.merge(cl[["sample_id", "cluster"]], on="sample_id", validate="1:1")
    fold_map = dict(zip(sp.gpsno, sp.fold))
    assert m.gpsno.isin(fold_map).all(), "存在折分表缺失的车辆"
    assert (m.fold.values == m.gpsno.map(fold_map).values).all(), "model_input.fold 与冻结 splits 不一致"
    assert set(m.gpsno) == set(tv.gpsno), "样本车辆集与 target_vehicles 不一致"
    assert m.y.isin([0, 1]).all(), "标签非 0/1"

    feats = [c for c in mi.columns if c not in META_COLS]
    X = m[feats].copy()
    if "energy_type" in X.columns:
        X["energy_type"] = X["energy_type"].fillna(X.energy_type.mode()[0])
        X = pd.get_dummies(X, columns=["energy_type"], drop_first=True)
    X = X.astype(float)

    grp = m.cluster.map({-1: "insufficient", 0: "low", 1: "high"})
    assert grp.notna().all(), "存在未知簇号"
    return Bundle(X=X, y=m.y.values.astype(int), fold=m.fold.values.astype(int),
                  sample_id=m.sample_id.values, gpsno=m.gpsno.values,
                  group=grp.values, raw=m, feature_cols=list(X.columns))


# ---------------------------------------------------------------- 工具
def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_manifest(name: str, extra: dict | None = None) -> dict:
    p = paths()
    manifest = {
        "name": name,
        "versions": VERSIONS,
        "prereg": PREREG,
        "inputs": {
            "model_input": file_sha256(p["model_input"]),
            "cluster_assignments": file_sha256(p["cluster"]),
            "splits": file_sha256(p["splits"]),
        },
        "created_at": pd.Timestamp.now().isoformat(),
    }
    if extra:
        manifest.update(extra)
    with open(os.path.join(OUT_DIR, f"manifest_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
