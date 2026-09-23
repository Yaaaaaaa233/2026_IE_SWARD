# -*- coding: utf-8 -*-
"""方案B V5（当前登记最优）共用层：F1 数据装载（只读）、版本与路径。

V5 = F1_v1 全 118 特征 × 5 成员（ebm2026/rf7/ebm7/ebm_i0_7/ebm_i0_2026）
     × logit 均值融合 × 分群校准（n0 折内按 Brier 选）。
判据状态（ADR-0004）：对冻结基线（RF，models/task1_baselines.py）车辆级配对
bootstrap 区间下界 > 0 —— **过门**，当前王健祺线登记最优；数字见受控台账。
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION_B = os.path.join(os.path.dirname(HERE), "solution_b")

VERSIONS = {
    "feature_version": "F1_v1",
    "split_version": "split_v0.1_strat5",
    "group_rule_version": "F1_v1_group_v1(聚类按 cluster_risk_rank 定向)",
    "model_version": "B/V5/five-member-fusion-group-calib",
    "eval_protocol": "EVAL-001 v1.0 + ADR-0004",
}

META_COLS = ["sample_id", "gpsno", "as_of", "lookback_days", "y", "fold"]
META_EXTRAS = {"feature_version", "source_version", "split_version", "label_version",
               "label_window", "horizon_days", "label_status", "group_rule_version",
               "feature_version_f1", "feature_version_f0"}


def _load_local_config() -> dict:
    path = os.path.join(HERE, "config_local.json")
    if not os.path.exists(path):
        raise RuntimeError("缺少本机路径配置：复制 models/solution_b_v5/config_local.template.json "
                           "为 config_local.json 并填写 feature_root（特征工程根目录，含 artifacts_f1_v1）。")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def paths() -> dict:
    root = _load_local_config()["feature_root"]
    return {
        "model_input": f"{root}/artifacts_f1_v1/model_interface/model_input.csv",
        "cluster": f"{root}/artifacts_f1_v1/clustering/cluster_assignments.csv",
        "splits": f"{root}/数据筛选/契约v0.1全量/v4_assessed/splits.csv",
        "vehicles": f"{root}/数据筛选/契约v0.1全量/v4_assessed/target_vehicles.csv",
    }


OUT_DIR = os.environ.get("SOLUTION_B_V5_OUT", os.path.join(HERE, "outputs"))
os.makedirs(OUT_DIR, exist_ok=True)


def load_f1() -> pd.DataFrame:
    """装载 F1 输入 + 聚类分组（断言齐全，失败即抛错）。"""
    p = paths()
    mi = pd.read_csv(p["model_input"])
    cl = pd.read_csv(p["cluster"])
    sp = pd.read_csv(p["splits"])
    tv = pd.read_csv(p["vehicles"])
    assert mi.sample_id.is_unique and len(mi) == 500, "model_input 需 500 行唯一 sample_id"
    m = mi.merge(cl[["sample_id", "cluster", "cluster_risk_rank"]], on="sample_id", validate="1:1")
    fold_map = dict(zip(sp.gpsno, sp.fold))
    assert (m.fold.values == m.gpsno.map(fold_map).values).all(), "折分与冻结 splits 不一致"
    assert set(m.gpsno) == set(tv.gpsno), "车辆集合与 target_vehicles 不一致"
    assert m.y.isin([0, 1]).all(), "标签非 0/1"
    rank_max = m.cluster_risk_rank.max()
    m["group"] = m.apply(
        lambda r: "insufficient" if r.cluster == -1 else ("high" if r.cluster_risk_rank == rank_max else "low"),
        axis=1)
    return m


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_manifest(extra: dict | None = None) -> dict:
    p = paths()
    man = {"versions": VERSIONS,
           "inputs": {k: file_sha256(v) for k, v in p.items()},
           "created_at": pd.Timestamp.now().isoformat()}
    if extra:
        man.update(extra)
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    return man


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
