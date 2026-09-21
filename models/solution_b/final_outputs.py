# -*- coding: utf-8 -*-
"""最终输出：按模型契约 §3/§6 产出 model_score 与 forecast_result。

- model_score.csv（内部溯源版，受控本地）：gpsno, prob, rank_pct, calib_flag,
  group_id, sample_id, fold, model, feature_version, group_rule_version。
- forecast_result.csv（官方三列）：gpsno, accident, risk_prob。
  accident 政策：固定名额 K=ceil(0.2·N)=100（预登记，不依赖标签），
  同分按 sample_id 字典序（EVAL-001 §3.3）。
- 最终采用层级由 validate.py 的预登记判优规则裁决后回写 adopted_tier.json；
  本模块默认 B1，若裁决为其他层级可传参覆盖。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from common import OUT_DIR, VERSIONS, PREREG

TIER_COL = {"B1": "p_b1", "B2": "p_b2"}  # B0 层级只有排序分数，无概率输出；概率交付至少到 B1


def write_outputs(b12: pd.DataFrame, tier: str = "B1") -> dict:
    assert tier in TIER_COL, f"未知层级 {tier}"
    df = b12.copy()
    df["prob"] = df[TIER_COL[tier]].astype(float)

    # ---- 输出合同断言（模型契约 §6） ----
    assert len(df) == 500, f"行数 {len(df)} ≠ 500"
    assert df.gpsno.is_unique, "gpsno 重复"
    assert df.prob.between(0, 1).all(), "概率越界"
    assert df.prob.notna().all(), "概率缺失"
    assert set(df.y.unique()) <= {0, 1}

    # 固定名额 accident（同分按 sample_id 字典序）
    K = int(np.ceil(0.2 * len(df)))
    order = df.sort_values(["prob", "sample_id"], ascending=[False, True], kind="mergesort")
    flagged = set(order.head(K)["sample_id"])
    df["accident"] = df.sample_id.isin(flagged).astype(int)
    df["rank_pct"] = df.prob.rank(pct=True, method="average")  # 越大越危险
    df["calib_flag"] = df.group.map({"low": "group_calib", "high": "shrunk_calib",
                                     "insufficient": "global_calib"})
    df["model"] = f"B/{tier}/rf+ebm_ranklogit_{tier.lower()}_calib"
    df["feature_version"] = VERSIONS["feature_version"]
    df["group_rule_version"] = VERSIONS["group_rule_version"]

    # ---- forecast_result.csv（官方三列） ----
    fr = df[["gpsno", "accident", "prob"]].rename(columns={"prob": "risk_prob"})
    fr_path = os.path.join(OUT_DIR, "forecast_result.csv")
    fr.to_csv(fr_path, index=False, encoding="utf-8", float_format="%.6f")

    # ---- model_score.csv（内部溯源版） ----
    ms = df[["gpsno", "prob", "rank_pct", "calib_flag", "group_id" if "group_id" in df else "group",
             "sample_id", "fold", "model", "feature_version", "group_rule_version"]]
    ms = ms.rename(columns={"group": "group_id"})
    ms_path = os.path.join(OUT_DIR, "model_score.csv")
    ms.to_csv(ms_path, index=False, encoding="utf-8", float_format="%.6f")

    meta = {
        "adopted_tier": tier,
        "K_quota": K,
        "flagged_n": int(df.accident.sum()),
        "files": {"forecast_result.csv": fr_path, "model_score.csv": ms_path},
        "versions": VERSIONS, "prereg_quota_policy": "fixed K=ceil(0.2N), tie-break sample_id lex",
    }
    with open(os.path.join(OUT_DIR, "adopted_tier.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    b12 = pd.read_csv(os.path.join(OUT_DIR, "oof_predictions_B1B2.csv"))
    meta = write_outputs(b12, tier="B1")
    print(json.dumps({k: meta[k] for k in ("adopted_tier", "K_quota", "flagged_n")},
                     ensure_ascii=False))
    fr = pd.read_csv(os.path.join(OUT_DIR, "forecast_result.csv"))
    print(fr.head(3).to_string(), "\n行数:", len(fr), " accident=1:", fr.accident.sum(),
          " 概率域:", (fr.risk_prob.min(), fr.risk_prob.max()))
