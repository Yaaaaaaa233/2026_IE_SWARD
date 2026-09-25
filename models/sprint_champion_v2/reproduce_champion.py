# -*- coding: utf-8 -*-
"""重建 sprint-g4v5-B2 冠军：三成员 OOF 加权融合 + model_score.csv。

数据路径为受控本地配置（E 盘与任务1工作冲刺/），公开仓库不含真实数据；
运行前确认协议 v2 标签/折分与三成员 OOF 已按 src/ 脚本生成。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

W = (0.65, 0.245, 0.105)  # g1融合臂 / g2面板 / g3 IMU
MEMBERS = {
    "g1": ("任务1工作冲刺/g1_事件序列组/g1v3/results/oof.csv", "p_C融合ACR"),
    "g2": ("任务1工作冲刺/g2_轨迹动态组/g2v1/results/oof.csv", "p_ebm_c"),
    "g3": ("任务1工作冲刺/g3_联合上下文组/g3v2/results/oof.csv", "p_ebm_c"),
}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main(root: str = ".") -> None:
    labels = pd.read_csv(os.path.join(root, "任务1工作冲刺/_common/protocol/labels.csv"),
                         dtype={"sample_id": str})
    lab = labels.assign(gpsno=labels.sample_id.str.split("_").str[0])
    z = None
    for (rel, col), w in zip(MEMBERS.values(), W):
        d = pd.read_csv(os.path.join(root, rel), dtype={"sample_id": str})[["sample_id", col]]
        d.columns = ["sample_id", "p"]
        m = lab[["sample_id"]].merge(d, on="sample_id", validate="1:1")
        z = w * logit(m.p.values) if z is None else z + w * logit(m.p.values)
    prob = 1 / (1 + np.exp(-z))
    ms = pd.DataFrame({"gpsno": lab.gpsno, "prob": prob.round(6),
                       "rank_pct": pd.Series(prob).rank(pct=True, method="average").round(4),
                       "calib_flag": "raw_oof_fusion", "model_version": "sprint-g4v5-B2"}
                      ).sort_values("gpsno").reset_index(drop=True)
    out = os.path.join(root, "models/sprint_champion_v2/model_score.csv")
    ms.to_csv(out, index=False, lineterminator="\n")
    print("rebuilt", out, ms.shape)


if __name__ == "__main__":
    main()
