# -*- coding: utf-8 -*-
"""通用版本运行器：加载特征表 → 绑协议标签/折分 → 模型评估 → 版本落盘。

用法：
  python run_version.py --group g1_事件序列组 --ver g1v1 \
      --features g1_事件序列组/g1v1/features/g1v1_features.csv \
      --modes ebm_a ebm_c nested
输出：gX/gXvY/results/{oof.csv, metrics.json} + 台账行 + 三行记录模板。
对照锚：RF@F0v2（逐车 OOF 可配对）；叶安 F3 为引用值（点估计对照，注明未逐车对账）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_stack import (ANCHOR_RF_F0V2, ANCHOR_YE_F3, CANDIDATES, EBM_C, REPO, SPRINT,
                        evaluate, feature_columns, fixed_model_oof, load_protocol,
                        nested_select, pair, save_version)

RF_OOF = os.path.join(REPO, "模型选型与融合前期调研", "方案B实现", "outputs",
                      "protocol_v2", "results_p2", "baseline_rf_oof.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--ver", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--modes", nargs="+", default=["ebm_a"],
                    choices=["ebm_a", "ebm_c", "nested", "lgbm_b", "nested_smooth"])
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    labels, splits = load_protocol()
    lab = labels.assign(gpsno=labels.sample_id.str.split("_").str[0])
    df = pd.read_csv(args.features, dtype={"sample_id": str, "gpsno": str})
    df = df.merge(lab[["sample_id", "y"]], on="sample_id", validate="1:1")
    assert "gpsno" in df.columns, "特征表缺少 gpsno 列"
    df = df.merge(splits[["gpsno", "fold"]], on="gpsno", validate="1:1")
    cols = [c for c in feature_columns(df) if c not in ("gpsno",)]
    x = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    y, fold, sid = df.y.values.astype(int), df.fold.values.astype(int), df.sample_id.values
    print(f"{args.ver}: {x.shape[1]} 列特征, 正类 {y.sum()}", flush=True)

    rf = pd.read_csv(RF_OOF, dtype={"sample_id": str})
    m = df[["sample_id"]].merge(rf, on="sample_id", validate="1:1")
    p_rf = m.p.values

    spec_a = next(s for s in CANDIDATES if s["name"] == "EBM-A")
    spec_l = next(s for s in CANDIDATES if s["name"] == "LightGBM-B")
    rows, pairs, oofs = [], [], {}
    if "ebm_a" in args.modes:
        p = fixed_model_oof(x, y, fold, spec_a)
        oofs["ebm_a"] = p
        rows.append(evaluate(f"{args.ver}-EBM-A", y, p, sid))
        pairs.append({"tag": f"{args.ver}-EBM-A", **pair(y, sid, p, p_rf, "RF锚")})
        print(f"  EBM-A: AUC={rows[-1]['AUC']:.4f} (Δ叶安F3锚 {rows[-1]['AUC'] - ANCHOR_YE_F3:+.4f})",
              flush=True)
    if "ebm_c" in args.modes:
        p = fixed_model_oof(x, y, fold, EBM_C)
        oofs["ebm_c"] = p
        rows.append(evaluate(f"{args.ver}-EBM-C", y, p, sid))
        pairs.append({"tag": f"{args.ver}-EBM-C", **pair(y, sid, p, p_rf, "RF锚")})
        print(f"  EBM-C: AUC={rows[-1]['AUC']:.4f} (Δ叶安F3锚 {rows[-1]['AUC'] - ANCHOR_YE_F3:+.4f})",
              flush=True)
    if "lgbm_b" in args.modes:
        p = fixed_model_oof(x, y, fold, spec_l)
        oofs["lgbm_b"] = p
        rows.append(evaluate(f"{args.ver}-LGBM-B", y, p, sid))
    if "nested_smooth" in args.modes:
        from eval_stack import SMOOTH_CANDIDATES
        p, diags = nested_select(x, y, fold, candidates=SMOOTH_CANDIDATES)
        oofs["nested_smooth"] = p
        rows.append(evaluate(f"{args.ver}-平滑嵌套", y, p, sid))
        pairs.append({"tag": f"{args.ver}-平滑嵌套", **pair(y, sid, p, p_rf, "RF锚")})
        print(f"  平滑嵌套: AUC={rows[-1]['AUC']:.4f} (Δ叶安F3锚 {rows[-1]['AUC'] - ANCHOR_YE_F3:+.4f})",
              flush=True)
    if "nested" in args.modes:
        p, diags = nested_select(x, y, fold)
        oofs["nested"] = p
        rows.append(evaluate(f"{args.ver}-嵌套", y, p, sid))
        pairs.append({"tag": f"{args.ver}-嵌套", **pair(y, sid, p, p_rf, "RF锚")})
        print(f"  嵌套: AUC={rows[-1]['AUC']:.4f} (Δ叶安F3锚 {rows[-1]['AUC'] - ANCHOR_YE_F3:+.4f})",
              flush=True)

    oof_df = pd.DataFrame({"sample_id": sid, "y": y, "fold": fold,
                           **{f"p_{k}": v for k, v in oofs.items()}})
    save_version(args.group, args.ver, rows, pairs, oof_df,
                 {"note": args.note, "n_features": int(x.shape[1]),
                  "reference_ye_f3": ANCHOR_YE_F3, "rf_anchor": ANCHOR_RF_F0V2})

    # 三行记录模板
    vdir = os.path.join(SPRINT, args.group, args.ver)
    with open(os.path.join(vdir, "results", "三行记录.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {args.ver} 三行记录\n\n- 设计：{args.note or '（待填）'}\n"
                 f"- 效果：{json.dumps({r['tag']: r['AUC'] for r in rows}, ensure_ascii=False)}\n"
                 f"- 问题：（每轮实验后填写：漏报/误报画像、压 AUC 的疑似因素、下一版调整）\n")
    print(f"[done] {vdir}")


if __name__ == "__main__":
    main()
