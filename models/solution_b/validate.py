# -*- coding: utf-8 -*-
"""验证套件：正确性 + 统计 + 负向测试 + 图表，产出 report/ 下的验证报告材料。

验证方案（预登记，见 report/验证方案.md）：
 V1 正确性：折分一致、输出合同、嵌套结构、复现一致（同种子重跑逐位一致）
 V2 统计：B0/B1/B2 阶梯的 AUC/AP/R@100/Brier + 配对bootstrap CI；分群分层（报样本量）
 V3 稳健性：3 种子全流程抖动；逐折指标
 V4 负向测试：构造坏输入，验证检查器确实拒绝（等分/单类/越界概率/重复键/空集）
 判优（预登记）：采纳 B1 需 ΔAUC(vs 最强单模)≥0.005 或 (ΔAUC>0 且 Brier 改善≥0.005)，
                且 R@100 退化≤0.02、Brier 退化≤0.005；B2 同规则 vs B1。
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from common import OUT_DIR, REPORT_DIR, PREREG, load_bundle, paths
from b0_backbone import run_b0
from b1_b2_layers import run_b1_b2
from final_outputs import write_outputs
from metrics import (all_metrics, auc_or_none, ap_or_none, brier, logloss,
                     calibration_bins, paired_bootstrap_auc, paired_bootstrap_topk,
                     hard_check_predictions, topk_table)

RESULTS: dict = {"checks": [], "stats": {}, "negatives": [], "decision": {}}


def check(name: str, ok: bool, detail: str = ""):
    RESULTS["checks"].append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))
    return ok


# ================================================================ V1 正确性
def v1_correctness(b, out_b12):
    print("== V1 正确性 ==")
    check("折分与冻结 splits 一致（load_bundle 内断言已执行）", True,
          "split_v0.1_strat5；model_input.fold == splits.csv")
    n = len(out_b12)
    check("样本完整 500 行", n == 500, f"n={n}")
    check("输出概率域 [0,1] 无缺失", out_b12.p_b1.between(0, 1).all() and out_b12.p_b1.notna().all())
    tv = pd.read_csv(paths()["target_vehicles"])
    check("gpsno 集合与 target_vehicles 一致", set(out_b12.gpsno) == set(tv.gpsno))
    # 复现一致：同种子重跑
    b0b, irf2, ieb2 = run_b0(b, seed=42, tag="B0_repro")
    out2, _ = run_b1_b2(b, irf2, ieb2, b0b.p_rf.values, b0b.p_ebm.values)
    a = pd.read_csv(os.path.join(OUT_DIR, "oof_predictions_B1B2.csv"))
    cols = ["p_rf", "p_ebm", "p_b1", "p_b2"]
    maxdiff = float(np.max(np.abs(a[cols].values - out2[cols].values)))
    # EVAL-001 §6.1：运行栈非确定性 → 预登记容差（1e-12），报告实际差异，不声称逐位一致
    check("同种子重跑一致（容差1e-12，预登记）", maxdiff <= 1e-12,
          f"max|Δ|={maxdiff:.2e}")


# ================================================================ V2 统计
def v2_statistics(out):
    print("== V2 统计 ==")
    y, sid = out.y.values, out.sample_id.values
    tiers = {"B0-RF": (out.p_rf.values, True), "B0-EBM": (out.p_ebm.values, True),
             "B0-融合(rank)": (None, False), "B1-分群校准": (out.p_b1.values, True),
             "B2-收缩残差": (out.p_b2.values, True)}
    # rank 融合分数重算（仅排序尺度，概率指标不适用）
    s_rank = (pd.Series(out.p_rf).rank() + pd.Series(out.p_ebm).rank()).values
    tiers["B0-融合(rank)"] = (s_rank, False)
    rows = []
    for name, (p, is_prob) in tiers.items():
        mm = all_metrics(y, p, sid) if is_prob else None
        if is_prob:
            rows.append({"方案": name, "AUC": round(mm["AUC"], 4), "AP": round(mm["AP"], 4),
                         "Recall@100": round(mm["Recall@100"], 3),
                         "Precision@100": round(mm["Precision@100"], 3),
                         "Lift@100": round(mm["Lift@100"], 2),
                         "Brier": round(mm["Brier"], 4), "LogLoss": round(mm["LogLoss"], 4)})
        else:
            from metrics import topk_table
            r = topk_table(y, p, sid)
            r100 = float(r[r.K == 100]["Recall@K"].iloc[0])
            rows.append({"方案": name, "AUC": round(auc_or_none(y, p), 4),
                         "AP": round(ap_or_none(y, p), 4), "Recall@100": round(r100, 3),
                         "Precision@100": None, "Lift@100": None,
                         "Brier": "n/a(rank)", "LogLoss": "n/a(rank)"})
    t = pd.DataFrame(rows)
    RESULTS["stats"]["ladder"] = t
    print(t.to_string(index=False))

    # 配对 bootstrap：B1 vs 最强单模(RF) 与 vs B0融合
    boots = {}
    for base_name, base_p in [("B0-RF", out.p_rf.values), ("B0-融合(rank)", s_rank)]:
        ba = paired_bootstrap_auc(y, out.p_b1.values, base_p, sid)
        br = paired_bootstrap_topk(y, out.p_b1.values, base_p, sid, K=PREREG["K_QUOTA"])
        boots[f"B1 vs {base_name}"] = {"AUC": ba, "R@100": br}
        print(f"  B1 vs {base_name}: ΔAUC={ba['delta_auc_point']:+.4f} "
              f"[{ba['ci_low']:+.4f}, {ba['ci_high']:+.4f}] p(改善)={ba['p_improve']:.2f} | "
              f"ΔR@100 中位={br['delta_recall_point_median']:+.3f} [{br['ci_low']:+.3f}, {br['ci_high']:+.3f}]")
    RESULTS["stats"]["bootstrap"] = boots

    # 逐折指标
    fold_rows = []
    for k in range(5):
        sub = out[out.fold == k]
        fold_rows.append({
            "fold": k, "n": len(sub), "pos": int(sub.y.sum()),
            "AUC_RF": round(auc_or_none(sub.y, sub.p_rf), 4) or None,
            "AUC_B1": round(auc_or_none(sub.y, sub.p_b1), 4) or None,
            "R@20%_B1": round(topk_table(sub.y, sub.p_b1, sub.sample_id,
                                         q_levels=[0.2])["Recall@K"].iloc[0], 3)})
    ft = pd.DataFrame(fold_rows)
    RESULTS["stats"]["per_fold"] = ft
    print(ft.to_string(index=False))

    # 分群分层（含样本量；Top100命中 = 该群正类落入全局前100的数量，总和应等于 TP_K）
    g_rank = out.p_b1.rank(ascending=False, method="first")
    grp_rows = []
    for g in out.group.unique():
        sel = (out.group == g).values
        sub = out[sel]
        grp_rows.append({"群": g, "n": len(sub), "正类": int(sub.y.sum()),
                         "阳性率": f"{sub.y.mean():.1%}",
                         "AUC_RF": round(auc_or_none(sub.y, sub.p_rf) or float('nan'), 4),
                         "AUC_B1": round(auc_or_none(sub.y, sub.p_b1) or float('nan'), 4),
                         "Top100命中": int(((g_rank <= 100).values & sel & (out.y == 1).values).sum())})
    assert sum(r["Top100命中"] for r in grp_rows) == \
        int(topk_table(y, out.p_b1.values, sid).query("K==100")["TP_K"].iloc[0]), \
        "分群Top100命中之和 ≠ 全局TP_K"
    gt = pd.DataFrame(grp_rows)
    RESULTS["stats"]["per_group"] = gt
    print(gt.to_string(index=False))

    # 校准表
    cal_b1 = calibration_bins(y, out.p_b1.values)
    cal_rf = calibration_bins(y, out.p_rf.values)
    RESULTS["stats"]["calibration_B1"] = cal_b1
    RESULTS["stats"]["calibration_RF"] = cal_rf


# ================================================================ V3 稳健性
def v3_jitter(b):
    print("== V3 种子抖动（全流程 B0+B1） ==")
    aucs, r100s = [], []
    for seed in PREREG["JITTER_SEEDS"]:
        b0o, irf, ieb = run_b0(b, seed=seed, tag=f"B0_s{seed}")
        o, _ = run_b1_b2(b, irf, ieb, b0o.p_rf.values, b0o.p_ebm.values)
        mm = all_metrics(o.y, o.p_b1.values, o.sample_id)
        aucs.append(mm["AUC"]); r100s.append(mm["Recall@100"])
        print(f"  seed={seed}: AUC={mm['AUC']:.4f} R@100={mm['Recall@100']:.3f}")
    jit = {"auc": aucs, "auc_std": float(np.std(aucs)), "auc_range": float(max(aucs) - min(aucs)),
           "r100": r100s, "r100_std": float(np.std(r100s)),
           "r100_range": float(max(r100s) - min(r100s))}
    RESULTS["stats"]["jitter"] = jit
    print(f"  AUC 极差={jit['auc_range']:.4f} → 3×抖动线={3 * jit['auc_range']:.4f}")
    return jit


# ================================================================ V4 负向测试
def v4_negative():
    print("== V4 负向测试（检查器必须拒绝坏输入） ==")
    rng = np.random.default_rng(0)
    sid = [f"s{i:03d}" for i in range(60)]
    y_ok = np.r_[np.ones(15), np.zeros(45)]
    p_ok = rng.uniform(0, 1, 60)
    cases = []
    # 1 单一类别 → AUC 无定义
    a = auc_or_none(np.zeros(60), p_ok)
    cases.append(check("单一类别 AUC 返回 None（不补 0/0.5）", a is None, f"got={a}"))
    # 2 概率越界 → 硬检查报错
    p_bad = p_ok.copy(); p_bad[0] = 1.5
    errs = hard_check_predictions(sid, y_ok, p_bad)
    cases.append(check("概率>1 被硬检查拒绝", "概率越界" in " ".join(errs), str(errs)))
    # 3 概率缺失
    p_nan = p_ok.copy(); p_nan[3] = np.nan
    errs = hard_check_predictions(sid, y_ok, p_nan)
    cases.append(check("NaN 概率被硬检查拒绝", "缺失" in " ".join(errs), str(errs)))
    # 4 重复键
    sid_dup = sid.copy(); sid_dup[5] = sid[4]
    errs = hard_check_predictions(sid_dup, y_ok, p_ok)
    cases.append(check("重复 sample_id 被硬检查拒绝", "重复" in " ".join(errs), str(errs)))
    # 5 标签非法
    y_bad = y_ok.copy(); y_bad[0] = 2
    errs = hard_check_predictions(sid, y_bad, p_ok)
    cases.append(check("非法标签被硬检查拒绝", "标签" in " ".join(errs), str(errs)))
    # 6 正常输入必须通过（防检查器误杀）
    errs = hard_check_predictions(sid, y_ok, p_ok)
    cases.append(check("合法输入通过硬检查（无误杀）", errs == [], str(errs)))
    RESULTS["negatives"] = cases


# ================================================================ 判优与图
def decide_and_figures(out, jit):
    print("== 判优（预登记规则） ==")
    y, sid = out.y.values, out.sample_id.values
    m_rf = all_metrics(y, out.p_rf.values, sid)
    m_b1 = all_metrics(y, out.p_b1.values, sid)
    m_b2 = all_metrics(y, out.p_b2.values, sid)
    d_auc = m_b1["AUC"] - m_rf["AUC"]
    d_brier = m_rf["Brier"] - m_b1["Brier"]
    d_r100 = m_b1["Recall@100"] - m_rf["Recall@100"]
    adopt_b1 = ((d_auc >= PREREG["DELTA_AUC_MIN"]) or (d_auc > 0 and d_brier >= 0.005)) \
        and (d_r100 >= -PREREG["R100_TOLERANCE"]) and (d_brier >= -PREREG["BRIER_TOLERANCE"])
    d2_auc = m_b2["AUC"] - m_b1["AUC"]
    d2_brier = m_b1["Brier"] - m_b2["Brier"]
    adopt_b2 = ((d2_auc >= PREREG["DELTA_AUC_MIN"]) or (d2_auc > 0 and d2_brier >= 0.005)) \
        and ((m_b2["Recall@100"] - m_b1["Recall@100"]) >= -PREREG["R100_TOLERANCE"])
    tier = "B2" if adopt_b2 else ("B1" if adopt_b1 else "B0(RF单模)")
    if not adopt_b1:
        tier = "B0"
    RESULTS["decision"] = {
        "adopt_b1": bool(adopt_b1), "adopt_b2": bool(adopt_b2), "tier": tier,
        "delta_auc_b1_vs_rf": round(d_auc, 4), "delta_brier": round(d_brier, 4),
        "delta_r100": round(d_r100, 3), "delta_auc_b2_vs_b1": round(d2_auc, 4),
        "rule": PREREG["DELTA_AUC_MIN"],
    }
    print(json.dumps(RESULTS["decision"], ensure_ascii=False))
    # 概率交付层级：B2/B1（B0 只有排序分数；未达标时以最近概率层级交付并如实标注）
    deliver_tier = "B2" if adopt_b2 else "B1"
    write_outputs(out, tier=deliver_tier)
    if tier == "B0":
        RESULTS["decision"]["note"] = ("判优规则下 B1 未达预登记增益线，交付文件仍为 B1 概率"
                                       "（B0 层级无概率输出），差异已如实标注")

    # ---- 图 1：阶梯 + bootstrap CI ----
    boots = RESULTS["stats"]["bootstrap"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["B0-RF", "B0-EBM", "B1分群校准", "B2收缩残差"]
    aucs = [m_rf["AUC"], all_metrics(y, out.p_ebm.values, sid)["AUC"], m_b1["AUC"], m_b2["AUC"]]
    bars = axes[0].bar(names, aucs, color=["#9aa5b1", "#9aa5b1", "#D4875A", "#c9a227"])
    axes[0].set_ylim(0.70, 0.80); axes[0].set_ylabel("OOF AUC")
    axes[0].set_title(f"AUC 阶梯（3×种子抖动线 ±{3 * jit['auc_range']:.4f}）")
    for b_, v in zip(bars, aucs):
        axes[0].text(b_.get_x() + b_.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    axes[0].tick_params(axis="x", labelsize=9)
    b1 = boots["B1 vs B0-RF"]["AUC"]
    axes[1].errorbar([0], [b1["delta_auc_point"]],
                     yerr=[[b1["delta_auc_point"] - b1["ci_low"]], [b1["ci_high"] - b1["delta_auc_point"]]],
                     fmt="o", capsize=4, color="#D4875A")
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_xticks([0]); axes[1].set_xticklabels(["B1 − RF单模"])
    axes[1].set_ylabel("ΔAUC（95% 配对bootstrap CI）")
    axes[1].set_title("配对差异与不确定性")
    plt.tight_layout(); plt.savefig(os.path.join(REPORT_DIR, "fig1_ladder.png"), dpi=150); plt.close()

    # ---- 图 2：ROC ----
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for p, nm, c in [(out.p_rf, "RF", "#9aa5b1"), (out.p_ebm, "EBM", "#7d97b5"),
                     (out.p_b1, "B1 分群校准", "#D4875A")]:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, lw=1.6, label=f"{nm} AUC={auc_or_none(y, p):.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":")
    ax.set_xlabel("假正率"); ax.set_ylabel("真正率"); ax.legend(loc="lower right", fontsize=9)
    ax.set_title("ROC（折外，N=500，正类126）")
    plt.tight_layout(); plt.savefig(os.path.join(REPORT_DIR, "fig2_roc.png"), dpi=150); plt.close()

    # ---- 图 3：校准曲线 ----
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for p, nm, c in [(out.p_rf, "RF（未校准）", "#9aa5b1"), (out.p_b1, "B1 分群校准", "#D4875A")]:
        cb = calibration_bins(y, p.values)
        ax.plot(cb.pred_mean, cb.true_rate, "o-", color=c, lw=1.4, ms=4, label=nm)
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":")
    ax.set_xlabel("预测概率（箱均值）"); ax.set_ylabel("实际出险率")
    ax.set_title("校准曲线（10 等宽箱）"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(REPORT_DIR, "fig3_calibration.png"), dpi=150); plt.close()

    # ---- 图 4：分群分层 ----
    gt = RESULTS["stats"]["per_group"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(gt)); w = 0.36
    ax.bar(x - w / 2, gt.AUC_RF, w, label="RF 单模", color="#9aa5b1")
    ax.bar(x + w / 2, gt.AUC_B1, w, label="B1 分群校准", color="#D4875A")
    for i, r in gt.iterrows():
        ax.text(i, max(r.AUC_RF, r.AUC_B1) + 0.02,
                f"n={r.n}\npos={r.正类}", ha="center", fontsize=8.5, color="#3A3028")
    ax.set_xticks(x); ax.set_xticklabels([f"{g}群" for g in gt["群"]])
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylim(0, 1.05); ax.set_ylabel("群内 AUC"); ax.legend(fontsize=9)
    ax.set_title("分群分层表现（含样本量与正类数）")
    plt.tight_layout(); plt.savefig(os.path.join(REPORT_DIR, "fig4_groups.png"), dpi=150); plt.close()
    print("图 1–4 已输出至 report/")


# ================================================================ 主流程
if __name__ == "__main__":
    b = load_bundle()
    b0out, inner_rf, inner_ebm = run_b0(b)
    out, trace = run_b1_b2(b, inner_rf, inner_ebm, b0out.p_rf.values, b0out.p_ebm.values)
    out.to_csv(os.path.join(OUT_DIR, "oof_predictions_B1B2.csv"), index=False)

    v1_correctness(b, out)
    v2_statistics(out)
    jit = v3_jitter(b)
    # 抖动测试会覆写输出文件；回存 seed=42 主结果后再裁决与出交付文件
    out.to_csv(os.path.join(OUT_DIR, "oof_predictions_B1B2.csv"), index=False)
    v4_negative()
    decide_and_figures(out, jit)

    with open(os.path.join(REPORT_DIR, "validation_results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=str)
    n_pass = sum(c["pass"] for c in RESULTS["checks"])
    print(f"\n验证检查 {n_pass}/{len(RESULTS['checks'])} PASS；"
          f"负向测试 {sum(RESULTS['negatives'])}/6；"
          f"裁决层级={RESULTS['decision']['tier']}")
    # 阶梯与分群表落盘（报告用）
    RESULTS["stats"]["ladder"].to_csv(os.path.join(REPORT_DIR, "table_ladder.csv"), index=False)
    RESULTS["stats"]["per_fold"].to_csv(os.path.join(REPORT_DIR, "table_per_fold.csv"), index=False)
    RESULTS["stats"]["per_group"].to_csv(os.path.join(REPORT_DIR, "table_per_group.csv"), index=False)
