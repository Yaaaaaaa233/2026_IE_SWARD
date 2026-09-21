# -*- coding: utf-8 -*-
"""误差归因分析：B1 相对单模 RF 改善有限的原因诊断（受控本地）。

诊断框架（5 个假设逐一检验）：
 H1 特征天花板：漏报正类是否为"特征上安静的普通车"（行为窗口内无可区分信号）
 H2 暴露混杂误报：前100 误报是否集中在高暴露车（跑得多事件多但没出事）
 H3 高风险群内排序饱和：全局 AUC 损失有多少来自群内(尤其 high)不可分
 H4 B1 vs RF 的实际改动：校准改了哪些车的相对位置，改动方向对错各多少
 H5 不可约部分的量级：群内排序若完美，AUC 上限是多少（oracle 分解）
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import OUT_DIR as OUT, paths

_mi = paths()["model_input"]
CL = paths()["cluster"]
EVENTS = paths()["splits"].replace("splits.csv", "events_clean.csv")
mi = pd.read_csv(_mi)
cl = pd.read_csv(CL)
oof = pd.read_csv(f"{OUT}/oof_predictions_B1B2.csv")
df = oof.merge(mi.drop(columns=["y", "fold", "gpsno"]), on="sample_id").merge(
    cl[["sample_id", "cluster"]], on="sample_id")
df["rank_b1"] = df.p_b1.rank(ascending=False, method="first").astype(int)
df["in_top100"] = df.rank_b1 <= 100
df["group"] = df.cluster.map({-1: "insufficient", 0: "low", 1: "high"})

TP = df[(df.y == 1) & df.in_top100]
FN = df[(df.y == 1) & ~df.in_top100]
FP = df[(df.y == 0) & df.in_top100]
TN = df[(df.y == 0) & ~df.in_top100]
print(f"[总览] TP={len(TP)} FN={len(FN)} FP={len(FP)} TN={len(TN)}  (名单100, 正类126)")

print("\n[H0] 漏报正类的排名分布（离名单有多远）")
bins = [(101, 150), (151, 200), (201, 300), (301, 500)]
for lo, hi in bins:
    n = ((FN.rank_b1 >= lo) & (FN.rank_b1 <= hi)).sum()
    print(f"   排名 {lo}-{hi}: {n} 台")
print(f"   漏报概率分布: 中位={FN.p_b1.median():.3f}  进入名单需≈{df[df.in_top100].p_b1.min():.3f}")

print("\n[H1] 特征天花板：漏报正类 vs 命中正类 vs 真负类 的画像（中位数）")
prof = pd.DataFrame({
    "漏报正类FN": FN[["evt_count", "evt_lane_count", "traj_km_20d", "monthly_avg_hours",
                       "night_hours_ratio", "accident_sensor_risk_index", "coverage_days"]].median(),
    "命中正类TP": TP[["evt_count", "evt_lane_count", "traj_km_20d", "monthly_avg_hours",
                       "night_hours_ratio", "accident_sensor_risk_index", "coverage_days"]].median(),
    "真负类TN": TN[["evt_count", "evt_lane_count", "traj_km_20d", "monthly_avg_hours",
                     "night_hours_ratio", "accident_sensor_risk_index", "coverage_days"]].median(),
})
print(prof.round(2).to_string())
# 漏报正类的特征是否与真负类无差别（若有差别，理论上可分）
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
X_fn = df[df.y == 0].append(FN) if False else None
mask = (df.y == 0) | ((df.y == 1) & ~df.in_top100)  # TN vs FN 二分类
sub = df[mask].copy()
feats = ["evt_count", "evt_lane_count", "traj_km_20d", "monthly_avg_hours",
         "night_hours_ratio", "accident_sensor_risk_index"]
sub["is_fn"] = ((sub.y == 1) & ~sub.in_top100).astype(int)
Xs = sub[feats].fillna(sub[feats].median())
auc_fn = cross_val_score(LogisticRegression(max_iter=2000), Xs, sub.is_fn, cv=5, scoring="roc_auc")
print(f"   [TN vs FN 可分性] 5折AUC={auc_fn.mean():.3f}±{auc_fn.std():.3f}（≈0.5=漏报者在特征上与安全车无差别）")

print("\n[H2] 暴露混杂误报：前100 误报(FP) vs 命中(TP) 的暴露画像")
for c in ["traj_km_20d", "monthly_avg_hours", "evt_count", "evt_lane_count"]:
    q_fp = FP[c].rank(pct=True).median()
    q_all = df[c].rank(pct=True).median()
    print(f"   {c:>16}: FP 全队列分位数中位={q_fp:.2f}（全体中位=0.50）  FP中位={FP[c].median():.1f} vs TP中位={TP[c].median():.1f}")

print("\n[H3] 群内/跨群排序质量分解（AUC 损失在哪）")
y, p, g = df.y.values, df.p_b1.values, df.group.values
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
total = correct = 0
by_pair = {}
for i in pos:
    for j in neg:
        gp = "群内:" + g[i] if g[i] == g[j] else "跨群"
        if gp not in by_pair:
            by_pair[gp] = [0, 0]
        total += 1
        by_pair[gp][1] += 1
        c = (p[i] > p[j]) + 0.5 * (p[i] == p[j])
        correct += c
        by_pair[gp][0] += c
print(f"   全局一致率={correct/total:.4f}（=AUC）")
order = sorted(by_pair.items(), key=lambda kv: -kv[1][1])
for gp, (c, n) in order:
    print(f"   {gp:>12}: 对数={n:6d} ({n/total:.1%})  一致率={c/n:.4f}  贡献损失={n/total*(1-c/n):.4f}")

print("\n[H4] B1 相对 RF 改了什么（排名变化与对错）")
df["rank_rf"] = df.p_rf.rank(ascending=False, method="first").astype(int)
df["drank"] = df.rank_rf - df.rank_b1  # >0 = B1 把它往前提了
chg = df[abs(df.drank) > 20]
print(f"   排名变动>20 位的车：{len(chg)} 台；其中 y=1 的 {int(chg.y.sum())} 台")
up_pos = chg[(chg.drank > 20) & (chg.y == 1)]
dn_pos = chg[(chg.drank < -20) & (chg.y == 1)]
up_neg = chg[(chg.drank > 20) & (chg.y == 0)]
dn_neg = chg[(chg.drank < -20) & (chg.y == 0)]
print(f"   正类被上提:{len(up_pos)} 被下压:{len(dn_pos)}｜负类被上提:{len(up_neg)} 被下压:{len(dn_neg)}")
print(f"   上提的正类群分布: {up_pos.group.value_counts().to_dict()}")
print(f"   上提的负类群分布: {up_neg.group.value_counts().to_dict()}")

print("\n[H5] Oracle 上限：跨群保持 B1、群内排序换成完美，AUC 能到多少")
p_or = np.zeros(len(df))
for grp in ["low", "high", "insufficient"]:
    m = (g == grp)
    sub = df[m]
    inner = sub.y * 1e6 + sub.p_b1.rank()  # 群内 y 优先
    p_or[m] = sub.p_b1.mean() + inner / 1e7
auc_or = roc_auc_score(y, p_or)
print(f"   群内完美化后 AUC = {auc_or:.4f}（现状 {roc_auc_score(y, p):.4f}，理论上限差 = +{auc_or - roc_auc_score(y, p):.4f}）")

print("\n[H6] 历史出险诊断：特征窗 [6/1,6/21) 内是否发生过事故/未遂（11803/11804）")
ev = pd.read_csv(EVENTS,
                 usecols=["gpsno", "event_type", "start_time"], parse_dates=["start_time"])
prior = ev[(ev.event_type.isin([11803, 11804])) & (ev.start_time < "2026-06-21")]
prior_cnt = prior.groupby("gpsno").size().rename("prior_acc")
df2 = df.merge(prior_cnt, left_on="gpsno", right_index=True, how="left")
df2["prior_acc"] = df2.prior_acc.fillna(0).astype(int)
has_prior = df2.prior_acc > 0
print(f"   特征窗内有历史出险的车: {has_prior.sum()} 台; 其中后续再出险 {int(df2.loc[has_prior & (df2.y==1)].shape[0])} 台"
      f"（复发率 {df2.loc[has_prior, 'y'].mean():.1%} vs 无历史者 {df2.loc[~has_prior, 'y'].mean():.1%}）")
from sklearn.metrics import roc_auc_score as ras
print(f"   prior_acc 单特征 AUC = {ras(df2.y, df2.prior_acc):.4f}")
fn_deep = df2[(df2.y == 1) & (~df2.in_top100)]
print(f"   漏报67台中有历史出险的: {int((fn_deep.prior_acc > 0).sum())} 台")
print(f"   命中59台中有历史出险的: {int(((df2.y==1) & df2.in_top100 & (df2.prior_acc>0)).sum())} 台")
# 增益模拟：把 prior_acc 加进 RF 的特征里重跑折外
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
mi_cols = ["evt_count", "evt_lane_count", "traj_km_20d", "accident_sensor_risk_index"]
Xp = df2[["p_b1", "prior_acc"]].values  # 简化：在 B1 概率 + 历史出险上做二特征逻辑残差
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
Xz = StandardScaler().fit_transform(np.c_[np.log(df2.p_b1.clip(1e-3, 1 - 1e-3) / (1 - df2.p_b1.clip(1e-3, 1 - 1e-3))), df2.prior_acc])
oof_p = np.zeros(len(df2))
for k in range(5):
    tr, va = df2.fold.values != k, df2.fold.values == k
    lr = LogisticRegression(max_iter=2000).fit(Xz[tr], df2.y.values[tr])
    oof_p[va] = lr.predict_proba(Xz[va])[:, 1]
print(f"   [增益模拟] logit(p_b1)+prior_acc 二特征逻辑融合 OOF AUC = {ras(df2.y, oof_p):.4f}（B1 现状 {roc_auc_score(y, p):.4f}）")

df.to_csv(f"{OUT}/error_analysis_vehicles.csv", index=False)
print("\n车辆级误差数据已存 outputs/error_analysis_vehicles.csv（含 FN/FP 标记与排名，供 demo 使用）")
