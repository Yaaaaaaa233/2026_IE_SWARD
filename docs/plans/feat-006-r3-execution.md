# FEAT-006 r3 执行方案：y 盲剪枝 bundle（瘦身版总体算法）＋配对判定

状态：executed（2026-09-22 晚执行完毕；两候选主种子均未过门、负结果登记，剪枝假设方向成立（点估计与 CI 均改善）；结果与判读见受控台账 §8 与 [worklog](../worklog/2026-09-22-feat-006-r3.md)）
上位方案：[task1-optimization-plan-v1](task1-optimization-plan-v1.md) §4 阶段 1；判据：ADR-0004 不放松。
背景：r1/r2 两轮"全员点估计为正、无人过门"；full_r2（全族 60 列叠加）+0.0094 未过门＝共线稀释实证。本轮假设：**剪枝后的小集**让 RF 在 500 样本下不被冗余分裂稀释，保留各族信息核心。

## 1. 预登记选特征规则（y 盲，执行前写死）

所有规则不接触标签 y（选特征不看结果，防多重比较/挑结果红线）。起点＝F2R2_v1 model_input（168 特征）。

**候选 A（pruned_bundle）**：
1. 基座全保：F0 全部＋F1 多窗口 15 列＋prior 计数（FEAT-003 裁定并入）＋F2 元数据（align_pass/var_ratio/低暴露旗标/any_scenario/prior 交叉）。
2. F2 六场景：每场景只保 per_1000km 版（弃 count/rate_10k/per_100h——同场景四版高度共线，保暴露归一版）。
3. r2 波动族：每基础信号只保 std 一版（弃 iqr/cv/rel_p90/p95——同一信号的分布统计变体）；wmagdesat_std 与 wmag_std 数值恒等（无饱和），弃；保语义不同的时段分波动（am/night）与前/后趋势比（fbratio）。
4. r2 时段交互：只保合并总量（am/night count 与 per_1000km 共 4 列，弃分场景拆分）。
5. r2 健康分：弃零方差列（sat/sat_rate/const——清洗层无饱和已定案）；保 dup_rate/glitch_rate/maxw/segments。
6. r2 方向无关：保 count_20d＋per_1000km（弃 per_100h）。
7. 收尾通则：保留下增量列内 |Spearman ρ|>0.95 的成对去后者（固定列序确定性），近零方差剔除。

**候选 B（lean）**：F0＋F1 多窗口＋prior＋各族信息核心各取 1-2 列：
波动 {wmag_std, gyroz_std, amagres_std}＋健康 {dup_rate, glitch_rate}＋方向无关 {turn_anom_per_1000km, long_anom_per_1000km}＋时段 {evt_night_count, evt_am_count}＋F2 六场景 per_1000km 6 列。（不做收尾剪枝，按声明执行）

## 2. 评估（判据不放松）

- 两候选各跑统一评估器（冻结 RF），对冻结基线 RF@F0_v1 车辆级配对 bootstrap（2,000/95%/seed42）＋种子 7/2026 旁证；
- 过门＝CI 下界>0 且点估计>0；有意义＝CI 下界≥0.01；容忍度同 ADR-0004；
- **预登记预期**：A 点估计 +0.010~0.020（剪枝抗稀释）、B +0.005~0.015；两者都不过门的概率不小——若是，按负结果登记，本轮结论仍成立（"剪枝也不解耦功效瓶颈"）。
- 两次判定（A、B）均为预登记假设，多重性在汇报中明示。

## 3. 交付

L1：构建脚本＋保留/剔除清单＋指纹；判定 JSON；台账 §8＋worklog＋FEAT-006 记录＋git 提交。L2/L3 视采纳与否按规程补（采纳才补全）。
