# T2R-005：R3 融合实验无数据部分交接

日期：2026-09-25
路线阶段：T2-R3（娄澜，09-27～09-29 的无数据前置部分）
负责人：娄澜
执行者：ZCode agent
状态：review（合成验收通过，等 Q0/Q1 数据阶段输入）

## 本次完成

在独立分支 `work/T2R-005-r3-fusion-datafree` 完成 R3 的无数据封闭工作包：

1. **融合机械**（`task2/fusion/fusion.py`）：
   - `fuse_scores`：S = alpha_q0·Q0 + (1−alpha_q0)·Q1，车辆集合一致性与 [0,100] 值域强制；
   - `run_alpha_grid`：五臂（1.0 纯规则 / 0.7 / 0.5 / 0.3 融合 / 0.0 纯模型参照）逐臂构造
     评价行并跑同一 `evaluate_scores`；概率列仅在 alpha_q0<1 挂接并按 manifest 声明；
     机械选择器=首要门通过者中 ROC AUC 最高、平手取更大 alpha_q0；
   - `rank_disagreements`：Q0/Q1/融合分三元风险排名与最大名次差 Top-K（人工复核弹药）；
   - `dimension_talking_point`：D20 司机话术格式（只呈现维度表现，不暴露模型内部）。
2. **章程落笔**（`fusion-charter-v1.md`）：D18 正反论证登记（正方融合=娄澜执笔，
   反方纯规则=② 立场代写）、D19 路线图 α 记号歧义的澄清（alpha_q0=Q0 权重，
   α=0 纯规则对应 alpha_q0=1.0）、四条红线（任务一概率不判优、台账复算保持、
   纯模型臂仅参照、平手偏解释性）。
3. **合成测试 8 项**：公式与端点、坏例拒绝、五臂同门通过、概率声明错配被评价
   硬检查拒绝（负向）、分歧排序、话术格式与校验。

## 验证记录

- `python -m unittest tests.test_task2_fusion -v`：8 项通过。
- `python -m unittest discover -s tests -v`：全量通过（提交前执行）。
- `python tools/check_governance.py` 与 `--scope index`：通过后提交。

合成分数为固定种子 fabricated，不代表任何真实对比结论；D18 裁决待数据阶段。

## 开放项与下一步（数据阶段）

1. Q0：T2R-004 数据阶段产物（label_v2 重学后的正式规则分）；
2. Q1：任务一当前采纳版本（solution_b_v5 或后续过门版本）同折 OOF 概率，
   经 `probability_to_safety_score` 转换；版本指纹按 ADR-0005 登记进 manifest；
3. 真实 α 网格对照 + 人因复核（5 人×10 案例，T2-R4 固定案例集）+ 三元散点
   与 Top 分歧车辆人工复核 → D18 裁决建议提交组长终审。

## 边界声明

未修改周航正的 R0/R4/R2 与 T2R-004 的任何文件；未使用或提交真实数据；
本任务不产生真实 AUC、Lift、一致性或人因结论。
