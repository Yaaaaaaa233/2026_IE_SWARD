# 2026-09-21 方案B集成学习 v1 实现与入库

- 任务／分支／基准提交：[MODEL-003](../tasks/MODEL-003.json)｜`work/MODEL-003-solution-b-v1`｜`0ebad4d`
- 人类责任人：王健祺（模型线；授权"作为 version1 按提交规范上传 git"）
- 执行者与 AI 协助：ZCode agent 执行；王健祺验收交付物后授权入库
- 复核状态与真实复核人：`review`——开发期自验证完成；正式复核随 R4 验收（待叶安或委托人）

## 做了什么

- 按用户批准的三层方案完成方案 B v1 实现并入库 [models/solution_b/](../../models/solution_b/README.md)：
  B0 全局骨架（RF+EBM 嵌套折外、每外层折独立 inner-OOF 校准材料）→ B1 分群校准（low 独立 / high 收缩 n/(n+80) /
  insufficient 纯全局并标记 calib_flag）→ B2 收缩残差（λ 折内网格，λ=0 恒等回退）。
- 指标库按 EVAL-001 v0.1 实现（AUC 主 + AP/Recall@20% 固定名额/Brier；同分 sample_id 字典序；
  车辆级配对 bootstrap 2000 次 95% 区间；判优门槛预登记于 common.py PREREG）。
- 验证套件四维分开报告：V1 正确性 11/11（含嵌套结构断言、同种子复现差异在浮点最低位）；
  V2 统计（阶梯/分群分层/校准）；V3 三种子抖动；V4 负向测试 6/6（检查器对坏输入全部拒绝）。
- 机器路径配置化：config_local.template.json 入库、config_local.json 走 .gitignore；入库目录重跑验证与本地结果一致。
- 路线提案修订：E-B1 硬路由 → 校准先行 + 收缩残差（§4.1，待 R0/R4 评审后同步总体路线图）。
- 新增 tests/test_solution_b_metrics.py（合成数据 11 项，含同分名单政策与配对 bootstrap 性质测试）。

## 验证

- `python models/solution_b/validate.py`：机器检查 11/11 PASS、负向测试 6/6、判优裁决采纳 B1（留痕 report/validation_results.json）。
- `python -m unittest discover -s tests -v`：28 项通过（既有 17 + 新增 11）。
- `python tools/check_governance.py` 与 `--scope index`：通过后提交。
- v1 结论边界：B1 相对最强单模 AUC 提升为迹象级（配对区间跨零、未超 3× 种子抖动线），概率质量（Brier/LogLoss）
  改善稳健；B2 残差层当前特征下全部折 λ 自动选 0 回退。真实数字仅存受控本地报告
  （方案B实现/docs/方案B_验证与效果报告.pdf 等），公开仓库只入方法与规则。

| 维度 | 状态与范围 |
| --- | --- |
| 实现与契约 | 代码入库可复跑（模板制路径）；输出合同断言全绿 |
| 实验效果 | B1 采纳（预登记规则）；AUC 迹象级、概率质量稳健改善；高风险群仍薄弱（特征升级是改善方向） |
| 验收与证据 | 四维报告与负向测试留痕；未经独立复核，随 R4 验收 |
| 环境与依赖 | 单机 CPU 全流程约 5 分钟；E 盘只读；interpret/sklearn 依赖已注明 |

## 决策与限制

- 无新 ADR：E-B 矩阵修订属模型线提案内部演进（原提案仍为 Proposed，未冻结），已在 §4.1 记录待评审。
- 预测结果、折外预测、manifest 指纹、图表与 PDF 均不入公开仓库（DATA_POLICY）。
- F0 特征升级（F0→F1）、splits 变更或 EVAL-001 冻结口径调整，均须全流程复测后更新结论。

## 接手者下一步

1. 叶安评审提案 §4.1 与本任务，确认后同步总体路线图 R4 条目（硬路由行）。
2. 方案 A/B 同折终局对照（E-B4）：共享成员库与折外预测存档，双指标择优。
3. 误差诊断与改进（用户已布置：基于预测不准车辆的归因分析 → 改进方案先报审后实现）与可视化 demo 在受控本地推进。
4. 特征线 F1（含历史出险、暴露双轨、强度加权候选）到位后触发 B2 残差层与全体复测。
