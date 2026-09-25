# FEAT-012 交接简报：新版 F3 特征与算法多轮优化

- 任务：[FEAT-012](../tasks/FEAT-012.json)
- 方案：[多轮迭代优化方案](../plans/feat-012-iterative-feature-model-optimization.md)
- 用户指令：不等待 EVAL-001 全项目统一复核，赶时间直接启动类似王健祺线的特征与算法多轮优化。
- 执行者：Codex；人类负责人：叶安。

## 范围与固定约束

任务以 FEAT-009 protocol-v2 Y1 F3 输入、FEAT-011 S0 逐列来源台账和 FEAT-009 Y3 F3 OOF 为只读源。沿用官方标签定义 `label_v2_record_count_20260923`、固定车辆折分 `split_v2_record_strat5_seed42` 及 20 天特征窗／40 天代理标签窗。任务本地参考 EVAL-001 v1.3 技术登记，不声称全项目评估规则已统一或接受。

预登记最多 19 个有效版本：第一轮 13 个单视角／模型版本，第二轮按第一轮 AUC 在不同视角选代表并做六个两两等权 logit 融合。所有预处理仅在外折训练行拟合；不重新提取原始特征，不使用最终 61 天训练窗，不修改 FEAT-009/011 或王健祺线产物。

多轮都重复使用相同代理车辆和折分，所以模型择优偏差会累积。AUC、配对 bootstrap 和图表都只用于开发假设排序，不构成独立测试、跨线胜负或正式采纳依据。真实预测、数值、图表和指纹保存在本机忽略的 `outputs/feat-012/<run-id>/`。

## 执行结果

- 受控 run `outputs/feat-012/20260925-r1/` 在候选运行前锁定了输入、列族视角、标签／折分身份与 19 版预算。13 个单模型视角／算法版本和 6 个自适应等权 logit 融合均已生成逐车 OOF。
- full EBM-A 与 FEAT-009 原始 F3 OOF 逐值相同；full EBM-C 与既有 EBM-C 探索 OOF 也逐值相同。输入没有在 FEAT-012 中重建或覆盖。
- full EBM-A 锚复现与既有 FEAT-009 OOF 的逐值对账、full EBM-C 与既有探索 OOF 的逐值对账均通过。具体真实数值和 OOF 仅在受控结果目录及其图表中查看。
- RF 与 HistGradientBoosting 在当前环境成功运行；LightGBM 本机动态库缺少 OpenMP，因此未纳入本轮运行，未临时安装依赖。
- 受控 `report.md`、`metrics.json`、配对 OOF、两张图及运行 manifest 留在 `outputs/feat-012/20260925-r1/`。脚本：`feature_engineering/experiments/feat012/run_exploration.py`；独立指标与融合复核：`feature_engineering/experiments/feat012/audit_exploration.py`。
- 独立复算重新检查 19 个版本的输入哈希、车辆／标签／折分对应、概率范围、pooled 与逐折 AUC、AP、Recall@100、Brier、配对区间、融合公式和 OOF／指标文件哈希，全部通过。机器单测与治理检查也通过。

FEAT-012 已完成两轮，任务转为 `review`。因为点估计与区间不足以收敛为稳健结论，用户已授权另开 FEAT-013 小型后续，单独测试种子平均、EBM 配方与固定权重；FEAT-013 的预测不混回本轮。
