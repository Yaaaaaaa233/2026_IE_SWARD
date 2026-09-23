# 2026-09-23 FEAT-009-S1 验收设计冻结

- 任务：[FEAT-009](../tasks/FEAT-009.json)；分支 `main`；执行基线 `0b39339`。
- 负责人：叶安；文档与合成样图整理：Codex。
- 决策：叶安于 2026-09-23 确认 S1 验收样图及建议判读标准。此确认仅冻结实施前的评估与呈现规则，不代表 S1 已放行或模型效果已验证。

## 冻结的执行与判定口径

S1 固定外层车辆五折，在每个外层训练集内以三折分层验证选择已登记的四个 EBM／LightGBM 候选。近零方差与 Spearman 筛列、同群特征重建、缺失处理和类别编码均在对应训练部分拟合；同群派生值不复用预计算结果。内层选型只依据内层平均 AUC，外折只产生最终 OOF 预测。

唯一主判定为 F3 自适应选择流程相对 V5 的同车配对 ΔAUC，使用 ADR-0004 的 2,000 次、95% 区间、seed 42。点估计及区间下界均大于 0 才称超过对照；下界至少 +0.01 才称有实际意义增益。Recall@100 与 Brier 分别受 −0.02、+0.005 退化容差约束。RF、F1、逐折表现和候选分数只作为次级诊断，不能代替主判定。

人工图表冻结为三组：同车 ΔAUC 区间及同源辅助指标表；标注“诊断”的逐外折样本数、正类数、内层选型及外折 AUC；同环境下、同车辆集合和外层折分的 F1／F3 运行时间、峰值内存、拟合次数与 OOF AUC。图表须来自同一 run ID，显示数据窗口、样本量、版本和单位。复核者需能区分 0 线与 +0.01 线、正确解释辅助指标方向，且不把单折最佳结果当成主成绩。

## 入库文件与数据边界

- [FEAT-009 执行方案](../plans/feat-009-feature-modeling-proposal.md) 已展开 S1 每一步的操作、机器验收、图表内容、人工通过标准与放行条件。
- [合成验收图](../plans/figures/feat-009-s1-acceptance-preview.png) 只展示布局与读图方式；所有数值均为编造示例，不是比赛数据或模型结果。
- [图样生成脚本](../plans/examples/generate_feat009_s1_acceptance_preview.py) 可重新生成该合成图。治理策略按用户已确认的图样，为该文件登记精确哈希例外；仓库中不允许其他未登记图片。
- FEAT-009 隔离依赖固定在 [requirements.txt](../../feature_engineering/experiments/task1_feature_modeling/requirements.txt)；macOS arm64 的 OpenMP 运行库单独记录在 [environment-macos-arm64.yml](../../feature_engineering/experiments/task1_feature_modeling/environment-macos-arm64.yml)，使用 `llvm-openmp=23.1.1`。
- 本 worklog 不记录真实数据统计、数据指纹、逐车值或本机数据路径；此前只读结构预检与环境细节仍留在受控本地目录。

## 验证与当前状态

本次工作仅修改评估文档、合成样图和任务环境记录，没有用比赛数据训练候选、拟合特征变换或计算真实效果。候选运行仍为 `not_run`。隔离环境用 CPython 3.11.6 安装清单中固定的 Python 包；macOS arm64 的 OpenMP 运行库单独由 conda-forge 提供。四个候选均通过编造数据上的环境冒烟拟合；LightGBM 原生参数名与 sklearn wrapper 参数别名在合成输入上的预测一致。该检查只验证依赖加载和 API 行为，不是 S1 实验。入库前检查结果：治理检查 `working-tree` 通过；通用单测 28 项通过；FEAT-009 审计合成测试 2 项通过；任务 JSON 解析和 `git diff --check` 通过。图样生成脚本在本地成功重建样图。

S1 仍未放行：FEAT-008、MODEL-005、EVAL-002 的具名复核记录尚缺。LightGBM 及 OpenMP 环境现已就绪；满足前置复核、S1 代码／负向测试要求后，再开始实际候选运行。
