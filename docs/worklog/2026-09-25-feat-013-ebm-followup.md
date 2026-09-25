# FEAT-013 交接简报：EBM 稳定性与融合后续

- 任务：[FEAT-013](../tasks/FEAT-013.json)
- 方案：[EBM 稳定性与固定融合](../plans/feat-013-ebm-stability-and-fusion-followup.md)
- 指令来源：用户授权先不等待 EVAL-001 全队复核，直接开始叶安线特征与模型多轮优化；FEAT-013 是 FEAT-012 两轮探索后的有界补充。
- 人类负责人：叶安；执行者：Codex。

## 为什么追加这一轮

FEAT-013 限定为多种子平均、EBM outer bagging／学习率、full 与 history 视角固定权重组合。前轮预测文件只读，不覆盖；所有补充配置在本轮 OOF 前写入受控预登记。

## 固定边界

继续使用 FEAT-009 v2 F3 表、官方 label v2 和固定 stratified 五折；只取 FEAT-011 S0 已登记候选列；所有预处理每折训练内拟合。最多 17 个版本（9 个新训练 OOF 与 8 个确定性组合）。权重和候选视角来自已经查看过的同一批代理 OOF，故不能用区间跨零与否消除多轮择优偏差，不能作为独立测试成绩或正式模型采纳。

## 执行结果

- 受控 run `outputs/feat-013/20260925-r1/` 完成预登记的 17 个版本：六个新种子 OOF、三个 EBM 配方 OOF、三个视角的三种子 logit 平均及五种固定融合。
- 本轮 17 个版本的逐车 OOF、候选指标和图表均已生成。具体表现与种子波动只留在受控 run 中，不写入公开工作日志。
- 独立复算通过 17 个 OOF 的输入和行绑定、概率范围、主辅指标、逐折 AUC、配对区间、注册融合公式及 OOF／指标哈希检查。FEAT-012 的锁定 F3 输入、OOF 及公开旧产物保持只读；本任务状态转 `review`，实际人审待叶安查看。

报告与逐车产物：受控 `outputs/feat-013/20260925-r1/`；运行脚本：`feature_engineering/experiments/feat013/run_followup.py`；复算入口：`feature_engineering/experiments/feat012/audit_exploration.py`。图片、真实指标、OOF 和指纹均不入公开 Git。
