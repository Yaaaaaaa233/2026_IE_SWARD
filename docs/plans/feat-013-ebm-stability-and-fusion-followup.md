# FEAT-013：FEAT-012 后续 EBM 稳定性与融合小迭代

- 状态：2026-09-25 按叶安线赶进度授权启动；独立于 FEAT-012 首轮与第二轮的补充探索。
- 负责人：叶安；执行：Codex；真实复核待登记。
- 任务：[FEAT-013](../tasks/FEAT-013.json)。

FEAT-012 的多视角与模型适配探索留下了 EBM 稳定性和融合权重这两个值得进一步检查的问题。FEAT-013 只验证三个更具体的问题：同一 EBM 结构换种子后做 logit 平均是否稳定、EBM-C 增加外袋或调整学习率是否有帮助、两视角融合的简单固定权重是否可能优于等权。它是事后续探，复用了前一轮已经查看过的代理 OOF，因此所有结果均为探索信号，偏差风险高于 FEAT-012。

## 输入和约束

输入保持 FEAT-009 v2 Y1 F3 表、FEAT-011 S0 列台账、FEAT-009 原 F3 OOF 与 FEAT-012 `20260925-r1` 受控 OOF。只读验证标签 `label_v2_record_count_20260923`、固定折分 `split_v2_record_strat5_seed42` 和 manifest／OOF 哈希。所有真实产物写入新目录 `outputs/feat-013/20260925-r1/`，绝不覆盖 FEAT-012。模型预处理继续按每折训练部分拟合；特征仍来自已登记的 F3 候选列；不改变 label、fold、窗口或全队评价器。

FEAT-012 full EBM-A 是配对参考。模型参数与所有组合规则在本轮真实 OOF 之前写入受控 `preregistration.json`。最多生成 17 个报告版本：9 个训练 OOF（6 个新增种子臂和 3 个 EBM 配方臂）及 8 个确定性组合。

## 固定实验组

### 新增训练 OOF：9 个

1. `full_ebm_c` 与 `slim_ebm_c`、`history_ebm_a` 分别增加种子 7 和 2026，共六个五折 OOF。种子 42 直接复用 FEAT-012 完整受控预测，不重训后覆盖原文件。
2. full EBM-C 增加 `outer_bags=28`；slim EBM-C 增加 `outer_bags=28`；full EBM-C 将 `learning_rate` 固定为 0.03（原配方为默认 0.015）。除列视角或所登记参数外保持模型配置不变。

### 确定性组合：8 个

- full EBM-C、slim EBM-C、history EBM-A 各自的三种子 logit 均值，共三版。
- 三种子 full EBM-C 与三种子 history EBM-A 的等权组合、full 权重 0.7 / history 权重 0.3、full 权重 0.3 / history 权重 0.7，共三版。
- 三种子 full + slim EBM-C 等权组合一版。
- 三种子 full + slim EBM-C + history EBM-A 等权组合一版。

本轮不在看到结果后重新调整配方、权重或成员。最高点估计不得自动变为候选冠军或提交模型。

## 评价与解释

沿用 FEAT-012 的 pooled OOF AUC、full EBM-A 配对分层 bootstrap（2000 次，种子 42，95% 百分位区间）、AP、Recall@100、Brier 和逐折 AUC。所有权重和成员在 FEAT-012 结果基础上形成，因此区间仅描述固定车辆重采样下的配对不确定性，不包含两轮择优、权重挑选或多种子筛选造成的偏差。

停止条件：输入与先前 run 指纹不一致则不训练；解释器或模型拟合失败时保留失败记录；完成 17 版或模型矩阵结束即收口。不存在基于 AUC 再开第三个补充搜索的自动授权。任何进一步探索须独立登记，且外部官方预测期才能验证独立泛化表现。

## 复现和回滚

使用 `feature_engineering/experiments/feat013/run_followup.py` 的 `prepare` 阶段锁定原始输入与 FEAT-012 OOF 指纹，之后运行 `run`。受控 run 保存每个新增种子的逐车预测、组合预测、主辅指标、配对区间、折内训练样本指纹、依赖、脚本哈希和两张图。删去或停止使用 `outputs/feat-013/20260925-r1/` 即可回滚；不更改既有结果。
