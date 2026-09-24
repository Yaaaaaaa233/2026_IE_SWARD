# T2R-002：任务二 T2-R4 MOE 与双基线交接

日期：2026-09-24
负责人：周航正
执行者：Codex
状态：待组长裁决 D21–D23

## 本次完成

1. 建立任务二评价合同，固定“硬检查 → 五分位单调 → AUC/AP/Lift → 跨折稳定 → 可复算与低证据保护 → 子群诊断 → 任务一一致性参照”的顺序。
2. 实现协议清单检查，强制特征窗早于评分截点和标签窗、标签／折分版本一致、所有学习操作只在训练折完成；任务一概率参与参照时必须为完整折外预测。
3. 实现评分评价器：ROC AUC、Average Precision、低分 Top20% Lift、五分位出险率、逐折 AUC、子群方向、Spearman、Kendall tau-b 和 Top20% 重合率。
4. 实现 Q0 等存在透明弱基线。它只使用通过暴露门的行为／情境编码，设备异常只影响置信度，证据不足返回保守先验 50，不替代 R1。
5. 实现 Q1 概率转换基线 `100 × (1 − p)`，并把它限定为一致性参照。
6. 实现 5 人 × 10 固定案例 × 5 问题的人因复核器，总均分须不低于 4，任何评分不低于 3。
7. 加入合成坏例，覆盖重复车辆、复算不一致、低证据虚假高分、未来特征、非折外任务一概率和不完整人因矩阵。

## 交付入口

- 方案：`task2/evaluation/moe-baseline-charter-v1.md`
- 评价合同：`task2/evaluation/evaluation_contract_v1.json`
- 协议清单示例：`task2/evaluation/protocol_manifest.example.json`
- 评分评价器：`task2/evaluation/evaluate_scores.py`
- 双基线：`task2/evaluation/baselines.py`
- 人因复核器：`task2/evaluation/human_review.py`
- 人因模板：`task2/evaluation/human-review-template.md`
- 合成测试：`tests/test_task2_evaluation.py`

## 边界

本任务没有使用真实数据，也没有生成真实 AUC、Lift、子群或人因结论。Q0 是刻意简单的效果下界；R1 仍由娄澜负责。Q1 不进入公开总分；是否融合属于 R3。历史事故／未遂的一票降级上限已经作为可选接口实现，但 T2-R0 D2 未经组长裁决前默认关闭。

## 下一步

组长确认 D21–D23 后，娄澜的 R1 输出按评价合同补齐 `gpsno`、`y`、`fold`、`safety_score`、`rule_recomputed_score`、`observation_status` 和可选任务一折外概率即可同口径评价。周航正后续的 T2-R2 仍需等待 R1 规则引擎可用；当前不提前修改其代码。
