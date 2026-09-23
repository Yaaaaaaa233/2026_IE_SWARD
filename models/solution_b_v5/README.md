# 方案B V5（任务一王健祺线·当前登记最优）

- 任务：[MODEL-005](../../docs/tasks/MODEL-005.json)｜对应 [ADR-0005](../../docs/decisions/ADR-0005-task1-dual-track-and-task2-split.md) 双人竞优·王健祺线
- 状态：**过门版本**——对冻结基线（[ADR-0004](../../docs/decisions/ADR-0004-baseline-freeze.md)，RF）车辆级配对 bootstrap 区间下界 > 0；数字见受控台账
- 消费方：任务二评分线（Q1 概率转换分的上游）与终局提交候选

## 构成

| 层 | 内容 |
| --- | --- |
| 特征 | F1_v1 全部有效特征（特征线交付，只读消费；含历史险情/暴露归一/多窗/事件语义族） |
| 成员 | 5 个：EBM(seed2026) + RF(seed7) + EBM(seed7) + EBM无交互(seed7) + EBM无交互(seed2026)——折内贪心选出 |
| 融合 | 成员概率 logit 均值 |
| 校准 | 全局 Platt + 分群收缩（low 独立 / high w=n/(n+n0) / insufficient 纯全局）；n0 每折在训练部分内按 Brier 选 |
| 防泄漏 | 嵌套折外（每外层折的训练部分内部再五折产校准材料）；一切学习型步骤折内拟合；同种子确定性 |

## 运行

```sh
cd models/solution_b_v5
cp config_local.template.json config_local.json   # 填 feature_root（含 artifacts_f1_v1 的根目录）
python run_v5.py
```

输出到 `models/solution_b_v5/outputs/`（git 忽略，受控）：`oof_predictions_v5.csv`、`model_score.csv`（评分线对接）、`forecast_result.csv`（官方三列）、`manifest.json`（输入 sha256 指纹——任务二按 [ADR-0005](../../docs/decisions/ADR-0005-task1-dual-track-and-task2-split.md) 登记消费指纹用）。

## 版本史（方法层，数字见受控台账）

- V1 全特征灌入 → V2 消融（确诊校准层缺陷）→ V3 12 成员池+折内贪心 → V4 校准修复（n0 按 Brier 选）→ **V5 成员扩展（当前）**。
- 已登记负结果：数据扩充四法（过采样/SMOTE/噪声/Mixup）全部失败；异构弱成员进池拖累融合；时间趋势特征无增量。详见 [MODEL-004](../../docs/tasks/MODEL-004.json) 与受控台账。

## 边界

开发期内部回测（冻结五折折外、单截点），非独立测试集成绩；正式表述以受控台账与 ADR-0004 判据登记为准。
