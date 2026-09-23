# FEAT-009 输入审计与建模实验

本目录实现 FEAT-009 的受控输入审计和后续建模入口。真实配置、输入指纹、审计数据、预测与图表只写入被忽略的本机配置和 `outputs/feat-009/`。

## S0 输入锁定与无泄漏审计

准备本机配置 `configs/feat-009.local.toml`（格式参照 [config.example.toml](config.example.toml)），然后运行：

```sh
python feature_engineering/experiments/task1_feature_modeling/audit.py --config configs/feat-009.local.toml
```

审计器只读 F1、F3R5、V5 OOF、冻结 RF OOF 和其清单，向受控输出目录写 `manifest.json`、`audit.json`、`issues.json`、一页 `acceptance.md` 与图表。真实指纹和统计只留本地。

### S0 图表预览与建议通过标准

- **输入流向与时间窗图**：显示 F1、F3、V5 OOF、RF OOF 汇入逐车主键／标签／折号对账，并标出特征窗 `[as_of-lookback, as_of)` 与标签窗 `[as_of, as_of+horizon)`。通过标准是四份输入主键唯一且逐车 `gpsno/y/fold` 完全相同，F1/F3 的 `as_of/lookback` 一致，特征时间严格早于 `as_of`；任何差异未解释则停止。
- **特征族与缺失图**：F1/F3 使用同一分类规则展示各族列数及缺失率，并列出未分类列。图只描述输入覆盖，不以缺失率本身判模型效果；缺失率达到 50% 的族或 F1/F3 差异明显的族须在报告中给出字段语义解释。
- **拟合边界表**：逐项列出时间窗提取、全量筛列、同群变换、V5 分群、模型和校准的输入范围与标签使用情况。目标标签进入特征构造、未来时间字段进入预测特征或折内拟合范围无法确认，均阻止 S1；仅使用无标签全量特征的操作标记为转导式，并由负责人决定是否接受该结论边界。

### 当前边界

S0 只审计输入与数据拟合范围，不运行候选模型或计算 FEAT-009 主效果判定。每次输出的本地 `acceptance.md` 会单列机器检查、实验效果、证据状态和环境状态。进入 S1 还须满足 [方案](../../../docs/plans/feat-009-feature-modeling-proposal.md) 中的放行条件。
