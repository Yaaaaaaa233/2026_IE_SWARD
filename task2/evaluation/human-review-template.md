# 任务二人因与运营复核模板

## 固定案例构成

| 案例类型 | 数量 | 复核目的 |
|---|---:|---|
| 低分且未来出险 | 2 | 检查规则能否解释真实高风险 |
| 高分且未来未出险 | 2 | 检查安全表现是否得到合理保留 |
| 高分但未来出险 | 2 | 专门寻找规则漏判和反直觉证据 |
| 低暴露或数据缺失 | 2 | 检查是否出现虚假高分 |
| 近期明显改善或恶化 | 2 | 检查变化方向及运营话术 |

案例在查看候选方案得分差异前固定。每位复核者都评价同一组 10 个案例，不能为某个方案单独替换案例。

## 每位复核者的五项问题

每项采用 1–5 分：

1. `reason_is_understandable`：是否看得懂为什么扣分。
2. `deduction_matches_evidence`：扣分是否与展示的证据一致。
3. `improvement_action_is_clear`：驾驶人是否知道下一步如何改善。
4. `manager_action_is_available`：管理者能否据此采取具体行动。
5. `missing_duplicate_stopped_handling_is_reasonable`：缺失、重复报警和停驶处理是否合理。

复核输入的每行对应一个“复核者—案例”组合，包含：

```text
reviewer_id, case_id, case_type,
reason_is_understandable, deduction_matches_evidence,
improvement_action_is_clear, manager_action_is_available,
missing_duplicate_stopped_handling_is_reasonable
```

验收要求：5 名复核者、10 个固定案例、共 50 行和 250 个评分；总均分不低于 4，任何一个评分不得低于 3。未达到时保留失败原因并修改规则或解释，不删除低分案例。

## 复核纪律

- 先展示证据、扣分条款、置信度和建议动作，再让复核者打分。
- 隐去未来标签，完成判断后才揭示是否出险，避免结果倒推评分。
- 高分但未来出险的案例必须保留在误差分析中。
- 真实车辆标识、明细和评分结果只保存在受控目录。
