# 任务二分类规则资产

本目录是 T2-R0 的公开、无数据实现，用于在扣分建模前冻结“同一证据只归一个主要计分位置”的语义合同。它不包含真实车辆记录、统计量、学习权重或评分结果。

- [分类总纲 v1](classification-charter-v1.md)：D1–D6、风险链、去重原则和实施边界。
- [三层结构一页图](three-layer-one-page.md)：公开五维、内部六组、32 个目录小类。
- [组长裁决表](leader-review-sheet.md)：M1–M8 与 D1–D6 的待确认选项。
- [机器契约](classification_v1.json)：24 个编码的唯一映射、32 个小类、十条跨事件规则和五条风险链。
- [契约校验器](validate_classification.py)：检查映射完整性、证据边界和条件项开关。

运行校验：

```bash
python task2/classification/validate_classification.py
python -m unittest tests.test_task2_classification_contract -v
```

校验通过只表示结构和边界完整，不表示组长已经接受裁决，也不表示任何扣分权重或模型效果已经验证。
