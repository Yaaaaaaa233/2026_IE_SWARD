# 任务二评价与双基线

本目录实现 T2-R4 的公开、无真实数据部分。它给 R1、R2、R3 提供同一把尺子，不修改任何候选扣分规则。

- [MOE 与双基线方案](moe-baseline-charter-v1.md)
- [评价合同](evaluation_contract_v1.json)
- [协议清单示例](protocol_manifest.example.json)
- [人因复核模板](human-review-template.md)
- `baselines.py`：Q0 等存在透明弱基线与 Q1 概率转换基线。
- `evaluate_scores.py`：硬检查、五分位、AUC、AP、Top20% Lift、分折方向、子群和任务一一致性。
- `human_review.py`：5 人 × 10 案例 × 5 问题验收。

真实运行时，评分 CSV、OOF 预测、指标 JSON 和图表均应写入受控目录，不得提交到公开仓库。命令入口：

```bash
python task2/evaluation/evaluate_scores.py scores.csv \
  --manifest protocol_manifest.json \
  --output evaluation_report.json

python task2/evaluation/human_review.py human_review.csv \
  --output human_review_report.json
```

退出码为零只代表输入合法且指标成功计算。是否通过首要门槛读取报告中的 `primary_gate_pass`；是否采用某个评分方案仍需结合规则复算、人因复核和组长裁决。
