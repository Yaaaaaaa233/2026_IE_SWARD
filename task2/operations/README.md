# 任务二动态运营修正

本目录实现 T2-R2 中不依赖 R1 内部算法的部分。R1 负责给出事件的基础扣分，R2 只处理时间、持续性、暴露、停驶和核查状态。

- [动态运营方案](operational-dynamics-charter-v1.md)
- [候选策略配置](operational_policy_v1.json)
- [R1 对接合同](r1-adapter-contract.md)
- [合成输入示例](input.example.json)
- `dynamic_score.py`：动态评分与命令行入口。

运行方式：

```bash
python task2/operations/dynamic_score.py input.json \
  --policy task2/operations/operational_policy_v1.json \
  --output dynamic_scores.json
```

真实输入和输出只能写入受控目录。当前配置中的 7 天半衰期、50 分先验、5 单位暴露门和 3 天核查期均为待验证候选；时段系数和持续性倍率默认保持中性。运行成功不表示这些候选值已经被团队接受。
