# 治理初始化验证记录

- 任务：[GOV-001](../tasks/GOV-001.json)
- 执行：Codex；人类维护责任人待确认
- 独立复核：未完成
- 数据范围：代码、文档、临时合成测试仓库；不读取真实比赛数据

## 检查入口

```sh
python tools/check_governance.py
python -m unittest discover -s tests -v
python tools/check_governance.py --scope index
```

实际执行结果在本次 [交接记录](../worklog/2026-09-16-governance-bootstrap.md) 登记。检查器读候选工作树、暂存区或 HEAD，发现问题非零退出。

## 证据边界

通过只支持：要求的治理文件存在、可检查链接可解析、任务记录符合结构、已知禁入路径／类型和部分秘密标记被拒绝。测试包含坏链接、强制加入数据、缺少复核人及版本差异等反例。

不支持：文件内容绝无泄密、模型正确、比赛阶段通过、团队已批准治理、远端保护规则已启用。真实数据实验不在 CI 中运行。
