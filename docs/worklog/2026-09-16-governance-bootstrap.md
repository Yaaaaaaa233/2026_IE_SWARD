# 2026-09-16 治理框架初始化

- 任务：[GOV-001](../tasks/GOV-001.json)
- 分支：`chore/agent-governance`，基于本地已有提交 `0ca7910`
- 人类需求方：本次会话用户；维护责任人待确认
- AI 协助：Codex，参考审查、文档与检查器实现
- 复核：实现者自测，待团队独立复核

## 变更

参考旧项目的统一入口、文档职责、交接与分层验收机制，建立本项目治理文件、任务记录、模板、跨平台检查和 CI。路线仍待确定，未建立模型或执行正式数据实验。

当前 GitHub 为公开仓库。保留原有官方资料，不新增比赛数据或实际统计图表；含本地统计的详细讨论稿保持本地。原工作目录未提交的 roadmap 修改和会话痕迹未纳入本次提交。

## 验证

- Python 3.11.6，`python3 tools/check_governance.py`：PASS，检查当前候选 36 个文件。
- `python3 -m unittest discover -s tests -v`：17 项通过，包括断链、强制加入数据、假复核状态、任务冲突、CLI 非零失败、HEAD／index／工作树差异。
- `git diff --check`：通过。
- `python3 tools/check_governance.py --scope index` 和 `git diff --cached --check`：通过，验证实际暂存内容。
- CI 配置为 Linux、Windows、macOS 三个平台，远端结果以实际 Actions 运行记录为准，不把本机通过等同于三平台均通过。

## 接手者下一步

1. 从 PR 分支或合并后的 main 读取 AGENTS 和状态页，在另一设备运行治理检查与自测。
2. 确认维护人、任务协调者和受控数据共享方式，真实复核后更新 GOV-001。
3. 对候选路线和官方歧义作独立决策，再认领 M0 实现任务；不把治理初始化当成算法阶段已通过。
