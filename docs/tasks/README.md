# 任务登记

每个任务单独一个 JSON 文件，避免所有人编辑同一个大看板。以下索引只登记任务位置，实时状态读各任务本身。

- [DATACON-001：任务一数据契约草案与合成接口模板](DATACON-001.json)
- [EVAL-001：MOE 与 MOP 评价指标及基线判优规则](EVAL-001.json)
- [GOV-001：治理框架初始化](GOV-001.json)
- [TASK-005：数据清洗版本与规则总说明（v0–v4）入库，docx 转 Markdown](TASK-005.json)
- [GOV-002：总体执行路线图 v0.2 与执行纪律落地](GOV-002.json)
- [GOV-003：协作模式切换——取消强制 PR，改为协调者定期监督（ADR-0003）](GOV-003.json)
- [GOV-004：数据交付规范（硬盘接口）与脚本归属（DATA_DELIVERY）](GOV-004.json)
- [GOV-005：目录归位——评估方案与 roadmap v2 迁入 docs/plans](GOV-005.json)
- [GOV-006：CI 单测依赖声明与治理工作流修复（requirements.txt + pip 安装步骤）](GOV-006.json)
- [MODEL-002：模型路线提案、模型契约与执行路线图 v2](MODEL-002.json)
- [MODEL-003：方案B集成学习 v1——全局骨架+分群校准+收缩残差实现与自验证](MODEL-003.json)
- [MODEL-004：方案B误差归因诊断与模型侧迭代实验（负结果记录）及F1特征需求输出](MODEL-004.json)
- [FEAT-001：整理特征工程当前工作、接口状态与执行路线图](FEAT-001.json)
- [FEAT-002：公开特征工程 F0/F1_v1 源码、合成测试与配置模板](FEAT-002.json)

使用 [任务模板](../templates/task.json)，文件名等于唯一 `id`。未来任务采用团队协调的前缀与编号，不在不同分支同时创建同名记录。

## 状态与责任

`proposed → ready → active → review → accepted`；阻塞用 `blocked`，取消用 `cancelled`。

- `owner`：真实人类责任人；未确认用 `null`，不能从 Git 身份猜测。
- `actor`：当前执行人员或工具。AI 可记录为执行者，不能替代人类最终责任。
- `write_scope`：允许写入的文件或目录，统一使用仓库相对路径；不是所有可能读取的路径。
- `dependencies`：前置任务 ID，不依赖“另一个聊天窗口里应该已经完成”。
- `reviewer`、`reviewed_at`：真实复核记录。`accepted` 不得为空，且应有 `evidence`。
- `branch`、`base_commit`：执行起点；`handoff` 指向最近交接。

领取任务后先共享认领记录并检查重叠。检查器可发现当前快照中两个 `active` 任务的写入范围重叠，但无法发现尚未推送的认领，也不能提供跨设备分布式锁。

任务主记录保留最新状态，历史理由放交接或决策，不追加互相矛盾的第二份状态表。
