# 文档导航与唯一职责

贡献与评审状态统一见 [初始化记录](decisions/ADR-0001-governance-bootstrap.md)。本体系已完成文件初始化，不表示团队已确认算法路线或全部治理细则。

## 先读什么

- 新成员／新 agent：[统一入口](../AGENTS.md) → [当前状态](DEVELOPMENT_STATUS.md) → [协作方式](COLLABORATION.md) → 本次任务。
- 准备实验：[数据边界](DATA_POLICY.md) → [接口入口](interfaces/README.md) → [验收规范](ACCEPTANCE.md)。
- 准备汇报：[当前状态](DEVELOPMENT_STATUS.md) → [证据索引](evidence/README.md)；交接日志不能单独证明数值结论。

## 每类信息只在一处定义

| 信息 | 权威位置 | 其他文档的职责 |
| --- | --- | --- |
| 赛题与提交要求 | 官方文件及后续正式澄清 | 摘要必须注明歧义，不自行裁决 |
| agent 基本规则 | [AGENTS.md](../AGENTS.md) | 兼容文件只链接 |
| 当前进度、阻塞、结论边界 | [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) | README 不复制具体实验成绩 |
| 任务认领、分支、写入范围、验收 | [tasks/](tasks/README.md) 中每个任务 JSON | 交接日志记录过程，不另立任务状态 |
| 协作流程与交接 | [COLLABORATION.md](COLLABORATION.md) | 客户端专属提示只做适配，不另定规则 |
| 数据存储和公开边界 | [DATA_POLICY.md](DATA_POLICY.md) | 忽略规则与检查器是辅助，不替代人工判断 |
| 硬盘与受控渠道交付规范 | [DATA_DELIVERY.md](DATA_DELIVERY.md) | 各线交付结构示例只作参考，权威以本表为准 |
| 路线与方案提议 | [plans/](plans/README.md) | 未接受的方案不能作为冻结合同 |
| 重大决策及变更理由 | [decisions/](decisions/README.md) | 接受时写明真实决策人和日期 |
| 数据、评估、评分公共契约 | [interfaces/](interfaces/README.md) | 实现代码引用版本，不私自改语义 |
| 验收与图表要求 | [ACCEPTANCE.md](ACCEPTANCE.md) | 阶段任务列出本次必交图表 |
| 可引用证据 | [evidence/](evidence/README.md) | 数值结果默认保存受控本地，仅记录允许公开的索引 |
| 会话历史 | [worklog/](worklog/README.md) | 追加，不冒充当前状态 |

## 发生变化时更新哪里

- 普通实现：任务记录、相关说明、交接日志；无需修改所有治理文件。
- 状态或主要阻塞变化：额外更新状态页，由负责归并的成员处理公共文件冲突。
- 标签、切分、公共字段或评分规则变化：增加决策记录，更新对应契约与失效实验清单。
- 新实验证据：保存受控报告，更新证据索引及可引用边界；未核验写“待复核”。
- 文档或目录变化：运行治理检查，确保新设备取得的是有效链接和真实状态。

空目录按需创建，不提前堆叠大量模块。尚未实现的命令、指标和页面不得写成“已可用”。
