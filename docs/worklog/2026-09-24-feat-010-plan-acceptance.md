# FEAT-010 最终窗口与观测量检验方案入库（2026-09-24）

- 任务／分支／基准提交：[FEAT-010](../tasks/FEAT-010.json)；`main`，按 ADR-0003 直接提交；`5a299a6`。
- 人类责任人：叶安。
- 执行者与 AI 协助：Codex 起草并按叶安指令将方案入库，未运行真实数据或模型。
- 复核状态与真实复核人：叶安于 2026-09-24 确认方案入库；仅接受方案，不构成数据、评估和算法验收。

## 做了什么

按用户提出的两个优先方向编成 [执行方案 v1.0](../plans/feat-010-final-window-and-observation-audit-plan.md)：A0 新版标签／折分及特征来源复核，A1 61 天无标签输入语义检查，B0 一项观测量列消融预登记，B1 同折 OOF 对照。每步列操作、机器验收、人工可视化验收与放行门；王健祺线新版 V5 不作依赖。按叶安“按照项目纪律入库”指令补 [ADR-0009](../decisions/ADR-0009-feat-010-final-window-and-observation-audit.md) 与任务、导航索引。保留工作区中既有未提交改动，不把它们纳入本任务。

## 验证

- `python3 tools/check_governance.py`：通过。
- `python3 -m unittest discover -s tests -v`：30 项通过。
- 尚无真实数据运行、合成样张、预登记或模型效果检查；此时只验证文档结构与任务记录。

## 决策与限制

遵循 [ADR-0006](../decisions/ADR-0006-official-qa-foundation.md)、[ADR-0007](../decisions/ADR-0007-near-miss-count-unit.md) 和 FEAT-009 新协议交接。[ADR-0009](../decisions/ADR-0009-feat-010-final-window-and-observation-audit.md) 仅接受本线方案，不改变公共标签、评估器、基线或最终模型决策。EVAL-001／EVAL-003／FEAT-009 的人工复核仍待完成。

## 接手者下一步

下次开工先按 [方案](../plans/feat-010-final-window-and-observation-audit-plan.md)更新执行者认领、具体写入范围和 A0/A1 合成验收样张；样张经实际复核后再实现对应步骤。真实数据与模型运行遵守 A0/B0 的前置门，受控产物仅保存于忽略目录 `outputs/feat-010/`。
