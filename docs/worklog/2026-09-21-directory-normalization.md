# 2026-09-21 目录归位（GOV-005）：根目录裸文件迁入 docs/plans

- 任务：[GOV-005](../tasks/GOV-005.json)
- 分支：`main`（直推，ADR-0003 模式）
- 基准提交：`cd3da2d`
- 人类责任人：叶安（组长，2026-09-21 指出"吴钊同的路线图裸露在根目录"）
- 执行者：ZCode agent
- 复核状态：`review`

## 做了什么

- **两处归位**（`git mv` 保留历史）：
  - `评价指标与基线对照方案（MOE与MOP）.md`（根目录 → `docs/plans/`）——EVAL-001 时期的根目录放置系当时"首页引流"安排（见 2026-09-20 navigation worklog），与 docs/README 权威位置表"路线与方案提议 → plans/"不符；首页 README 的醒目入口保留，仅改路径。
  - `docs/roadmap.md`（roadmap v2 顶层拓扑 → `docs/plans/roadmap.md`）——与总体执行路线图、requirements、模型提案同处 plans/。
- **出站链接改写**：两个搬移文件内部的相对链接全部按新位置重算（roadmap.md 13 组、评估方案 7 处 `docs/` 前缀 → `../`），逐组断言命中数，无漏改误改。
- **入站引用同步**：根 README（3 处含目录树）、ACCEPTANCE、roadmap_overall、plans/README、interfaces/README、ADR-0002、DEVELOPMENT_STATUS、EVAL-001 与 MODEL-002 任务 JSON、三份历史 worklog——共 12 个文件。
- **plans 索引补漏**：评估方案此前不在 plans/README 索引中，补入并注明 R0 接受后升格 interfaces/ 的预期路径。

## 关于"是否预建各线文件夹"

不预建。治理既有原则（docs/README 末条）："空目录按需创建，不提前堆叠"。各线代码目录（特征线 src、评估器、模型线）在 R2/R3 开工时按 DATA_DELIVERY §3 的 L2/L3 结构模板随首批提交建立；文档侧的权威位置（plans/interfaces/cleaning/modules/evidence/tasks/worklog/decisions）已齐备且本次已对齐。

## 验证

- 治理检查 `--scope head` / `--scope index` PASS（含候选文件链接校验）；单元测试 17/17。
- 替换脚本内置命中数断言，任何"期望 N 处实际 M 处"都会中止。

| 维度 | 状态与范围 |
| --- | --- |
| 实现与契约 | 两文件归位、25 组链接修正、索引补漏完成 |
| 实验效果 | `not_run`，纯治理任务 |
| 验收与证据 | 检查器链接校验与断言脚本通过 |
| 环境与依赖 | 无 |

## 决策与限制

- 历史文档（ADR、worklog、已合入任务的 JSON）只改链接路径，不改叙述——路径修正是导航修复，不构成历史改写。
- `赛题资料/`（官方 PDF，DATA_POLICY 固定哈希登记的既有资产）保持在根目录不动。

## 接手者下一步

1. 群里知会吴钊同（评估方案新路径）与王健祺（roadmap v2 新路径）；R0 评审时按新路径引用。
2. 后续任何方案/路线文档直接落 `docs/plans/`，不再进根目录。
