# 2026-09-24 FEAT-009 新协议特征与模型适配方案入库

- 任务／分支／基准提交：[FEAT-009](../tasks/FEAT-009.json)；`main`（ADR-0003 直推）；`edbd29a`
- 人类责任人、方案复核人：叶安；叶安确认“入库吧”
- 执行者：Codex（仅方案、决策记录与导航更新）；阶段实施 agent 待认领
- 复核范围：方案内容获确认；特征、模型、真实数据与 L2/L3 均未验收

## 做了什么

将待复核草案定为 [FEAT-009 新协议适配方案 v1.0](../plans/feat-009-protocol-v2-adaptation-plan.md)，并以 [ADR-0008](../decisions/ADR-0008-feat-009-protocol-v2-route.md) 记录叶安对 Y0–Y4 路线、依赖、预算和验收门的接受。更新旧 FEAT-009 方案入口、任务记录、方案与决策索引及当前状态。新方案先审计 F0/F1/F3 的画像时间可见性，再重建公共新标签同折输入；评估负责人重新登记 RF，王健祺线自主交付 V5，主对照与候选预登记锁定后才允许一次真实 S1。

本次不改 `feature_engineering/` 或 `models/` 代码，不运行数据或模型，不覆盖旧产物。工作树中其他任务的 `EVAL-002`、`FEAT-008`、`MODEL-005` 复核改动及 FEAT-009 本地建模草稿／依赖改动均未暂存、提交或代为复核。

## 验证

`python3 tools/check_governance.py`：working-tree PASS；`python3 tools/check_governance.py --scope index`：精确暂存后 PASS；`python3 -m unittest discover -s tests -q`：30 项通过；FEAT-009 任务 JSON 解析、新增文档相对链接检查、`git diff --check` 与暂存 diff 检查：通过。上述检查只验证文档与仓库规则，不代表方案中的拟实现检查器或真实模型已经通过。

## 决策与限制

ADR-0006/0007 的公共规则不变；ADR-0008 只接受叶安线工作路线。旧标签／折分下的 S0、RF、V5、F3 数字和过门身份保留历史版本。FEAT-009 维持 `blocked`；公共代理标签待人工复核，EVAL-001 升版、新版基线与画像可见性审计未完成。真实指纹和数据派生图表不入公开仓库。

## 接手者下一步

先读取已入库的 Y0–Y4 方案与 [roadmap 纪律](../plans/roadmap_requirements.md)，声明具体阶段、输入版本与写入范围；从合成 Y0 前置样张及叶安人工确认开始，再进行字段来源审计。Y1 依赖公共标签／折分复核；Y3 不得绕过新版 S0、主对照、预登记和人工放行门。终评使用的 EVAL-003 编号与当前标签任务冲突，进入终评前由协调者另行消除。
