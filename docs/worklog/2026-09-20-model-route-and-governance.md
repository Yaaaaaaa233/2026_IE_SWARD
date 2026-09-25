# 2026-09-20 模型路线提案、模型契约与执行路线图 v2

- 任务／分支／基准提交：MODEL-002｜`work/MODEL-002-route-and-governance`｜`a4211d9`
- 人类责任人：叶安（纪律管理；待登记 owner）
- 执行者与 AI 协助：ZCode agent 执行；叶安提供会议口径、授权与评审输入
- 复核状态与真实复核人：未复核

## 做了什么

- 重写 [执行路线图 v2](../plans/roadmap.md)：顶层设计（端到端管线 Mermaid 图、五条工作线、阶段门槛 S0–S4、接口总览、新 AI 十分钟上手路径、维护规则）。v1（M0–M4 工程视角）并入结构，原文在 Git 历史。
- 新增 [模块登记表](../modules/README.md)：五条工作线 × 模块 × 入口 × 依赖 × 生命周期的单一登记源；未实现模块标 planned，不虚构入口。
- 新增 [模型与评估契约草案 v0.1](../interfaces/model_contract.md)：输入（引用数据契约）、评估协议参数化（lookback/horizon/split 显式配置）、oof 命名规范（A/* 与 B/群/模型）、group_assignments 合成示例、ensemble_config 字段、输出（model_score 与官方三列模板）、守恒断言。
- 新增 [模型路线提案 v0.1](../plans/model-route-proposal.md)：方案 A 八组合 C1–C8、方案 B 分群集成要点、E-B 实验矩阵与择优规则、五个待决策问题。方法层表述，不含本地实验数值。
- 索引与状态最小更新：tasks/plans/interfaces 三处 README 各增索引行（顺带补登 DATACON-001 此前缺失的索引行）、根 README 当前阶段句与目录树、状态页工作线表与日期。
- 本地受控产物（不入库）：模型选型调研报告 V4 与集成方案设计 V1.1（含全部实验数字与图表），保存于模型线本地目录（已通过 .git/info/exclude 忽略，不改共享 .gitignore）。

## 验证

- `python tools/check_governance.py`（working-tree）与 `--scope index`（暂存后）通过。
- `python -m unittest discover -s tests -v` 通过。
- 未运行任何模型或数据实验；本任务为方法与治理层。

## 决策与限制

- 无新增 ADR：路线提案状态为 Proposed，接受与否由团队评审决定；契约草案冻结同样待评审后走 ADR。
- 口径对齐：模型侧评估默认值跟随数据线落地口径（lookback 20 / horizon 40 / 冻结五折）；0918 会议的 40+20 与 350/150 保留为 harness 对照参数，未替团队删除任何口径。
- 数据边界：提案与契约仅含合成示例；V4/V1.1 详细报告的数字不进入公开仓库，其结论在公开文档中只以方向性表述出现。
- 治理修复：TASK-005 直接提交于 main 的《数据清洗版本与规则总说明.docx》此前未被登记，导致任何索引级治理检查失败；本次将其 sha256 登记进 tools/governance_policy.json 的 existing_assets（理由注明"公开复核仍待协调者确认"），仅恢复检查可通过，不构成公开授权追认。该文档含全量实测统计，建议协调者补任务记录并正式复核公开边界；DATACON-001 的索引缺失已由本次补登。

## 接手者下一步

1. 团队评审模型路线提案（重点：C3 默认推荐、Recall 档位、方案 B 群组规则），结论走 ADR。
2. 特征线按契约五表对接消费，F00 冻结后触发模型复测（C1→C3，复测脚本在模型线本地目录）。
3. 评分线依据 model_score 接口启动 B0/Q1 原型；Q0 名称保留给 R1 学习规则分。
4. 协调者确认 MODEL 任务前缀与模块登记表中各线入口的填报责任。
