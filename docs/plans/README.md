# 路线讨论入口

**状态：0923 官方答疑公共规则见 [ADR-0006](../decisions/ADR-0006-official-qa-foundation.md)，团队未遂计数单位见 [ADR-0007](../decisions/ADR-0007-near-miss-count-unit.md)；旧任务一效果结论冻结，原算法路线图中旧口径仅作历史设计。**

- [0923 答疑后项目公共口径调整](2026-09-23-official-qa-adjustment-plan.md)（GOV-008，待复核）：统一标签、时间、时段、评估和交付口径；不指定两条算法线的优化及重跑顺序。

- [总体执行路线图](roadmap_overall.md)：R0–R6 阶段、汇合门槛、红线、三段交接面接口、执行纪律——项目的**执行基线**；配套受控数值页 `roadmap_overall_numbers.md`（git 忽略，不入库）。
- [Roadmap 编写与验收要求](roadmap_requirements.md)：每步必备字段（方案/机器验收/可视化验收三文档）、AI 开工协议、状态回填纪律——各线细化 roadmap 的格式标准。
- [全项目执行路线图 v2](roadmap.md)：**顶层路线入口**（端到端管线、五条工作线、阶段门槛 S0–S4、接口总览、模块登记链接）；与总体执行路线图的分工见其 §7 待拍板第 8 项。
- [特征工程当前工作、接口状态与路线图 v0.1](feature-engineering-roadmap.md)（任务 [FEAT-001](../tasks/FEAT-001.json)，待复核）：特征线本地实现盘点、A/B/C 三段接口状态、问题边界及 F-R0～F-R4 执行路线；不含真实数据派生统计和模型产物。
- [任务二执行路线图 v0.2](task2-roadmap.md)：行为大类→扣分规则→可运营/动态微调→白盒黑盒融合→MOE（不冻结）；0923 已校正夜间命名及无固定 CSV 模板的交付口径，执行=任务二三组（ADR-0005）
- [模型路线提案 v0.1](model-route-proposal.md)（任务 [MODEL-002](../tasks/MODEL-002.json)，评审中）：方案 A 八种组合方案库 C1–C8 与方案 B 分群集成设计、E-B 实验矩阵与择优规则、待决策问题清单。不含本地实验数值。
- [FEAT-009 新口径下叶安线特征与模型适配执行方案](feat-009-protocol-v2-adaptation-plan.md)（v1.0，[ADR-0008](../decisions/ADR-0008-feat-009-protocol-v2-route.md) 接受方案）：Y0–Y4 技术运行已完成，[FEAT-009](../tasks/FEAT-009.json) 处于 `review`，真实人工复核待完成。
- [FEAT-010 最终历史窗口适配与观测量依赖检验方案](feat-010-final-window-and-observation-audit-plan.md)（v1.0，[ADR-0009](../decisions/ADR-0009-feat-010-final-window-and-observation-audit.md) 仅接受方案）：A0/A1 技术运行和机器检查完成，真实来源／图表复核待完成；B0/B1 仍待前置责任人复核，不依赖王健祺线新版 V5。
- [FEAT-011 既有 F3 特征的小规模建模适配方案](feat-011-small-attribution-model-adaptation-plan.md)（v1.0，[ADR-0010](../decisions/ADR-0010-feat-011-small-attribution-model-adaptation.md) 仅接受方案）：S0 只读审计和机器检查已完成，人工来源／图表复核待完成；S1–S3 未运行，真实 OOF 仍待前置复核门。
- [FEAT-012 新版 F3 特征与算法多轮迭代优化](feat-012-iterative-feature-model-optimization.md)：叶安线任务内开发期探索；按用户授权直接启动，不等待 EVAL-001 全项目统一复核；上限 19 个版本，结果不改变公共评价基座。
- [FEAT-013 FEAT-012 后续 EBM 稳定性与融合小迭代](feat-013-ebm-stability-and-fusion-followup.md)：补做三种子、EBM 平滑与固定权重融合对照；上限 17 版，结果仍为开发期探索。
- [FEAT-014 叶安线特征与模型持续迭代探索](feat-014-adaptive-feature-model-exploration.md)：按 2026-09-26 用户指示启动；E0、B001–B003 的独立审计和融合诊断机器检查通过，下一步按方案执行稳定性检查；真实指标和图表仅在受控 `outputs/feat-014/`，实际图表复核待记录。
- [FEAT-009 叶安线已有特征建模优化方案](feat-009-feature-modeling-proposal.md)（旧协议设计存档）：旧 S0 复核只保留历史身份，不能放行新版 S1。
- [评价指标与基线对照方案 v1.3](评价指标与基线对照方案（MOE与MOP）.md)：公共规则按 0923 答疑及团队记录计次裁定校正；v1.3 已登记 FEAT-009 新协议技术锚点，但全项目正式评价基座仍待责任人复核。

各线细化 roadmap（DATACON 线等）陆续按上述要求补齐后在此登记。历史路线草案（v1，M0–M4 工程视角）已并入路线图 v2 与本入口，原文见 Git 历史。

接受路线时通过 [决策记录](../decisions/README.md) 写明：决策人、日期、路线版本、依赖、阶段门槛、资源预算与未决口径，再更新状态页。含数据统计的详细本地讨论稿不通过公开 Git 同步。
