# 模块登记表（单一登记源）

> 本文是全项目工作线与模块的**唯一登记处**：清单、职责、入口、状态。快照日期：2026-09-20。
> 实时任务状态以 [tasks/](../tasks/README.md) 各任务 JSON 为准；本文维护结构，不复制任务状态细节。
> 生命周期取值：`planned / building / runnable / accepted / dropped`（与任务状态独立，"runnable"表示入口可运行，不代表算法验收通过）。
> 0923 公共赛题规则见 [ADR-0006](../decisions/ADR-0006-official-qa-foundation.md)。下表旧标签分层折和 20＋40 评估入口仅说明历史实现状态，不能据此发布新标签下的效果结论；两条算法线自行调整。

## 登记规则

- 新增模块先在此登记一行（工作线、模块名、入口、依赖、生命周期），再开工；入口路径必须真实存在，未实现的写 `planned` 并留空入口。
- 模块完成或废弃时更新生命周期并注明任务编号；不在多份文档重复维护同一清单。
- 受控本地目录（数据、特征、模型产物）只在"产物位置"注明相对约定，不写机器绝对路径。

## 数据线（任务一数据地基）

| 模块 | 职责 | 入口 / 契约 | 依赖 | 生命周期 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 五版清洗管线（v0–v4） | 一阶段单类型标注 + 二阶段跨类型互证 + 处置可信度 | 数据线受控目录（脚本 `10_全量清洗_契约v0.1.py`、`11_二阶段跨类型标注分级.py`） | 官方数据 2.0 | runnable | 规则总说明见 [cleaning/](../cleaning/)；TASK-005 |
| 数据契约五表 | target_vehicles / events_clean / vehicle_day / features / labels + splits / oof | [interfaces/data_contract.md](../interfaces/data_contract.md) | DATACON-001 | review（草案 v0.3） | 公共标签与未遂记录计次语义已校正，新版产物待发布 |
| 旧版冻结折分 | 车辆级按旧标签分层五折（哈希确定） | 契约 splits 表 | 五版清洗 | runnable（历史） | 新标签下不得直接用于判优 |
| 新版代理标签与折分 | ADR-0007 按有效记录计次、车辆级分层五折 | [build_task1_proxy_labels.py](../../tools/build_task1_proxy_labels.py) | 受控 `events_clean`、features 与车辆名单 | runnable（受控产物待复核） | 历史 6/21–7/31 代理窗；非官方 8/1 后标签，见 [证据](../evidence/task1-proxy-label-v2.md) |

## 特征线（特征工程与降维）

| 模块 | 职责 | 入口 / 契约 | 依赖 | 生命周期 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 核心特征层 F00–F04 | 画像合规字段、事件计数、事件段、语义组+平滑率、新近度趋势 | 特征线受控目录（待登记入口） | 契约五表 | planned | 规范见特征线文档（受控） |
| 轨迹增量 T01–T02 | 真实暴露更正、场景与节律 | 同上 | F 系列 + 轨迹清洗 | planned | T1 通过后 T2 才可归因 |
| IMU 增量 I01 | 四道门（可测/合理/新增/有效）+ 动作段 | 同上 | v3/v4 标注与处置列 | planned | 死区窗与缺口日按 v4 处置豁免 |
| 联合与降维 X01/D01 | 跨源联合特征、相关聚类与稳定选择 | 同上 | 各单源通过 | planned | 最终入模列数受控 |
| 事故语义特征管线 F0/F1_v1 | IMU流式聚合、统一模型接口、历史险情、暴露归一、多窗口与聚类 | [feature_engineering/](../../feature_engineering/README.md) | 契约五表 + 冻结折分 | runnable | 公开源码与合成测试；真实产物受控；FEAT-002，待团队复核 |

## 模型线（基线、组合选型、方案 A/B）

| 模块 | 职责 | 入口 / 契约 | 依赖 | 生命周期 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 评估 harness | 同一标签、截点、折分和指标的统一比较入口 | [interfaces/model_contract.md](../interfaces/model_contract.md)（评估契约节） | 新版 labels / splits 待发布 | planned | 旧 20＋40 仅为内部代理回测设计；通用 AUC 方法可复用 |
| 组合方案库 C1–C8 | 八种可执行组合（极简基线→双主力→稳健三角→…→概率结构） | [plans/model-route-proposal.md](../plans/model-route-proposal.md) | harness + F00 | planned | 默认推荐 C3 稳健三角（提案中，待评审） |
| 方案 A 全局组合 | 全局模型 + 折外秩平均融合 | 模型线受控目录（run_ab_test.py A 臂，待建仓） | harness | planned | |
| 方案 B 分群集成 | 三层递进：B0 全局骨架（RF+EBM）→ B1 分群校准 → B2 收缩残差（E-B v2，硬路由弃用） | [models/solution_b/](../../models/solution_b/README.md)（validate.py 一键全流程） | 方案 A 对照 | runnable（v1，开发期自验证） | 群组=特征线聚类 v0；数字见受控报告；MODEL-003 |
| 提交导出 | 官方模板 forecast_result.csv 三列 + 内部溯源版 | models/solution_b/final_outputs.py | 终局模型冻结 | building（v1 已产开发期版本，终局冻结待 R6） | accident=固定名额前 20%（预登记政策） |

## 评分线（任务二）

| 模块 | 职责 | 入口 / 契约 | 依赖 | 生命周期 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 双基线评分 B0/Q1 | 三维度等权规则评分 / 概率转换分 | 待登记 | model_score | planned | B0 为透明弱基线；Q1 消费模型线 model_score |
| 五维子分与扣分制 | 疲劳/分心/激进/盲区车距/历史险情 + IV 定权 | 待登记 | 特征底座 | planned | |
| 动态更新与运营包 | EWMA 日/周/月更新、分档排名奖惩培训四件套 | 待登记 | 评分冻结 | planned | |

## 治理线

| 模块 | 职责 | 入口 | 生命周期 | 备注 |
| --- | --- | --- | --- | --- |
| 治理检查器 | 禁入路径/类型/凭据/索引检查 | [tools/check_governance.py](../../tools/check_governance.py) | accepted | GOV-001 |
| 治理自测 | 合成内容的负向与快照测试 | [tests/test_governance.py](../../tests/test_governance.py) | accepted | |
| CI | PR 上自动跑检查 | [.github/workflows/governance.yml](../../.github/workflows/governance.yml) | accepted | |
