# 模型与评估契约草案 v0.1（未冻结）

状态：草案（draft），随任务 [MODEL-002](../tasks/MODEL-002.json) 提交评审；冻结与修订走[决策流程](../decisions/README.md)。
本文只定义模型线的输入、评估协议、输出与命名规则；字段示例全部为**合成数据**。上游数据语义以[数据契约](data_contract.md)与 [ADR-0006](../decisions/ADR-0006-official-qa-foundation.md) 为准。下列旧窗口与折分默认值属历史代理回测配置，不能作为 0923 后的当前评估基座直接使用。

## 1. 输入（模型线消费什么）

| 输入 | 来源 | 约定 |
| --- | --- | --- |
| features（*_v1_raw / *_v2_filled） | 数据契约 | 每个模型/组合声明所用版本，比较实验内不得混用两版 |
| labels | 数据契约 | y 与 label_status 原样消费；标签口径变更走 ADR |
| splits | 数据契约（旧版已冻结） | 同一比较共用车辆级折分；旧折分依旧标签分层，不能直接作为新版判优折分 |
| 群组键标记（方案 B） | v1 标注版标记信息 | 盲区硬件等设备标记由数据线口径维护，模型线不自行从原始计数推断 |

## 2. 评估协议（harness 配置）

| 项 | 默认值 | 说明 |
| --- | --- | --- |
| lookback_days | 20（旧代理回测） | 官方允许 6/1–7/31 共 61 天历史；具体模型特征长度由各线验证，不由本契约规定 |
| horizon_days | 40 | 标签窗长度；与数据线落地口径一致 |
| split | 旧契约 splits 冻结版（历史） | 新标签下须发布版本化新折分；车辆级分组和同折对照原则不变 |
| 主指标 | ROC AUC | 折外（OOF）口径，各折与整体并报 |
| 辅指标 | Recall@K（默认 K=100/150）、AP、Brier | K 按新基率重估，见提案决策项 |
| 拟合边界 | 一切学习型处理折内拟合 | 平滑先验、标准化、校准、路由权重均不得跨折 |
| 配置载体 | experiments 配置文件显式携带上述参数 | 禁止硬编码窗口与折分 |

## 3. 折外预测与命名（oof_predictions 扩展）

沿用数据契约 oof_predictions 表结构，`model` 字段命名规范：

- 方案 A：`A/<组合成员>`（如 `A/enet`、`A/ebm`、`A/rf`、`A/rankavg3`）
- 方案 B：`B/<群组>/<模型>`（如 `B/G-high/enet`、`B/global/enet`）、路由输出 `B/router`
- 每条记录必带 source_version、feature_version；方案 B 另带 group_rule_version。

## 4. 群组分派表 group_assignments（方案 B 专用，合成示例）

```csv
gpsno,as_of,group_id,group_rule_version,assignment_method,fit_fold,fallback_flag,source_version,feature_version
DEMO0001,2026-07-11,G-high,group_rule_v1,rule,,false,2.0,F0_event_v2
DEMO0002,2026-07-11,G-radar,group_rule_v1,rule,,false,2.0,F0_event_v2
DEMO0003,2026-07-11,G-global,group_rule_v1,rule,,true,2.0,F0_event_v2
```

群组规则（v1 提案）：G-radar=有盲区硬件标记；G-low/G-high=无标记且月均里程低/高于中位数；G-global=兜底（含小群回退）。最小群规模 80 台，跌破即并入 G-global；聚类分派时 KMeans 参数必须折内拟合并记录 fit_fold。

## 5. 组合配置 ensemble_config（JSON，随实验冻结）

关键字段：`scheme_id`、`group_rule`（键/阈值/最小群）、`sub_models`（每群模型与冻结超参）、`calibration`（per-group sigmoid，仅在群内折外预测上拟合）、`router`（R1 硬路由默认；R2a 元学习器 / R2b 误差倒数加权需满足启用条件）、`feature_set`、`split_set`、`threshold`（accident 列阈值规则与冻结值）。

## 6. 输出（模型线交付什么）

| 输出 | 消费方 | 格式 |
| --- | --- | --- |
| model_score | 评分线（任务二 Q1 概率转换分） | gpsno, prob（折外校准后）, rank_pct, calib_flag |
| forecast_result.csv | 组委会（官方模板） | gpsno, accident, risk_prob 三列且按此顺序；500 行全覆盖、概率 [0,1]、无重复；线上 AUC 只用 risk_prob，accident 仅作格式检查 |
| 内部溯源版 | 复核 | 附 fold、model_version、feature_version、group_id 列，仅存受控目录 |

accident 阈值规则：在验证折上按 Recall 目标（预警名单规模）反推并冻结；选择依据写入说明文档，不使用 0.5 默认值。

## 7. 守恒断言

- 输出行数 = target_vehicles 行数；gpsno 集合完全一致；无缺失无重复。
- 概率全部落在 [0,1]；同版本配置两次运行输出一致（随机种子冻结）。
- 方案 B 输出可追溯：任一车辆可回查群组分派、子模型版本与校准参数。

## 8. 开放问题

- Recall@K 的 K 值档位按新基率重估（列入提案决策项）。
- 方案 B 群组规则在数据 2.0 上的复核结论未出，group_rule_v1 视为提案。
- 本契约冻结需配合模型路线提案的评审结论，走 ADR。
