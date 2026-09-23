# 数据契约草案 v0.3（公共语义及未遂计数单位已校正，未冻结）

状态：草案（draft），待团队评审后冻结。冻结与修订流程见 [决策入口](../decisions/README.md)。

来源：任务一数据线（[DATACON-001](../tasks/DATACON-001.json)）；0923 公共规则按 [ADR-0006](../decisions/ADR-0006-official-qa-foundation.md) 校正，未遂计数单位按团队 [ADR-0007](../decisions/ADR-0007-near-miss-count-unit.md) 裁定。本文只定义字段、类型、语义与规则层方法；全部示例值为**合成数据**，仅示意格式。含真实统计的详细分析报告保存于受控本地目录，按 [数据边界](../DATA_POLICY.md) 不入公开仓库。新版标签与折分文件尚未发布。

## 1. 通用语义

| 项 | 约定 |
| --- | --- |
| 车辆主键 | `gpsno`，一律按字符串处理；明细表中 gpsno 重复属正常，车辆主表中唯一 |
| 设备串号 | `imei` 仅作设备关联审计字段，不入模型输入 |
| 时间 | 原始数据无时区标记；统一按本地业务时间解析，假设写入配置；格式 `YYYY-MM-DD HH:MM:SS` |
| 窗口 | 一律左闭右开；特征事件时间严格早于预测截点 `as_of`，标签事件时间 ≥ `as_of` |
| 官方时段 | 夜间 `[21:00, 次日 06:00)`；早晨 `[06:00, 09:00)`；黄昏 `[18:00, 21:00)`；“深夜”如另设须独立命名与登记 |
| 官方时间轴 | 历史可用 2026-06-01 至 2026-07-31（61 天）；2026-08-01 起预测未来 40 天。旧 20＋40 仅属内部代理回测 |
| 缺失 | 空串与 `\N` 统一为 null；0 本身不是缺失值；缺失必须伴随 `_flag` 标记列 |
| 版本 | 每张表带 `source_version`（原始数据版本）；实验另记 `cleaning_version / feature_version / split_version` |
| 守恒断言 | 坐标无效只置字段缺失，不得删行删车；标签行数与正类车数在清洗前后必须相等 |

## 2. 表清单与主键

| 表 | 一行 = | 主键 |
| --- | --- | --- |
| `target_vehicles` | 一台目标车辆 | `gpsno` |
| `events_clean` | 一条清洗后事件 | `row_id` |
| `vehicle_day` | 一车一天 | `gpsno` + `date` |
| `features` | 一车一个历史窗口 | `sample_id` |
| `labels` | 一车一个未来窗口 | `sample_id`（与 features 一对一） |

配套约定表：`splits`（车辆级折分，同车所有窗口同折）与 `oof_predictions`（折外预测统一输出）。

## 3. 字段契约

### 3.1 target_vehicles

| 字段 | 类型 | 单位 | 说明 |
| --- | --- | --- | --- |
| gpsno | string | — | 车辆关联键 |
| energy_type | category | — | 能源类型；低频类别不单独建模 |
| monthly_avg_mileage | float | km | 画像月均里程；画像近半年轨迹统计期为 2026-02-01 至 2026-07-31，早期回测需核可见性 |
| monthly_avg_hours | float | 小时 | 画像月均时长；同上 |
| monthly_avg_stops | float | 次 | 画像月均停留次数；同上 |
| highway_ratio 等 5 类占比 | float | 0–1 小数 | 不再除 100；范围校验为硬断言 |

### 3.2 events_clean

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| row_id | string | 清洗层唯一行号 |
| gpsno | string | 车辆关联键 |
| event_type | int | 官方事件编码；编码无强度含义 |
| event_name | string | 保留原始名称，另按编码映射标准名 |
| start_time | datetime | 事件开始时间 |
| speed | float | km/h；负值无效置空；超 200 视为物理不可能置空并标记 |
| lat / lng | float | 度；越界或占位值置空；坐标系未确认前不与轨迹坐标互算 |
| coord_flag / speed_flag | category | 见第 4 节枚举 |
| source_version | string | 原始数据版本 |

### 3.3 vehicle_day

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| gpsno + date | — | 主键 |
| evt_raw_count / evt_segments30 / evt_days_flag | int | 事件三种计数形态并存；段合并仅作特征候选，不用于标签 |
| traj_km / traj_hours | float | 窗口内真实公里（distance 厘米÷1e5）与驾驶小时（run_time 增量秒÷3600） |
| traj_observed_seconds / imu_observed_seconds | int | 各源当日有效观测秒数；不等于在线时长 |
| imu_traj_overlap_seconds | int | 两源有效区间交集，动作率分母的候选口径 |
| quality_flags | string | 当日质量标记集合 |

### 3.4 features

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| sample_id | string | `gpsno_asof_lookback` 组合，唯一 |
| gpsno / as_of / lookback_days | — | 窗口定义；窗口长度训练与推理必须一致 |
| evt_* | float/int | 事件特征，前缀 `evt_`；长尾计数同时提供原值与 log1p 变换列 |
| traj_* / imu_* / profile_* | — | 各源特征按来源前缀；画像字段在统计期确认前默认关闭 |
| feature_version | string | 特征规则版本 |

### 3.5 labels

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| sample_id | string | 与 features 一对一 |
| label_window | string | 未来窗起止（左闭右开） |
| horizon_days | int | 预测跨度；窗口口径变更须记录决策 |
| y | int | 1 = 窗口内 `11803` 事故记录至少 1 条，或 `11804` 未遂有效记录至少 2 条；否则 0（日志完整假设下）。`11804` 同日不同记录逐条计次，精确重复行去重后只计一次；见 ADR-0007 |
| label_status | category | 日志完整假设确认前使用枚举；见第 4 节 |
| label_version | string | 标签定义版本 |

## 4. 标记枚举

- `coord_flag`：`ok / missing / zero_placeholder / out_of_range`
- `speed_flag`：`ok / missing / negative_invalid / impossible_gt200 / suspicious_gt160`
- `quality_flags`（vehicle_day）：`not_observed / partial_coverage / cross_day_boundary` 等，分号分隔
- `label_status`：`log_complete / unknown / proxy_negative`（口径变更须记录决策并标注受影响实验）
- `imu_calibration_status`：`calibrated / failed / pending`；校准失败车辆仅使用非方向性统计

## 5. 双版输出规约

| 版本 | 内容 | 用途 |
| --- | --- | --- |
| `*_v1_raw` | 保留 null 与全部标记列 | 供原生支持缺失的模型与审计 |
| `*_v2_filled` | 对 null 按规则填充（如分组中位数），保留标记列 | 供需要完整矩阵的模型 |

两版同主键、同口径，仅缺失处理不同；比较实验必须声明使用哪一版。

## 6. 清洗规则层（方法，不含数据结论）

1. 名单以画像车辆全集为输出骨架，缺源车辆不删除、以标记表达。
2. 精确重复行删除；逻辑矛盾字段（如物理不可能的速度）置空加标记，不删行。
3. 重复报警同时保留"原始条数 / 时间段合并数 / 出现天数"三种形态；未遂标签按 ADR-0007 对精确去重后的有效 `11804` 记录逐条计次，不使用 30 秒并段数或发生天数。坐标／速度异常只标记字段，不因此删去标签事件。
4. 轨迹驾驶小时采用增量秒数求和估计，须先排序、去重、处理乱序与跨分片，并以相邻时间差交叉验证；停车心跳不计入里程与时长。
5. IMU 逐车用静止段估计重力方向做坐标对齐、扣除角速度零偏；动作判定阈值人为约定并做敏感性，不交由模型学习。
6. 学习型处理（填补、标准化、平滑、降维）参数只在训练折内估计，预测端复用。

## 7. 示例

合成示例见 [templates.md](templates.md)。示例值全部为构造数据，不代表任何真实车辆或真实统计。

## 8. 开放问题

日期/标签口径等未确认事项以 [状态页](../DEVELOPMENT_STATUS.md) 的待确认清单为准，本文不重复维护数值性结论。
