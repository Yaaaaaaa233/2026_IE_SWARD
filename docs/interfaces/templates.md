# 接口模板（合成示例）

本文给出 [数据契约](data_contract.md) 五张表及两张配套表的**格式示例**。全部字段值均为**合成数据**，不代表任何真实车辆、真实统计或实验结果；字段类型与语义以契约为准。

## 1. target_vehicles —— 一行一台目标车辆

主键 `gpsno`；500 台目标车全部保留，缺源以标记表达，不因缺数据减少行。

```csv
gpsno,energy_type,monthly_avg_mileage,monthly_avg_hours,monthly_avg_stops,highway_ratio,profile_status,source_version
DEMO0001,electric,8300.50,150.20,64.30,0.1210,reference_only,2.0
DEMO0002,diesel,5200.75,98.60,51.05,0.0873,reference_only,2.0
```

## 2. events_clean —— 一行一条清洗后事件

主键 `row_id`；坐标或速度无效时仅字段置空并打标记，整行保留。

```csv
row_id,gpsno,event_type,event_name,start_time,speed,lat,lng,coord_flag,speed_flag,source_version
EVT-000001,DEMO0001,30017,lane_occupation,2026-06-05 09:12:30,56,30.52,114.85,ok,ok,2.0
EVT-000002,DEMO0002,60292,blind_zone_rear,2026-06-07 21:40:11,,,zero_placeholder,zero_placeholder,ok,2.0
EVT-000003,DEMO0001,11402,curve_overspeed,2026-06-24 02:45:27,,43.72,87.49,ok,impossible_gt200,2.0
```

## 3. vehicle_day —— 一行一车一天

主键 `gpsno + date`；传感器字段在未全量计算前保留空值并说明，不用 0 冒充。

```csv
gpsno,date,evt_raw_count,evt_segments30,evt_days_flag,traj_km,traj_hours,traj_observed_seconds,imu_observed_seconds,imu_traj_overlap_seconds,quality_flags,source_version
DEMO0001,2026-06-05,121,96,1,,,,,not_observed,2.0
DEMO0002,2026-06-06,0,0,0,182.40,7.15,25740,,partial_coverage,2.0
```

## 4. features —— 一行一车一个历史窗口

主键 `sample_id`；特征列随 feature_version 增长，此处仅示意命名与窗口字段。

```csv
sample_id,gpsno,as_of,lookback_days,evt_fatigue_count,evt_distraction_count,evt_lane_count,event_observed_days,traj_hours,imu_coverage_pct,feature_version
DEMO0001_20260711_40d,DEMO0001,2026-07-11,40,12,305,1440,38,,pending,F0_event_v2
DEMO0002_20260711_40d,DEMO0002,2026-07-11,40,3,88,402,60,41.30,pending,F0_event_v2
```

## 5. labels —— 一行一车一个未来窗口

主键 `sample_id`，与 features 一对一；`y` 的定义与 `label_status` 口径变更须记录决策。

```csv
sample_id,gpsno,label_window,horizon_days,y,label_status,label_version
DEMO0001_20260711_40d,DEMO0001,2026-07-11|2026-07-31,20,1,log_complete,2.0
DEMO0002_20260711_40d,DEMO0002,2026-07-11|2026-07-31,20,0,log_complete,2.0
```

## 6. splits —— 车辆级折分

同车所有窗口在同一折；折分文件冻结前，不同实验的成绩不可比较。

```csv
gpsno,fold,split_version
DEMO0001,0,stratified5_v1
DEMO0002,1,stratified5_v1
```

## 7. oof_predictions —— 折外预测统一输出

一行一个样本一次模型的折外预测；真实预测文件按 [数据边界](../DATA_POLICY.md) 保存在受控目录，不入公开仓库。

```csv
sample_id,gpsno,fold,y,prob,model,source_version,feature_version
DEMO0001_20260711_40d,DEMO0001,0,1,0.42,baseline_L2LR,2.0,F0_event_v2
DEMO0002_20260711_40d,DEMO0002,1,0,0.11,baseline_L2LR,2.0,F0_event_v2
```

## 使用说明

- 真实数据管线以契约为准生成同名结构的文件；本页示例可直接用于合成数据测试。
- 字段命名在冻结前可能调整；调整时新增 [决策](../decisions/README.md) 并在本文更新版本号。
