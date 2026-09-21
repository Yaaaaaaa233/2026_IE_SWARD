# FEAT-003 执行方案：阶段 0——F1 真实运行与配对评估＋双探针

状态：ready（2026-09-22 登记，[ADR-0005](../decisions/ADR-0005-task1-dual-track-and-task2-split.md) 双人竞优·叶安线）
上位方案：[task1-optimization-plan-v1](task1-optimization-plan-v1.md) §4 阶段 0；验收：[task1-optimization-acceptance-v1](task1-optimization-acceptance-v1.md) §5 矩阵 FEAT-003 行。
本文不含真实数据派生统计；运行数字与图只落受控目录（DATA_POLICY）。

## 1. 目标与范围

1. 首次在真实契约数据上运行已入库的 F1 管线（FEAT-002 代码，合成测试已通过），产出 F1_v1 `model_input`；
2. 用统一评估器对冻结基线做同折配对比较，按 ADR-0004 判据得出"过门/未过门"，并做特征族级消融；
3. 双探针为第一轮迭代（FEAT-004-r1，F2 主管线）提供设计输入：陀螺质量审计定 stage 0 规则，对齐可行性定设备系→车辆系方案；
4. 顺带核查 IMU 行数两套口径差异（ADR-0005 善后项）；FEAT-002 的 review 以本任务证据收口。

## 2. 输入与环境

- 代码：仓库 `feature_engineering/`（F1 入口 `src/accident_pipeline_f1/cli.py`，命令 `build-features / build-interface / cluster / run-all`）；配置模板 `configs/pipeline_f1_v1.example.yaml`。
- 数据：本地受控契约目录 `v4_assessed`（events_clean 等）与 IMU 层，路径写进本地配置文件（从模板复制为 `configs/pipeline_f1_v1.local.yaml`，**不入库**——含本机路径）。
- 解释器：仓库外 venv（requirements.txt 同口径＋pytest/PyYAML/pyarrow，已装齐）。
- F1 产物输出目录：特征工程受控目录下**新增子目录** `artifacts/feature_f1_v1/`（管线默认约定；不触碰既有 F0/聚类产物，符合"冻结只读"）。

## 3. 步骤 A：F1 真实运行

1. 复制配置模板为 local 版，填本机数据路径与输出目录；核对 `feature_version=F1_v1`、窗口边界（特征窗 `[as_of−20d, as_of)`、标签窗事件排除）与模板一致。
2. `run-all`（或分步 build-features → build-interface）：扫描 events_clean 产 `f1_event_features.csv`，合成 `model_input.csv`（F1_v1）＋字段字典＋manifest。
3. 硬检查：sample_id 唯一、每车一行、y∈{0,1}、fold 与冻结折分一致、元数据不入特征；行数与车辆数对齐。
4. 指纹登记：F1_v1 `model_input.csv`/schema 的 sha256 落受控台账（进版本登记三件套）。
5. 预期偏差（不作阻塞）：特征窗内金标准事故数低于回放校准门槛，IMU 阈值校准按设计输出 `deferred_insufficient_gold_accidents`。

## 4. 步骤 B：配对评估（统一评估器）

1. 以 F1_v1 `model_input` 为输入跑 `models/task1_baselines.py`（同一冻结协议：折内预处理与调参、种子 42），得候选 OOF 预测，落 `outputs/feat-003/round-1/`。
2. 车辆级配对 bootstrap（2,000 次、95%、种子 42）：候选（RF@F1_v1）对冻结基线（RF@F0_v1，既有受控 OOF）；判据与辅助容差按 ADR-0004；结果（过门/未过门＋区间）登记任务 notes 与受控台账。
3. 特征族消融：F0＋逐族叠加（历史险情族／碰撞预警族／多尺度窗口／暴露归一）各跑一次 OOF，配对区间落受控；定位增益来源，反哺 FEAT-004-r1 取舍。

## 5. 步骤 C：探针一——陀螺质量审计（产物 `outputs/probe/gyro_quality/`）

抽样：按 有/无窗内 IMU × 方向可靠/不可靠 分层抽样（规模以单机可完成为限，登记抽样框）。
统计：陀螺三轴量级分布；尖峰率（单样本超物理合理界）；卡死率（常值游程）；死区窗复核（轨迹 ≥20km/h 而加速度+陀螺方差同近零，C3 判据复刻，与上游 C3 清单可得部分对照）。
行数口径核查：本地 IMU 目录实际行数（按文件统计）对特征线 manifest 总行数；L1 校验报告数字不在本机时，输出差额口径登记"待娄澜侧对照"，不硬下结论。

## 6. 步骤 D：探针二——设备系→车辆系对齐可行性（产物 `outputs/probe/frame_alignment/`）

方法：每车以低通重力方向定垂直轴、行驶主加速度方向定前向轴，构造旋转矩阵并正交化。
检验准则（物理常识，无标签）：对齐后垂直轴均值≈1g 且集中；前向加速度方差显著大于横向；横向均值≈0。**必含方向异常车**（按受控台账清单）与分能源类型分层。
产出：逐车对齐质量指标、通过/失败名单、失败车保留方向无关特征的处置建议——直接决定 FEAT-004-r1 的对齐方案与阈值策略。

## 7. 产物清单（全部受控，不入公开仓库）

`outputs/feat-003/round-1/`（OOF 预测、配对区间、manifest）＋ `outputs/probe/gyro_quality/`、`outputs/probe/frame_alignment/`（统计表与图）＋ 受控台账迭代登记行＋图（L3 用）落 `outputs/figures/feat-003/`。

## 8. 验收落地（对照规程矩阵 FEAT-003 行）

- L1：治理检查、单测、`pytest feature_engineering/tests`、F1 产物重跑一致性（同配置二次运行 manifest 对齐）、schema 硬检查、配对评估落目录与台账。
- L2：抽 10 辆车从 events_clean 人工数历史险情计数对 `f1_prior_*` 特征；探针数据抽样复核（对齐抽 3-5 辆人工看窗段）。
- L3：图组 A（基线诊断复用＋候选对照）、B（F1 新特征分布与单调性）、D（如含与官方事件交叉）。
- 记录：`outputs/acceptance/FEAT-003.md`（受控），任务 JSON 登记 accepted 时只写结论＋复核人＋日期。

## 9. 完成后交接

FEAT-002 review 收口（引用本任务证据）；FEAT-004-r1（F2 主管线）执行方案依据双探针结论成文后登记开工；若 F1 过门，基线棘轮前进并群内知会双线。
