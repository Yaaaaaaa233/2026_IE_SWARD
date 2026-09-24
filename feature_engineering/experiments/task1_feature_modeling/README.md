# FEAT-009 输入审计与建模实验

本目录实现 FEAT-009 的受控输入审计和后续建模入口。真实配置、输入指纹、审计数据、预测与图表只写入被忽略的本机配置和 `outputs/feat-009/`。

## S0 输入锁定与无泄漏审计

准备本机配置 `configs/feat-009.local.toml`（格式参照 [config.example.toml](config.example.toml)），然后运行：

```sh
python feature_engineering/experiments/task1_feature_modeling/audit.py --config configs/feat-009.local.toml
```

审计器只读 F1、F3R5、V5 OOF、冻结 RF OOF 和其清单，向受控输出目录写 `manifest.json`、`audit.json`、`issues.json`、一页 `acceptance.md` 与图表。真实指纹和统计只留本地。

### S0 图表预览与建议通过标准

- **输入流向与时间窗图**：显示 F1、F3、V5 OOF、RF OOF 汇入逐车主键／标签／折号对账，并标出特征窗 `[as_of-lookback, as_of)` 与标签窗 `[as_of, as_of+horizon)`。通过标准是四份输入主键唯一且逐车 `gpsno/y/fold` 完全相同，F1/F3 的 `as_of/lookback` 一致，特征时间严格早于 `as_of`；任何差异未解释则停止。
- **特征族与缺失图**：F1/F3 使用同一分类规则展示各族列数及缺失率，并列出未分类列。图只描述输入覆盖，不以缺失率本身判模型效果；缺失率达到 50% 的族或 F1/F3 差异明显的族须在报告中给出字段语义解释。
- **拟合边界表**：逐项列出时间窗提取、全量筛列、同群变换、V5 分群、模型和校准的输入范围与标签使用情况。目标标签进入特征构造、未来时间字段进入预测特征或折内拟合范围无法确认，均阻止 S1；仅使用无标签全量特征的操作标记为转导式，并由负责人决定是否接受该结论边界。

### 当前边界

S0 只审计输入与数据拟合范围，不运行候选模型或计算 FEAT-009 主效果判定。每次输出的本地 `acceptance.md` 会单列机器检查、实验效果、证据状态和环境状态。进入 S1 还须满足[方案](../../../docs/plans/feat-009-feature-modeling-proposal.md)中的放行条件。

## S1 隔离环境

四个 EBM／LightGBM 候选和图表生成依赖按[任务专用 requirements](requirements.txt)固定，避免改动项目共享测试环境。版本与当前本机已核对的 EBM 技术栈一致，LightGBM 固定为 `4.7.0`；真实运行 manifest 仍须登记实际 Python、完整包版本、硬件、种子与配置。

在 Git checkout 外创建独立环境，然后执行：

```sh
uv venv --python 3.11 <env>
uv pip install --python <env>/bin/python -r feature_engineering/experiments/task1_feature_modeling/requirements.txt
```

macOS 上的 PyPI LightGBM wheel 还需要 OpenMP。此 Apple Silicon 主机使用 conda-forge 的 `llvm-openmp=23.1.1`，运行库定义见[平台依赖文件](environment-macos-arm64.yml)；调用任务环境中的 Python 时，将该运行库的 `lib` 目录加入 `DYLD_LIBRARY_PATH`。其他 macOS 主机可按 [LightGBM 安装指南](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)使用 Homebrew 的 `libomp`。

## 新协议 v2 执行入口

用户于 2026-09-24 授权按 [FEAT-009 新协议方案](../../../docs/plans/feat-009-protocol-v2-adaptation-plan.md)连续执行。新版流程使用本目录的 `prepare_protocol_v2.py`、`run_protocol_v2.py`、`audit_protocol_v2_run.py` 和 `diagnose_protocol_v2.py`；旧 `run.py` 保留旧 S1 前置门，不是本轮入口。状态、边界与人工待复核项见[执行交接](../../../docs/worklog/2026-09-24-feat-009-protocol-v2-execution.md)。

先按本机配置生成隔离输入并重训新版 RF 锚点：

```sh
python feature_engineering/experiments/task1_feature_modeling/prepare_protocol_v2.py
python models/task1_baselines.py --feature-root "../特征工程" \
  --model-input outputs/feat-009/v2/y1/f0_model_input.csv \
  --output-dir outputs/feat-009/v2/y2/new_rf_anchor --seed 42 --inner-folds 3
```

查看受控 `outputs/feat-009/v2/y2/pre_registration.json` 中锁定的标签、折分、主对照、候选池与门槛后，运行一次嵌套 OOF 并复核：

```sh
DYLD_LIBRARY_PATH="<llvm-openmp-lib-dir>" PYTHONPATH=feature_engineering/experiments/task1_feature_modeling \
  <env>/bin/python feature_engineering/experiments/task1_feature_modeling/run_protocol_v2.py \
  --confirm-preregistration --output outputs/feat-009/v2/y3/<run-id>
<env>/bin/python feature_engineering/experiments/task1_feature_modeling/audit_protocol_v2_run.py \
  --run-dir outputs/feat-009/v2/y3/<run-id>
<env>/bin/python feature_engineering/experiments/task1_feature_modeling/diagnose_protocol_v2.py \
  --run-dir outputs/feat-009/v2/y3/<run-id>
```

输出目录在 `.gitignore` 覆盖范围内。不要把本机输入、标签、预测、统计、图表或指纹加入 Git。当前正式比较仍是开发期代理回测；标签人工复核、字段来源抽查、任务二 V5 新口径 OOF 和最终 61 天训练另有边界，不由本轮 OOF 自动满足。
