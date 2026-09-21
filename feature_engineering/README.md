# 特征工程公开实现（F0 / F1_v1）

本目录保存可公开、可复查的特征工程方法与合成测试。真实数据、真实特征表、标签、聚类分派、模型文件和实验报告不进入公开仓库；它们按项目的数据政策保存在受控目录。

当前实现包括：

- IMU分块读取和事故语义聚合。
- 设备安装方向可靠性处理。
- 事件、轨迹、车辆属性、覆盖质量和IMU特征的统一接口。
- 标签与折号隔离、防泄漏时间窗口检查。
- 数据不足样本隔离和多源特征聚类。
- F1历史险情、暴露归一、5/10/20天窗口及新近度特征。
- 完全重复特征检测和逐列字段字典。

## 本地安装

```sh
python -m pip install -e "./feature_engineering[test]"
```

复制公开模板为本地配置，并在本机填写数据位置。真实本地配置应保持未跟踪：

```sh
cp feature_engineering/configs/pipeline.example.yaml feature_engineering/configs/pipeline.local.yaml
cp feature_engineering/configs/pipeline_f1_v1.example.yaml feature_engineering/configs/pipeline_f1_v1.local.yaml
```

Windows PowerShell 可使用 `Copy-Item` 完成同样操作。

## 运行

```sh
accident-pipeline --config feature_engineering/configs/pipeline.local.yaml run-all
accident-pipeline-f1 --config feature_engineering/configs/pipeline_f1_v1.local.yaml run-all
```

也可以按 `extract-imu`、`build-interface`、`cluster` 或 `build-features` 分步运行。正式运行前必须核对数据契约、时间窗口、折分版本和输出位置。

## 测试

```sh
python -m pytest feature_engineering/tests -q
```

测试只构造合成车辆、事件和传感器数据，不读取比赛数据。根目录的治理CI没有安装本模块依赖，因此特征测试作为模块级检查单独运行。

## 结论边界

- 代码可运行不等于特征已经通过模型验收。
- IMU阈值必须使用预测时点前可见、经过核验的事故时刻校准。
- 若金标准事故数量低于配置门槛，程序只输出暂缓结论，不应使用标签窗口事件调参。
- 单特征监督筛选必须放在训练折内部，不能先看全量标签再复用同一折评估。
- F1相对F0的真实增益需要在冻结折分上做配对消融后确认。

公开方法说明见 [F1方法边界](../docs/plans/feature-engineering-f1-method.md)，接口语义以项目公共数据和模型契约为准。
