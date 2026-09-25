# FEAT-014 执行入口

方案：[叶安线特征与模型持续迭代探索](../../../docs/plans/feat-014-adaptive-feature-model-exploration.md)。所有真实数据输入只读；运行产物写入受控 `outputs/feat-014/<run-id>/`，不进 Git。

执行环境沿用 FEAT-009 v2 的隔离 Python 环境，并通过命令行传入本机路径，不在代码中硬编码成员目录。先运行 `python3 run.py prepare` 完成 E0 锁定，再用预登记的 `run.py batch` 执行候选。E0 的受控报告与 A0/A1 图位于 run 目录。E1 每版保存完整五折 OOF、折内拟合审计、配置、指标、哈希及时间。

当前程序提供 E0 `prepare` 和通用预登记 `batch`。后续 E2 错误反馈、融合、E3 收口审计将随真实批次补全；没有对应命令实现前，不把它们描述为已完成。
