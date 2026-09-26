# FEAT-014 执行入口

方案：[叶安线特征与模型持续迭代探索](../../../docs/plans/feat-014-adaptive-feature-model-exploration.md)。真实数据输入只读；运行产物写入受控 `outputs/feat-014/<run-id>/`，由 `.gitignore` 排除，不进 Git。

执行环境沿用 FEAT-009 v2 的隔离 Python 环境，并通过命令行传入本机路径，不在代码中硬编码成员目录。E0 用 `run.py prepare` 锁定输入；E1 用 `run.py batch` 执行预登记版本；`audit.py` 对 E0/E1 独立检查。E2 由 `diagnose.py` 生成 OOF 误差反馈，`fusion.py` 生成预登记固定融合，并由 `audit_e2.py` 复算。E3 由 `finalize.py` 生成全版本台账、稳定性报告与 D0–D3 图，再由 `audit_e3.py` 独立检查计划指纹、指标、融合重算及同种子复跑。E1 每版保存完整五折 OOF、折内拟合审计、配置、指标、哈希及时间。

本轮实际 run ID 为 `20260926-e0-r2`，25 个新评估版本；E0–E3 独立机器审计通过。复现命令和依赖版本见受控 `input_lock.json`、`environment_reconstruction.json` 与 E3 报告；运行环境和本地数据路径不写入公开仓库。
