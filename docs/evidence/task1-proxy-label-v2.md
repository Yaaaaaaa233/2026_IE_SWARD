# 任务一 ADR-0007 代理标签与折分：公开验收元信息

- 任务：[EVAL-003](../tasks/EVAL-003.json)
- 规则：[ADR-0006](../decisions/ADR-0006-official-qa-foundation.md)＋[ADR-0007](../decisions/ADR-0007-near-miss-count-unit.md)
- 生成器：[build_task1_proxy_labels.py](../../tools/build_task1_proxy_labels.py)
- 文件身份：`label_v2_record_count_20260923`、`split_v2_record_strat5_seed42`
- 范围：既有历史数据的内部代理回测 `[2026-06-21, 2026-07-31)`；不是官方 8/1 后的保留标签
- 本机受控产物：`outputs/protocol_20260923/label_v2_record_proxy/`（Git 忽略，不随仓库分发）
- 可视化验收：同目录 `acceptance.png`，显示旧／新正类迁移与新版五折双类分布；真实派生图只在本机受控查看
- 状态：实现者完成生成与自查，待数据／评估负责人复核；未运行模型

## 已完成的检查

从受控 2.0 版 `events_clean.csv`、`features.csv`、`target_vehicles.csv` 生成新的标签、五折与 manifest，原 `labels.csv`、`splits.csv` 保持不变。检查 500 车 `sample_id` 对齐、旧负类不变为新正类、新版正类仅为旧正类子集、五折均有正负样本、输出校验和与 manifest 一致；另核对源版本防混用。合成测试覆盖同日两条不同未遂记录、精确重复记录、单条未遂、事故、右边界排除及折分复现。

## 尚未证明

本次只证明标签生成规则与受控源表的对应，不证明这些标签与现有 F1/F3 模型输入已对齐，更不证明旧基线或 V5 在新标签下的效果。旧 6/21 回测所用画像可能晚于预测截点，需另做特征可见性检查；7/31 的源数据此前被隔离，正式 8/1 预测所需输入也未在此任务中恢复。真实类别数、逐车变化表和全部输入／输出指纹保留在受控 manifest 与源目录，不进入公开仓库。
