# T2R-001：任务二分类总纲 v1 交接

日期：2026-09-24
路线阶段：T2-R0
负责人：周航正
执行者：Codex
状态：待组长人工裁决

## 本次完成

在独立分支 `work/T2-R0-classification-charter` 完成无真实数据的分类规则资产：

1. 建立公开五维、内部六组、32 个目录小类的三层结构。
2. 将 22 个行为／警示编码和 2 个结果编码共 24 个编码唯一映射到主要语义组与公开维度。
3. 为每个小类写明证据来源、一句话边界和启用状态；C25–C32 条件／预留项默认关闭。
4. 写入 D1–D6 推荐意见、M1–M8 歧义裁决表、十条跨事件规则和五条风险链。
5. 将 `11803`、`11804` 固定为结果门，并由机器检查 ADR-0006/0007 的阈值和未来标签窗隔离。
6. 对盲区报警强制检查装配／可观测车群和变道暴露分母，对 IMU 与连续驾驶条件项强制保持关闭。
7. 提供标准库校验器与六个合成单测，其中坏例覆盖重复编码、缺失边界、盲区不可比、预留项误启用和未遂阈值被弱化。

## 交付入口

- 分类总纲：`task2/classification/classification-charter-v1.md`
- 三层一页图：`task2/classification/three-layer-one-page.md`
- 组长裁决表：`task2/classification/leader-review-sheet.md`
- 机器契约：`task2/classification/classification_v1.json`
- 校验器：`task2/classification/validate_classification.py`
- 合成测试：`tests/test_task2_classification_contract.py`

## 验证记录

- `python task2/classification/validate_classification.py`：通过，零条契约错误。
- `python -m unittest tests.test_task2_classification_contract -v`：6 项通过。
- `python -m unittest discover -s tests -v`：36 项通过。
- `python tools/check_governance.py`：任务 ID 修正为仓库允许的 `T2R-001` 后通过。

上述结果只证明公共结构与合成边界可复算，不代表真实效果、评分权重或组长验收。

## 开放项与下一步

路线图引用的“① 8 大类 32 小类”原件当前不在公开仓库。本版根据路线图、24 编码和现有两份调研重建 C01–C32，其中 C25–C32 是明确标记的条件／预留类。组长冻结 D1 前应拿原件逐项对照；若有差异，升版契约并保留变更记录。

组长需要在 `leader-review-sheet.md` 勾选或调整 D1–D6、M1–M8。人工裁决完成前，R1/R4 可以读取本版做接口准备，但不能称为冻结总纲。娄澜负责的 R1/R3 文件未修改；本任务未运行真实数据、未学习扣分权重、未发布效果数字。
