# 任务一终选与正式交付首轮交接（2026-09-27）

- 任务：[TASK-008](../tasks/TASK-008.json)；责任人叶安，执行者 Codex，状态 `active`。叶安指示停止刷分，转为跨线合理性终选与全历史输入正式推理。
- 审查入口：[跨线终选审查](../reviews/task1-final-selection.md)；格式适配入口：[任务一正式提交适配器](../../models/task1_final/README.md)。真实数据与运行结果不入公开 Git。

## 已完成

读取公共口径、两线任务和交接、经验材料及本机受控 FEAT-014 E3 报告。核对后，历史泄漏版本排除；叶安 FEAT-014 候选虽完成独立机器审计，开发集配对区间仍跨零；王健祺 g10 是最高开发期点分，但多轮同集筛选且相对 g8v6 的配对区间跨零；g8v6 泄漏已修正、结构较简单、已报告的窗口检验变化较小。因此暂定 g8v6-corrected 为首选移植对象，g10 作备选，发布须经逐车与真实推理核验。吴钊同为评估线，没有另一条独立任务一新模型。

核对仓库现有脚本：`models/solution_b_v5/run_v5.py` 输出的 `forecast_result.csv` 来自开发期 OOF；`models/sprint_champion_v2/reproduce_champion.py` 仍重建旧 g4v5-B2 的 OOF 融合。两者都不能作为 8 月 1 日正式预测。FEAT-010 受控 A1 是 61 天无标签输入审计候选，不覆盖所审查候选的全部训练特征；事件／车辆日来源对 7 月 31 日仍有待解释的完整性边界。

新增 `models/task1_final/prepare_submission.py`，接收未来预测、固定 500 车清单和来源 manifest；检查未来截点、列序、车集合、概率、预测哈希，输出官方三列 UTF-8 无 BOM/LF 的 CSV 和受控回执。五项合成测试包含 OOF、错误截点、哈希篡改、标签列混入、非法概率和 CRLF 负例。适配器不声称已经训练或推理模型，也不能独立证明来源 manifest 的真实性。

## 检查及未完成

`python3 -m unittest discover -s tests -v`：100 项通过；`python3 -m unittest tests/test_task1_final_submission.py -v`：5 项通过；`git diff --check` 与 `py_compile` 通过。治理检查将在本交接写入后重新运行。

王健祺 g8v6/g10 完整训练与特征脚本、逐车主窗和 40+20 OOF、配置、模型与运行指纹在当前 Mac 未取得；已向叶安列出受控传递清单。缺少这些产物时不能完成同车配对终选或正式模型移植，更不能从现有 OOF 冒充正式 CSV。拿到后先核对标签／折分／时窗与哈希，再冻结一版配置，构建 61 天特征、重训、8 月 1 日推理并由适配器产出提交文件；失败尝试留在忽略的 `outputs/task1-final/`。本任务保持 `active`，不冒充人工 `accepted`。
