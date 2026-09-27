# 任务一最终算法版本确认与提交 CSV 核验（2026-09-27）

- 任务：[TASK-008](../tasks/TASK-008.json)；叶安明确决定采用王健祺 `g8v6-corrected` 作为任务一最终算法版本。
- 评审记录：[跨线终选与正式输出审查](../reviews/task1-final-selection.md)。
- 受控正式结果：`/Users/yea/Desktop/Study/比赛/2609_IE亮剑/交付_L3-模型_任务一正式预测g8v6-61d_20260927_叶安/正文/03_fullfit/submission/forecast_result.csv`。

按叶安确定的模型版本核验已有文件，没有重训或改写结果。对照赛题说明的“每车输出二分类结果和 0–1 概率、CSV 含车辆 ID”及官方答疑的严格三列规则，检查到 500 行、500 个唯一目标车辆、`gpsno,accident,risk_prob` 列序、二元 `accident`、有限概率、UTF-8 无 BOM、LF；车辆集合与受控目标清单相同，文件 SHA-256 与回执一致。来源 manifest 记录 `future_inference`、2026-08-01 截点及 `[2026-06-01,2026-08-01)` 特征期，确认不是 OOF 文件。`accident` 取风险排序前 100 标 1，仅为确定性格式输出；正式线上 AUC 只计算 `risk_prob` 排序。

最终模型选择已由叶安确认，项目任务仍置于 `review`：官方未来标签不可见，官方 AUC 未知；任务二及算法说明文档也不在此 CSV 的范围内，赛事系统尚未上传。
