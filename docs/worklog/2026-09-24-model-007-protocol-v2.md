# MODEL-007 交接简报：任务一协议 v2 模型线适配与全版本复跑（2026-09-24）

- 任务：[MODEL-007](../tasks/MODEL-007.json)
- 依据：[ADR-0006](../decisions/ADR-0006-official-qa-foundation.md)、[ADR-0007](../decisions/ADR-0007-near-miss-count-unit.md)、[ADR-0008](../decisions/ADR-0008-feat-009-protocol-v2-route.md)（"王健祺线自行交付新版 V5 对照"义务）
- 执行者：ZCode（王健祺线）
- 状态：复跑完成待人工复核；本文只登记范围、方法与公开边界，真实数字见受控 summary

## 做了什么

1. **协议 v2 输入重建（受控目录 `outputs/protocol_v2/`）**
   - 用仓库登记生成器 `tools/build_task1_proxy_labels.py` 从 E 盘受控 v4_assessed 三源表本地重生成新标签/折分：`label_v2_record_count_20260923` / `split_v2_record_strat5_seed42`，**54 正类（11/11/11/11/10）**，与 EVAL-003 登记版本同源同规则（生成确定性，manifest 记录源表 sha256）。
   - 时间可见性排除集对齐 FEAT-009 `prepare_protocol_v2.py`：8 个动态画像列 + `energy_type`（+画像派生列，本线特征中无）。三套输入：F0v2（54 特征列）、F1v2（113）、升级版 v2（113）。
   - **分组路由重拟合**：旧路由（E 盘 F1 聚类）含画像列不可复用。改为可见特征 kmeans（K=2、seed42、rank-gauss、语义组等权）+ insufficient 规则（evt=0 且 traj<1km 且 imu<1000 行）。配方验证：全特征复算与旧聚类一致率仅 33.6%，说明旧路由无法从特征表复现——差异如实登记，路由视为协议 v2 下的重拟合组件。新路由对新标签分离良好（low 组正类率 30.2% / high 组 3.7% / 无数据组 0%）。
2. **冻结设计全版本复跑**（`run_p2.py`，不在新标签上重新探索配置）
   - 基线：仓库登记实现 `models/task1_baselines.py`（LR/RF，F0v2 输入，`--feature-root` 传参绕过其默认路径立即求值的小缺陷，未改登记脚本）。
   - V1（RF+EBM→B1）、V3（12 成员池+折内贪心）、V4（贪心组合+S1）、V5（登记 SEL5 固定配置+S1）、V5-纯全局校准（诊断臂）、V12（升级特征×V5）、V13（C1/C2/C3 三配置）。
   - 判据：各版本 vs 基线 RF、V12/V13 另加 vs V5 的车辆级配对 bootstrap（2000×，seed42）。
3. **提交格式修正**：`models/solution_b/final_outputs.py` 与 `models/solution_b_v5/run_v5.py` 的 `forecast_result.csv` 输出统一 UTF-8 无 BOM + LF（此前 CRLF 会挂官方格式校验）。
4. **经验汇总报告**：受控生成《任务一特征工程与模型选择经验总结》PDF（旧协议 14 版本全表 + 新协议复跑表 + 失败清单）。

## 尚未证明 / 边界

- 复跑结果为内部 20 天窗代理回测，不代表 61 天正式窗口或 8 月后真实成绩；V5 等版本在新协议下的"采纳"身份仍需组长人工复核后再定。
- 叶安线 FEAT-009 的 RF 锚点未参与本轮交叉对数（各自独立生成）；如需跨线合并终评，按 ADR-0008 由评估负责人归并。
- 61 天窗口训练方案（FEAT-010 A1）与 8 月 1 日推理端到端检查未在本任务范围。
- run_v5.py 复跑版尚未在新输入上生成新 model_score/forecast_result（待版本采纳身份确定后再产出正式预测文件）。

## 接手入口

- 复跑日志与逐版本 OOF：受控 `模型选型与融合前期调研/方案B实现/outputs/protocol_v2/`（`run_p2.log`、`results_p2/summary_p2.json`、`model_inputs/manifest.json`）。
- 复跑脚本：受控 `versions/prep_protocol_v2_inputs.py` + `versions/run_p2.py`。
- 经验报告 PDF：受控 `docs/任务一特征工程与模型选择经验总结_2026-09-24.pdf`。
