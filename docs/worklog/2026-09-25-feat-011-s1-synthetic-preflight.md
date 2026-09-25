# 2026-09-25 FEAT-011 S1 合成预检与成本样本交接

- 任务／分支／基准提交：[FEAT-011](../tasks/FEAT-011.json)；按 ADR-0003 直接提交 `main`；本轮起点 `b514288f606b68c8384e089ecf512bf05220dd1c`。
- 人类责任人：叶安。执行者与 AI 协助：Codex，按 FEAT-011 v1.0 继续 S0 并完成 S1 合成预检。
- 复核状态：叶安已确认 S0 的两张合成图与真实特征族图／台账样张通过；本交接内 S1 矩阵／融合图和区间图待叶安复核。

## 做了什么

在 `feature_engineering/experiments/feat011/` 新增 S1 合成预检器和测试。预检使用确定性生成的 500 行、150 列通用合成数据，不读取真实 F3 特征、标签、折分、OOF 或指标。它用项目实际 EBM 工厂完成 A–D 四个固定臂各自的五折训练，逐折拟合缺失值处理器并记录训练行指纹；E 依照固定公式从 A/D 的折外预测生成，没有训练第五个模型。

合成预登记包含完整 EBM 参数、列顺序与合成哈希，并显式标记为 `synthetic_preflight_only`、`real_preregistration_locked=false`。检查器核对配置／输入指纹、A–D 2×2 因子矩阵、五折类别、每折拟合范围和 E 的等权 logit 复算。配置篡改、改变融合权重、交换折号、把验证行加入预处理拟合范围均被负向检查拒绝。

受控最终样张和记录位于 `outputs/feat-011/s1-synthetic/20260925-r4/`：`s1_matrix_fusion_synthetic.png`、`s1_interval_synthetic.png`、`synthetic_preregistration.json`、`preflight_result.json` 与 `negative_checks.json`。全部带合成标识。一个 A 外折的合成运行样本约 3.9 秒，进程峰值 RSS 约 198 MiB（包含解释器和库导入基线）；这是资源流程检查，不足以预测真实输入耗时或内存。合成 AUC／区间只证明报告代码能生成，不构成本轮模型效果结论。

初次扩展合成输入时，缺失值夹具意外创建了一个未登记列，检查器拒绝运行；该失败保存在 `outputs/feat-011/s1-synthetic/20260925-r2/failure.json`。修正后使用新 run ID 完成最终预检，没有覆盖失败记录。

## 验证

- `python3 -m unittest discover -s feature_engineering/experiments/feat011 -p 'test_s1_synthetic_preflight.py' -v`：4 项通过，含完整配置漂移、矩阵／折图、融合复算及注册的坏例。
- `python3 feature_engineering/experiments/feat011/s1_synthetic_preflight.py --output-dir outputs/feat-011/s1-synthetic/20260925-r4`：完成，A–D 各 5 次合成折训练，E 确定性派生；配置篡改、融合权重篡改、折号交换、预处理泄漏均拒绝。
- 最终运行环境包版本：interpret 0.7.8、interpret-core 0.7.8、NumPy 2.4.6、pandas 3.0.6、scikit-learn 1.9.1、Matplotlib 3.10.9。包来自本机已有的离线缓存；未安装或改写仓库依赖。
- 公开仓库未增加真实 OOF、真实分数、输入指纹或数据派生图。
- 尚未解除的门：EVAL-001 责任人／规则复核、EVAL-003 标签与折分实际复核、FEAT-009 真实字段来源／标签／图表复核。因此没有锁定真实 `preregistration.json`，没有进入 S2，也没有运行任何真实 A–D OOF。

## 决策与限制

本轮遵循 [ADR-0010](../decisions/ADR-0010-feat-011-small-attribution-model-adaptation.md) 和 [FEAT-011 方案](../plans/feat-011-small-attribution-model-adaptation-plan.md)。S0 的三项图表／来源样张人工验收已完成；S1 的合成样张仍待人工验收。方案验收不等于真实运行授权，当前 FEAT-011 总体仍为 `blocked`。合成评分和小样本成本数字均不可引用为实际模型或资源表现，也不据此推断 A–E 优劣。

## 接手者下一步

叶安先复核最终合成样张 `outputs/feat-011/s1-synthetic/20260925-r4/s1_matrix_fusion_synthetic.png` 与 `s1_interval_synthetic.png`，重点确认 2×2 归因关系、唯一主比较 E−A、零线／+0.01 参考线和辅助容差表达清楚。再分别完成 EVAL-001、EVAL-003、FEAT-009 的实际责任人复核登记。两类人工检查都齐备后，才从受控 FEAT-009 Y1 输入生成单独的真实 S1 预登记；未完成前继续保持真实 OOF 阻塞。
