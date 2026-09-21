# 方案B：全局骨架 + 分群校准 + 收缩残差（v1）

- 任务：[MODEL-003](../../docs/tasks/MODEL-003.json)｜路线图位置：总体执行路线图 R4 · 方案 B（E-B 矩阵 v2：校准先行、收缩残差替代硬路由）
- 执行：王健祺（模型线）· ZCode 辅助｜2026-09-21｜状态：开发期自验证完成（v1），待 R4 阶段验收
- 数据边界：特征工程目录全程只读；预测结果、折外预测、manifest 与图表仅存本目录 `outputs/`、`report/`（git 忽略），不入公开仓库

## 结构

| 文件 | 职责 |
| --- | --- |
| `common.py` | 路径配置（模板制）、版本与预登记参数、装载断言（折分一致/样本守恒/标签合法） |
| `metrics.py` | EVAL-001 指标库：AUC/AP/Top-K（固定名额、同分按 sample_id 字典序）/Brier/LogLoss/校准箱/车辆级配对 bootstrap |
| `b0_backbone.py` | B0：RF+EBM 嵌套折外；每个外层折独立生成训练部分内部折外（inner-OOF）校准材料 |
| `b1_b2_layers.py` | B1 分群校准（low 独立 / high 收缩 w=n/(n+80) / insufficient 全局）+ B2 收缩残差（λ 网格折内选择，λ=0 恒等回退 B1） |
| `final_outputs.py` | 输出合同：`forecast_result.csv`（官方三列）与 `model_score.csv`（评分线溯源版）+ 断言 |
| `validate.py` | 验证套件 V1 正确性 / V2 统计 / V3 种子抖动 / V4 负向测试 + 预登记判优裁决 + 图表 |
| `analyze_errors.py` | 误差归因诊断（H0–H6：漏报/误报画像、损失分解、oracle 上限、历史出险信号）｜MODEL-004 |
| `improve_b.py` | 模型侧迭代实验（多种子包/单调LGBM/n0 网格；负结果，按预登记规则不采纳）｜MODEL-004 |

## 运行

```sh
cp config_local.template.json config_local.json   # 填写本机特征工程目录，勿提交
python validate.py                                # 全流程：B0→B1/B2→验证→裁决→输出（CPU 约 5 分钟）
```

依赖：scikit-learn、interpret（EBM）、pandas、numpy、matplotlib。

## v1 开发期结论（定性；实测数字见受控本地报告）

- 判优裁决采纳 **B1（全局骨架 + 分群校准）**：相对最强单模，AUC 提升为迹象级（配对 bootstrap 区间跨零、未超 3× 种子抖动线），概率质量（Brier/LogLoss）改善稳健——后者同时服务任务二概率转换评分。
- B2 残差层在当前特征版本下全部折 λ 自动选 0：按预登记规则回退 B1，代码保留待特征升级复测。
- 结构性发现（指导后续）：高风险簇内单独建模不可行（差于随机，簇定义特征已饱和）；数据不足群预测为全局插补水平，输出以 calib_flag 显式标记。
- 验证：机器检查 11/11、负向测试 6/6、同种子复现差异在浮点最低位（预登记容差内）。

## 边界

全部结果为开发期内部回测（单一截点、冻结折分折外），非独立验证成绩；评价协议 EVAL-001 为 Proposed；
特征底座升级（F0→F1）后须全流程复测；方案 A/B 终局对照（E-B4）执行前，本方案数字不构成择优结论。
