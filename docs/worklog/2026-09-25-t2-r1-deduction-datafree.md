# T2R-004：R1 扣分引擎无数据部分交接

日期：2026-09-25
路线阶段：T2-R1（娄澜，09-25～09-27）
负责人：娄澜
执行者：ZCode agent
状态：review（合成验收通过，等数据阶段与人工复核）

## 本次完成

在独立分支 `work/T2R-004-r1-deduction-datafree` 完成 R1 的无数据封闭工作包：

1. **引擎骨架**（`task2/deduction/engine.py`）：D7 七阶段管道全部为确定性纯函数——
   风险链并段（RC1-RC4 120s / RC5 600s / 场景超速 60s 近邻，结果门编码隔离）、
   EB 平滑（先验仅折内估计）、分位分档、加权 PAVA 单调、相邻等风险压缩、
   组内最大单项+0.25×其余均值的边际递减聚合。
2. **契约驱动**（`contract_loader.py`）：归属/角色/分母/风险链全部只读自 T2-R0 的
   `classification_v1.json`，本模块零硬编码分类。
3. **三份输出契约**（`writers.py`）：T2-R4 评分行（含台账复算列）、T2-R2 适配器载荷
   （含 r1-adapter-contract 四条约束的强制校验）、逐条扣分台账（gpsno/小类/证据段/率/档/分值/规则版本）。
4. **D8-D11 落笔**：章程给建议+备选；D9 首版分母=总里程代理；G6 预算置零（证据质量角色）；
   事件级扣分等分摊约定。
5. **合成测试 18 项**：不变量断言（单调性/精确复算/值域）、合同坏例、
   evaluate_scores 端到端（50 合成车、5 折折内拟合，primary_gate_pass=真、AUC>0.75）、
   dynamic_score 实跑联调。

## 验证记录

- `python -m unittest tests.test_task2_deduction -v`：18 项通过。
- `python -m unittest discover -s tests -v`：全量通过（提交前执行）。
- `python tools/check_governance.py` 与 `--scope index`：通过后提交。

合成数据为固定种子 fabricated，不代表任何真实效果；正式数字待数据阶段入受控报告。

## 开放项与下一步（数据阶段）

1. ② 引擎对照（向周航正索取受控代码）或直接以本管道在真实数据上出正式档位；
2. label_v2（EVAL-003）确认后 `fit_ruleset` 全量重学并升 `rule_version`；
3. 暴露分母切换真实行为分母（升版并重学）；
4. EXP-002（EBM 切档蒸馏挑战者）登记与执行；
5. 折外五分位单调+AUC 入受控报告（T2-R1 放行条件）；
6. 提请对齐 Q0 命名冲突（建议弱基线更名 B0，Q0 保留给 R1 学习分）。

## 边界声明

未修改周航正的 R0/R4/R2 任何文件；未使用或提交真实数据；本任务不产生真实 AUC、
Lift 或人因结论；R3（T2R-005）另分支进行。
