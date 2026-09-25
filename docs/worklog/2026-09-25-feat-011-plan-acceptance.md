# 2026-09-25 FEAT-011 小规模建模适配方案入库

- 任务／分支／基准提交：[FEAT-011](../tasks/FEAT-011.json)；`main`，按 ADR-0003 直接提交；整理起点 `373b5f35cf7095973bdb2218fb5de68d52d79db4`。
- 人类责任人：叶安。执行者与 AI 协助：Codex 整理方案、决策和索引。
- 复核状态与真实复核人：叶安于 2026-09-25 指示“入库”；仅确认方案，尚无执行结果复核。

## 做了什么

将 [FEAT-011 方案](../plans/feat-011-small-attribution-model-adaptation-plan.md) 从草案升为 v1.0，按 [ADR-0010](../decisions/ADR-0010-feat-011-small-attribution-model-adaptation.md) 登记真实决策人、范围、依赖、预算、停止与回退边界；同步任务和文档索引。此前的[代拟记录](2026-09-25-feat-011-plan-draft.md)保留为历史，不再代表当前状态。

本机 `main` 有其他线未提交文件且落后远端，故从最新 `origin/main` 建独立检出整理本次文件；没有暂存、覆盖或提交那些已有改动。没有改 FEAT-009、FEAT-010、王健祺线代码或公共评估实现。

## 验证

- `python3 tools/check_governance.py`：通过（working-tree）。
- `python3 -m unittest discover -s tests -v`：82 项通过。
- `git diff --check`、`git diff --cached --check`：通过；只暂存本次 10 个文档文件。
- `python3 tools/check_governance.py --scope index`：通过（index）。
- S0–S3 尚未执行；没有新预登记、真实 OOF、模型指标或真实数据图。

## 决策与限制

方案接受仅允许按既定步骤继续准备；真实效果运行仍需 EVAL-001／EVAL-003／FEAT-009 的实际复核记录。固定四训练臂、一个等权融合主候选与唯一 `E−A` 主比较，不能在结果出来后改称诊断臂为主候选。20＋40 代理结果不等于官方未来 40 天成绩，61 天最终输入不接现有代理标签算 AUC。

## 接手者下一步

先核查远端任务状态与写入冲突，从 [方案 S0](../plans/feat-011-small-attribution-model-adaptation-plan.md) 做输入／列集只读审计及合成样张；实际复核与成本门齐备后才锁 S1 预登记并运行真实候选。结果只进受控 `outputs/feat-011/`，失败和负结果也留 run ID。
