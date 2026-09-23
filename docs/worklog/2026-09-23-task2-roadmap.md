# 2026-09-23 任务二执行路线图入库

- 任务／分支／基准提交：[SCORE-001](../tasks/SCORE-001.json)｜`work/SCORE-001-task2-roadmap`｜`7b473d9`
- 人类责任人：王健祺（组长；0923 批示三点修改后授权上传）
- 执行者与 AI 协助：ZCode agent
- 复核状态：`review`——组长审过 v0.1 并给批示；v0.2 按批示修订后上传，正式复核由任务二执行组（周航正/娄澜/吴钊同）在开工时确认

## 做了什么

- 入库 [任务二执行路线图 v0.2](../plans/task2-roadmap.md)：按 0922 会议口径的执行主线
  （T2-R0 行为大类 → R1 扣分规则 Q0 → R2 可运营/动态微调 → R3 白盒黑盒融合 → R4 MOE 与基线（独立任务）→ R5 交付），
  含行为大类-事件类型-特征三列映射底稿、接口与红线（追溯率/单调性/无数据不满分/不承诺因果）。
- 按组长批示修订三处：待拍板 1–3 项改为**执行期决策点**（不预先拍死）；Q1 改为**持续跟进任务一最新采纳版本**
  （按 manifest 指纹登记、版本演进即换用重算）；**MOE 不冻结**（不走 ADR，随任务二迭代自由演进）。
- 对齐新治理：写入域按 [ADR-0005](../decisions/ADR-0005-task1-dual-track-and-task2-split.md)（task2/ 与 docs/plans/task2-*）；
  任务一消费与判据引用 [ADR-0004](../decisions/ADR-0004-baseline-freeze.md)。
- 新起任务前缀 SCORE（评分线），与 DATACON 先例同法，待任务二组确认沿用。

## 验证

- `python tools/check_governance.py --scope index` 与 `python -m unittest discover -s tests`：通过后提交；
- 文档自检：不含真实数据统计（G7 白皮书为公开行业数字）；相对链接指向仓库内文件。

## 决策与限制

- 无新 ADR（路线图为执行基线提案，非口径冻结；MOE 按批示明确不冻结）。

## 接手者下一步

1. 任务二三组按路线图开工：R0 映射表（决策点 1）→ R1 规则基线；代码入 `task2/`；
2. Q1 消费 [MODEL-005](../tasks/MODEL-005.json) 的 model_score（当前任务一登记最优），登记 manifest 指纹；
3. 执行期决策点 1–3 在对应阶段评审后回填路线图。
