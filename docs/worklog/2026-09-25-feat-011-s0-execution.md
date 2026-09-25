# 2026-09-25 FEAT-011 S0 只读审计与样张交接

- 任务／分支／基准提交：[FEAT-011](../tasks/FEAT-011.json)；按 ADR-0003 直接提交 `main`；本轮起点 `a0abad5eeaca2bf0e258df776d7eaf5b1bfbdc8f`。
- 人类责任人：叶安。执行者与 AI 协助：Codex，按 FEAT-011 v1.0 执行 S0。
- 复核状态：机器审计通过；叶安对合成样张、真实来源台账与特征族图的人工复核待完成。

## 做了什么

新增 [S0 输入审计器](../../feature_engineering/experiments/feat011/audit_s0.py)、合成样张生成器和对应的合成坏例检查。审计器只读锁定的 FEAT-009 v2 F3 Y1 表及 Y0 可见性台账，核对标签／折分／时间合同、输入哈希、车辆键、五折类别、禁止字段，并按预定规则生成全量与精简列集台账。精简集合仅移除 `f3_night_*` 的细分列，四个预定夜间摘要列保留；未重算夜间特征。

受控运行入口为 `outputs/feat-011/s0/20260925-r4/acceptance.md`，同时保存 `input_check.json`、`column_ledger.csv` 和特征族计数图。该图只反映字段结构，不含模型效果。两张合成样张保存在 `outputs/feat-011/s0/synthetic-preview/`，展示时间窗／字段来源边界和固定 2×2 特征视角矩阵；它们不是数据或模型结果。具体输入指纹、字段清单和数据派生图只留在 Git 忽略的受控目录。

首次审计实现误以为输入清单保存完整字段顺序；检查后发现清单只保存字段数，随后按实际合同校正检查器。最终 `r4` 运行重新读取原输入并通过。此前尝试未修改输入或 FEAT-009 输出。

## 验证与边界

- S0 机器检查：通过；完整视角、精简视角和差集符合方案中的规则，Y0 台账可映射全部 F3 预测列。字段来源的实际语义仍待叶安抽查。
- `python3 -m unittest discover -s feature_engineering/experiments/feat011 -p 'test_*.py' -v`：2 项通过；合成测试覆盖合法样例及未来画像列、重复车辆、标签／折分版本漂移、非整数标签／折号和列差集漂移。
- `py_compile`：S0 审计器与样张生成器通过。
- `python3 tools/check_governance.py`：通过（working-tree）。
- `python3 tools/check_governance.py --scope index`：通过（index）；只暂存本任务 10 个文件。
- `python3 -m unittest discover -s tests -v`：82 项通过。
- EVAL-001／EVAL-003 责任人真实复核仍未登记；FEAT-009 标签来源、字段来源和真实图表人工复核仍待完成。故 S1 预登记与真实 OOF 保持阻塞；没有运行模型、生成新 OOF 或计算新分数。
- `outputs/feat-009/v2/y1/` 保持只读；没有修改王健祺线、公共基线或任务二接口。

## 接手者下一步

叶安先复核合成样张 `outputs/feat-011/s0/synthetic-preview/s0_timeline_synthetic.png`、`s0_feature_views_synthetic.png`，再查看真实结构图 `outputs/feat-011/s0/20260925-r4/feature_family_counts.png` 与 `column_ledger.csv`。复核只确认图表是否准确表达输入边界和来源，不代表模型效果通过。再完成 EVAL-001、EVAL-003 和 FEAT-009 的真实责任人复核；所有门齐备后才锁定 S1 预登记。
