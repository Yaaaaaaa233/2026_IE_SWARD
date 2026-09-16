# 治理检查器

`check_governance.py` 使用 Python 3.10+ 标准库和 Git。不会联网，也不读取被忽略的比赛数据目录。

```sh
python tools/check_governance.py                     # 当前已跟踪 + 未忽略的新文件
python tools/check_governance.py --scope index       # 实际暂存内容
python tools/check_governance.py --scope head        # 已提交内容；CI 使用
python -m unittest discover -s tests -v              # 合成反例测试
```

检查必要文件、仓库内 Markdown 文件链接、任务结构、当前活动任务写入冲突、禁入路径和产物、部分凭据标记、既有 PDF 指纹。代码块中的示例链接不检查；标题锚点和外链可用性不检查。只检查当前候选快照，不扫描完整 Git 历史。

工作区、暂存区和 HEAD 可能不同，提交前必须查 index。`governance_policy.json` 维护结构与已存在资产的清单，不能为了绕过失败放宽规则；例外要在 PR 写明公开依据并真实复核。

任务互斥检查仅看本快照的 `active` 任务，不是跨设备锁。文本和 JSON 中所有可能的敏感信息无法自动穷举，因此仍须按数据规范做内容审查。
