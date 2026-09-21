# 2026-09-21 CI 单测依赖修复（GOV-006）

- 任务：[GOV-006](../tasks/GOV-006.json)
- 分支：`main`（直推，ADR-0003 模式）
- 基准提交：`5fdcbc9`
- 人类责任人：叶安（组长，授权"做吧"）
- 执行者：ZCode agent
- 复核状态：`review`——组长知情并授权；CI 绿与否由推送后 Actions 结果客观呈现

## 背景与诊断

组长收到 GitHub 邮件 "Run failed: Governance - main (32639a5)"。排查结论：

- 自 MODEL-003（`4342561`，经 `97af80e` 进 main）起，`tests/test_solution_b_metrics.py` 及其被测模块依赖 numpy/pandas/scikit-learn；CI（governance.yml）裸机仅装 Python 3.11，`unittest discover` 在导入阶段即 `ModuleNotFoundError: No module named 'pandas'`，main 上每次 push 三系统矩阵全红。
- 治理检查本身在 `32639a5`（80 文件）与 `5fdcbc9`（110 文件）均 PASS——链接、任务登记、发布护栏无违规，问题是隐性契约"tests/ 纯标准库"被打破且依赖从未声明（仓库此前无 requirements/pyproject）。
- 本机干净复现了同一导入错误，确认与 OS 无关。

## 做了什么

1. 新增 `requirements.txt`：仅登记单测最小依赖 numpy/pandas/scikit-learn；注明检查器仍仅标准库、各线完整算法环境与锁文件由实现任务自选（维持 COLLABORATION §1 原则）。
2. `.github/workflows/governance.yml`：setup-python 启用 pip 缓存；新增 "Install declared test dependencies"（`pip install -r requirements.txt`）；测试步骤名从名不符实的 "Test checks with synthetic repositories" 改为 "Run unit tests"。
3. `docs/COLLABORATION.md`：§1 启动命令块加入 `pip install -r requirements.txt`，依赖政策成文（依赖先登记、CI 同口径、未登记会在 CI 失败）；§4 推送纪律同步一句。

## 验证

- 干净 venv 按 requirements.txt 安装后 `unittest discover`：**Ran 28 tests, OK**——17 项标准库测试无回归，solution_b 指标 11 项首次在裸环境通过（此前被导入错误挡住从未在 CI 执行）。
- 推送前 `--scope index` 检查通过（见提交）。
- CI 最终绿以推送后 Actions 三系统结果为准；若仍有红，属新问题另行登记。

| 维度 | 状态与范围 |
| --- | --- |
| 实现与契约 | CI 基础设施修复三件套完成；依赖政策写入协作文档 |
| 实验效果 | `not_run`，纯基础设施任务（单测通过即本任务效果） |
| 验收与证据 | venv 端到端 28 项通过；导入错误本机复现留档于任务 notes |
| 环境与依赖 | 本任务新增依赖仅 numpy/pandas/scikit-learn（版本下限声明） |

## 决策与限制

- 选方案 B（声明依赖 + CI 安装）而非跳过守卫（丢覆盖）或测试分层（当前规模过重）：数字诚实性守卫（配对 bootstrap、Top-K 同分政策）应每次 push 都被 CI 执行。
- requirements.txt 定位为"CI 与单测最小依赖"，不是项目环境锁文件；interpret/lightgbm 等仅被本地脚本使用的库不进此清单，待相应实现任务需要 CI 覆盖时再登记。
- interpret/lightgbm/matplotlib 未登记：现有测试不导入它们；`models/solution_b/` 中使用这些库的脚本不受 CI 覆盖，属模型线自验范围。

## 受阻记录（2026-09-21 追加）

- 提交被 Mimosa 钩子强制拦截：4 个 high 级"路径穿越"，全部位于 MODEL-003/004 已入库的 `models/solution_b/`（`common.py:154`、`final_outputs.py:68`、`improve_b.py:255`、`validate.py:305`），均为 `open(os.path.join(OUT_DIR, "字面量.json"))` 形态写入模块内 gitignore 目录。
- 评估：判定为静态分析误报（路径为 `__file__` 派生的常量目录 + 字面量文件名，无不可信输入、无网络面）。但钩子扫描全项目且强制拦截，**本机所有后续提交均被阻塞**，与本任务改动内容无关。
- 尝试过并被拦：① Edit 修改告警行——钩子对被改区域做修改前扫描，告警行自身拦截针对它的编辑（死循环）；② Write 整文件重写并在 common.py 增加 `safe_output_path` 校验助手（commonpath 限制在受控目录内）——钩子污点分析不识别该校验为净化，仍按 `OUT_DIR` 污染源拦截。半成品改动已回退（`git checkout --`），代码保持 MODEL-003/004 原样。
- 解堵执行（2026-09-21，组长拍板）：宿主侧调整——`launchctl setenv MIMOSA_GIT_GATE_MODE warn`（提交前安全门降为"警告不拦截"，写入前门禁 `MIMOSA_HOOK_BLOCK` 保持默认 graded，agent 新写代码的高危拦截不变），对 ZCode 重启后生效；本次 GOV-006 提交由组长在终端执行（ZCode 钩子不作用于用户终端）。**王健祺加固补丁落地后应 `launchctl unsetenv MIMOSA_GIT_GATE_MODE` 并重启 ZCode，恢复 graded 强拦截**。注意 launchctl setenv 不跨重启保留，若重启电脑后钩子恢复强拦截且补丁未落地，需重设一次。

## 接手者下一步

1. 各线新增重依赖测试时，同步把依赖加进 requirements.txt（一行 + 注释说明用途），否则 CI 红。
2. 王健祺：后续单测可直接使用已登记三件套；如需 CI 跑 interpret/lightgbm 相关测试，先扩清单。
3. 组长监督时可看 GitHub Actions 页确认三系统矩阵全绿。
