# T2R-003：任务二 T2-R2 动态运营骨架交接

日期：2026-09-24
负责人：周航正
执行者：Codex
状态：接口和中性骨架完成，真实联调等待 R1

## 完成内容

1. 定义 R1 → R2 最小接口，只消费事件段基础扣分和核查状态，不复制 R1 的分箱、权重或组内聚合。
2. 按官方边界实现夜间、早晨、白天和黄昏分类；默认系数均为 1。
3. 实现以有效暴露日计的 EWMA 风险负担，支持 5、7、14 日半衰期候选。
4. 实现跨日持续性倍率接口；默认步长 0、倍率 1，等待折内学习。
5. 实现安全恢复、单日回分上限和“风险日不得反向加分”。
6. 实现低暴露向保守先验收缩、停驶冻结和从未有效观察车辆的保守处理。
7. 实现 `valid`、`pending`、`exempt` 三种核查状态及到期未核自动转扣。
8. 实现 JSON 命令行入口、合成输入示例和八项边界测试。

## 机器验证

- 合成示例成功生成一辆车的动态分数和逐日轨迹。
- `python -m unittest discover -s tests -v`：52 项通过。
- `python tools/check_governance.py`：通过。
- `git diff --check`：通过。

## 交付入口

- 方案：`task2/operations/operational-dynamics-charter-v1.md`
- R1 对接合同：`task2/operations/r1-adapter-contract.md`
- 策略候选：`task2/operations/operational_policy_v1.json`
- 实现：`task2/operations/dynamic_score.py`
- 合成示例：`task2/operations/input.example.json`
- 测试：`tests/test_task2_operational_dynamics.py`

## 未完成依赖

仓库目前没有娄澜的 T2-R1 正式任务记录或基础扣分接口，因此尚不能做真实联调、参数选择或微调前后 MOE 比较。当前配置明确标记为占位候选；在 R1 进入仓库后，需要先核对基础扣分单位、暴露单位和版本字段，再在固定训练折内确定参数。
