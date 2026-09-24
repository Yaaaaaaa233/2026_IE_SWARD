# R3 白盒黑盒融合实验（无数据部分）

本目录实现 T2-R3 中不依赖真实分数的部分：融合公式、同 MOE 的 α 网格对照、分歧清单与司机话术格式。D18 的裁决（模型概率进不进总分）由数据阶段的对照实验与组长终审作出。

- [R3 章程（D18 正反论证、D19 记号与网格、D20 话术、红线）](fusion-charter-v1.md)
- `fusion.py`：`fuse_scores` / `run_alpha_grid` / `rank_disagreements` / `dimension_talking_point`。
- 合成测试：`tests/test_task2_fusion.py`。

边界：任务一概率只作 Q1 输入与一致性参照，绝不参与判优；纯模型臂只是参照不是候选公开分；真实运行产物只入受控目录。
