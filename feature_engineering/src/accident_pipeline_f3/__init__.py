# -*- coding: utf-8 -*-
"""FEAT-007 r4 特征工程包：新信息特征族（stage-0 修正／Tier 1／Tier 2／Tier 3／综合分）。

core.py／trajectory.py 提供 Tier 1/2/3 纯函数实现（不做文件 IO）；
pipeline.py 一遍扫描编排（分片读取→聚合），interface.py 组装 F3_v1 model_input、
执行收尾通则与契约断言，cli.py 提供 scan / interface / run-all。
预登记定义见 docs/plans/feat-007-r4-execution.md §3.0。
"""
from .interface import (MAX_NEW_COLUMNS, RHO_THRESHOLD, assert_contract,
                        build_f3_interface, finalize_columns, load_base_columns,
                        select_new_columns)
from .pipeline import F3Config, load_f3_config, run_all, run_interface, run_scan

__all__ = [
    "MAX_NEW_COLUMNS", "RHO_THRESHOLD", "assert_contract", "build_f3_interface",
    "finalize_columns", "load_base_columns", "select_new_columns",
    "F3Config", "load_f3_config", "run_all", "run_interface", "run_scan",
]
