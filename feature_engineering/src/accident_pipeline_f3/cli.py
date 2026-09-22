# -*- coding: utf-8 -*-
"""F3 cli：scan / interface / run-all（一遍扫描：分片读取→聚合→interface 组装，FEAT-007 r4）。

真实数据路径一律来自本地配置（configs/pipeline_f3_v1.local.yaml，复制 example 模板填写；
example.yaml 仅占位示例、不写本机路径）。
"""
from __future__ import annotations

import argparse

from .pipeline import load_f3_config, run_all, run_interface, run_scan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F3_v1 lean-base + new-family assembly (FEAT-007 r4)")
    parser.add_argument("--config", default="configs/pipeline_f3_v1.local.yaml")
    parser.add_argument("command", choices=["scan", "interface", "run-all"])
    args = parser.parse_args()
    cfg = load_f3_config(args.config)
    if args.command == "scan":
        run_scan(cfg)
    elif args.command == "interface":
        run_interface(cfg)
    else:
        run_all(cfg)


if __name__ == "__main__":
    main()
