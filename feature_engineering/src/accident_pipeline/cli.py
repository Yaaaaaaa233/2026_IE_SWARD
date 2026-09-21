from __future__ import annotations

import argparse
from pathlib import Path

from .clustering import cluster_features
from .config import load_config
from .dataset import DatasetBuilder
from .imu_features import extract_imu_features
from .pipeline import run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="事故特征、聚类与模型数据契约流水线")
    parser.add_argument("--config", default="config/pipeline.yaml", help="YAML 配置文件")
    commands = parser.add_subparsers(dest="command", required=True)
    imu = commands.add_parser("extract-imu", help="流式提取 IMU 事故语义特征")
    imu.add_argument("--limit-files", type=int, default=None, help="仅调试：只读取前 N 个分片")
    interface = commands.add_parser("build-interface", help="生成模型输入数据契约")
    interface.add_argument("--allow-missing-imu", action="store_true")
    commands.add_parser("cluster", help="对全部模型特征聚类")
    commands.add_parser("run-all", help="依次执行 IMU、接口、聚类")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "extract-imu":
        files = sorted(config.imu_dir.glob("part-*"))
        if args.limit_files:
            files = files[: args.limit_files]
        extract_imu_features(config, files=files)
    elif args.command == "build-interface":
        DatasetBuilder(config).materialize(DatasetBuilder(config).build(require_imu=not args.allow_missing_imu))
    elif args.command == "cluster":
        builder = DatasetBuilder(config)
        cluster_features(builder.materialize(), config)
    elif args.command == "run-all":
        run_pipeline(config)


if __name__ == "__main__":
    main()
