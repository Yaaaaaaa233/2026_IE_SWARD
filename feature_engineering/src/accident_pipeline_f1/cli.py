from __future__ import annotations

import argparse
import json
from dataclasses import replace

from accident_pipeline.clustering import cluster_features

from .config import load_f1_config
from .event_features import build_event_features
from .interface import build_f1_interface
from .pipeline import run_f1_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="F1_v1 versioned feature upgrade")
    parser.add_argument("--config", default="config/pipeline_f1_v1.yaml")
    parser.add_argument("command", choices=["build-features", "build-interface", "cluster", "run-all"])
    args = parser.parse_args()
    config = load_f1_config(args.config)
    if args.command == "build-features":
        build_event_features(config)
    elif args.command == "build-interface":
        build_f1_interface(config)
    elif args.command == "cluster":
        bundle = build_f1_interface(config)
        target = replace(config.base, output_dir=config.output_dir)
        cluster_features(bundle, target)
        path = config.output_dir / "clustering" / "cluster_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report.update(feature_version=config.feature_version, group_rule_version=config.group_rule_version)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        run_f1_pipeline(config)


if __name__ == "__main__":
    main()
