from __future__ import annotations

import json
from dataclasses import replace

from accident_pipeline.clustering import cluster_features

from .config import F1Config
from .event_features import build_event_features
from .interface import build_f1_interface


def run_f1_pipeline(config: F1Config) -> None:
    build_event_features(config)
    bundle = build_f1_interface(config)
    clustering_config = replace(config.base, output_dir=config.output_dir)
    cluster_features(bundle, clustering_config)
    report_path = config.output_dir / "clustering" / "cluster_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["feature_version"] = config.feature_version
    report["group_rule_version"] = config.group_rule_version
    report["parent_feature_version"] = "F0"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
