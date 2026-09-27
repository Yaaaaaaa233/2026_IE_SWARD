"""Build the frozen g8 member feature tables for the official 61-day history.

Loads the transferred, hash-audited feature recipes without modifying them.
All path overrides and resulting data remain in the controlled output directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from .extend_cleaned_history import sha256
except ImportError:  # direct script entry point
    from extend_cleaned_history import sha256


START = "2026-06-01"
END = "2026-08-01"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import feature recipe: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_table(path: Path, expected_ids: set[str]) -> None:
    table = pd.read_csv(path, dtype={"gpsno": str, "sample_id": str})
    if len(table) != len(expected_ids) or set(table.gpsno) != expected_ids or table.gpsno.duplicated().any():
        raise ValueError(f"member table does not match the fixed vehicle set: {path.name}")
    if table.sample_id.tolist() != (table.gpsno + "_20260801_61d").tolist():
        raise ValueError(f"member table has stale as-of sample IDs: {path.name}")
    if {"y", "fold", "label_status"} & set(table):
        raise ValueError(f"member table contains label metadata: {path.name}")


def restore_training_transitions(
    score: pd.DataFrame, events_path: Path, train_g1_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Evaluate training-selected transition columns absent from the 61d top 12.

    The transferred g1 builder chooses top pairs anew on the score window.  A
    displaced training pair remains a valid fixed feature; calculate its true
    61-day count instead of silently filling zero or changing model selection.
    """
    train_columns = pd.read_csv(train_g1_path, nrows=0).columns
    missing = [c for c in train_columns if c not in score]
    if not missing:
        return score, []
    if any(re.fullmatch(r"tr_\d+_\d+", c) is None for c in missing):
        raise ValueError(f"g1 has non-transition training columns missing: {missing}")
    events = pd.read_csv(
        events_path, usecols=["row_id", "gpsno", "event_type", "start_time"],
        dtype={"gpsno": str, "event_type": str},
    )
    events["t"] = pd.to_datetime(events.start_time, errors="raise")
    events = events[(events.t >= pd.Timestamp(START)) & (events.t < pd.Timestamp(END))]
    events = events.sort_values(["gpsno", "t"])
    events["prev_type"] = events.groupby("gpsno").event_type.shift(1)
    for column in missing:
        _, prior, current = column.split("_")
        selected = events[(events.prev_type == prior) & (events.event_type == current)]
        score[column] = score.gpsno.map(selected.groupby("gpsno").row_id.count()).fillna(0)
    return score, missing


def join_imu_features(labels: pd.DataFrame, imu: pd.DataFrame) -> pd.DataFrame:
    output = labels[["sample_id"]].copy()
    output["gpsno"] = output.sample_id.str.split("_").str[0]
    output = output.merge(imu, on="gpsno", how="left", validate="1:1")
    output["imu_missing"] = output.imu_rows.isna().astype(int)
    # The frozen training artifact retains imu_rows, though the transferred
    # standalone join script drops it. The artifact is the fitted schema.
    return output


def build(args: argparse.Namespace) -> None:
    if args.phase in ("all", "trajectory") and args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.phase in ("imu", "members") and not args.output_dir.is_dir():
        raise FileNotFoundError(args.output_dir)
    cleaner = json.loads(args.cleaning_manifest.read_text(encoding="utf-8"))
    if cleaner.get("july30_control") != "exact_except_at_most_0.001km_rounding" or cleaner.get("july31_event_rows", 0) <= 0:
        raise ValueError("61-day cleaning input has not passed its control-day check")
    if cleaner.get("july31_vehicle_day_rows", 0) <= 0:
        raise ValueError("July 31 vehicle-day source is empty")
    if cleaner["output_sha256"].get(args.events.name) != sha256(args.events):
        raise ValueError("event input differs from cleaning manifest")
    if cleaner["output_sha256"].get(args.vehicle_day.name) != sha256(args.vehicle_day):
        raise ValueError("vehicle-day input differs from cleaning manifest")

    script_root = args.package_root / "01_scripts"
    common = script_root / "_common"
    feature_dir = script_root / "特征构建"
    protocol = args.protocol_dir
    labels = pd.read_csv(protocol / "labels.csv", dtype={"sample_id": str})
    ids = set(labels.sample_id.str.split("_").str[0])
    if len(labels) != 500 or len(ids) != 500:
        raise ValueError("frozen protocol must contain exactly 500 vehicles")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models" / "solution_b"))
    sys.path.insert(0, str(common))
    eval_stack = load_module(common / "eval_stack.py", "eval_stack")
    eval_stack.PROTOCOL_DIR = str(protocol)

    if args.phase in ("all", "trajectory"):
        args.output_dir.mkdir(parents=True)
    traj_out = args.output_dir / "trajectory"
    imu_out = args.output_dir / "imu"
    if args.phase in ("all", "trajectory"):
        if traj_out.exists():
            raise FileExistsError(traj_out)
        traj_scan = load_module(common / "scan_trajectory.py", "task1_score_scan_trajectory")
        traj_scan.TRAJ_DIR = str(args.trajectory_dir)
        traj_scan.main(w_start=START, w_end=END, out_dir=str(traj_out))
        if args.phase == "trajectory":
            return
    if args.phase in ("all", "imu"):
        if imu_out.exists():
            raise FileExistsError(imu_out)
        imu_scan = load_module(common / "scan_imu.py", "task1_score_scan_imu")
        imu_scan.IMU_DIR = str(args.imu_dir)
        imu_scan.main(w_start=START, w_end=END, out_dir=str(imu_out))
        if args.phase == "imu":
            return
    if not (traj_out / "traj_vehicle.csv").is_file() or not (imu_out / "imu_vehicle.csv").is_file():
        raise ValueError("frozen trajectory and IMU scans must both complete before member build")

    g1 = load_module(feature_dir / "build_g1v1.py", "task1_score_g1")
    g1.EV, g1.VD = str(args.events), str(args.vehicle_day)
    g1_table, _ = g1.build(w_start=START, w_end=END)
    g1_table, restored_transitions = restore_training_transitions(
        g1_table, args.events,
        args.package_root / "04_models" / "中间特征表" / "20天窗" / "g1v1_features.csv",
    )
    g1_path = args.output_dir / "g1.csv"
    g1_table.to_csv(g1_path, index=False, lineterminator="\n")

    g2 = load_module(feature_dir / "build_g2v1.py", "task1_score_g2")
    g2.VD = str(args.vehicle_day)
    g2_path = args.output_dir / "g2.csv"
    g2.build(w_start=START, w_end=END).to_csv(g2_path, index=False, lineterminator="\n")

    imu = pd.read_csv(imu_out / "imu_vehicle.csv", dtype={"gpsno": str})
    g3 = join_imu_features(labels, imu)
    g3_path = args.output_dir / "g3.csv"
    g3.to_csv(g3_path, index=False, lineterminator="\n")

    m4 = load_module(feature_dir / "build_member4_clean.py", "task1_score_m4")
    m4.EV, m4.VD, m4.TV = str(args.events), str(args.vehicle_day), str(args.target_vehicles)
    m4_path = args.output_dir / "m4.csv"
    m4.build(w_start=START, w_end=END, traj_csv=str(traj_out / "traj_vehicle.csv")).to_csv(
        m4_path, index=False, lineterminator="\n"
    )

    member_paths = {"g1": g1_path, "g2": g2_path, "g3": g3_path, "m4": m4_path}
    for path in member_paths.values():
        table = pd.read_csv(path, dtype={"gpsno": str, "sample_id": str})
        table["sample_id"] = table.gpsno + "_20260801_61d"
        table.to_csv(path, index=False, lineterminator="\n")
        check_table(path, ids)
    score_manifest = {
        "role": "future_features",
        "as_of": "2026-08-01",
        "feature_start": START,
        "feature_end_exclusive": END,
        "member_sha256": {name: sha256(path) for name, path in member_paths.items()},
        "source_coverage": {
            "events_clean": "2026-07-31",
            "vehicle_day": "2026-07-31",
            "trajectory": "2026-07-31" if cleaner["july31_sensor_days"]["trajectory"] else "2026-07-30",
            "imu": "2026-07-31" if cleaner["july31_sensor_days"]["imu"] else "2026-07-30",
        },
        "cleaning_manifest_sha256": sha256(args.cleaning_manifest),
        "restored_training_transition_columns": restored_transitions,
        "recipe_sha256": {
            path.name: sha256(path) for path in [
                common / "scan_trajectory.py", common / "scan_imu.py",
                feature_dir / "build_g1v1.py", feature_dir / "build_g2v1.py",
                feature_dir / "build_member4_clean.py", feature_dir / "build_g3v2_join.py",
            ]
        },
    }
    (args.output_dir / "score_manifest.json").write_text(
        json.dumps(score_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("61-day g8 score features saved", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "trajectory", "imu", "members"), default="all")
    for name in (
        "package_root", "protocol_dir", "events", "vehicle_day", "cleaning_manifest",
        "trajectory_dir", "imu_dir", "target_vehicles", "output_dir",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
