#!/usr/bin/env python3
"""Rebind existing 20-day feature candidates to the 0923 proxy protocol.

This creates private, reproducible inputs. It never changes source feature files.
Portrait columns and their known descendants are removed before model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
PORTRAIT_DYNAMIC = {
    "monthly_avg_mileage", "monthly_avg_hours", "monthly_avg_stops", "highway_ratio",
    "morning_ratio", "dusk_ratio", "night_hours_ratio", "night_mileage_ratio",
}
PORTRAIT_STATIC_UNVERIFIED = {"energy_type"}
PORTRAIT_DERIVED = {"f2_prior_incident_x_night", "f2_prior_incident_x_highway"}
META = {
    "sample_id", "gpsno", "y", "fold", "as_of", "window_start", "window_end",
    "lookback_days", "horizon_days", "label_window", "label_status", "label_version",
    "split_version", "source_version", "feature_version", "feature_version_f1",
    "feature_version_f2", "feature_version_f3", "feature_version_f2r2", "cohort_fallback",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/feat-009.local.toml"))
    ap.add_argument("--output", default=str(REPO / "outputs/feat-009/v2"))
    args = ap.parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    y0, y1, y2 = (out / "y0", out / "y1", out / "y2")
    for p in (y0, y1, y2):
        p.mkdir(parents=True, exist_ok=True)

    labels_dir = REPO / "outputs/protocol_20260923/label_v2_record_proxy"
    labels_path, splits_path = labels_dir / "labels.csv", labels_dir / "splits.csv"
    f0_path = Path(cfg.get("f0_model_input", REPO.parent / "特征工程/artifacts/model_interface/model_input.csv")).expanduser().resolve()
    f1_path = Path(cfg["f1_model_input"])
    f3_path = Path(cfg["f3_model_input"])
    new_path = Path(cfg["f3_new_features"])
    manifest_path = Path(cfg["f3_manifest"])
    drop_path = Path(cfg["f3_dropped_columns"])
    source_paths = {"labels": labels_path, "splits": splits_path, "f0": f0_path,
                    "f1": f1_path, "f3": f3_path, "f3_new": new_path,
                    "f3_manifest": manifest_path, "f3_drop_ledger": drop_path}
    missing = [f"{k}: {p}" for k, p in source_paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing controlled inputs: " + "; ".join(missing))

    frames = {k: read(p) for k, p in (("f0", f0_path), ("f1", f1_path), ("f3", f3_path), ("f3_new", new_path))}
    labels, splits = read(labels_path), read(splits_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ledger = json.loads(drop_path.read_text(encoding="utf-8"))
    base_value = manifest.get("inputs", {}).get("base_input")
    if not base_value:
        raise ValueError("F3 manifest has no registered base_input")
    base_path = Path(base_value)
    if not base_path.is_absolute():
        base_path = (manifest_path.parent / base_path).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"registered F3 base input missing: {base_path}")
    base = read(base_path)
    source_paths["f3_base"] = base_path

    for name, frame in frames.items():
        for c in ("sample_id", "gpsno"):
            if c not in frame or frame[c].isna().any():
                raise ValueError(f"{name}: missing/null {c}")
        if frame.sample_id.duplicated().any() or frame.gpsno.duplicated().any():
            raise ValueError(f"{name}: expected one unique row per vehicle")
    if labels.sample_id.duplicated().any() or splits.gpsno.duplicated().any():
        raise ValueError("label or split keys are duplicated")
    if set(labels.y.dropna().astype(int).unique()) != {0, 1}:
        raise ValueError("labels are not binary")
    if labels.label_version.nunique() != 1 or splits.split_version.nunique() != 1:
        raise ValueError("protocol versions are not unique")
    if len(labels) != 500 or len(splits) != 500:
        raise ValueError("expected exactly 500 target vehicles")

    split_by_car = splits[["gpsno", "fold", "split_version"]]
    target = labels.merge(frames["f0"][["sample_id", "gpsno"]], on="sample_id", validate="1:1")
    target = target.merge(split_by_car, on="gpsno", validate="1:1")
    if target.gpsno.duplicated().any() or target.y.isna().any() or target.fold.isna().any():
        raise ValueError("new protocol key binding is incomplete or duplicated")
    if set(target.fold.astype(int).unique()) != set(range(5)):
        raise ValueError("split must contain folds 0..4")
    if any(set(target.loc[target.fold == k, "y"].astype(int)) != {0, 1} for k in range(5)):
        raise ValueError("a frozen fold lacks one class")
    expected_sample = set(target.sample_id.astype(str))
    for name, frame in {**frames, "base": base}.items():
        if set(frame.sample_id.astype(str)) != expected_sample:
            raise ValueError(f"{name}: vehicle/sample set differs from protocol v2")
        if not frame.gpsno.astype(str).isin(target.gpsno.astype(str)).all():
            raise ValueError(f"{name}: gpsno is outside target population")

    # Existing interfaces declare a common 20-day [2026-06-01, 2026-06-21) view.
    for name, frame in frames.items():
        if "as_of" in frame:
            dates = pd.to_datetime(frame.as_of, errors="coerce").dt.strftime("%Y-%m-%d")
            if dates.isna().any() or set(dates) != {"2026-06-21"}:
                raise ValueError(f"{name}: as_of is not uniformly 2026-06-21")
        if "lookback_days" in frame and set(pd.to_numeric(frame.lookback_days, errors="coerce").dropna()) != {20}:
            raise ValueError(f"{name}: lookback is not uniformly 20 days")

    portrait_descendants = {c for f in [*frames.values(), base] for c in f.columns
                            if c.startswith("f3_profile_") or "_cohort_" in c or c == "cohort_fallback"}
    blocked = PORTRAIT_DYNAMIC | PORTRAIT_STATIC_UNVERIFIED | PORTRAIT_DERIVED | portrait_descendants
    feature_tables = {}
    visibility: dict[str, dict] = {}
    for name in ("f0", "f1"):
        frame = frames[name].copy()
        feature_cols = [c for c in frame.columns if c not in META]
        bad = set(feature_cols) & blocked
        feature_cols = [c for c in feature_cols if c not in blocked]
        if bad:
            pass
        if not feature_cols:
            raise ValueError(f"{name}: no features remain")
        feature_tables[name] = frame[["sample_id", "gpsno", *feature_cols]].copy()
    # Rebuild F3 from the declared candidate set, not the old full-population kept list.
    # Use the registered lean F0/F1/F2 base plus the full raw F3 feature table.
    # The old r5 kept-column list was selected on all 500 vehicles, so it is
    # evidence for the historical interface only and is deliberately not reused.
    base_candidates = [c for c in base.columns if c not in META]
    new_candidates = [c for c in frames["f3_new"].columns if c not in META]
    raw_f3 = pd.concat([base[base_candidates].reset_index(drop=True),
                        frames["f3_new"][new_candidates].reset_index(drop=True)], axis=1)
    if raw_f3.columns.duplicated().any():
        raise ValueError("F3 base/new candidates overlap")
    # Identifier order comes from the verified base input; any pre-screened F3 frame is not used.
    raw_f3 = pd.concat([base[["sample_id", "gpsno"]].reset_index(drop=True), raw_f3], axis=1)
    f3cols = [c for c in raw_f3.columns if c not in META and c not in blocked]
    if not f3cols:
        raise ValueError("F3: no usable candidates remain after time-visibility exclusions")
    feature_tables["f3"] = raw_f3[["sample_id", "gpsno", *f3cols]].copy()

    binding = target.set_index("sample_id")[["y", "fold", "label_version", "split_version", "label_window", "horizon_days"]]
    for name, table in feature_tables.items():
        ordered = table.assign(sample_id=table.sample_id.astype(str)).set_index("sample_id", drop=False)
        ordered = ordered.loc[binding.index.astype(str)].reset_index(drop=True)
        ordered["y"] = binding.y.astype(int).to_numpy()
        ordered["fold"] = binding.fold.astype(int).to_numpy()
        ordered["label_version"] = binding.label_version.astype(str).to_numpy()
        ordered["split_version"] = binding.split_version.astype(str).to_numpy()
        ordered["label_window"] = binding.label_window.astype(str).to_numpy()
        ordered["horizon_days"] = binding.horizon_days.astype(int).to_numpy()
        outpath = y1 / f"{name}_model_input.csv"
        ordered.to_csv(outpath, index=False, lineterminator="\n")
        feature_tables[name] = ordered

    all_columns = sorted(set().union(*(set(f.columns) - META - {"y", "fold"}
                                      for f in [*frames.values(), base])))
    dynamic_parents = {
        "f3_profile_km_dev": ["traj_km_20d", "monthly_avg_mileage"],
        "f3_profile_hours_dev": ["traj_hours_20d", "monthly_avg_hours"],
        "f3_profile_night_dev": ["f3_night_exposure_share", "night_hours_ratio"],
        "f2_prior_incident_x_highway": ["f2_prior_incident_per_1000km", "highway_ratio"],
        "f2_prior_incident_x_night": ["f2_prior_incident_per_1000km", "night_hours_ratio"],
    }
    for col in all_columns:
        if col in PORTRAIT_DYNAMIC:
            reason, source_end, decision = "target_vehicles dynamic portrait; official aggregate through 2026-07-31", "2026-07-31", "exclude_future"
        elif col in PORTRAIT_STATIC_UNVERIFIED:
            reason, source_end, decision = "target_vehicles static status not independently proven at proxy cutoff", "unknown", "exclude_unverified"
        elif col in PORTRAIT_DERIVED:
            reason, source_end, decision = "interaction depends on future target_vehicles dynamic portrait input", "2026-07-31", "exclude_dependency"
        elif col in portrait_descendants:
            reason, source_end, decision = "descends from full-population portrait/profile/cohort source", "2026-07-31 or full fleet", "exclude_dependency"
        else:
            reason, source_end, decision = "per-vehicle historical feature table; producer declares [as_of-20d, as_of); cutoff verified from row metadata", "2026-06-21", "retain_candidate"
        parents = dynamic_parents.get(col, [])
        if "_cohort_" in col:
            parents = [col.rsplit("_cohort_", 1)[0], "energy_type", "highway_ratio"]
        visibility[col] = {"feature": col, "source": "existing F0/F1/F3 20-day feature outputs" if decision == "retain_candidate" else "target_vehicles/profile descendants",
                           "source_end": source_end, "as_of": "2026-06-21", "parents": parents,
                           "fit_scope": "per-vehicle window; selectors/imputers fitted within training fold",
                           "decision": decision, "reason": reason}
    pd.DataFrame(visibility.values()).sort_values("feature").to_csv(y0 / "visibility.csv", index=False, lineterminator="\n")
    (y0 / "issues.json").write_text(json.dumps({"status": "partial", "unresolved": [
        "Historical source table timestamps were not rescanned; cutoff is supported by pipeline window guards and 20-day model-input metadata.",
        "energy_type is excluded because its static-at-cutoff provenance is not independently verified.",
        "Human review of public proxy labels/splits and the visibility map remains pending.",
    ], "excluded_dynamic_columns": sorted(PORTRAIT_DYNAMIC), "excluded_static_unverified": sorted(PORTRAIT_STATIC_UNVERIFIED),
       "excluded_portrait_interactions": sorted(PORTRAIT_DERIVED),
       "excluded_profile_descendants": sorted(portrait_descendants)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fold_counts = target.groupby(["fold", "y"]).size().unstack(fill_value=0).to_dict(orient="index")
    in_manifest = {k: {"path": str(p), "sha256": sha(p)} for k, p in source_paths.items()}
    out_manifest = {p.name: {"sha256": sha(p), "rows": int(len(pd.read_csv(p, nrows=0).columns))} for p in y1.glob("*_model_input.csv")}
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": "label_v2_record_count_20260923",
              "split_version": str(splits.split_version.iloc[0]), "as_of": "2026-06-21T00:00:00+08:00",
              "feature_window": "[2026-06-01,2026-06-21)", "label_window": str(labels.label_window.iloc[0]),
              "rows": len(target), "positives": int(target.y.sum()), "fold_counts": fold_counts,
              "inputs": in_manifest, "outputs": {p.name: {"sha256": sha(p), "columns": len(pd.read_csv(p, nrows=0).columns)} for p in sorted(y1.glob("*_model_input.csv"))},
              "excluded": {"dynamic_portrait": sorted(PORTRAIT_DYNAMIC), "static_unverified": sorted(PORTRAIT_STATIC_UNVERIFIED),
                           "portrait_interactions": sorted(PORTRAIT_DERIVED), "profile_descendants": sorted(portrait_descendants)},
              "git_head": __import__("subprocess").run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
              "worktree_dirty": bool(__import__("subprocess").run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())}
    (y1 / "input_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (y1 / "check.json").write_text(json.dumps({"machine_status": "pass", "human_review": "pending", "rows": len(target),
        "positive_count": int(target.y.sum()), "feature_columns": {k: len(v.columns) - 8 for k, v in feature_tables.items()},
        "fold_counts": fold_counts, "all_ids_equal": True, "old_y_fold_rebound": True,
        "blocked_columns_absent": all(not (set(t.columns) & blocked) for t in feature_tables.values())}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
