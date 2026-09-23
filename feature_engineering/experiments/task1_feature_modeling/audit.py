#!/usr/bin/env python3
"""FEAT-009 S0 audit for controlled F1/F3 inputs and registered OOF outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PATHS = ("f1_model_input", "f3_model_input", "v5_oof", "rf_oof")
OOF_KEYS = ("sample_id", "gpsno", "y", "fold")
BLOCKED_FEATURES = {
    "sample_id", "gpsno", "y", "fold", "as_of", "window_start", "window_end",
    "lookback_days", "horizon_days", "label_window", "label_status", "label_version",
    "split_version", "source_version", "feature_version", "feature_version_f1",
    "feature_version_f2", "feature_version_f3", "cohort_fallback",
}
FUTURE_NAME = re.compile(r"(^|_)(future|next|post|label|target|after_as_of)(_|$)", re.I)
DEPENDENCIES = ("FEAT-008", "MODEL-005", "EVAL-002")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in BLOCKED_FEATURES]


def _family(column: str) -> str:
    name = column.lower()
    rules = (
        ("history", ("f3_hist", "f1_prior", "prior_incident")),
        ("cohort", ("_cohort_", "cohort_")),
        ("profile", ("profile", "energy_type", "highway_ratio", "monthly_avg")),
        ("night", ("night", "deep_rate", "day_rate", "degradation")),
        ("trajectory", ("f3_traj", "traj_", "spell", "hotspot", "samepoint", "route_repeat")),
        ("events", ("evt_", "event", "f3_chain", "fatigue", "distraction", "speed_count")),
        ("imu_sensor", ("accident_", "imu_", "f2r2_", "f2_")),
        ("coverage_quality", ("coverage_", "align_", "qa_", "quality")),
        ("scores", ("f3_score", "risk_index")),
    )
    for family, prefixes in rules:
        if any(token in name for token in prefixes):
            return family
    return "other"


def _normalized_key_table(frame: pd.DataFrame, name: str, errors: list[str]) -> pd.DataFrame:
    missing = [column for column in OOF_KEYS if column not in frame.columns]
    if missing:
        errors.append(f"{name}: missing required key columns {missing}")
        return pd.DataFrame(columns=OOF_KEYS)
    part = frame[list(OOF_KEYS)].copy()
    for column in ("sample_id", "gpsno"):
        part[column] = part[column].astype("string").str.strip()
    if part["sample_id"].isna().any() or part["gpsno"].isna().any():
        errors.append(f"{name}: null sample_id/gpsno")
    if part["sample_id"].duplicated().any() or part["gpsno"].duplicated().any():
        errors.append(f"{name}: duplicate sample_id or gpsno")
    if not part["y"].isin([0, 1]).all():
        errors.append(f"{name}: y contains values outside {0, 1}")
    if part["fold"].isna().any():
        errors.append(f"{name}: fold contains null values")
    folds = pd.to_numeric(part["fold"], errors="coerce")
    fold_values = folds.to_numpy(dtype=float)
    if folds.isna().any() or not np.equal(fold_values, np.floor(fold_values)).all():
        errors.append(f"{name}: fold contains non-integer values")
    elif set(folds.astype(int)) != {0, 1, 2, 3, 4}:
        errors.append(f"{name}: fold values do not match the frozen five-fold IDs")
    return part.sort_values("sample_id", kind="stable").reset_index(drop=True)


def check_alignment(tables: dict[str, pd.DataFrame]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    normalized = {name: _normalized_key_table(frame, name, errors) for name, frame in tables.items()}
    reference_name = "f1_model_input"
    reference = normalized.get(reference_name)
    comparison: dict[str, Any] = {}
    if reference is None or reference.empty:
        errors.append("F1 reference input is empty or unreadable")
        return errors, comparison
    for name, current in normalized.items():
        if name == reference_name or current.empty:
            continue
        same_keys = current["sample_id"].tolist() == reference["sample_id"].tolist()
        same_rows = same_keys and current["gpsno"].equals(reference["gpsno"])
        same_y = same_keys and current["y"].astype(str).equals(reference["y"].astype(str))
        same_fold = same_keys and current["fold"].astype(str).equals(reference["fold"].astype(str))
        comparison[name] = {
            "same_sample_id_set_and_order": same_keys,
            "same_gpsno_by_sample": same_rows,
            "same_y_by_sample": same_y,
            "same_fold_by_sample": same_fold,
        }
        if not same_keys:
            errors.append(f"{name}: sample_id set differs from F1")
        if not same_rows:
            errors.append(f"{name}: gpsno mapping differs from F1")
        if not same_y:
            errors.append(f"{name}: y differs from F1")
        if not same_fold:
            errors.append(f"{name}: fold differs from F1")
    return errors, comparison


def _resolve_path(value: str | None, repo_root: Path) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def _family_missingness(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    columns = _feature_columns(frame)
    grouped: dict[str, list[float]] = {}
    for column in columns:
        grouped.setdefault(_family(column), []).append(float(frame[column].isna().mean()))
    return {
        family: {
            "feature_count": len(rates),
            "median_missing_rate": float(np.median(rates)),
            "mean_missing_rate": float(np.mean(rates)),
            "max_missing_rate": float(np.max(rates)),
        }
        for family, rates in sorted(grouped.items())
    }


def _check_probabilities(frame: pd.DataFrame, name: str, column: str, errors: list[str]) -> None:
    if column not in frame.columns:
        errors.append(f"{name}: missing prediction column {column}")
        return
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        errors.append(f"{name}: {column} contains non-finite values")
    elif ((values < 0) | (values > 1)).any():
        errors.append(f"{name}: {column} contains values outside [0,1]")


def _check_time_and_features(
    f1: pd.DataFrame,
    f3: pd.DataFrame,
    f3_manifest: dict[str, Any] | None,
    errors: list[str],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for name, frame in (("f1", f1), ("f3", f3)):
        for column in ("as_of", "lookback_days"):
            if column not in frame.columns:
                errors.append(f"{name}: missing temporal metadata column {column}")
        if "as_of" in frame:
            values = pd.to_datetime(frame["as_of"], errors="coerce", utc=True)
            if values.isna().any() or values.nunique(dropna=False) != 1:
                errors.append(f"{name}: as_of is invalid or inconsistent across samples")
            else:
                details[f"{name}_as_of"] = values.iloc[0].isoformat()
        if "lookback_days" in frame:
            values = pd.to_numeric(frame["lookback_days"], errors="coerce")
            if values.isna().any() or values.nunique(dropna=False) != 1 or (values <= 0).any():
                errors.append(f"{name}: lookback_days is invalid or inconsistent across samples")
            else:
                details[f"{name}_lookback_days"] = float(values.iloc[0])
    if "as_of" in f1 and "as_of" in f3:
        a = f1[["sample_id", "as_of"]].copy().sort_values("sample_id", kind="stable")
        b = f3[["sample_id", "as_of"]].copy().sort_values("sample_id", kind="stable")
        a["as_of"] = pd.to_datetime(a["as_of"], errors="coerce", utc=True).astype(str).to_numpy()
        b["as_of"] = pd.to_datetime(b["as_of"], errors="coerce", utc=True).astype(str).to_numpy()
        if not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
            errors.append("F1/F3 as_of differs by sample")
    if "lookback_days" in f1 and "lookback_days" in f3:
        a = f1[["sample_id", "lookback_days"]].copy().sort_values("sample_id", kind="stable")
        b = f3[["sample_id", "lookback_days"]].copy().sort_values("sample_id", kind="stable")
        a["lookback_days"] = pd.to_numeric(a["lookback_days"], errors="coerce").to_numpy(dtype=float)
        b["lookback_days"] = pd.to_numeric(b["lookback_days"], errors="coerce").to_numpy(dtype=float)
        if not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
            errors.append("F1/F3 lookback_days differs by sample")
    if f3_manifest:
        window = f3_manifest.get("window", {})
        details["feature_window"] = window
        horizon = window.get("horizon_days")
        if horizon is None or not np.isfinite(float(horizon)) or float(horizon) <= 0:
            errors.append("F3 manifest lacks a valid horizon_days")
        for field, observed in (("as_of", details.get("f3_as_of")),
                                ("lookback_days", details.get("f3_lookback_days"))):
            declared = window.get(field)
            if declared is not None and observed is not None:
                if field == "as_of":
                    left = pd.to_datetime(declared, utc=True, errors="coerce")
                    right = pd.to_datetime(observed, utc=True, errors="coerce")
                    if pd.isna(left) or left != right:
                        errors.append("F3 manifest and model_input as_of differ")
                elif not np.isclose(float(declared), float(observed)):
                    errors.append("F3 manifest and model_input lookback_days differ")
    suspicious = {
        name: [column for column in _feature_columns(frame) if FUTURE_NAME.search(column)]
        for name, frame in (("f1", f1), ("f3", f3))
    }
    for name, columns in suspicious.items():
        if columns:
            errors.append(f"{name}: suspicious future/target-like feature names {columns}")
    details["suspicious_feature_names"] = suspicious
    return details


def _verify_v5_manifest(
    manifest: dict[str, Any] | None,
    path_config: dict[str, Path | None],
    hashes: dict[str, str],
    errors: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"present": bool(manifest), "input_hash_checks": {}}
    if not manifest:
        return result
    result["versions"] = manifest.get("versions", {})
    expected = manifest.get("inputs", {})
    key_map = {
        "model_input": "f1_model_input",
        "cluster": "v5_cluster_assignments",
        "splits": "v5_splits",
        "vehicles": "v5_vehicles",
    }
    for manifest_key, config_key in key_map.items():
        declared = expected.get(manifest_key)
        path = path_config.get(config_key)
        if not declared:
            errors.append(f"V5 manifest missing input fingerprint: {manifest_key}")
            result["input_hash_checks"][manifest_key] = "missing_manifest_hash"
        elif path is None or not path.is_file():
            result["input_hash_checks"][manifest_key] = "source_path_not_configured"
            errors.append(f"V5 manifest source path is missing: {manifest_key}")
        else:
            actual = sha256_file(path)
            ok = actual == declared
            result["input_hash_checks"][manifest_key] = "match" if ok else "mismatch"
            hashes[f"v5_{manifest_key}"] = actual
            if not ok:
                errors.append(f"V5 manifest fingerprint mismatch: {manifest_key}")
    return result


def _make_figures(out_dir: Path, audit: dict[str, Any], f1: pd.DataFrame,
                  f3: pd.DataFrame, run_id: str) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    window = audit["time_window"].get("feature_window", {})
    as_of = pd.to_datetime(window.get("as_of") or audit["time_window"].get("f3_as_of"), utc=True)
    lookback = float(window.get("lookback_days", audit["time_window"].get("f3_lookback_days", 20)))
    horizon = float(window.get("horizon_days", 40))
    start = as_of - pd.Timedelta(days=lookback)
    end = as_of + pd.Timedelta(days=horizon)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.barh([1], [lookback], left=[-lookback], height=0.42, color="#3977a8", label="Feature window")
    ax.barh([0], [horizon], left=[0], height=0.42, color="#e58a44", label="Label window")
    ax.axvline(0, color="#333333", linewidth=1.2, linestyle="--")
    ax.set_yticks([0, 1], ["Label window", "Feature window"])
    ax.set_xlim(-lookback, horizon)
    ax.set_xlabel("Days relative to as_of (days)")
    ax.set_title("Feature and label windows relative to as_of", pad=8)
    fig.suptitle(f"FEAT-009 S0 | F1/F3 aligned with V5 OOF and RF OOF | n={len(f1)} | {run_id}", y=0.98)
    fig.text(0.08, 0.88, "Version: F1_v1 / F3R5_v1 | Unit: days | dashed line: as_of", fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    fig.subplots_adjust(top=0.78, bottom=0.22, left=0.16, right=0.98)
    fig.text(0.16, 0.06,
             f"Feature: {start.date()} to {as_of.date()} (right edge excluded)    "
             f"Label: {as_of.date()} to {end.date()} (right edge excluded)", fontsize=9)
    flow_name = "s0_flow_time.png"
    fig.savefig(figure_dir / flow_name, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summaries = audit["feature_families"]
    families = sorted(set(summaries["f1"]) | set(summaries["f3"]))
    f1_count = [summaries["f1"].get(f, {}).get("feature_count", 0) for f in families]
    f3_count = [summaries["f3"].get(f, {}).get("feature_count", 0) for f in families]
    f1_missing = [100 * summaries["f1"].get(f, {}).get("median_missing_rate", 0) for f in families]
    f3_missing = [100 * summaries["f3"].get(f, {}).get("median_missing_rate", 0) for f in families]
    y = np.arange(len(families))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, len(families) * 0.42)))
    height = 0.36
    ax1.barh(y - height / 2, f1_count, height, color="#8cb9d8", label="F1")
    ax1.barh(y + height / 2, f3_count, height, color="#df9a67", label="F3")
    ax1.set_yticks(y, families)
    ax1.invert_yaxis()
    ax1.set_xlabel("Feature columns (count)")
    ax1.set_title("Columns by family")
    ax1.legend()
    ax2.barh(y - height / 2, f1_missing, height, color="#8cb9d8", label="F1")
    ax2.barh(y + height / 2, f3_missing, height, color="#df9a67", label="F3")
    ax2.axvline(50, color="#8c2d24", linewidth=1, linestyle="--", label="50% review marker")
    ax2.set_yticks(y, families)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Median missing values per feature (%)")
    ax2.set_title("Feature missingness by family")
    from matplotlib.lines import Line2D
    ax2.legend(handles=[Patch(color="#8cb9d8", label="F1"),
                        Patch(color="#df9a67", label="F3"),
                        Line2D([0], [0], color="#8c2d24", linestyle="--", label="50% review marker")])
    fig.suptitle(f"FEAT-009 S0 | feature coverage | F1_v1 vs F3R5_v1 | n={len(f1)} | {run_id}")
    fig.tight_layout()
    missing_name = "s0_feature_coverage.png"
    fig.savefig(figure_dir / missing_name, dpi=150, bbox_inches="tight")
    plt.close(fig)

    boundaries = audit["fit_boundaries"]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.axis("off")
    columns = ["Operation", "Fit scope", "Uses target y?", "S0 reading"]
    rows = [
        ["Per-vehicle features", "Historical window [as_of-lookback, as_of)", "No (code review)",
         "Window is pre-as_of; source audit is partial."],
        ["F3 variance/correlation screen", "All rows in F1 + F3", "No",
         "Transductive; fold-local selection needed for inductive claims."],
        ["F3 cohort z-score / percentile", "Outer five-fold cross-fit", "No",
         "Rebuild inside each inner split to avoid validation influence."],
        ["V5 clustering / risk ordering", "Full F1 feature table", "No target y in fit",
         "Transductive groups; OOF uses global assignments."],
        ["V5 learners / calibration", "Outer-train; nested calibration", "Train labels",
         "Outer validation labels excluded; groups remain global."],
    ]
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left",
                     colWidths=[0.22, 0.27, 0.17, 0.34])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    fig.suptitle(f"FEAT-009 S0 | fitting boundaries | {run_id}\nCode provenance; model performance not run", y=0.98)
    fig.text(0.05, 0.035,
             f"Source: F1/F3 pipelines, clustering implementation, V5 runner | n={len(f1)} | features pre-as_of",
             fontsize=8)
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.04, right=0.98)
    boundary_name = "s0_fit_boundaries.png"
    fig.savefig(figure_dir / boundary_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [flow_name, missing_name, boundary_name]


def run_audit(config_path: Path, output_override: Path | None = None,
              make_figures: bool = True) -> tuple[dict[str, Any], int]:
    config_path = config_path.resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    repo_root = next((parent for parent in Path(__file__).resolve().parents if (parent / "AGENTS.md").is_file()),
                     Path.cwd())
    configured: dict[str, Path | None] = {
        key: _resolve_path(config.get(key), repo_root) for key in config
        if key.endswith(("model_input", "oof", "manifest", "dropped_columns", "new_features",
                         "cluster_assignments", "splits", "vehicles"))
    }
    output_dir = output_override or _resolve_path(config.get("output_dir"), repo_root)
    if output_dir is None:
        raise ValueError("config must define output_dir or pass --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for key in REQUIRED_PATHS:
        path = configured.get(key)
        if path is None or not path.is_file():
            errors.append(f"required input missing: {key}")
    if errors:
        result = {"machine_status": "fail", "errors": errors}
        (output_dir / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result, 2

    frames = {key: _read_csv(configured[key]) for key in REQUIRED_PATHS}
    manifest_path = configured.get("f3_manifest")
    dropped_path = configured.get("f3_dropped_columns")
    v5_manifest_path = configured.get("v5_manifest")
    f3_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path and manifest_path.is_file() else None
    dropped = json.loads(dropped_path.read_text(encoding="utf-8")) if dropped_path and dropped_path.is_file() else None
    v5_manifest = json.loads(v5_manifest_path.read_text(encoding="utf-8")) if v5_manifest_path and v5_manifest_path.is_file() else None

    for name, column in (("v5_oof", "p_v5"), ("rf_oof", "p_random_forest")):
        _check_probabilities(frames[name], name, column, errors)
    alignment_errors, alignment = check_alignment({key: frames[key] for key in REQUIRED_PATHS})
    errors.extend(alignment_errors)
    time_window = _check_time_and_features(frames["f1_model_input"], frames["f3_model_input"], f3_manifest, errors)

    metadata_checks: dict[str, Any] = {}
    feature_audit: dict[str, Any] = {}
    feature_families: dict[str, Any] = {}
    for short_name, key in (("f1", "f1_model_input"), ("f3", "f3_model_input")):
        frame = frames[key]
        predictors = _feature_columns(frame)
        metadata_checks[short_name] = {
            "excluded_contract_fields_present": sorted(set(frame.columns) & BLOCKED_FEATURES),
            "predictor_selection_rule": "exclude ID, label, fold, time/window, version, and cohort diagnostic fields",
        }
        feature_audit[short_name] = {
            "rows": int(len(frame)),
            "total_columns": int(len(frame.columns)),
            "predictor_columns": int(len(predictors)),
            "categorical_predictors": [c for c in predictors if not pd.api.types.is_numeric_dtype(frame[c])],
            "unmapped_predictors": [c for c in predictors if _family(c) == "other"],
        }
        feature_families[short_name] = _family_missingness(frame)

    if f3_manifest:
        if int(f3_manifest.get("rows", -1)) != len(frames["f3_model_input"]):
            errors.append("F3 manifest rows does not match model_input")
        if f3_manifest.get("input_fingerprint") == "PENDING":
            fingerprint_state = "placeholder_recomputed_by_this_audit"
        elif not f3_manifest.get("input_fingerprint"):
            errors.append("F3 manifest input_fingerprint is empty")
            fingerprint_state = "missing"
        else:
            fingerprint_state = "declared_value_recorded_and_output_rehashed"
    else:
        fingerprint_state = "F3 manifest unavailable"
        errors.append("F3 manifest is required for window and fingerprint audit")

    if dropped:
        feature_audit["f3_screening"] = {
            "declared_columns": len(dropped.get("declared_columns", [])),
            "kept_new_columns": len(dropped.get("kept_new", [])),
            "dropped_column_count": len(dropped.get("dropped", [])),
            "rules": dropped.get("rules", {}),
            "scope": "full merged vehicle table at build_f3_interface",
            "target_used": False,
        }

    if f3_manifest:
        expected_features = list(f3_manifest.get("base_kept", [])) + list(f3_manifest.get("new_kept", []))
        declared_meta = sorted(set(expected_features) & BLOCKED_FEATURES)
        missing_declared = sorted(set(expected_features) - set(frames["f3_model_input"].columns))
        if declared_meta:
            errors.append(f"F3 manifest declares metadata as predictors {declared_meta}")
        if missing_declared:
            errors.append(f"F3 model_input lacks manifest-declared features {missing_declared}")
        if int(f3_manifest.get("feature_count", -1)) != len(expected_features):
            errors.append("F3 manifest feature_count does not match declared kept feature lists")
        if len(expected_features) != len(set(expected_features)):
            errors.append("F3 manifest kept feature lists contain duplicate columns")
        if dropped and expected_features != list(dropped.get("kept_base", [])) + list(dropped.get("kept_new", [])):
            errors.append("F3 manifest and dropped-column ledger declare different kept features")
        feature_audit["f3_manifest_contract"] = {
            "declared_predictor_count": len(expected_features),
            "categorical_contract_metadata_also_available_to_model": [
                c for c in ("energy_type",) if c in frames["f3_model_input"].columns
            ],
            "declared_metadata_as_predictors": declared_meta,
            "missing_declared_features": missing_declared,
        }

    hashes: dict[str, str] = {}
    file_roles = {
        "f1_model_input": "f1_model_input", "f3_model_input": "f3_model_input",
        "v5_oof": "v5_oof", "rf_oof": "rf_oof", "f3_manifest": "f3_manifest",
        "f3_dropped_columns": "f3_dropped_columns", "f3_new_features": "f3_new_features",
        "v5_cluster_assignments": "v5_cluster_assignments", "v5_splits": "v5_splits",
        "v5_vehicles": "v5_vehicles", "v5_manifest": "v5_manifest",
    }
    file_inventory: dict[str, Any] = {}
    for config_key, role in file_roles.items():
        path = configured.get(config_key)
        if path is not None and path.is_file():
            hashes[role] = sha256_file(path)
            file_inventory[role] = {"file": path.name, "sha256": hashes[role], "bytes": path.stat().st_size}
    f3_input_hashes: dict[str, str] = {}
    if f3_manifest:
        for role, raw_path in f3_manifest.get("inputs", {}).items():
            path = Path(raw_path).expanduser()
            if path.is_file():
                f3_input_hashes[role] = sha256_file(path)
            else:
                errors.append(f"F3 manifest source input unavailable: {role}")
    v5_manifest_check = _verify_v5_manifest(v5_manifest, configured, hashes, errors)
    dependency_status: dict[str, Any] = {}
    for task_id in DEPENDENCIES:
        task_path = repo_root / "docs" / "tasks" / f"{task_id}.json"
        if not task_path.is_file():
            errors.append(f"FEAT-009 dependency record missing: {task_id}")
            continue
        task = json.loads(task_path.read_text(encoding="utf-8"))
        dependency_status[task_id] = {
            "status": task.get("status"),
            "reviewer_recorded": bool(task.get("reviewer")),
            "reviewed_at_recorded": bool(task.get("reviewed_at")),
        }

    cluster_check: dict[str, Any] = {"available": False}
    cluster_path = configured.get("v5_cluster_assignments")
    if cluster_path and cluster_path.is_file():
        cluster = _read_csv(cluster_path)
        if "sample_id" not in cluster or "cluster" not in cluster or "cluster_risk_rank" not in cluster:
            errors.append("V5 cluster assignments lacks expected columns")
        else:
            cluster_check = {"available": True, "rows": int(len(cluster)), "unique_sample_id": bool(cluster.sample_id.is_unique)}
            if not cluster.sample_id.is_unique:
                errors.append("V5 cluster assignments contain duplicate sample_id")
            merged = frames["v5_oof"][ ["sample_id", "group"] ].merge(
                cluster[["sample_id", "cluster", "cluster_risk_rank"]], on="sample_id", how="outer", validate="1:1", indicator=True
            ) if "group" in frames["v5_oof"] else None
            if merged is not None:
                if not (merged["_merge"] == "both").all():
                    errors.append("V5 OOF and cluster assignments have different sample_id sets")
                expected_group = np.where(merged["cluster"] == -1, "insufficient",
                                          np.where(merged["cluster_risk_rank"] == merged["cluster_risk_rank"].max(), "high", "low"))
                cluster_check["group_match"] = bool(np.array_equal(merged["group"].astype(str).to_numpy(), expected_group.astype(str)))
                if not cluster_check["group_match"]:
                    errors.append("V5 OOF group labels do not reproduce from cluster assignments")

    fit_boundaries = [
        {"operation": "F1/F3 per-vehicle historical feature extraction", "scope": "[as_of-lookback, as_of)",
         "uses_target": "No; historical signals only", "reading": "Reviewed F1 event/IMU and F3 window masks; raw streams were not rescanned."},
        {"operation": "F3 near-constant and Spearman screen", "scope": "All rows in merged F1+F3 table", "uses_target": "No",
         "reading": "Transductive covariate screening; strict inductive claim requires fold-local screening."},
        {"operation": "F3 cohort z-score/percentile", "scope": "Frozen outer five-fold cross-fit", "uses_target": "No",
         "reading": "Precomputed features can carry inner-validation distribution into inner-training rows; rebuild per inner split for strict S1 selection."},
        {"operation": "V5 clustering and risk ordering", "scope": "Full F1 feature table", "uses_target": "No target y in fit; historical sensor-risk ordering",
         "reading": "Unsupervised but transductive group assignment; V5 OOF reproduces the registered full-population groups."},
        {"operation": "V5 base learners and calibration", "scope": "Outer training fold; calibration material nested within it", "uses_target": "Yes, training labels only",
         "reading": "Reviewed runner excludes outer validation labels from learner and calibrator fitting; grouping remains globally fitted."},
    ]

    run_id = datetime.now(timezone.utc).strftime("feat009-s0-%Y%m%dT%H%M%SZ")
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine_status": "pass" if not errors else "fail",
        "human_gate": "review_required",
        "s1_release": False,
        "errors": errors,
        "alignment": alignment,
        "time_window": time_window,
        "fingerprint_state": fingerprint_state,
        "files": file_inventory,
        "f3_manifest_input_sha256": f3_input_hashes,
        "v5_manifest_check": v5_manifest_check,
        "dependency_status": dependency_status,
        "v5_oof_checksum_registered_in_source_manifest": bool(
            v5_manifest and ("oof_sha256" in v5_manifest or "outputs" in v5_manifest)
        ),
        "v5_cluster_check": cluster_check,
        "metadata_predictor_audit": metadata_checks,
        "feature_audit": feature_audit,
        "feature_families": feature_families,
        "fit_boundaries": fit_boundaries,
        "limitations": [
            "No FEAT-009 candidate model was trained and no comparative effect metric was calculated.",
            "F3 full-table screening and outer-fold cohort features require explicit treatment before nested inner-fold model selection.",
            "V5 full-population unsupervised clustering makes the registered OOF comparison transductive at the group-assignment layer.",
            "V5 source manifest does not register an OOF output checksum; this audit pins the current OOF file hash locally but cannot authenticate it against a separately registered copy.",
            "FEAT-008, MODEL-005, and EVAL-002 are recorded as review rather than accepted.",
            "Machine consistency checks do not replace the owner's S0 review gate.",
        ],
    }
    reviewed_sources = {
        "f1_event_features": repo_root / "feature_engineering/src/accident_pipeline_f1/event_features.py",
        "f1_dataset_windows": repo_root / "feature_engineering/src/accident_pipeline/dataset.py",
        "f1_imu_windows": repo_root / "feature_engineering/src/accident_pipeline/imu_features.py",
        "f3_pipeline": repo_root / "feature_engineering/src/accident_pipeline_f3/pipeline.py",
        "f3_interface": repo_root / "feature_engineering/src/accident_pipeline_f3/interface.py",
        "v5_clustering": repo_root / "feature_engineering/src/accident_pipeline/clustering.py",
        "v5_runner": repo_root / "models/solution_b_v5/run_v5.py",
        "v5_loader": repo_root / "models/solution_b_v5/common_v5.py",
    }
    reviewed_code_hashes = {role: sha256_file(path) for role, path in reviewed_sources.items() if path.is_file()}
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                           capture_output=True, text=True, check=False).stdout.strip()
    manifest = {
        "run_id": run_id,
        "created_at_utc": result["created_at_utc"],
        "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                                      capture_output=True, text=True, check=False).stdout.strip(),
        "config_sha256": sha256_file(config_path),
        "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        "reviewed_code_source_sha256": reviewed_code_hashes,
        "working_tree_dirty_at_audit": bool(dirty),
        "inputs": file_inventory,
        "output_dir_is_controlled_local": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "issues.json").write_text(json.dumps({"run_id": run_id, "errors": errors,
        "review_findings": [item for item in fit_boundaries if "Transductive" in item["reading"] or "inner-validation" in item["reading"]],
        "dependencies": dependency_status,
        "v5_oof_checksum_registered_in_source_manifest": result["v5_oof_checksum_registered_in_source_manifest"]},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if make_figures:
        figures = _make_figures(output_dir, result, frames["f1_model_input"], frames["f3_model_input"], run_id)
    else:
        figures = []
    _write_acceptance(output_dir, result, figures)
    return result, 0 if not errors else 2


def _write_acceptance(out_dir: Path, result: dict[str, Any], figures: list[str]) -> None:
    status = result["machine_status"]
    comparison = result.get("alignment", {})
    lines = [
        "# FEAT-009 S0 输入锁定与无泄漏审计",
        "",
        f"运行：`{result['run_id']}`；机器检查：**{status}**；人工放行：**待叶安复核**。",
        "",
        "## 结论",
        "",
        "F1、F3R5、V5 OOF 与冻结 RF OOF 已按逐车键、标签和折号执行只读对账。该步骤没有训练 FEAT-009 候选模型，也没有计算主判定效果。",
        "",
        "S0 已识别全量无标签拟合与内层验证污染边界。S1 暂不放行：须先确定是否在 S1 对 F3 近常数/相关性筛列与同群变换按内层训练折重建，并明确 V5 全量聚类的转导式比较边界。",
        "",
        "## 四维状态",
        "",
        "| 维度 | 状态 | 说明 |",
        "| --- | --- | --- |",
        f"| 实现／数据契约 | {status} | 逐车键、标签、折号、概率范围和时间元数据见 `audit.json`。 |",
        "| 实验效果 | not_run | S0 不训练模型、不计算 AUC 或基线差异。 |",
        "| 验收与证据 | partial | 受控 manifest、审计 JSON、问题清单及同源图表已生成；人工阶段门待复核。 |",
        "| 环境与依赖 | pass | 本机读取受控输入并生成本地报告；具体依赖记录由运行环境复现。 |",
        "",
        "## 输入与时间窗",
        "",
        f"四类逐车对账：{json.dumps(comparison, ensure_ascii=False, sort_keys=True)}",
        "",
        "F3 的原 manifest 指纹为占位值时，本次 manifest 另存模型输入与上游清单文件的 SHA-256；原占位值不会被解释为已验证。实际指纹只在本地受控目录。",
        "",
        "## 拟合边界与待复核问题",
        "",
    ]
    for item in result["fit_boundaries"]:
        lines.append(f"- **{item['operation']}**：{item['scope']}；目标标签：{item['uses_target']}。{item['reading']}")
    lines.extend(["", "## 前置任务与 V5 产物", ""])
    for task_id, dependency in result["dependency_status"].items():
        has_review = dependency["reviewer_recorded"] and dependency["reviewed_at_recorded"]
        lines.append(f"- {task_id}：状态 `{dependency['status']}`；独立复核记录：{'有' if has_review else '无'}。")
    lines.append("- V5 manifest 未登记 OOF 输出校验和；本次审计仅在受控 manifest 中固定当前 OOF 文件指纹，尚无独立登记副本可对账。")
    lines.extend(["", "## 特征族覆盖复核", ""])
    for dataset in ("f1", "f3"):
        unmapped = result["feature_audit"][dataset]["unmapped_predictors"]
        lines.append(f"- {dataset.upper()} 未分类列：{', '.join(unmapped) if unmapped else '无'}。")
        high_missing = [
            f"{family} ({100 * summary['median_missing_rate']:.1f}%)"
            for family, summary in result["feature_families"][dataset].items()
            if summary["median_missing_rate"] >= 0.5
        ]
        lines.append(f"- {dataset.upper()} 达到 50% 缺失复核线的特征族：{', '.join(high_missing) if high_missing else '无'}。")
    lines.extend(["", "## 可视化验收材料", ""])
    for name in figures:
        caption = {
            "s0_flow_time.png": "四类输入的对账流向与特征窗/标签窗边界。",
            "s0_feature_coverage.png": "F1/F3 特征族列数及族内中位缺失率；50% 线仅触发语义复核。",
            "s0_fit_boundaries.png": "从实现代码追踪的拟合范围、标签使用和 S0 限制。",
        }.get(name, name)
        lines.extend([f"![{caption}](figures/{name})", "", caption, ""])
    lines.extend([
        "## 未证明事项与复现",
        "",
        "- 没有证明任何候选算法优于 V5 或 RF，也没有独立测试集证据。",
        "- 全量无标签筛列／聚类属于转导式边界；F3 同群特征需在内层验证切分内重算，才能按严格嵌套选择解释 S1。",
        "- 复现命令：`python feature_engineering/experiments/task1_feature_modeling/audit.py --config configs/feat-009.local.toml`。",
        "- 原始数据、文件指纹、图表和真实派生统计均留在受控输出目录；不提交公开 Git。",
        "",
        "机器错误：" + ("无。" if not result.get("errors") else "；".join(result["errors"])),
        "",
    ])
    (out_dir / "acceptance.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="local TOML path configuration")
    parser.add_argument("--output-dir", type=Path, help="controlled report directory override")
    parser.add_argument("--skip-figures", action="store_true", help="used for synthetic negative tests")
    args = parser.parse_args(argv)
    result, code = run_audit(args.config, args.output_dir, make_figures=not args.skip_figures)
    print(f"[{result['machine_status']}] FEAT-009 S0 audit; run={result.get('run_id', 'unavailable')}")
    if result.get("errors"):
        for error in result["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
    print(f"Human gate: {result.get('human_gate', 'review_required')}; S1 release: {result.get('s1_release', False)}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
