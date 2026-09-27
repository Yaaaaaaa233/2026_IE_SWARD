"""Fit the frozen g8v6 recipe on proxy training rows and score August 1 rows.

No candidate search is performed. Feature construction is upstream; this runner
requires the same named feature columns at training and inference time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

try:
    from .prepare_submission import digest, package
except ImportError:  # direct ``python3 models/task1_final/run_g8v6_fullfit.py``
    from prepare_submission import digest, package

SEEDS = (42, 7, 2026)
WEIGHTS = (0.45, 0.17, 0.06, 0.32)
MEMBERS = ("g1", "g2", "g3", "m4")
BLOCKED = {
    "monthly_avg_mileage", "monthly_avg_hours", "monthly_avg_stops", "highway_ratio",
    "morning_ratio", "dusk_ratio", "night_hours_ratio", "night_mileage_ratio",
    "energy_type", "f2_prior_incident_x_night", "f2_prior_incident_x_highway",
}
META = {
    "sample_id", "gpsno", "y", "fold", "group", "as_of", "lookback_days",
    "label_window", "horizon_days", "label_version", "split_version", "label_status",
    "feature_version", "source_version",
}


def logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def expit(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def fold_prune(x: pd.DataFrame, rho: float = 0.95) -> list[str]:
    """Mirror the supplied g8v6/eval_stack full-training feature pruning."""
    keep: list[str] = []
    for column in x.columns:
        series = x[column]
        if series.std(ddof=0) < 1e-12:
            continue
        if all(not (abs(series.corr(x[prior])) > rho) for prior in keep):
            keep.append(column)
    if not keep:
        raise ValueError("no usable feature columns after pruning")
    return keep


def read_vehicles(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype={"gpsno": str})
    if "gpsno" not in frame or len(frame) != 500:
        raise ValueError("vehicle list must have 500 gpsno rows")
    ids = frame.gpsno.astype(str).str.strip()
    if ids.eq("").any() or ids.nunique() != 500:
        raise ValueError("vehicle list contains empty or duplicate gpsno")
    # Preserve the frozen protocol's row order: RF/bootstrap draws depend on it.
    return ids.tolist()


def read_labels(path: Path, vehicles: list[str]) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={"sample_id": str})
    required = {"sample_id", "y", "label_version", "label_window"}
    if not required <= set(labels) or len(labels) != 500:
        raise ValueError("expected 500 proxy labels with sample_id, y, version and window")
    labels["gpsno"] = labels.sample_id.str.split("_").str[0]
    if labels.gpsno.nunique() != 500 or set(labels.gpsno) != set(vehicles):
        raise ValueError("proxy label vehicle set differs from official targets")
    if set(labels.y.unique()) != {0, 1} or int(labels.y.sum()) != 54:
        raise ValueError("proxy labels do not match protocol v2")
    if set(labels.label_version) != {"label_v2_record_count_20260923"}:
        raise ValueError("proxy label version differs from frozen protocol")
    if set(labels.label_window) != {"2026-06-21/2026-07-31"}:
        raise ValueError("proxy label window differs from frozen protocol")
    return labels.set_index("gpsno").loc[vehicles]


def read_member(path: Path, vehicles: list[str], *, score: bool,
                expected_sample_ids: pd.Series | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"gpsno": str, "sample_id": str})
    if not {"gpsno", "sample_id"} <= set(frame) or len(frame) != 500:
        raise ValueError(f"{path.name}: expected 500 rows with sample_id and gpsno")
    if frame.gpsno.nunique() != 500 or set(frame.gpsno) != set(vehicles):
        raise ValueError(f"{path.name}: vehicle set differs from official targets")
    if score and ({"y", "fold", "label_window", "label_status"} & set(frame)):
        raise ValueError(f"{path.name}: inference features contain label or fold columns")
    frame = frame.set_index("gpsno").loc[vehicles]
    if expected_sample_ids is not None and not frame.sample_id.equals(expected_sample_ids):
        raise ValueError(f"{path.name}: training sample_id differs from proxy labels")
    columns = [c for c in frame if c not in META and c not in BLOCKED]
    if not columns:
        raise ValueError(f"{path.name}: no model features")
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if (frame[column].notna() & converted.isna()).any():
            raise ValueError(f"{path.name}: nonnumeric feature {column}")
        frame[column] = converted.astype(float)
    return frame[columns]


def prepare_member(train: pd.DataFrame, score: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict, SimpleImputer]:
    missing = sorted(set(train) - set(score))
    if missing:
        raise ValueError(f"inference features missing {len(missing)} training columns: {missing[:8]}")
    cols = fold_prune(train)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(train[cols]).astype(float)
    x_score = imputer.transform(score[cols]).astype(float)
    if not np.isfinite(x_train).all() or not np.isfinite(x_score).all():
        raise ValueError("imputed model matrix contains nonfinite values")
    schema = {"train_columns": list(train), "selected_columns": cols,
              "unused_score_columns": sorted(set(score) - set(train))}
    return x_train, x_score, schema, imputer


def ebm(seed: int, *, bins: int, leaf: int):
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(
        interactions=0, max_bins=bins, min_samples_leaf=leaf,
        random_state=seed, n_jobs=1,
    )


def fit_predict_member(name: str, seed: int, x_train: np.ndarray,
                       y: np.ndarray, x_score: np.ndarray) -> tuple[np.ndarray, list]:
    if name == "g1":
        models = [ebm(seed, bins=64, leaf=10), ebm(seed, bins=32, leaf=20),
                  RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                         max_features="sqrt", class_weight="balanced",
                                         random_state=seed, n_jobs=-1)]
        for model in models:
            model.fit(x_train, y)
        score = expit(np.mean([logit(model.predict_proba(x_score)[:, 1]) for model in models], axis=0))
    else:
        models = [ebm(seed, bins=32, leaf=20)]
        models[0].fit(x_train, y)
        score = models[0].predict_proba(x_score)[:, 1]
    return score, models


def hash_inputs(paths: dict[str, Path]) -> str:
    entries = {key: digest(path) for key, path in sorted(paths.items())}
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def validate_score_manifest(path: Path, score_paths: dict[str, Path]) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {"role": "future_features", "as_of": "2026-08-01",
                "feature_start": "2026-06-01", "feature_end_exclusive": "2026-08-01"}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"score manifest {key} must be {value!r}")
    hashes = manifest.get("member_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(MEMBERS):
        raise ValueError("score manifest must hash each frozen member table")
    for name in MEMBERS:
        if hashes[name] != digest(score_paths[name]):
            raise ValueError(f"score manifest hash mismatch for {name}")
    coverage = manifest.get("source_coverage")
    if not isinstance(coverage, dict) or not {"events_clean", "vehicle_day", "trajectory", "imu"} <= set(coverage):
        raise ValueError("score manifest must declare each source's observed coverage")
    for source in ("events_clean", "vehicle_day", "trajectory", "imu"):
        value = coverage[source]
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError(f"score manifest invalid observed date for {source}")
        if value > "2026-07-31":
            raise ValueError(f"score manifest {source} extends beyond July 31")
    if coverage["events_clean"] < "2026-07-31" or coverage["vehicle_day"] < "2026-07-31":
        raise ValueError("event and vehicle-day sources do not cover July 31")
    return manifest


def run(labels_path: Path, vehicles_path: Path, train_paths: dict[str, Path],
        score_paths: dict[str, Path], score_manifest_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    score_manifest = validate_score_manifest(score_manifest_path, score_paths)
    vehicles = read_vehicles(vehicles_path)
    labels = read_labels(labels_path, vehicles)
    y = labels.y.to_numpy(dtype=int)
    prepared = {}
    schemas = {}
    for name in MEMBERS:
        train = read_member(train_paths[name], vehicles, score=False,
                            expected_sample_ids=labels.sample_id)
        score = read_member(score_paths[name], vehicles, score=True)
        x_train, x_score, schemas[name], imp = prepare_member(train, score)
        prepared[name] = (x_train, x_score, imp)
    chains = []
    bundle = {"recipe": "sprint-g8v6-corrected-fullfit", "seeds": SEEDS,
              "weights": WEIGHTS, "members": {}, "schema": schemas}
    for seed in SEEDS:
        member_scores = []
        for name in MEMBERS:
            x_train, x_score, imp = prepared[name]
            probability, models = fit_predict_member(name, seed, x_train, y, x_score)
            member_scores.append(probability)
            bundle["members"][(name, seed)] = {"imputer": imp, "models": models}
        chains.append(sum(weight * logit(p) for weight, p in zip(WEIGHTS, member_scores)))
    probability = expit(np.mean(chains, axis=0))
    if not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
        raise ValueError("invalid final probability")
    output_dir.mkdir(parents=True)
    model_path = output_dir / "trained_bundle.joblib"
    joblib.dump(bundle, model_path)
    schema_path = output_dir / "feature_schema.json"
    schema_path.write_text(json.dumps(schemas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions = output_dir / "predictions.csv"
    pd.DataFrame({"gpsno": vehicles, "risk_prob": probability}).to_csv(
        predictions, index=False, lineterminator="\n", float_format="%.17g"
    )
    pd.DataFrame({"gpsno": vehicles}).to_csv(output_dir / "vehicles.csv", index=False,
                                               lineterminator="\n")
    provenance = {
        "prediction_role": "future_inference",
        "model_version": "sprint-g8v6-corrected-fullfit",
        "as_of": "2026-08-01",
        "feature_start": "2026-06-01",
        "feature_end_exclusive": "2026-08-01",
        "training_feature_end_exclusive": "2026-06-21",
        "training_label_start": "2026-06-21",
        "training_label_end_exclusive": "2026-07-31",
        "model_sha256": digest(model_path),
        "inference_features_sha256": hash_inputs(score_paths),
        "training_features_sha256": hash_inputs(train_paths),
        "training_labels_sha256": digest(labels_path),
        "feature_schema_sha256": digest(schema_path),
        "predictions_sha256": digest(predictions),
        "vehicle_list_sha256": digest(vehicles_path),
        "score_manifest_sha256": digest(score_manifest_path),
        "source_coverage": score_manifest["source_coverage"],
        "python": platform.python_version(),
        "package_versions": {name: importlib.metadata.version(name) for name in
                             ("interpret", "numpy", "pandas", "scikit-learn", "scipy", "joblib")},
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return package(predictions, output_dir / "vehicles.csv", provenance_path,
                   output_dir / "submission")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--vehicles", type=Path, required=True)
    for name in MEMBERS:
        parser.add_argument(f"--train-{name}", type=Path, required=True)
        parser.add_argument(f"--score-{name}", type=Path, required=True)
    parser.add_argument("--score-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train_paths = {name: getattr(args, f"train_{name}") for name in MEMBERS}
    score_paths = {name: getattr(args, f"score_{name}") for name in MEMBERS}
    print(run(args.labels, args.vehicles, train_paths, score_paths,
              args.score_manifest, args.output_dir))


if __name__ == "__main__":
    main()
