#!/usr/bin/env python3
"""Strictly nested, train-fold-only FEAT-009 candidate selection utilities."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()), Path.cwd())
SRC_ROOT = REPO_ROOT / "feature_engineering" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from accident_pipeline_f3.core import COHORT_MIN_VEHICLES, NEAR_CONST_VAR
from accident_pipeline_f3.interface import META_COLUMNS as F3_META, RHO_THRESHOLD, finalize_columns, select_new_columns

META = {
    "sample_id", "gpsno", "y", "fold", "as_of", "window_start", "window_end",
    "lookback_days", "horizon_days", "label_window", "label_status", "label_version",
    "split_version", "source_version", "feature_version", "feature_version_f1",
    "feature_version_f2", "feature_version_f3", "cohort_fallback",
}
COHORT_DERIVED = re.compile(r"_cohort_(?:z|pct)$")
CANDIDATES = (
    {"name": "EBM-A", "kind": "ebm", "interactions": 0, "max_bins": 64, "min_samples_leaf": 10},
    {"name": "EBM-B", "kind": "ebm", "interactions": 5, "max_bins": 64, "min_samples_leaf": 10},
    {"name": "LightGBM-A", "kind": "lgbm", "num_leaves": 4, "min_data_in_leaf": 30},
    {"name": "LightGBM-B", "kind": "lgbm", "num_leaves": 8, "min_data_in_leaf": 40},
)


def fingerprint_rows(values: pd.DataFrame) -> str:
    raw = values.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def fingerprint_sample_ids(sample_ids) -> str:
    return hashlib.sha256("\n".join(map(str, sample_ids)).encode("utf-8")).hexdigest()


def cohort_sources_for_features(columns: list[str]) -> list[str]:
    """Return source stems whose cohort outputs belong to this registered feature pool."""
    sources: list[str] = []
    for column in columns:
        if COHORT_DERIVED.search(column):
            source = COHORT_DERIVED.sub("", column)
            if source not in sources:
                sources.append(source)
    return sources


def _resolved(path: str | Path, relative_to: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (relative_to / candidate).resolve()


def load_datasets(config: dict[str, Any], *, s0_audit: dict[str, Any] | None = None,
                  config_path: Path | None = None) -> tuple[
                      dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]
                  ]:
    """Read controlled inputs, rebuild F3 from its registered base, and align all tables."""
    config_dir = config_path.resolve().parent if config_path else Path.cwd()
    f1 = pd.read_csv(config["f1_model_input"], dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)
    f3 = pd.read_csv(config["f3_model_input"], dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)
    new = pd.read_csv(config["f3_new_features"], dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)
    manifest_path = _resolved(config["f3_manifest"], config_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_value = config.get("f3_base_input") or manifest.get("inputs", {}).get("base_input")
    if not base_value:
        raise ValueError("F3 registered base_input is missing from local config and manifest")
    base_path = _resolved(base_value, REPO_ROOT if config.get("f3_base_input") else manifest_path.parent)
    registered_base_path = manifest.get("inputs", {}).get("base_input")
    if registered_base_path and _resolved(registered_base_path, manifest_path.parent) != base_path:
        raise ValueError("configured f3_base_input differs from the registered F3 manifest")
    if s0_audit is not None:
        expected_base_hash = s0_audit.get("f3_manifest_input_sha256", {}).get("base_input")
        if not expected_base_hash or hashlib.sha256(base_path.read_bytes()).hexdigest() != expected_base_hash:
            raise ValueError("F3 registered base_input fingerprint differs from the accepted S0 audit")
    base = pd.read_csv(base_path, dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)
    dropped_path = _resolved(config["f3_dropped_columns"], config_dir)
    dropped = json.loads(dropped_path.read_text(encoding="utf-8"))
    v5 = pd.read_csv(config["v5_oof"], dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)
    rf = pd.read_csv(config["rf_oof"], dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)
    for name, frame in (("F1", f1), ("F3 base", base), ("F3", f3),
                        ("F3 new features", new), ("V5", v5), ("RF", rf)):
        if "sample_id" not in frame or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
            raise ValueError(f"{name}: sample_id is absent, null, or duplicated")
    order = f1.sample_id.astype(str).tolist()
    by_id = {name: frame.assign(sample_id=frame.sample_id.astype(str)).set_index("sample_id", drop=False)
             for name, frame in (("f1", f1), ("base", base), ("f3", f3),
                                 ("new", new), ("v5", v5), ("rf", rf))}
    if any(set(table.index.astype(str)) != set(order) for table in by_id.values()):
        raise ValueError("F1/F3-base/F3/new/V5/RF sample_id sets differ")
    tables = {name: table.loc[order].reset_index(drop=True) for name, table in by_id.items()}
    ref = tables["f1"]
    for name in ("base", "f3", "v5", "rf"):
        cur = tables[name]
        for col in ("gpsno", "y", "fold"):
            if col not in cur or not cur[col].astype(str).equals(ref[col].astype(str)):
                raise ValueError(f"{name}: {col} differs from F1")
    if not tables["new"].gpsno.astype(str).equals(ref.gpsno.astype(str)):
        raise ValueError("new F3 feature table: gpsno mapping differs from F1")

    blocked_future = [c for c in f1.columns if re.search(r"(^|_)(future|next|post|target|after_as_of)(_|$)", c, re.I)]
    if blocked_future:
        raise ValueError(f"F1 has suspicious future/target features: {blocked_future}")
    labels = pd.to_numeric(ref.y, errors="raise").astype(int).to_numpy()
    folds = pd.to_numeric(ref.fold, errors="raise").astype(int).to_numpy()
    if set(np.unique(labels)) != {0, 1} or set(np.unique(folds)) != {0, 1, 2, 3, 4}:
        raise ValueError("expected binary labels and frozen fold IDs 0..4")

    def clean_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return frame[[c for c in columns if c not in META]].copy()

    f1_cols = [c for c in tables["f1"].columns if c not in META]
    x1 = clean_features(tables["f1"], f1_cols)
    # Rebuild every registered pre-screen declaration from the actual F3 base and
    # raw new-feature table; do not reuse the all-row screening outcomes.
    new_candidates = [c for c in tables["new"].columns if c not in F3_META]
    declared_new = select_new_columns(new_candidates)
    declared = list(dropped.get("declared_columns", []))
    if not declared_new or declared[-len(declared_new):] != declared_new:
        raise ValueError("F3 declaration ledger differs from the registered r5 candidate selector")
    base_cols = declared[:-len(declared_new)]
    missing_base = [c for c in base_cols if c not in tables["base"]]
    if missing_base:
        raise ValueError(f"registered F3 base lacks declared columns: {missing_base}")
    if list(manifest.get("base_kept", [])) != list(dropped.get("kept_base", [])) or \
       list(manifest.get("new_kept", [])) != list(dropped.get("kept_new", [])):
        raise ValueError("F3 manifest kept columns differ from the dropped-column ledger")
    x3 = pd.concat([tables["base"][base_cols].reset_index(drop=True),
                    clean_features(tables["new"], declared_new).reset_index(drop=True)], axis=1)
    for context_column in ("energy_type", "highway_ratio"):
        if context_column in tables["base"] and context_column not in x3:
            x3[context_column] = tables["base"][context_column].to_numpy()
    if x3.columns.duplicated().any():
        raise ValueError("F3 base and new feature columns overlap")
    group_info = tables["base"][["sample_id", "gpsno", "energy_type", "highway_ratio"]].copy()
    group_info["gpsno"] = group_info.gpsno.astype(str)
    y_frame = pd.DataFrame({"sample_id": order, "gpsno": ref.gpsno.astype(str), "y": labels, "fold": folds})
    models = {
        "F1": pd.DataFrame(x1),
        "F3": pd.DataFrame(x3),
    }
    cohort_values: dict[str, pd.DataFrame] = {}
    for name, frame, source_table in (("F1", models["F1"], tables["f1"]),
                                      ("F3", models["F3"], tables["base"])):
        sources = cohort_sources_for_features(list(frame.columns))
        missing_sources = [source for source in sources if source not in source_table]
        if missing_sources:
            raise ValueError(f"{name} cohort source columns are missing from registered source: {missing_sources}")
        cohort_values[name] = source_table[sources].copy() if sources else pd.DataFrame(index=source_table.index)
    # Attach the split/identity fields separately so they can never enter estimators.
    for table in models.values():
        table.insert(0, "sample_id", order)
        table.insert(1, "gpsno", ref.gpsno.astype(str).to_numpy())
    return models, y_frame, group_info, cohort_values


@dataclass
class FittedTransform:
    features: list[str]
    numeric: list[str]
    categorical: list[str]
    num_imputer: SimpleImputer
    cat_imputer: SimpleImputer
    encoder: OneHotEncoder
    selected: list[str]
    cohort_median: float | None
    cohort_stats: dict[str, dict[str, Any]]
    global_stats: dict[str, Any]
    cohort_sources: list[str]
    training_fingerprint: str

    def _engineer_cohorts(self, x: pd.DataFrame, groups: pd.DataFrame,
                          source_values: pd.DataFrame | None) -> pd.DataFrame:
        out = x.copy()
        if not self.cohort_sources:
            return out
        if source_values is None and all(source in x for source in self.cohort_sources):
            source_values = x[self.cohort_sources]
        if source_values is None or len(source_values) != len(x):
            raise ValueError("cohort source values are missing or misaligned")
        energy = groups.energy_type.fillna("unknown").astype(str).to_numpy()
        highway = pd.to_numeric(groups.highway_ratio, errors="coerce").to_numpy(float)
        median = self.cohort_median
        branch = (np.full(len(highway), "na", dtype=object) if median is None or not np.isfinite(median)
                  else np.where(~np.isfinite(highway), "na", np.where(highway > median, "hi", "lo")))
        keys = np.char.add(np.char.add(energy.astype(str), "|"), branch.astype(str))
        for source in self.cohort_sources:
            if source not in source_values:
                raise ValueError(f"cohort source values lack {source}")
            vals = pd.to_numeric(source_values[source], errors="coerce").to_numpy(float)
            z = np.full(len(x), np.nan)
            pct = np.full(len(x), np.nan)
            for i, (key, value) in enumerate(zip(keys, vals)):
                if not np.isfinite(value):
                    continue
                stat = self.cohort_stats.get(source, {}).get(str(key), self.global_stats[source])
                if stat["fallback"]:
                    stat = self.global_stats[source]
                if np.isfinite(stat["mean"]) and np.isfinite(stat["std"]) and stat["std"] > 1e-12:
                    z[i] = (value - stat["mean"]) / stat["std"]
                pool = stat["pool"]
                if len(pool):
                    pct[i] = float(np.mean(pool <= value))
            if f"{source}_cohort_z" in out:
                out[f"{source}_cohort_z"] = z
            if f"{source}_cohort_pct" in out:
                out[f"{source}_cohort_pct"] = pct
        return out

    def transform(self, x: pd.DataFrame, groups: pd.DataFrame,
                  source_values: pd.DataFrame | None = None) -> np.ndarray:
        frame = self._engineer_cohorts(x, groups, source_values)
        nums = self.num_imputer.transform(frame[self.numeric]) if self.numeric else np.empty((len(frame), 0))
        if self.categorical:
            cats = self.cat_imputer.transform(frame[self.categorical].astype("object")).astype(str)
            cats = self.encoder.transform(cats)
            return np.column_stack([nums, cats]).astype(float, copy=False)
        return nums.astype(float, copy=False)


def fit_transformer(x_train: pd.DataFrame, groups_train: pd.DataFrame,
                    cohort_sources: list[str] | None = None,
                    cohort_values_train: pd.DataFrame | None = None) -> FittedTransform:
    sources = cohort_sources if cohort_sources is not None else cohort_sources_for_features(list(x_train.columns))
    if sources and cohort_values_train is None and all(source in x_train for source in sources):
        cohort_values_train = x_train[sources]
    if sources and (cohort_values_train is None or any(c not in cohort_values_train for c in sources)):
        raise ValueError("cohort source values do not cover all declared cohort features")
    group_source_values = (cohort_values_train[sources].apply(pd.to_numeric, errors="coerce")
                           if sources else pd.DataFrame(index=x_train.index))
    highway = (pd.to_numeric(groups_train.highway_ratio, errors="coerce").to_numpy(float)
               if "highway_ratio" in groups_train else np.full(len(groups_train), np.nan))
    finite_highway = highway[np.isfinite(highway)]
    median = float(np.median(finite_highway)) if len(finite_highway) else float("nan")
    energy = (groups_train.energy_type.fillna("unknown").astype(str).to_numpy()
              if "energy_type" in groups_train else np.full(len(groups_train), "unknown", dtype=object))
    branch = (np.full(len(highway), "na", dtype=object) if not np.isfinite(median)
              else np.where(~np.isfinite(highway), "na", np.where(highway > median, "hi", "lo")))
    keys = np.char.add(np.char.add(energy, "|"), branch)
    stats: dict[str, dict[str, Any]] = {}
    global_stats: dict[str, Any] = {}
    for source in sources:
        values = group_source_values[source].to_numpy(float)
        all_values = values[np.isfinite(values)]
        global_stats[source] = {"mean": float(np.mean(all_values)) if len(all_values) else np.nan,
                                "std": float(np.std(all_values, ddof=0)) if len(all_values) else np.nan,
                                "pool": all_values, "fallback": False}
        stats[source] = {}
        for key in np.unique(keys):
            pool = values[(keys == key) & np.isfinite(values)]
            n_vehicles = groups_train.loc[keys == key, "gpsno"].nunique()
            stats[source][str(key)] = {"mean": float(np.mean(pool)) if len(pool) else np.nan,
                                       "std": float(np.std(pool, ddof=0)) if len(pool) else np.nan,
                                       "pool": pool, "fallback": n_vehicles < 30}

    engineered = x_train.copy()
    if sources:
        stub = FittedTransform([], [], [], SimpleImputer(), SimpleImputer(), OneHotEncoder(), [],
                               median, stats, global_stats, sources, "")
        engineered = stub._engineer_cohorts(x_train, groups_train, cohort_values_train)
    numeric = [c for c in engineered if pd.api.types.is_numeric_dtype(engineered[c])]
    categorical = [c for c in engineered if c not in numeric]
    kept_numeric, _ = finalize_columns(
        engineered[numeric], numeric,
        rho_threshold=RHO_THRESHOLD, var_threshold=NEAR_CONST_VAR,
    ) if numeric else ([], [])
    kept_numeric_set = set(kept_numeric)
    kept_categorical = {
        col for col in categorical
        if engineered[col].dropna().astype(str).nunique() > 1
    }
    selected = [col for col in engineered.columns
                if (col in kept_numeric_set if col in numeric else col in kept_categorical)]
    numeric = [c for c in selected if c in numeric]
    categorical = [c for c in selected if c in categorical]
    num_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    cat_imputer = SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)
    if numeric:
        num_imputer.fit(engineered[numeric])
    if categorical:
        cat_values = engineered[categorical].astype("object")
        cat_imputer.fit(cat_values)
        cat_train = cat_imputer.transform(cat_values).astype(str)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(cat_train)
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if "sample_id" in groups_train:
        training_fingerprint = fingerprint_sample_ids(groups_train.sample_id.astype(str).tolist())
    else:
        training_fingerprint = fingerprint_rows(x_train.reset_index(drop=True))
    return FittedTransform(list(x_train.columns), numeric, categorical, num_imputer, cat_imputer,
                           encoder, selected, median, stats, global_stats, sources, training_fingerprint)


def make_candidate(spec: dict[str, Any], seed: int = 42):
    if spec["kind"] == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(
            interactions=spec["interactions"], max_bins=spec["max_bins"],
            min_samples_leaf=spec["min_samples_leaf"], random_state=seed, n_jobs=1,
        )
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        num_leaves=spec["num_leaves"], min_child_samples=spec["min_data_in_leaf"],
        learning_rate=0.05, n_estimators=200, reg_lambda=1.0, colsample_bytree=0.8,
        objective="binary", random_state=seed, n_jobs=1, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )


def fit_predict(estimator, x_train, y_train, x_test, fit_sample_ids: list[str] | None = None,
                candidate_name: str = "unknown") -> tuple[np.ndarray, dict[str, Any]]:
    estimator.fit(x_train, y_train)
    row_ids = fit_sample_ids if fit_sample_ids is not None else [str(i) for i in range(len(y_train))]
    fit_scope = {
        "candidate": candidate_name,
        "fit_rows": int(len(y_train)),
        "fit_sample_ids_sha256": hashlib.sha256("\n".join(sorted(map(str, row_ids))).encode("utf-8")).hexdigest(),
    }
    setattr(estimator, "_feat009_fit_scope", fit_scope)
    setattr(estimator, "_feat009_fit_stamp", type("FitStamp", (), fit_scope)())
    classes = list(estimator.classes_)
    prediction = np.asarray(estimator.predict_proba(x_test)[:, classes.index(1)], dtype=float)
    return prediction, fit_scope


def select_and_fit_fold(name: str, x: pd.DataFrame, y: np.ndarray, folds: np.ndarray,
                        outer_fold: int, groups: pd.DataFrame,
                        cohort_sources: list[str] | None = None,
                        candidates: tuple[dict[str, Any], ...] = CANDIDATES,
                        estimator_factory=make_candidate, *,
                        cohort_values: pd.DataFrame | None = None) -> dict[str, Any]:
    train_idx = np.flatnonzero(folds != outer_fold)
    test_idx = np.flatnonzero(folds == outer_fold)
    if not len(train_idx) or not len(test_idx) or set(np.unique(y[train_idx])) != {0, 1}:
        raise ValueError(f"outer fold {outer_fold} has empty test or single-class train")
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores: dict[str, list[float]] = {spec["name"]: [] for spec in candidates}
    fit_scopes: list[dict[str, Any]] = []
    fit_count = 0
    for inner_fold, (inner_train_rel, inner_valid_rel) in enumerate(inner.split(np.zeros(len(train_idx)), y[train_idx])):
        itr = train_idx[inner_train_rel]
        iva = train_idx[inner_valid_rel]
        if np.intersect1d(itr, iva).size:
            raise AssertionError("inner train and validation indices overlap")
        itr_cohort = cohort_values.iloc[itr] if cohort_values is not None else None
        iva_cohort = cohort_values.iloc[iva] if cohort_values is not None else None
        for spec in candidates:
            transformer = fit_transformer(x.iloc[itr], groups.iloc[itr], cohort_sources, itr_cohort)
            a = transformer.transform(x.iloc[itr], groups.iloc[itr], itr_cohort)
            b = transformer.transform(x.iloc[iva], groups.iloc[iva], iva_cohort)
            fit_ids = (groups.iloc[itr].sample_id.astype(str).tolist() if "sample_id" in groups
                       else [str(i) for i in itr])
            pred, model_scope = fit_predict(estimator_factory(spec), a, y[itr], b,
                                            fit_ids, spec["name"])
            scores[spec["name"]].append(float(roc_auc_score(y[iva], pred)))
            fit_scopes.append({"stage": "inner", "inner_fold": int(inner_fold),
                               "preprocessor_fit_rows": len(itr),
                               "preprocessor_fit_sample_ids_sha256": transformer.training_fingerprint,
                               **model_scope})
            fit_count += 1
    means = {key: float(np.mean(vals)) for key, vals in scores.items()}
    # Python's stable max retains the registered order for exact ties.
    selected = next(spec for spec in candidates if spec["name"] == max(means, key=means.get))
    train_cohort = cohort_values.iloc[train_idx] if cohort_values is not None else None
    test_cohort = cohort_values.iloc[test_idx] if cohort_values is not None else None
    final_transform = fit_transformer(x.iloc[train_idx], groups.iloc[train_idx], cohort_sources, train_cohort)
    train_matrix = final_transform.transform(x.iloc[train_idx], groups.iloc[train_idx], train_cohort)
    test_matrix = final_transform.transform(x.iloc[test_idx], groups.iloc[test_idx], test_cohort)
    fit_ids = (groups.iloc[train_idx].sample_id.astype(str).tolist() if "sample_id" in groups
               else [str(i) for i in train_idx])
    pred, final_scope = fit_predict(estimator_factory(selected), train_matrix, y[train_idx],
                                    test_matrix, fit_ids, selected["name"])
    fit_scopes.append({"stage": "outer_final", "preprocessor_fit_rows": len(train_idx),
                       "preprocessor_fit_sample_ids_sha256": final_transform.training_fingerprint,
                       **final_scope})
    fit_count += 1
    return {
        "dataset": name, "outer_fold": int(outer_fold), "train_indices": train_idx,
        "test_indices": test_idx, "prediction": pred, "selected_candidate": selected["name"],
        "inner_auc": means, "inner_auc_by_fold": scores,
        "selected_features": final_transform.selected,
        "train_fingerprint": final_transform.training_fingerprint,
        "fit_scopes": fit_scopes,
        "fit_count": fit_count, "test_indices_disjoint": not bool(set(train_idx) & set(test_idx)),
        "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
        "test_positive": int(np.asarray(y)[test_idx].sum()),
        "outer_auc": float(roc_auc_score(np.asarray(y)[test_idx], pred)),
    }


def nested_oof(name: str, x: pd.DataFrame, y: np.ndarray, folds: np.ndarray,
               groups: pd.DataFrame, cohort_sources: list[str] | None = None,
               candidates: tuple[dict[str, Any], ...] = CANDIDATES,
               estimator_factory=make_candidate, *,
               cohort_values: pd.DataFrame | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if set(np.unique(np.asarray(folds, dtype=int))) != {0, 1, 2, 3, 4}:
        raise ValueError("outer folds must use the frozen IDs 0..4")
    predictions = np.full(len(y), np.nan, dtype=float)
    prediction_counts = np.zeros(len(y), dtype=int)
    diagnostics = []
    for outer_fold in sorted(np.unique(folds)):
        result = select_and_fit_fold(name, x, y, folds, int(outer_fold), groups,
                                     cohort_sources, candidates, estimator_factory,
                                     cohort_values=cohort_values)
        idx = result.pop("test_indices")
        result.pop("train_indices")
        predictions[idx] = result.pop("prediction")
        prediction_counts[idx] += 1
        diagnostics.append(result)
    if not np.isfinite(predictions).all() or not np.all(prediction_counts == 1):
        raise RuntimeError("nested OOF predictions do not cover every sample exactly once")
    return predictions, diagnostics
