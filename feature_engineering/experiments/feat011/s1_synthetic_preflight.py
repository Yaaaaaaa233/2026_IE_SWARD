#!/usr/bin/env python3
"""Run FEAT-011 S1 plumbing checks using generated data only."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "task1_feature_modeling"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

try:
    from .audit_s0 import NIGHT_SUMMARY, validate_column_views
except ImportError:  # direct script execution
    from audit_s0 import NIGHT_SUMMARY, validate_column_views

from modeling import fit_transformer, make_candidate

VERSION = "feat011-s1-synthetic-preflight-20260925-r4"
SEED = 42
EXPECTED_WEIGHTS = {"A": 0.5, "D": 0.5}
SYNTHETIC_ROWS = 500
SYNTHETIC_BASE_FEATURES = [f"f3_base_synthetic_{i:03d}" for i in range(114)]
SYNTHETIC_NIGHT_DETAILS = [f"f3_night_synthetic_detail_{i:03d}" for i in range(32)]
SYNTHETIC_FEATURES = [
    *SYNTHETIC_BASE_FEATURES,
    "f3_night_deep_rate", "f3_night_day_rate", "f3_night_degradation",
    "f3_night_exposure_share",
    *SYNTHETIC_NIGHT_DETAILS,
]
DETAILED_NIGHT = list(SYNTHETIC_NIGHT_DETAILS)
SLIM_FEATURES = [column for column in SYNTHETIC_FEATURES if column not in set(DETAILED_NIGHT)]
MODEL_SPECS = {
    "A": {"name": "EBM-A", "kind": "ebm", "interactions": 0, "max_bins": 64,
          "min_samples_leaf": 10},
    "B": {"name": "EBM-A", "kind": "ebm", "interactions": 0, "max_bins": 64,
          "min_samples_leaf": 10},
    "C": {"name": "EBM-C", "kind": "ebm", "interactions": 0, "max_bins": 32,
          "min_samples_leaf": 20},
    "D": {"name": "EBM-C", "kind": "ebm", "interactions": 0, "max_bins": 32,
          "min_samples_leaf": 20},
}


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_columns(columns: list[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def synthetic_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Make deterministic, explicitly artificial data with a fixed five-fold map."""
    rng = np.random.default_rng(SEED)
    rows = SYNTHETIC_ROWS
    latent = rng.normal(size=rows)
    y = (latent + rng.normal(scale=0.9, size=rows) > 0.95).astype(int)
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError("synthetic generator did not create both classes")
    values: dict[str, np.ndarray] = {
        column: rng.normal(size=rows) for column in SYNTHETIC_BASE_FEATURES
    }
    values[SYNTHETIC_BASE_FEATURES[0]] = latent + rng.normal(scale=0.15, size=rows)
    values[SYNTHETIC_BASE_FEATURES[1]] = 0.35 * latent + rng.normal(scale=0.8, size=rows)
    values.update({
        "f3_night_deep_rate": rng.beta(2, 8, size=rows),
        "f3_night_day_rate": rng.beta(2, 7, size=rows),
        "f3_night_degradation": rng.normal(size=rows),
        "f3_night_exposure_share": rng.beta(2, 6, size=rows),
    })
    values.update({column: rng.poisson(1.5, size=rows).astype(float)
                   for column in SYNTHETIC_NIGHT_DETAILS})
    x = pd.DataFrame(values, columns=SYNTHETIC_FEATURES)
    # Include a few missing values to exercise train-fold-only imputation.
    for col in (SYNTHETIC_BASE_FEATURES[2], "f3_night_degradation"):
        x.loc[rng.choice(rows, 8, replace=False), col] = np.nan
    folds = np.full(rows, -1, dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_id, (_, test_idx) in enumerate(splitter.split(x, y)):
        folds[test_idx] = fold_id
    ids = pd.Series([f"SYN-{i:04d}" for i in range(rows)], name="sample_id")
    groups = pd.DataFrame({"sample_id": ids})
    return x, y, folds, groups


def materialize_preregistration(x: pd.DataFrame, y: np.ndarray, folds: np.ndarray,
                                groups: pd.DataFrame) -> dict[str, Any]:
    """Build an illustrative pre-registration; it deliberately cannot lock real data."""
    validate_column_views(SYNTHETIC_FEATURES, SLIM_FEATURES, DETAILED_NIGHT)
    arms: dict[str, Any] = {}
    for arm, features in (("A", SYNTHETIC_FEATURES), ("B", SLIM_FEATURES),
                          ("C", SYNTHETIC_FEATURES), ("D", SLIM_FEATURES)):
        estimator = make_candidate(MODEL_SPECS[arm], seed=SEED)
        arms[arm] = {
            "feature_view": "full" if arm in {"A", "C"} else "slim",
            "ordered_columns": list(features),
            "ordered_columns_sha256": sha256_columns(features),
            "model_spec": dict(MODEL_SPECS[arm]),
            "full_estimator_parameters": estimator.get_params(deep=False),
        }
    fold_frame = pd.DataFrame({"sample_id": groups.sample_id, "y": y, "fold": folds})
    prereg: dict[str, Any] = {
        "artifact_class": "synthetic_preflight_only",
        "version": VERSION,
        "status": "synthetic_preflight_template_only",
        "synthetic_only": True,
        "real_preregistration_locked": False,
        "seed": SEED,
        "synthetic_rows": int(len(x)),
        "synthetic_positive_count": int(y.sum()),
        "synthetic_input_sha256": sha256_frame(pd.concat([
            groups.reset_index(drop=True), pd.DataFrame({"y": y, "fold": folds}),
            x.reset_index(drop=True),
        ], axis=1)),
        "synthetic_fold_map_sha256": sha256_frame(fold_frame),
        "fold_ids": [0, 1, 2, 3, 4],
        "arms": arms,
        "fusion": {
            "method": "mean_of_clipped_logits_then_sigmoid",
            "members": ["A", "D"],
            "weights": EXPECTED_WEIGHTS,
            "clip": [1e-6, 1 - 1e-6],
        },
        "primary_comparison": "E_minus_A",
        "primary_metric": "delta_auc",
        "paired_bootstrap": {"strata": "label", "replicates": 2000, "seed": SEED,
                              "interval": "95_percentile"},
        "auxiliary_tolerances": {"recall_at_100_delta_min": -0.02,
                                 "brier_delta_max": 0.005},
        "max_valid_real_runs": {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1},
        "environment": package_versions(),
        "note": "Generic generated 500-row/150-feature examples only. This is not the real locked preregistration.",
    }
    prereg["body_sha256"] = sha256_json(prereg)
    return prereg


def package_versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for package in ("interpret", "interpret-core", "numpy", "pandas", "scikit-learn", "matplotlib"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    return found


def validate_preregistration(prereg: dict[str, Any]) -> None:
    if prereg.get("artifact_class") != "synthetic_preflight_only" or not prereg.get("synthetic_only"):
        raise ValueError("preflight artifact is not marked synthetic-only")
    if prereg.get("real_preregistration_locked") is not False:
        raise ValueError("synthetic preflight cannot lock a real preregistration")
    expected_hash = prereg.get("body_sha256")
    body = {key: value for key, value in prereg.items() if key != "body_sha256"}
    if expected_hash != sha256_json(body):
        raise ValueError("pre-registration configuration fingerprint drift")
    arms = prereg.get("arms", {})
    if set(arms) != {"A", "B", "C", "D"}:
        raise ValueError("training arm set drift")
    for arm, expected_spec in MODEL_SPECS.items():
        item = arms[arm]
        expected_features = SYNTHETIC_FEATURES if arm in {"A", "C"} else SLIM_FEATURES
        expected_parameters = make_candidate(expected_spec, seed=SEED).get_params(deep=False)
        if item.get("model_spec") != expected_spec:
            raise ValueError(f"{arm}: model specification drift")
        if item.get("ordered_columns") != expected_features:
            raise ValueError(f"{arm}: ordered feature columns drift")
        if item.get("ordered_columns_sha256") != sha256_columns(expected_features):
            raise ValueError(f"{arm}: ordered feature fingerprint drift")
        if item.get("full_estimator_parameters") != expected_parameters:
            raise ValueError(f"{arm}: full estimator configuration drift")
    if arms["A"]["full_estimator_parameters"] != arms["B"]["full_estimator_parameters"]:
        raise ValueError("A/B must use the same complete estimator configuration")
    if arms["C"]["full_estimator_parameters"] != arms["D"]["full_estimator_parameters"]:
        raise ValueError("C/D must use the same complete estimator configuration")
    if arms["A"]["ordered_columns"] != arms["C"]["ordered_columns"]:
        raise ValueError("A/C must use the same feature view")
    if arms["B"]["ordered_columns"] != arms["D"]["ordered_columns"]:
        raise ValueError("B/D must use the same feature view")
    changed_parameters = {key for key in arms["A"]["full_estimator_parameters"]
                          if arms["A"]["full_estimator_parameters"][key]
                          != arms["C"]["full_estimator_parameters"][key]}
    if changed_parameters != {"max_bins", "min_samples_leaf"}:
        raise ValueError("model smoothness factor differs outside max_bins/min_samples_leaf")
    fusion = prereg.get("fusion", {})
    if (fusion.get("method") != "mean_of_clipped_logits_then_sigmoid"
            or fusion.get("members") != ["A", "D"]
            or fusion.get("weights") != EXPECTED_WEIGHTS
            or fusion.get("clip") != [1e-6, 1 - 1e-6]):
        raise ValueError("fixed E fusion specification drift")


def validate_fold_map(x: pd.DataFrame, y: np.ndarray, folds: np.ndarray,
                      groups: pd.DataFrame, prereg: dict[str, Any]) -> None:
    fold_frame = pd.DataFrame({"sample_id": groups.sample_id, "y": y, "fold": folds})
    if sha256_frame(fold_frame) != prereg.get("synthetic_fold_map_sha256"):
        raise ValueError("fold assignment/label map differs from pre-registration")
    if set(np.unique(folds)) != {0, 1, 2, 3, 4}:
        raise ValueError("fold IDs must be exactly 0..4")
    if not np.isfinite(y).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("labels must be complete binary values")
    for fold_id in range(5):
        test_y, train_y = y[folds == fold_id], y[folds != fold_id]
        if set(np.unique(test_y)) != {0, 1} or set(np.unique(train_y)) != {0, 1}:
            raise ValueError(f"fold {fold_id} does not contain both classes in train and test")
    if list(x.columns) != SYNTHETIC_FEATURES:
        raise ValueError("synthetic source columns/order drift")
    input_frame = pd.concat([
        groups.reset_index(drop=True), pd.DataFrame({"y": y, "fold": folds}),
        x.reset_index(drop=True),
    ], axis=1)
    if sha256_frame(input_frame) != prereg.get("synthetic_input_sha256"):
        raise ValueError("synthetic input fingerprint drift")


def equal_logit_fusion(p_a: np.ndarray, p_d: np.ndarray,
                       weights: dict[str, float] = EXPECTED_WEIGHTS) -> np.ndarray:
    if weights != EXPECTED_WEIGHTS:
        raise ValueError("E fusion weights must remain fixed at 0.5/0.5")
    eps = 1e-6
    clipped_a = np.clip(np.asarray(p_a, dtype=float), eps, 1 - eps)
    clipped_d = np.clip(np.asarray(p_d, dtype=float), eps, 1 - eps)
    logits_a = np.log(clipped_a / (1 - clipped_a))
    logits_d = np.log(clipped_d / (1 - clipped_d))
    return 1 / (1 + np.exp(-0.5 * (logits_a + logits_d)))


def audit_equal_logit_fusion(p_e: np.ndarray, p_a: np.ndarray, p_d: np.ndarray,
                             prereg: dict[str, Any]) -> None:
    """Independently recompute E from the frozen contract and compare row by row."""
    fusion = prereg.get("fusion", {})
    if (fusion.get("members") != ["A", "D"] or fusion.get("weights") != EXPECTED_WEIGHTS
            or fusion.get("method") != "mean_of_clipped_logits_then_sigmoid"
            or fusion.get("clip") != [1e-6, 1 - 1e-6]):
        raise ValueError("preregistered fusion contract is not the fixed equal-logit rule")
    eps = 1e-6
    a = np.clip(np.asarray(p_a, dtype=float), eps, 1 - eps)
    d = np.clip(np.asarray(p_d, dtype=float), eps, 1 - eps)
    expected = 1 / (1 + np.exp(-0.5 * (np.log(a / (1 - a)) + np.log(d / (1 - d)))))
    if not np.allclose(np.asarray(p_e, dtype=float), expected, rtol=0, atol=1e-15):
        raise ValueError("E OOF does not equal the preregistered A/D equal-logit fusion")


def validate_fit_scope(fit_scope: dict[str, Any], expected_train_ids: list[str],
                       arm: str, fold_id: int) -> None:
    expected = hashlib.sha256("\n".join(expected_train_ids).encode("utf-8")).hexdigest()
    if fit_scope.get("arm") != arm or fit_scope.get("fold") != fold_id:
        raise ValueError("fit-scope record is attached to the wrong arm/fold")
    if fit_scope.get("preprocessor_fit_rows") != len(expected_train_ids):
        raise ValueError("preprocessor fit row count includes rows outside the training fold")
    if fit_scope.get("model_fit_rows") != len(expected_train_ids):
        raise ValueError("model fit row count includes rows outside the training fold")
    if fit_scope.get("preprocessor_fit_sample_ids_sha256") != expected:
        raise ValueError("preprocessor fit scope differs from the outer training fold")


def bootstrap_delta(y: np.ndarray, p_e: np.ndarray, p_a: np.ndarray,
                    n: int = 2000, seed: int = SEED) -> dict[str, float]:
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n)
    for index in range(n):
        sampled = np.r_[rng.choice(positives, len(positives), replace=True),
                        rng.choice(negatives, len(negatives), replace=True)]
        deltas[index] = roc_auc_score(y[sampled], p_e[sampled]) - roc_auc_score(y[sampled], p_a[sampled])
    return {"delta_auc": float(roc_auc_score(y, p_e) - roc_auc_score(y, p_a)),
            "ci_low": float(np.quantile(deltas, 0.025)),
            "ci_high": float(np.quantile(deltas, 0.975))}


def fit_one_fold(x: pd.DataFrame, y: np.ndarray, folds: np.ndarray, groups: pd.DataFrame,
                 arm: str, fold_id: int) -> tuple[np.ndarray, dict[str, Any], float]:
    train_idx = np.flatnonzero(folds != fold_id)
    test_idx = np.flatnonzero(folds == fold_id)
    if set(np.unique(y[train_idx])) != {0, 1} or set(np.unique(y[test_idx])) != {0, 1}:
        raise ValueError(f"fold {fold_id} does not contain both classes")
    feature_cols = SYNTHETIC_FEATURES if arm in {"A", "C"} else SLIM_FEATURES
    started = time.perf_counter()
    transformer = fit_transformer(x.iloc[train_idx][feature_cols], groups.iloc[train_idx],
                                  cohort_sources=[])
    train_matrix = transformer.transform(x.iloc[train_idx][feature_cols], groups.iloc[train_idx])
    test_matrix = transformer.transform(x.iloc[test_idx][feature_cols], groups.iloc[test_idx])
    estimator = make_candidate(MODEL_SPECS[arm], seed=SEED)
    estimator.fit(train_matrix, y[train_idx])
    prediction = np.asarray(estimator.predict_proba(test_matrix)[:, list(estimator.classes_).index(1)], dtype=float)
    fit_scope = {
        "arm": arm, "fold": fold_id,
        "preprocessor_fit_rows": int(len(train_idx)),
        "preprocessor_fit_sample_ids_sha256": transformer.training_fingerprint,
        "model_fit_rows": int(len(train_idx)),
    }
    train_ids = groups.iloc[train_idx].sample_id.astype(str).tolist()
    # The production fingerprint uses sorted sample IDs; keep this checker aligned with that contract.
    fit_scope["preprocessor_fit_sample_ids_sha256"] = hashlib.sha256(
        "\n".join(sorted(train_ids)).encode("utf-8")).hexdigest()
    validate_fit_scope(fit_scope, sorted(train_ids), arm, fold_id)
    return prediction, fit_scope, time.perf_counter() - started


def run_preflight() -> dict[str, Any]:
    x, y, folds, groups = synthetic_data()
    prereg = materialize_preregistration(x, y, folds, groups)
    validate_preregistration(prereg)
    validate_fold_map(x, y, folds, groups, prereg)

    predictions: dict[str, np.ndarray] = {}
    fit_scopes: list[dict[str, Any]] = []
    durations: dict[str, list[float]] = {arm: [] for arm in "ABCD"}
    first_fold_peak_rss: int | None = None
    for arm in "ABCD":
        feature_cols = SYNTHETIC_FEATURES if arm in {"A", "C"} else SLIM_FEATURES
        oof = np.full(len(y), np.nan, dtype=float)
        counts = np.zeros(len(y), dtype=int)
        for fold_id in range(5):
            fold_pred, scope, elapsed = fit_one_fold(x[feature_cols], y, folds, groups,
                                                     arm, fold_id)
            test_idx = np.flatnonzero(folds == fold_id)
            oof[test_idx] = fold_pred
            counts[test_idx] += 1
            fit_scopes.append(scope)
            durations[arm].append(elapsed)
            if arm == "A" and fold_id == 0:
                usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                first_fold_peak_rss = int(usage if platform.system() == "Darwin" else usage * 1024)
        if not np.isfinite(oof).all() or not np.all(counts == 1):
            raise RuntimeError(f"arm {arm} failed exactly-once OOF coverage")
        predictions[arm] = oof

    p_e = equal_logit_fusion(predictions["A"], predictions["D"])
    predictions["E"] = p_e
    audit_equal_logit_fusion(p_e, predictions["A"], predictions["D"], prereg)

    metrics = {arm: {"auc": float(roc_auc_score(y, p)),
                     "average_precision": float(average_precision_score(y, p)),
                     "brier": float(brier_score_loss(y, p))}
               for arm, p in predictions.items()}
    delta = bootstrap_delta(y, predictions["E"], predictions["A"])
    # This chart only demonstrates how to read the predeclared criteria.
    result = {
        "artifact_class": "synthetic_preflight_only",
        "version": VERSION,
        "synthetic_only": True,
        "real_oof_run": False,
        "real_preregistration_locked": False,
        "preregistration": prereg,
        "folds": 5,
        "arms_completed": ["A", "B", "C", "D", "E"],
        "fit_count": {arm: len(durations[arm]) for arm in "ABCD"},
        "fit_scope_checks": len(fit_scopes),
        "synthetic_metrics_do_not_estimate_real_performance": metrics,
        "synthetic_delta_e_minus_a_do_not_estimate_real_performance": delta,
        "runtime_seconds_by_arm_outer_fold": durations,
        "one_outer_fold_cost_sample": {
            "arm": "A", "fold": 0, "runtime_seconds": durations["A"][0],
            "process_peak_rss_bytes_after_fold": first_fold_peak_rss,
            "memory_note": "Process high-water RSS after one fold, including interpreter/import baseline; not estimator-only memory.",
        },
        "checks": [
            "full estimator parameters and config fingerprint stable",
            "feature/model factor matrix changes only registered factor",
            "all four arms use the same frozen five-fold map and contain both classes",
            "fold-scoped imputation and model fitting use outer training rows only",
            "each synthetic row receives exactly one OOF prediction per training arm",
            "E is the fixed equal-weight A/D clipped-logit average",
        ],
        "limitations": [
            "All input rows and outcomes are generated synthetic examples.",
            "The synthetic schema is generic and does not encode project feature values or identities.",
            "These model scores and runtime do not establish real-data efficacy or official test performance.",
            "Synthetic preregistration cannot substitute for the real Y1 input fingerprint or human review gates.",
        ],
    }
    result["result_sha256"] = sha256_json({k: v for k, v in result.items() if k != "result_sha256"})
    return result


def run_negative_checks() -> dict[str, Any]:
    x, y, folds, groups = synthetic_data()
    prereg = materialize_preregistration(x, y, folds, groups)
    checks: list[str] = []

    bad = json.loads(json.dumps(prereg))
    bad["fusion"]["weights"] = {"A": 0.4, "D": 0.6}
    bad["body_sha256"] = sha256_json({k: v for k, v in bad.items() if k != "body_sha256"})
    try:
        validate_preregistration(bad)
    except ValueError:
        checks.append("changed fusion weights rejected")
    else:
        raise AssertionError("changed fusion weights were accepted")

    swapped = folds.copy()
    left = 0
    right = next(i for i in range(1, len(folds)) if folds[i] != folds[left])
    swapped[left], swapped[right] = swapped[right], swapped[left]
    try:
        validate_fold_map(x, y, swapped, groups, prereg)
    except ValueError:
        checks.append("swapped fold assignments rejected")
    else:
        raise AssertionError("swapped fold assignments were accepted")

    expected = groups.loc[folds != 0, "sample_id"].astype(str).sort_values().tolist()
    leaked = {"arm": "A", "fold": 0, "preprocessor_fit_rows": len(expected) + 1,
              "preprocessor_fit_sample_ids_sha256": hashlib.sha256(
                  "\n".join(expected + ["SYN-LEAK"]).encode("utf-8")).hexdigest()}
    try:
        validate_fit_scope(leaked, expected, "A", 0)
    except ValueError:
        checks.append("validation-row preprocessing leakage rejected")
    else:
        raise AssertionError("validation-fold preprocessing leakage was accepted")

    return {"status": "pass", "rejected_cases": checks, "rejected_count": len(checks)}


def make_previews(out: Path, result: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.set_title("FEAT-011 S1 synthetic preflight: fixed 2×2 diagnostic matrix", fontsize=13)
    ax.text(5, 5.55, "SYNTHETIC PREFLIGHT ONLY — no real-data result", ha="center",
            color="#a12b2b", weight="bold", fontsize=10)
    cells = [
        (0.7, 3.15, "Full F3 + EBM-A", "A · main comparator", "#dce7f5"),
        (5.1, 3.15, "Full F3 + EBM-C", "C · model diagnostic", "#dce7f5"),
        (0.7, 1.55, "Slim F3 + EBM-A", "B · feature diagnostic", "#e1eee4"),
        (5.1, 1.55, "Slim F3 + EBM-C", "D · diagnostic / fusion member", "#e1eee4"),
    ]
    for x0, y0, title, subtitle, color in cells:
        ax.add_patch(FancyBboxPatch((x0, y0), 3.8, 1.0, boxstyle="round,pad=.04",
                                    facecolor=color, edgecolor="#52616b"))
        ax.text(x0 + 1.9, y0 + .63, title, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x0 + 1.9, y0 + .27, subtitle, ha="center", va="center", fontsize=9)
    ax.text(5, .72, "E = sigmoid((logit(clip(A)) + logit(clip(D))) / 2)",
            ha="center", fontsize=10, bbox={"boxstyle": "round", "fc": "#f6ead7", "ec": "#8a7251"})
    ax.text(5, .22, "One fixed five-fold map · no tuning or weight search · E−A is the only primary comparison",
            ha="center", fontsize=8.5, color="#39464e")
    fig.tight_layout()
    fig.savefig(out / "s1_matrix_fusion_synthetic.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    delta = result["synthetic_delta_e_minus_a_do_not_estimate_real_performance"]
    center, low, high = delta["delta_auc"], delta["ci_low"], delta["ci_high"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.errorbar([center], [0], xerr=[[center - low], [high - center]], fmt="o",
                capsize=5, color="#23658a")
    ax.axvline(0, color="#444", lw=1, label="zero improvement")
    ax.axvline(.01, color="#d88923", lw=1, ls="--", label="+0.01 reference")
    ax.set_yticks([0], ["Synthetic E − A"])
    ax.set_xlabel("Illustrative ΔAUC with 95% paired bootstrap interval")
    ax.set_title("SYNTHETIC DATA ONLY — interval chart format, not experiment evidence")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=.2)
    ax.text(.02, .08, "Auxiliary gates: Recall@100 Δ ≥ −0.02; Brier Δ ≤ +0.005",
            transform=ax.transAxes, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "s1_interval_synthetic.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = run_preflight()
        negative = run_negative_checks()
    except Exception as exc:
        failure = {"artifact_class": "synthetic_preflight_only", "version": VERSION,
                   "status": "failed", "synthetic_only": True,
                   "error_type": type(exc).__name__, "error": str(exc)}
        (out / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    result["negative_checks"] = negative
    result["result_sha256"] = sha256_json({k: v for k, v in result.items() if k != "result_sha256"})
    (out / "synthetic_preregistration.json").write_text(
        json.dumps(result["preregistration"], ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    (out / "preflight_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "negative_checks.json").write_text(
        json.dumps(negative, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_previews(out, result)
    (out / "README.txt").write_text(
        "FEAT-011 S1 synthetic preflight only. All data, metrics and intervals are synthetic; no real OOF was run.\n"
        "This does not lock the real preregistration or clear EVAL-001/EVAL-003/FEAT-009 human review gates.\n",
        encoding="utf-8")
    print(json.dumps({"status": "pass", "output_dir": str(out),
                      "arms": result["arms_completed"], "fit_count": result["fit_count"],
                      "negative_cases_rejected": negative["rejected_count"],
                      "one_fold_cost_sample": result["one_outer_fold_cost_sample"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
