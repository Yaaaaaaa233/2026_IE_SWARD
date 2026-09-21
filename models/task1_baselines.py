# -*- coding: utf-8 -*-
"""Task 1 frozen-fold baselines: L2 logistic regression and random forest.

The script reads local controlled data, fits only on outer-train folds, chooses
hyperparameters inside each outer train split, and writes out-of-fold metrics.
Real predictions and numeric reports are written to an ignored local output
directory and must not be committed to the public repository.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SOLUTION_B_DIR = REPO_ROOT / "models" / "solution_b"
sys.path.insert(0, str(SOLUTION_B_DIR))
from metrics import topk_table  # noqa: E402


META_COLS = {
    "sample_id",
    "gpsno",
    "as_of",
    "lookback_days",
    "y",
    "fold",
    "label_window",
    "horizon_days",
    "label_status",
    "label_version",
    "feature_version",
    "source_version",
    "split_version",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        default=os.environ.get("FEATURE_ROOT"),
        help="Local feature-engineering root. Defaults to FEATURE_ROOT.",
    )
    parser.add_argument(
        "--model-input",
        default=None,
        help="Optional model_input.csv path. Defaults to <feature-root>/artifacts/model_interface/model_input.csv if present.",
    )
    parser.add_argument(
        "--contract-dir",
        default=None,
        help="Optional contract v4 directory containing features.csv, labels.csv, and splits.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "task1_baselines"),
        help="Directory for local reports and OOF predictions. Keep real reports in ignored or controlled storage.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-folds", type=int, default=3)
    return parser.parse_args()


def default_paths(feature_root: str) -> tuple[Path, Path]:
    if not feature_root:
        raise ValueError("Set FEATURE_ROOT or pass --feature-root.")
    root = Path(feature_root)
    model_input = root / "artifacts" / "model_interface" / "model_input.csv"
    contract_dir = root / "数据筛选" / "契约v0.1全量" / "v4_assessed"
    return model_input, contract_dir


def load_model_frame(args: argparse.Namespace) -> pd.DataFrame:
    default_model_input, default_contract = default_paths(args.feature_root)
    model_input = Path(args.model_input) if args.model_input else default_model_input
    contract_dir = Path(args.contract_dir) if args.contract_dir else default_contract

    if model_input.exists():
        df = pd.read_csv(model_input)
        required = {"sample_id", "gpsno", "y", "fold"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"model_input missing required columns: {sorted(missing)}")
        return df

    features = pd.read_csv(contract_dir / "features.csv")
    labels = pd.read_csv(contract_dir / "labels.csv")
    splits = pd.read_csv(contract_dir / "splits.csv")
    df = features.merge(labels[["sample_id", "y", "label_status"]], on="sample_id", validate="1:1")
    df = df.merge(splits[["gpsno", "fold", "split_version"]], on="gpsno", validate="m:1")
    return df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = [c for c in df.columns if c not in META_COLS]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in candidates if c not in numeric]
    if not numeric and not categorical:
        raise ValueError("No usable feature columns found after excluding metadata.")
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool) -> ColumnTransformer:
    num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, object, list[str]]] = [("num", Pipeline(num_steps), numeric)]
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def model_specs(numeric: list[str], categorical: list[str], seed: int) -> dict[str, tuple[Pipeline, dict]]:
    lr = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical, scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )
    rf = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical, scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {
        "l2_logistic_regression": (lr, {"model__C": [0.1, 1.0, 10.0]}),
        "random_forest": (
            rf,
            {
                "model__max_depth": [4, 6, None],
                "model__min_samples_leaf": [5, 10],
            },
        ),
    }


def hard_validate(df: pd.DataFrame) -> None:
    if df["sample_id"].duplicated().any():
        raise ValueError("sample_id duplicated")
    if not set(df["y"].dropna().unique()) <= {0, 1}:
        raise ValueError("y must be binary 0/1")
    if df["fold"].isna().any():
        raise ValueError("fold contains missing values")
    if df["gpsno"].duplicated().any():
        raise ValueError("This runner expects one row per vehicle/gpsno for task 1.")


def safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def all_report_metrics(y: np.ndarray, p: np.ndarray, sample_id: np.ndarray) -> dict:
    topk = topk_table(y, p, sample_id, q_levels=[0.10, 0.20])
    topk_records = []
    for row in topk.to_dict(orient="records"):
        topk_records.append({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()})
    return {
        "N": int(len(y)),
        "N_pos": int(y.sum()),
        "positive_rate": float(y.mean()),
        "AUC": safe_auc(y, p),
        "AP": float(average_precision_score(y, p)) if y.sum() > 0 and len(np.unique(y)) > 1 else None,
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
        "TopK": topk_records,
    }


def fit_oof(df: pd.DataFrame, numeric: list[str], categorical: list[str], seed: int, inner_folds: int) -> tuple[pd.DataFrame, dict]:
    X = df[numeric + categorical]
    y = df["y"].astype(int).to_numpy()
    folds = df["fold"].astype(int).to_numpy()
    sample_ids = df["sample_id"].astype(str).to_numpy()
    specs = model_specs(numeric, categorical, seed)

    pred = pd.DataFrame({"sample_id": sample_ids, "gpsno": df["gpsno"].to_numpy(), "y": y, "fold": folds})
    report: dict[str, object] = {
        "protocol": {
            "outer_split": "provided frozen vehicle fold",
            "inner_selection": f"StratifiedKFold(n_splits={inner_folds}, shuffle=True, random_state={seed}) inside each outer train fold",
            "tie_break": "sample_id lexicographic for Top-K",
            "baselines": {
                "l2_logistic_regression": "L2 LogisticRegression, C in {0.1, 1, 10}, max_iter=2000",
                "random_forest": "RandomForestClassifier, n_estimators=300, max_features=sqrt, max_depth in {4, 6, None}, min_samples_leaf in {5, 10}",
            },
        },
        "feature_columns": {"numeric": numeric, "categorical": categorical},
        "models": {},
    }

    for name, (pipe, grid) in specs.items():
        p_oof = np.full(len(df), np.nan, dtype=float)
        fold_rows = []
        for fold in sorted(np.unique(folds)):
            valid_mask = folds == fold
            train_mask = ~valid_mask
            cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
            search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
            search.fit(X.loc[train_mask], y[train_mask])
            p_oof[valid_mask] = search.predict_proba(X.loc[valid_mask])[:, 1]
            fold_rows.append(
                {
                    "fold": int(fold),
                    "n_valid": int(valid_mask.sum()),
                    "n_pos": int(y[valid_mask].sum()),
                    "best_params": search.best_params_,
                    "inner_best_auc": float(search.best_score_),
                    "outer_auc": safe_auc(y[valid_mask], p_oof[valid_mask]),
                }
            )

        if np.isnan(p_oof).any():
            raise RuntimeError(f"{name} has missing OOF predictions")
        pred[f"p_{name}"] = p_oof
        report["models"][name] = {
            "oof": all_report_metrics(y, p_oof, sample_ids),
            "folds": fold_rows,
        }
    return pred, report


def main() -> None:
    args = parse_args()
    df = load_model_frame(args)
    hard_validate(df)
    numeric, categorical = feature_columns(df)
    pred, report = fit_oof(df, numeric, categorical, args.seed, args.inner_folds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "task1_baseline_oof_predictions.csv"
    report_path = output_dir / "task1_baseline_metrics.json"
    pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({"metrics_path": str(report_path), "predictions_path": str(pred_path), "models": report["models"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
