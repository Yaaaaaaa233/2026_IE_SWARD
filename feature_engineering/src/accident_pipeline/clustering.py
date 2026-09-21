from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from .config import PipelineConfig
from .dataset import DatasetBundle


def _cluster_preprocessor(
    bundle: DatasetBundle, random_state: int, fit_frame: pd.DataFrame
) -> tuple[ColumnTransformer, dict[str, list[str]]]:
    groups: dict[str, list[str]] = {
        "behavior": [], "vehicle": [], "coverage": [], "imu_accident": [], "categorical": bundle.categorical_columns,
    }
    for column in bundle.numeric_columns:
        if column.startswith(("accident_", "imu_")):
            groups["imu_accident"].append(column)
        elif column.startswith("coverage_"):
            groups["coverage"].append(column)
        elif column.startswith(("evt_", "traj_")):
            groups["behavior"].append(column)
        else:
            groups["vehicle"].append(column)

    transformers = []
    weights: dict[str, float] = {}
    for name in ("behavior", "vehicle", "coverage", "imu_accident"):
        columns = groups[name]
        if not columns:
            continue
        transformers.append((
            name,
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("rank_normalize", QuantileTransformer(
                    n_quantiles=min(200, len(fit_frame)),
                    output_distribution="normal",
                    random_state=random_state,
                )),
            ]),
            columns,
        ))
        # Equalize total expected squared contribution of semantic groups.
        weights[name] = 1.0 / np.sqrt(len(columns))
    if bundle.categorical_columns:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            bundle.categorical_columns,
        ))
        encoded_width = sum(max(1, fit_frame[column].nunique(dropna=True)) for column in bundle.categorical_columns)
        weights["categorical"] = 1.0 / np.sqrt(encoded_width)
    return ColumnTransformer(
        transformers,
        remainder="drop",
        transformer_weights=weights,
        verbose_feature_names_out=False,
    ), groups


def cluster_features(bundle: DatasetBundle, config: PipelineConfig) -> pd.DataFrame:
    """Cluster every model feature; labels and folds are used only for post-hoc diagnostics."""
    settings = config.clustering
    observed = pd.Series(False, index=bundle.frame.index)
    rules = []
    if "evt_count" in bundle.frame:
        observed |= bundle.frame["evt_count"].fillna(0) > 0
        rules.append("evt_count > 0")
    if "traj_km_20d" in bundle.frame:
        observed |= bundle.frame["traj_km_20d"].fillna(0) >= settings.min_trajectory_km
        rules.append(f"traj_km_20d >= {settings.min_trajectory_km}")
    if "imu_rows_window" in bundle.frame:
        observed |= bundle.frame["imu_rows_window"].fillna(0) >= settings.min_imu_rows
        rules.append(f"imu_rows_window >= {settings.min_imu_rows}")
    if not rules:
        observed[:] = True
    eligible_frame = bundle.frame.loc[observed]
    preprocessing, feature_groups = _cluster_preprocessor(bundle, settings.random_state, eligible_frame)
    matrix = np.asarray(preprocessing.fit_transform(eligible_frame[bundle.feature_columns]), dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Preprocessed clustering matrix contains non-finite values")

    upper = min(settings.max_clusters, len(eligible_frame) - 1)
    if upper < settings.min_clusters:
        raise ValueError("Not enough samples for the requested clustering range")
    trials: list[dict[str, float | int]] = []
    fitted: dict[int, KMeans] = {}
    for k in range(settings.min_clusters, upper + 1):
        model = KMeans(n_clusters=k, random_state=settings.random_state, n_init=settings.n_init)
        labels = model.fit_predict(matrix)
        fitted[k] = model
        trials.append({
            "k": k,
            "silhouette": float(silhouette_score(matrix, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
            "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
            "inertia": float(model.inertia_),
        })
    best = max(trials, key=lambda item: (item["silhouette"], -item["k"]))
    best_k = int(best["k"])
    kmeans = fitted[best_k]
    eligible_labels = kmeans.labels_.astype(int)
    labels = np.full(len(bundle.frame), -1, dtype=int)
    labels[np.flatnonzero(observed.to_numpy())] = eligible_labels

    risk_column = "accident_sensor_risk_index"
    if risk_column in bundle.frame:
        risk_by_cluster = eligible_frame.assign(cluster=eligible_labels).groupby("cluster")[risk_column].mean()
        ordered = list(risk_by_cluster.sort_values().index)
    else:
        ordered = list(pd.Series(eligible_labels).value_counts().sort_values(ascending=False).index)
    rank = {int(cluster): position + 1 for position, cluster in enumerate(ordered)}

    assignments = bundle.frame[["sample_id", "gpsno", "y", "fold"]].copy()
    assignments["cluster"] = labels
    assignments["clustering_eligible"] = observed.to_numpy().astype(int)
    assignments["cluster_risk_rank"] = [rank.get(int(value), 0) for value in labels]
    assignments["cluster_risk_tier"] = [
        f"risk_{rank[int(value)]}_of_{best_k}" if value >= 0 else "insufficient_data" for value in labels
    ]

    numeric_profile = bundle.frame[bundle.numeric_columns].assign(cluster=labels).groupby("cluster").mean()
    overview = assignments.groupby("cluster").agg(
        sample_count=("sample_id", "size"),
        positive_count=("y", "sum"),
        positive_rate=("y", "mean"),
        cluster_risk_rank=("cluster_risk_rank", "first"),
    )
    profiles = overview.join(numeric_profile).reset_index().sort_values("cluster_risk_rank")

    transformed_names = list(preprocessing.get_feature_names_out())
    top_drivers: dict[str, list[dict[str, float | str]]] = {}
    for cluster in range(best_k):
        center = kmeans.cluster_centers_[cluster]
        top = np.argsort(np.abs(center))[::-1][:10]
        top_drivers[str(cluster)] = [
            {"feature": transformed_names[index], "standardized_center": float(center[index])}
            for index in top
        ]

    output = config.output_dir / "clustering"
    output.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output / "cluster_assignments.csv", index=False)
    profiles.to_csv(output / "cluster_profiles.csv", index=False)
    pd.DataFrame(trials).to_csv(output / "cluster_selection.csv", index=False)
    model_pipeline = Pipeline([("preprocess", preprocessing), ("cluster", kmeans)])
    joblib.dump(model_pipeline, output / "cluster_model.joblib")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "maximum silhouette; lower k wins exact ties",
        "selected_k": best_k,
        "selected_metrics": best,
        "eligible_samples": int(observed.sum()),
        "insufficient_data_samples": int((~observed).sum()),
        "eligibility_rule": " OR ".join(rules),
        "input_features": bundle.feature_columns,
        "feature_groups": feature_groups,
        "group_weighting": "each semantic group scaled by 1/sqrt(number of transformed dimensions)",
        "numeric_transform": "median imputation + empirical quantile to normal distribution",
        "label_used_for_fit": False,
        "fold_used_for_fit": False,
        "top_cluster_drivers": top_drivers,
    }
    (output / "cluster_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return assignments
