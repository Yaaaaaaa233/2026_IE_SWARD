"""Distil task-1 OOF probabilities into a monotone, auditable score.

The teacher probability is a soft learning target.  Each behaviour rate first
becomes a monotone severity curve, related behaviours are fused inside their
semantic group with diminishing marginal influence, and a non-negative ridge
fit assigns one point budget per group.  The resulting D0 score is independent
code: it does not modify or replace the R1/Q0 engine.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


DEDUCTIBLE_ROLES = frozenset({"behavior", "context_warning"})


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    name: str
    semantic_group: str
    public_dimension: str
    column: str


@dataclass(frozen=True)
class MonotoneCurve:
    boundaries: tuple[float, ...]
    severity: tuple[float, ...]
    teacher_means: tuple[float, ...]
    support: tuple[int, ...]

    def transform(self, value: float) -> float:
        return float(self.severity[bisect.bisect_right(self.boundaries, value)])


@dataclass
class DistilledRuleset:
    rule_version: str
    teacher_model_version: str
    teacher_score_fingerprint: str
    feature_specs: tuple[FeatureSpec, ...]
    curves: dict[str, MonotoneCurve]
    group_weights: dict[str, float]
    base_deduction: float
    group_agg_fraction: float
    min_exposure: float
    low_evidence_prior: float
    n_bins: int
    ridge: float
    fit_vehicle_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_version": self.rule_version,
            "teacher_model_version": self.teacher_model_version,
            "teacher_score_kind": "out_of_fold_probability",
            "teacher_score_fingerprint": self.teacher_score_fingerprint,
            "feature_specs": [asdict(spec) for spec in self.feature_specs],
            "curves": {
                key: {
                    "boundaries": list(curve.boundaries),
                    "severity": list(curve.severity),
                    "teacher_means": list(curve.teacher_means),
                    "support": list(curve.support),
                }
                for key, curve in sorted(self.curves.items())
            },
            "group_weights": dict(sorted(self.group_weights.items())),
            "base_deduction": self.base_deduction,
            "group_agg_fraction": self.group_agg_fraction,
            "min_exposure": self.min_exposure,
            "low_evidence_prior": self.low_evidence_prior,
            "n_bins": self.n_bins,
            "ridge": self.ridge,
            "fit_vehicle_count": self.fit_vehicle_count,
        }


@dataclass
class DistilledScoring:
    gpsno: str
    safety_score: float
    teacher_probability: float | None
    raw_deduction: float
    applied_deduction: float
    observation_status: str
    feature_severity: dict[str, float] = field(default_factory=dict)
    class_points: dict[str, float] = field(default_factory=dict)
    group_deductions: dict[str, float] = field(default_factory=dict)
    top_reasons: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CrossFitResult:
    rows: list[dict[str, Any]]
    fold_rulesets: dict[str, DistilledRuleset]
    metrics: dict[str, Any]


def build_feature_specs(
    classification_contract: Mapping[str, Any],
    *,
    column_prefix: str = "rate_",
    column_suffix: str = "_per_1000km",
) -> tuple[FeatureSpec, ...]:
    """Build the D0 input list from the frozen T2-R0 ownership contract."""

    specs: list[FeatureSpec] = []
    for row in classification_contract.get("event_mappings", []):
        if str(row.get("scoring_role")) not in DEDUCTIBLE_ROLES:
            continue
        code = str(row["code"])
        specs.append(
            FeatureSpec(
                key=str(row["catalog_class"]),
                name=str(row["name"]),
                semantic_group=str(row["semantic_group"]),
                public_dimension=str(row["public_dimension"]),
                column=f"{column_prefix}{code}{column_suffix}",
            )
        )
    _validate_specs(specs)
    return tuple(specs)


def fit_ruleset(
    rows: Sequence[Mapping[str, Any]],
    teacher_scores: Mapping[str, float],
    feature_specs: Sequence[FeatureSpec],
    *,
    teacher_model_version: str,
    teacher_scores_are_oof: bool,
    rule_version: str = "D0-task1-distilled-v1",
    n_bins: int = 4,
    ridge: float = 0.05,
    group_agg_fraction: float = 0.25,
    min_exposure: float = 0.0,
    low_evidence_prior: float = 50.0,
) -> DistilledRuleset:
    """Fit a transparent D0 ruleset against task-1 OOF soft targets.

    All bin boundaries, monotone curves and weights are learned from ``rows``.
    Cross-fitted evaluation must call this function separately inside each
    training fold; fitting once on all vehicles is only for the delivery rule.
    """

    if not teacher_scores_are_oof:
        raise ValueError("teacher scores must be declared out-of-fold")
    if not str(teacher_model_version).strip():
        raise ValueError("teacher_model_version must be non-empty")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")
    if not 0.0 <= group_agg_fraction <= 1.0:
        raise ValueError("group_agg_fraction must be within [0, 1]")
    if not math.isfinite(min_exposure) or min_exposure < 0:
        raise ValueError("min_exposure must be finite and non-negative")
    if not math.isfinite(low_evidence_prior) or not 0 <= low_evidence_prior <= 100:
        raise ValueError("low_evidence_prior must be within [0, 100]")

    specs = tuple(feature_specs)
    _validate_specs(specs)
    ids, matrix, exposure = _validated_matrix(rows, specs)
    teacher_map = {str(key): value for key, value in teacher_scores.items()}
    teacher = _validated_teacher(ids, teacher_map)
    usable = exposure > min_exposure
    if int(usable.sum()) < max(10, len({spec.semantic_group for spec in specs}) + 2):
        raise ValueError("too few exposed vehicles to fit the distilled rules")

    fit_x = matrix[usable]
    fit_teacher = teacher[usable]
    curves: dict[str, MonotoneCurve] = {}
    transformed = np.zeros_like(fit_x, dtype=float)
    for index, spec in enumerate(specs):
        curve = _fit_monotone_curve(fit_x[:, index], fit_teacher, n_bins=n_bins)
        curves[spec.key] = curve
        transformed[:, index] = [curve.transform(float(value)) for value in fit_x[:, index]]

    group_names = sorted({spec.semantic_group for spec in specs})
    group_matrix = _group_matrix(transformed, specs, group_names, group_agg_fraction)
    target_deduction = 100.0 * fit_teacher
    base, weights = _fit_nonnegative_ridge(group_matrix, target_deduction, ridge)
    return DistilledRuleset(
        rule_version=str(rule_version),
        teacher_model_version=str(teacher_model_version),
        teacher_score_fingerprint=teacher_fingerprint(teacher_map),
        feature_specs=specs,
        curves=curves,
        group_weights={group: float(weight) for group, weight in zip(group_names, weights)},
        base_deduction=float(base),
        group_agg_fraction=float(group_agg_fraction),
        min_exposure=float(min_exposure),
        low_evidence_prior=float(low_evidence_prior),
        n_bins=int(n_bins),
        ridge=float(ridge),
        fit_vehicle_count=int(usable.sum()),
    )


def score_vehicle(
    row: Mapping[str, Any],
    ruleset: DistilledRuleset,
    *,
    teacher_probability: float | None = None,
) -> DistilledScoring:
    """Apply a frozen ruleset and emit an exactly reproducible deduction ledger."""

    gpsno = str(row.get("gpsno", "")).strip()
    if not gpsno:
        raise ValueError("vehicle row must contain a non-empty gpsno")
    exposure = _finite_nonnegative(row.get("exposure"), f"exposure for {gpsno}")
    if teacher_probability is not None:
        teacher_probability = _probability(teacher_probability, f"teacher score for {gpsno}")
    if exposure <= ruleset.min_exposure:
        applied = 100.0 - ruleset.low_evidence_prior
        return DistilledScoring(
            gpsno=gpsno,
            safety_score=ruleset.low_evidence_prior,
            teacher_probability=teacher_probability,
            raw_deduction=applied,
            applied_deduction=applied,
            observation_status="insufficient_evidence",
        )

    severity: dict[str, float] = {}
    by_group: dict[str, list[tuple[FeatureSpec, float]]] = {}
    for spec in ruleset.feature_specs:
        value = _finite_nonnegative(row.get(spec.column), f"{spec.column} for {gpsno}")
        level = ruleset.curves[spec.key].transform(value)
        severity[spec.key] = level
        by_group.setdefault(spec.semantic_group, []).append((spec, level))

    class_points: dict[str, float] = {}
    group_deductions: dict[str, float] = {}
    reason_names = {spec.key: spec.name for spec in ruleset.feature_specs}
    for group in sorted(by_group):
        components = _group_components(by_group[group], ruleset.group_agg_fraction)
        weight = ruleset.group_weights.get(group, 0.0)
        for key, component in components.items():
            class_points[key] = float(weight * component)
        group_deductions[group] = float(sum(class_points[key] for key in components))

    raw = float(ruleset.base_deduction + sum(group_deductions.values()))
    applied = float(min(100.0, max(0.0, raw)))
    top_reasons = [
        {"catalog_class": key, "name": reason_names[key], "points": points}
        for key, points in sorted(class_points.items(), key=lambda item: (-item[1], item[0]))
        if points > 0
    ][:3]
    return DistilledScoring(
        gpsno=gpsno,
        safety_score=100.0 - applied,
        teacher_probability=teacher_probability,
        raw_deduction=raw,
        applied_deduction=applied,
        observation_status="observed",
        feature_severity=severity,
        class_points=class_points,
        group_deductions=group_deductions,
        top_reasons=top_reasons,
    )


def recompute_score(scoring: DistilledScoring, ruleset: DistilledRuleset) -> float:
    if scoring.observation_status == "insufficient_evidence":
        return float(ruleset.low_evidence_prior)
    deduction = ruleset.base_deduction + sum(scoring.class_points.values())
    return float(100.0 - min(100.0, max(0.0, deduction)))


def cross_fit_distillation(
    rows: Sequence[Mapping[str, Any]],
    teacher_scores: Mapping[str, float],
    folds: Mapping[str, str],
    feature_specs: Sequence[FeatureSpec],
    *,
    teacher_model_version: str,
    teacher_scores_are_oof: bool,
    rule_version: str = "D0-task1-distilled-v1",
    actual_labels: Mapping[str, int] | None = None,
    **fit_kwargs: Any,
) -> CrossFitResult:
    """Generate honest D0 scores with every learned operation inside a fold."""

    ids, _, _ = _validated_matrix(rows, tuple(feature_specs))
    teacher_map = {str(key): value for key, value in teacher_scores.items()}
    _validated_teacher(ids, teacher_map)
    fold_map = _validated_folds(ids, folds)
    label_map = None if actual_labels is None else {str(key): int(value) for key, value in actual_labels.items()}
    if label_map is not None:
        if set(ids) != set(label_map):
            raise ValueError("actual_labels must cover exactly the scored vehicles")
        if not set(label_map.values()) <= {0, 1}:
            raise ValueError("actual labels must be binary")
    by_id = {str(row["gpsno"]): row for row in rows}
    fold_rulesets: dict[str, DistilledRuleset] = {}
    output: list[dict[str, Any]] = []
    for fold in sorted(set(fold_map.values())):
        train_ids = [gpsno for gpsno in ids if fold_map[gpsno] != fold]
        valid_ids = [gpsno for gpsno in ids if fold_map[gpsno] == fold]
        if not train_ids or not valid_ids:
            raise ValueError(f"fold {fold} must have both train and validation vehicles")
        trained = fit_ruleset(
            [by_id[gpsno] for gpsno in train_ids],
            {gpsno: teacher_map[gpsno] for gpsno in train_ids},
            feature_specs,
            teacher_model_version=teacher_model_version,
            teacher_scores_are_oof=teacher_scores_are_oof,
            rule_version=f"{rule_version}/fit_without_{fold}",
            **fit_kwargs,
        )
        fold_rulesets[fold] = trained
        for gpsno in valid_ids:
            scored = score_vehicle(
                by_id[gpsno], trained, teacher_probability=float(teacher_map[gpsno])
            )
            output.append(
                {
                    "gpsno": gpsno,
                    "fold": fold,
                    "teacher_probability": float(teacher_map[gpsno]),
                    "task1_risk_probability": float(teacher_map[gpsno]),
                    "safety_score": scored.safety_score,
                    "rule_recomputed_score": recompute_score(scored, trained),
                    "raw_deduction": scored.raw_deduction,
                    "observation_status": scored.observation_status,
                    "top_reasons": scored.top_reasons,
                    "rule_version": trained.rule_version,
                    **({"y": label_map[gpsno]} if label_map is not None else {}),
                }
            )
    output.sort(key=lambda row: row["gpsno"])
    metrics = evaluate_fidelity(output, actual_labels=label_map)
    metrics.update(
        {
            "score_name": "D0",
            "teacher_score_name": "Q1_probability",
            "teacher_model_version": teacher_model_version,
            "teacher_scores_are_oof": bool(teacher_scores_are_oof),
            "evaluation_mode": "cross_fitted_teacher_distillation",
        }
    )
    return CrossFitResult(rows=output, fold_rulesets=fold_rulesets, metrics=metrics)


def evaluate_fidelity(
    rows: Sequence[Mapping[str, Any]],
    *,
    actual_labels: Mapping[str, int] | None = None,
    top_fraction: float = 0.2,
) -> dict[str, Any]:
    """Measure D0 fidelity to Q1 and, when supplied, real-outcome separation."""

    if not rows:
        raise ValueError("at least one score row is required")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be within (0, 1]")
    ids = [str(row["gpsno"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("score rows contain duplicate gpsno values")
    teacher = np.asarray(
        [_probability(row["teacher_probability"], f"teacher score for {row['gpsno']}") for row in rows]
    )
    scores = np.asarray([float(row["safety_score"]) for row in rows], dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 100)).any():
        raise ValueError("safety scores must be finite and within [0, 100]")
    student_risk = 1.0 - scores / 100.0
    k = max(1, int(math.ceil(len(rows) * top_fraction)))
    teacher_top = set(np.argsort(-teacher, kind="stable")[:k].tolist())
    student_top = set(np.argsort(-student_risk, kind="stable")[:k].tolist())
    order = np.argsort(scores, kind="stable")
    quintile_means = [float(np.mean(teacher[indexes])) for indexes in np.array_split(order, 5)]
    result: dict[str, Any] = {
        "vehicle_count": len(rows),
        "teacher_risk_spearman": _spearman(teacher, student_risk),
        "teacher_risk_kendall": _kendall_tau_b(teacher, student_risk),
        "teacher_penalty_mae": float(np.mean(np.abs(100.0 * teacher - 100.0 * student_risk))),
        "top_risk_overlap": len(teacher_top & student_top) / k,
        "top_fraction": top_fraction,
        "teacher_probability_by_score_quintile_low_to_high": quintile_means,
        "quintile_direction_pass": all(
            left + 1e-12 >= right for left, right in zip(quintile_means, quintile_means[1:])
        ),
        "teacher_fidelity_only": actual_labels is None,
        "actual_outcome_validated": actual_labels is not None,
    }
    if actual_labels is not None:
        label_map = {str(key): int(value) for key, value in actual_labels.items()}
        if set(ids) != set(label_map):
            raise ValueError("actual_labels must cover exactly the scored vehicles")
        labels = np.asarray([label_map[gpsno] for gpsno in ids], dtype=int)
        if not set(labels.tolist()) <= {0, 1}:
            raise ValueError("actual labels must be binary")
        if len(set(labels.tolist())) < 2:
            result.update({"actual_auc": None, "actual_average_precision": None})
        else:
            from sklearn.metrics import average_precision_score, roc_auc_score

            result.update(
                {
                    "actual_auc": float(roc_auc_score(labels, student_risk)),
                    "actual_average_precision": float(average_precision_score(labels, student_risk)),
                }
            )
    return result


def teacher_fingerprint(scores: Mapping[str, float]) -> str:
    payload = [
        [str(gpsno), format(_probability(prob, f"teacher score for {gpsno}"), ".12g")]
        for gpsno, prob in sorted(scores.items(), key=lambda item: str(item[0]))
    ]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _fit_monotone_curve(x: np.ndarray, teacher: np.ndarray, *, n_bins: int) -> MonotoneCurve:
    low = float(np.min(x))
    high = float(np.max(x))
    if low == high:
        mean = float(np.mean(teacher))
        return MonotoneCurve((), (0.0,), (mean,), (len(x),))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    boundaries = sorted(
        {
            float(value)
            for value in np.quantile(x, quantiles)
            if low < float(value) < high
        }
    )
    indexes = np.searchsorted(boundaries, x, side="right")
    raw_means: list[float] = []
    support: list[int] = []
    for bin_index in range(len(boundaries) + 1):
        values = teacher[indexes == bin_index]
        if len(values) == 0:
            raise RuntimeError("quantile bin unexpectedly has no observations")
        raw_means.append(float(np.mean(values)))
        support.append(int(len(values)))
    monotone = _pava(raw_means, support)
    span = monotone[-1] - monotone[0]
    severity = [0.0 for _ in monotone] if span <= 1e-12 else [
        float((value - monotone[0]) / span) for value in monotone
    ]
    return MonotoneCurve(
        boundaries=tuple(boundaries),
        severity=tuple(severity),
        teacher_means=tuple(float(value) for value in monotone),
        support=tuple(support),
    )


def _pava(values: Sequence[float], weights: Sequence[int]) -> list[float]:
    blocks: list[list[float]] = []
    for value, weight in zip(values, weights):
        blocks.append([float(value), float(weight), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left[1] + right[1]
            mean = (left[0] * left[1] + right[0] * right[1]) / total_weight
            blocks.append([mean, total_weight, left[2] + right[2]])
    result: list[float] = []
    for mean, _, length in blocks:
        result.extend([float(mean)] * int(length))
    return result


def _group_matrix(
    transformed: np.ndarray,
    specs: Sequence[FeatureSpec],
    group_names: Sequence[str],
    fraction: float,
) -> np.ndarray:
    output = np.zeros((transformed.shape[0], len(group_names)), dtype=float)
    for group_index, group in enumerate(group_names):
        indexes = [index for index, spec in enumerate(specs) if spec.semantic_group == group]
        for row_index in range(transformed.shape[0]):
            members = [(specs[index], float(transformed[row_index, index])) for index in indexes]
            output[row_index, group_index] = sum(_group_components(members, fraction).values())
    return output


def _group_components(
    members: Sequence[tuple[FeatureSpec, float]], fraction: float
) -> dict[str, float]:
    if not members:
        return {}
    ranked = sorted(members, key=lambda item: (-item[1], item[0].key))
    if len(ranked) == 1:
        return {ranked[0][0].key: float(ranked[0][1])}
    denominator = 1.0 + fraction
    result = {ranked[0][0].key: float(ranked[0][1] / denominator)}
    secondary_scale = fraction / (len(ranked) - 1) / denominator
    for spec, value in ranked[1:]:
        result[spec.key] = float(value * secondary_scale)
    return result


def _fit_nonnegative_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[float, np.ndarray]:
    if x.ndim != 2 or len(y) != x.shape[0]:
        raise ValueError("invalid regression matrix")
    weights = np.zeros(x.shape[1], dtype=float)
    intercept = float(np.clip(np.mean(y), 0.0, 100.0))
    for _ in range(10000):
        old_intercept = intercept
        old_weights = weights.copy()
        intercept = float(np.clip(np.mean(y - x @ weights), 0.0, 100.0))
        prediction = intercept + x @ weights
        for index in range(x.shape[1]):
            column = x[:, index]
            residual = y - prediction + column * weights[index]
            denominator = float(column @ column + len(y) * ridge)
            updated = 0.0 if denominator <= 0 else max(0.0, float(column @ residual) / denominator)
            prediction += column * (updated - weights[index])
            weights[index] = updated
        delta = max(abs(intercept - old_intercept), float(np.max(np.abs(weights - old_weights))))
        if delta < 1e-10:
            break
    return intercept, weights


def _validated_matrix(
    rows: Sequence[Mapping[str, Any]], specs: Sequence[FeatureSpec]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("at least one vehicle row is required")
    ids = [str(row.get("gpsno", "")).strip() for row in rows]
    if any(not gpsno for gpsno in ids):
        raise ValueError("every vehicle row must contain gpsno")
    if len(ids) != len(set(ids)):
        raise ValueError("vehicle rows contain duplicate gpsno values")
    matrix = np.empty((len(rows), len(specs)), dtype=float)
    exposure = np.empty(len(rows), dtype=float)
    for row_index, row in enumerate(rows):
        exposure[row_index] = _finite_nonnegative(row.get("exposure"), f"exposure for {ids[row_index]}")
        for column_index, spec in enumerate(specs):
            matrix[row_index, column_index] = _finite_nonnegative(
                row.get(spec.column), f"{spec.column} for {ids[row_index]}"
            )
    return ids, matrix, exposure


def _validated_teacher(ids: Sequence[str], scores: Mapping[str, float]) -> np.ndarray:
    normalised = {str(key): value for key, value in scores.items()}
    if set(ids) != set(normalised):
        raise ValueError("teacher scores must cover exactly the feature vehicles")
    return np.asarray([_probability(normalised[gpsno], f"teacher score for {gpsno}") for gpsno in ids])


def _validated_folds(ids: Sequence[str], folds: Mapping[str, str]) -> dict[str, str]:
    normalised = {str(key): str(value).strip() for key, value in folds.items()}
    if set(ids) != set(normalised):
        raise ValueError("folds must cover exactly the feature vehicles")
    if any(not fold for fold in normalised.values()):
        raise ValueError("fold values must be non-empty")
    if len(set(normalised.values())) < 2:
        raise ValueError("at least two folds are required")
    return normalised


def _validate_specs(specs: Sequence[FeatureSpec]) -> None:
    if not specs:
        raise ValueError("at least one feature specification is required")
    for attribute in ("key", "column"):
        values = [getattr(spec, attribute) for spec in specs]
        if len(values) != len(set(values)):
            raise ValueError(f"feature specifications contain duplicate {attribute}")
    if any(not spec.semantic_group or not spec.public_dimension for spec in specs):
        raise ValueError("feature specifications require group and public dimension")


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _probability(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label} must be finite and within [0, 1]")
    return number


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = _rank_average(left)
    right_rank = _rank_average(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _kendall_tau_b(left: np.ndarray, right: np.ndarray) -> float | None:
    concordant = discordant = left_ties = right_ties = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            delta_left = np.sign(left[first] - left[second])
            delta_right = np.sign(right[first] - right[second])
            if delta_left == 0 and delta_right == 0:
                continue
            if delta_left == 0:
                left_ties += 1
            elif delta_right == 0:
                right_ties += 1
            elif delta_left == delta_right:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + left_ties) *
        (concordant + discordant + right_ties)
    )
    return None if denominator == 0 else (concordant - discordant) / denominator
