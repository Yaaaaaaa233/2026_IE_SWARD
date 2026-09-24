"""Evaluate task-2 safety scores under the public T2-R4 contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LOW_EVIDENCE_STATUSES = {"insufficient_evidence", "unobservable", "low_confidence"}


def _parse_time(value: Any, field: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field} must use ISO-8601 format") from exc


def validate_protocol_manifest(
    manifest: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[str]:
    """Validate the time, label, fold and fitting boundary for one run."""

    errors: list[str] = []
    required = contract["evaluation"]["required_protocol_fields"]
    for field in required:
        if field not in manifest or manifest[field] in (None, ""):
            errors.append(f"protocol manifest lacks {field}")
    if errors:
        return errors
    try:
        as_of = _parse_time(manifest["as_of_time"], "as_of_time")
        feature_start = _parse_time(manifest["feature_window"]["start"], "feature_window.start")
        feature_end = _parse_time(manifest["feature_window"]["end"], "feature_window.end")
        label_start = _parse_time(manifest["label_window"]["start"], "label_window.start")
        label_end = _parse_time(manifest["label_window"]["end"], "label_window.end")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
        return errors
    if not feature_start < feature_end <= as_of <= label_start < label_end:
        errors.append("time windows must satisfy feature_start < feature_end <= as_of <= label_start < label_end")
    if manifest["label_version"] != contract["dependencies"]["label_version"]:
        errors.append("label_version does not match the evaluation contract")
    if manifest["split_version"] != contract["dependencies"]["split_version"]:
        errors.append("split_version does not match the evaluation contract")
    if manifest.get("all_fit_operations_train_fold_only") is not True:
        errors.append("all learned weights, bins, smoothing and thresholds must be fit in training folds")
    if manifest.get("uses_task1_probability") and manifest.get("task1_predictions_are_oof") is not True:
        errors.append("task1 probability must be out-of-fold when used for consistency checks")
    if manifest.get("timezone") != contract["dependencies"]["timezone"]:
        errors.append("timezone does not match the evaluation contract")
    try:
        if int(manifest.get("target_count", 0)) <= 0:
            errors.append("target_count must be positive")
    except (TypeError, ValueError):
        errors.append("target_count must be an integer")
    return errors


def _as_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _auc(labels: Sequence[int], risks: Sequence[float]) -> float | None:
    positives = [risk for label, risk in zip(labels, risks) if label == 1]
    negatives = [risk for label, risk in zip(labels, risks) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _average_precision(labels: Sequence[int], risks: Sequence[float]) -> float | None:
    positive_count = sum(labels)
    if positive_count == 0:
        return None
    grouped: dict[float, list[int]] = defaultdict(list)
    for label, risk in zip(labels, risks):
        grouped[risk].append(label)
    true_positive = 0
    seen = 0
    previous_recall = 0.0
    ap = 0.0
    for risk in sorted(grouped, reverse=True):
        bucket = grouped[risk]
        true_positive += sum(bucket)
        seen += len(bucket)
        recall = true_positive / positive_count
        precision = true_positive / seen
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_ss = sum((a - left_mean) ** 2 for a in left)
    right_ss = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float | None:
    concordant = discordant = left_ties = right_ties = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            left_delta = left[i] - left[j]
            right_delta = right[i] - right[j]
            if left_delta == 0 and right_delta == 0:
                continue
            if left_delta == 0:
                left_ties += 1
            elif right_delta == 0:
                right_ties += 1
            elif left_delta * right_delta > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + left_ties)
        * (concordant + discordant + right_ties)
    )
    return (concordant - discordant) / denominator if denominator else None


def _quintiles(rows: Sequence[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (row["safety_score"], row["gpsno"]))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(5)]
    for index, row in enumerate(ordered):
        bucket = min(4, index * 5 // len(ordered))
        buckets[bucket].append(row)
    table = []
    rates = []
    for index, bucket in enumerate(buckets, start=1):
        rate = sum(row["y"] for row in bucket) / len(bucket) if bucket else None
        rates.append(rate)
        table.append(
            {
                "quintile": index,
                "meaning": "lowest_safety" if index == 1 else "highest_safety" if index == 5 else "middle",
                "count": len(bucket),
                "event_rate": rate,
                "min_safety_score": min((row["safety_score"] for row in bucket), default=None),
                "max_safety_score": max((row["safety_score"] for row in bucket), default=None),
            }
        )
    monotonic = all(
        rates[index] is not None
        and rates[index + 1] is not None
        and rates[index] + tolerance >= rates[index + 1]
        for index in range(4)
    )
    return {"table": table, "monotonic_nonincreasing": monotonic}


def _top_fraction_metrics(rows: Sequence[dict[str, Any]], fraction: float) -> dict[str, Any]:
    count = max(1, math.ceil(len(rows) * fraction))
    ordered = sorted(rows, key=lambda row: (row["safety_score"], row["gpsno"]))
    selected = ordered[:count]
    overall_rate = sum(row["y"] for row in rows) / len(rows)
    selected_rate = sum(row["y"] for row in selected) / len(selected)
    lift = selected_rate / overall_rate if overall_rate else None
    return {
        "fraction": fraction,
        "selected_count": count,
        "event_rate": selected_rate,
        "lift": lift,
        "selected_ids": [row["gpsno"] for row in selected],
    }


def _normalize_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        gpsno = str(raw.get("gpsno", "")).strip()
        if not gpsno:
            errors.append(f"row {index}: gpsno must be nonempty")
            continue
        if gpsno in seen:
            errors.append(f"duplicate gpsno: {gpsno}")
        seen.add(gpsno)
        try:
            label_value = _as_float(raw.get("y"), "y")
            if label_value not in {0.0, 1.0}:
                raise ValueError
            label = int(label_value)
        except (TypeError, ValueError):
            errors.append(f"row {index}: y must be 0 or 1")
            continue
        try:
            safety_score = _as_float(raw.get("safety_score"), "safety_score")
        except ValueError as exc:
            errors.append(f"row {index}: {exc}")
            continue
        if not 0.0 <= safety_score <= 100.0:
            errors.append(f"row {index}: safety_score must be within [0, 100]")
        row = dict(raw)
        fold = str(raw.get("fold", "")).strip()
        if not fold:
            errors.append(f"row {index}: fold must be nonempty")
        row.update(gpsno=gpsno, y=label, safety_score=safety_score, fold=fold)
        for field in ("rule_recomputed_score", "task1_risk_probability"):
            if raw.get(field) not in (None, ""):
                try:
                    row[field] = _as_float(raw[field], field)
                except ValueError as exc:
                    errors.append(f"row {index}: {exc}")
        normalized.append(row)
    return normalized, errors


def evaluate_scores(
    rows: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate score rows and return a JSON-serializable report."""

    normalized, errors = _normalize_rows(rows)
    errors.extend(validate_protocol_manifest(manifest, contract))
    policy = contract["evaluation"]
    if len(normalized) < policy["minimum_rows"]:
        errors.append(f"at least {policy['minimum_rows']} rows are required")
    if normalized and len({row["y"] for row in normalized}) < 2:
        errors.append("both label classes are required")
    try:
        if int(manifest.get("target_count", -1)) != len(normalized):
            errors.append("target_count does not match score row count")
    except (TypeError, ValueError):
        pass
    fold_ids = sorted({row["fold"] for row in normalized if row["fold"]})
    if len(fold_ids) < policy["minimum_fold_count"]:
        errors.append(f"at least {policy['minimum_fold_count']} folds are required")

    tolerance = policy["recomputation_tolerance"]
    for row in normalized:
        recomputed = row.get("rule_recomputed_score")
        if recomputed is not None and abs(row["safety_score"] - recomputed) > tolerance:
            errors.append(f"score recomputation mismatch for {row['gpsno']}")
        status = str(row.get("observation_status", "observed"))
        if status in LOW_EVIDENCE_STATUSES and row["safety_score"] > policy["max_low_evidence_score"]:
            errors.append(f"low-evidence vehicle {row['gpsno']} has a falsely high score")
        probability = row.get("task1_risk_probability")
        if probability is not None and not 0.0 <= probability <= 1.0:
            errors.append(f"task1_risk_probability out of range for {row['gpsno']}")
    has_probabilities = ["task1_risk_probability" in row for row in normalized]
    if manifest.get("uses_task1_probability") is True and not all(has_probabilities):
        errors.append("task1 probability was declared but is not complete for all vehicles")
    if manifest.get("uses_task1_probability") is not True and any(has_probabilities):
        errors.append("task1 probability is present but not declared in the protocol manifest")

    if errors:
        return {"status": "fail", "errors": sorted(set(errors)), "row_count": len(normalized)}

    labels = [row["y"] for row in normalized]
    risks = [100.0 - row["safety_score"] for row in normalized]
    quintiles = _quintiles(normalized, policy["quintile_rate_tolerance"])
    top = _top_fraction_metrics(normalized, policy["top_fraction"])
    fold_report: dict[str, Any] = {}
    for fold in fold_ids:
        group = [row for row in normalized if row["fold"] == fold]
        fold_auc = _auc(
            [row["y"] for row in group],
            [100.0 - row["safety_score"] for row in group],
        )
        fold_report[fold] = {
            "count": len(group),
            "roc_auc": fold_auc,
            "direction_ok": fold_auc is not None and fold_auc >= 0.5,
        }
    fold_direction_consistent = all(
        item["direction_ok"] for item in fold_report.values()
    )
    report: dict[str, Any] = {
        "status": "pass",
        "errors": [],
        "row_count": len(normalized),
        "roc_auc": _auc(labels, risks),
        "average_precision": _average_precision(labels, risks),
        "quintiles": quintiles,
        "top_fraction": {key: value for key, value in top.items() if key != "selected_ids"},
        "folds": fold_report,
        "fold_direction_consistent": fold_direction_consistent,
        "primary_gate_pass": quintiles["monotonic_nonincreasing"]
        and fold_direction_consistent,
        "subgroups": {},
    }

    for field in policy["subgroup_fields"]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in normalized:
            value = str(row.get(field, "")).strip()
            if value:
                grouped[value].append(row)
        field_report = {
            key: {
                "count": len(group),
                "roc_auc": _auc(
                    [item["y"] for item in group],
                    [100.0 - item["safety_score"] for item in group],
                )
                if len(group) >= policy["minimum_subgroup_rows"]
                else None,
                "status": "evaluated"
                if len(group) >= policy["minimum_subgroup_rows"]
                else "insufficient_rows",
            }
            for key, group in sorted(grouped.items())
        }
        report["subgroups"][field] = field_report
    evaluated_subgroups = [
        group
        for field_report in report["subgroups"].values()
        for group in field_report.values()
        if group["status"] == "evaluated" and group["roc_auc"] is not None
    ]
    report["fairness_direction_pass"] = (
        all(group["roc_auc"] >= 0.5 for group in evaluated_subgroups)
        if evaluated_subgroups
        else None
    )

    if all("task1_risk_probability" in row for row in normalized):
        model_risks = [row["task1_risk_probability"] for row in normalized]
        model_order = sorted(
            normalized,
            key=lambda row: (-row["task1_risk_probability"], row["gpsno"]),
        )[: top["selected_count"]]
        model_ids = {row["gpsno"] for row in model_order}
        score_ids = set(top["selected_ids"])
        report["task1_consistency"] = {
            "spearman": _spearman(risks, model_risks),
            "kendall_tau_b": _kendall_tau_b(risks, model_risks),
            "top_fraction_overlap": len(model_ids & score_ids) / len(score_ids),
            "role": "reference_only_not_a_selection_gate",
        }
    else:
        report["task1_consistency"] = {
            "status": "not_available",
            "role": "reference_only_not_a_selection_gate",
        }
    return report


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("evaluation_contract_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = evaluate_scores(_read_csv(args.scores), contract, manifest)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
