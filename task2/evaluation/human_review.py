"""Validate the fixed 5-reviewer by 10-case human-factors review."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def validate_human_review(
    rows: Iterable[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    policy = contract["human_review"]
    questions = list(policy["questions"])
    rows = list(rows)
    errors: list[str] = []

    reviewer_ids = {str(row.get("reviewer_id", "")) for row in rows}
    case_ids = {str(row.get("case_id", "")) for row in rows}
    pairs = [(str(row.get("reviewer_id", "")), str(row.get("case_id", ""))) for row in rows]
    if "" in reviewer_ids or "" in case_ids:
        errors.append("reviewer_id and case_id must be nonempty")
    if len(set(pairs)) != len(pairs):
        errors.append("each reviewer-case pair must appear exactly once")
    if len(reviewer_ids) != policy["reviewer_count"]:
        errors.append(f"expected {policy['reviewer_count']} reviewers")
    if len(case_ids) != policy["case_count"]:
        errors.append(f"expected {policy['case_count']} cases")
    if len(rows) != policy["reviewer_count"] * policy["case_count"]:
        errors.append("review matrix must be complete")

    by_case_type: dict[str, set[str]] = defaultdict(set)
    case_type_seen: dict[str, str] = {}
    ratings: list[float] = []
    question_ratings: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        case_id = str(row.get("case_id", ""))
        case_type = str(row.get("case_type", ""))
        previous = case_type_seen.setdefault(case_id, case_type)
        if previous != case_type:
            errors.append(f"case {case_id} has inconsistent case_type")
        by_case_type[case_type].add(case_id)
        for question in questions:
            try:
                value = float(row[question])
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid rating for {question}")
                continue
            if not math.isfinite(value) or not 1.0 <= value <= 5.0:
                errors.append(f"rating for {question} must be within [1, 5]")
                continue
            ratings.append(value)
            question_ratings[question].append(value)

    for case_type, expected_count in policy["case_type_quotas"].items():
        if len(by_case_type.get(case_type, set())) != expected_count:
            errors.append(f"case type {case_type} must contain {expected_count} cases")
    unexpected_types = set(by_case_type) - set(policy["case_type_quotas"])
    if unexpected_types:
        errors.append(f"unexpected case types: {sorted(unexpected_types)}")

    grand_mean = sum(ratings) / len(ratings) if ratings else None
    minimum_rating = min(ratings) if ratings else None
    question_means = {
        key: sum(values) / len(values) for key, values in question_ratings.items() if values
    }
    if grand_mean is None or grand_mean < policy["minimum_mean"]:
        errors.append(f"overall mean must be at least {policy['minimum_mean']}")
    if minimum_rating is None or minimum_rating < policy["minimum_single_rating"]:
        errors.append(
            f"every single rating must be at least {policy['minimum_single_rating']}"
        )

    return {
        "status": "pass" if not errors else "fail",
        "errors": sorted(set(errors)),
        "reviewer_count": len(reviewer_ids - {""}),
        "case_count": len(case_ids - {""}),
        "rating_count": len(ratings),
        "overall_mean": grand_mean,
        "minimum_rating": minimum_rating,
        "question_means": question_means,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviews", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("evaluation_contract_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    with args.reviews.open("r", encoding="utf-8-sig", newline="") as handle:
        report = validate_human_review(list(csv.DictReader(handle)), contract)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
