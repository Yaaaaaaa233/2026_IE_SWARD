"""Validate the public task-2 classification contract.

The checker deliberately uses only the Python standard library.  It validates
the semantic contract and never reads competition data or derived metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_CODES = {
    "11401", "11402", "11403", "11405", "11406",
    "30000", "30002", "30003", "30005", "30017",
    "41001", "41002", "41003", "41004", "41005", "41006",
    "41009", "41021", "41023", "41029", "60292", "60294",
    "11803", "11804",
}
EVIDENCE_STATUSES = {"event", "condition", "reserved"}
SCORING_ROLES = {"behavior", "evidence_quality", "context_warning", "outcome_gate"}
ALLOWED_WINDOWS = {60, 120, 600}


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    return {value for value in values if value in seen or seen.add(value)}


def validate_contract(contract: dict[str, Any]) -> list[str]:
    """Return all contract violations; an empty list means the contract passes."""

    errors: list[str] = []
    dimensions = contract.get("public_dimensions", [])
    groups = contract.get("semantic_groups", [])
    classes = contract.get("catalog_classes", [])
    mappings = contract.get("event_mappings", [])

    dimension_ids = [item.get("id") for item in dimensions]
    group_ids = [item.get("id") for item in groups]
    class_ids = [item.get("id") for item in classes]
    codes = [str(item.get("code")) for item in mappings]

    if len(dimensions) != 5 or len(set(dimension_ids)) != 5:
        errors.append("public_dimensions must contain five unique entries")
    if len(groups) != 6 or len(set(group_ids)) != 6:
        errors.append("semantic_groups must contain six unique entries")
    if len(classes) != 32 or len(set(class_ids)) != 32:
        errors.append("catalog_classes must contain 32 unique entries")
    if len(mappings) != 24:
        errors.append("event_mappings must contain exactly 24 entries")
    if set(codes) != EXPECTED_CODES:
        missing = sorted(EXPECTED_CODES - set(codes))
        extra = sorted(set(codes) - EXPECTED_CODES)
        errors.append(f"event code set mismatch; missing={missing}, extra={extra}")
    duplicate_codes = sorted(_duplicates(codes))
    if duplicate_codes:
        errors.append(f"event codes must be unique; duplicates={duplicate_codes}")
    mapped_class_ids = [item.get("catalog_class") for item in mappings]
    if set(mapped_class_ids) != {f"C{i:02d}" for i in range(1, 25)}:
        errors.append("the 24 event mappings must cover catalog classes C01-C24 exactly once")

    for item in groups:
        if item.get("public_dimension") not in dimension_ids:
            errors.append(f"semantic group {item.get('id')} has an unknown public dimension")

    for item in classes:
        ident = item.get("id", "<missing>")
        if item.get("public_dimension") not in dimension_ids:
            errors.append(f"catalog class {ident} has an unknown public dimension")
        if item.get("semantic_group") not in group_ids:
            errors.append(f"catalog class {ident} has an unknown semantic group")
        if item.get("evidence_status") not in EVIDENCE_STATUSES:
            errors.append(f"catalog class {ident} has an invalid evidence status")
        if not str(item.get("evidence_source", "")).strip():
            errors.append(f"catalog class {ident} lacks evidence_source")
        if not str(item.get("boundary", "")).strip():
            errors.append(f"catalog class {ident} lacks a one-sentence boundary")
        if item.get("evidence_status") == "reserved" and item.get("enabled") is not False:
            errors.append(f"reserved catalog class {ident} must remain disabled")

    class_by_id = {item.get("id"): item for item in classes}
    for item in mappings:
        code = str(item.get("code"))
        class_item = class_by_id.get(item.get("catalog_class"))
        if class_item is None:
            errors.append(f"event {code} maps to an unknown catalog class")
            continue
        if item.get("semantic_group") != class_item.get("semantic_group"):
            errors.append(f"event {code} and its catalog class disagree on semantic_group")
        if item.get("public_dimension") != class_item.get("public_dimension"):
            errors.append(f"event {code} and its catalog class disagree on public_dimension")
        if item.get("evidence_status") != class_item.get("evidence_status"):
            errors.append(f"event {code} and its catalog class disagree on evidence_status")
        if item.get("scoring_role") not in SCORING_ROLES:
            errors.append(f"event {code} has an invalid scoring role")
        if not str(item.get("exposure_denominator", "")).strip():
            errors.append(f"event {code} lacks an exposure denominator")
        if not str(item.get("deduplication", "")).strip():
            errors.append(f"event {code} lacks a deduplication rule")

    for code in ("11803", "11804"):
        item = next((row for row in mappings if str(row.get("code")) == code), None)
        if item and item.get("scoring_role") != "outcome_gate":
            errors.append(f"event {code} must be an outcome_gate")

    label_contract = contract.get("label_contract", {})
    if (
        label_contract.get("accident_code") != "11803"
        or label_contract.get("accident_min_count") != 1
        or label_contract.get("near_miss_code") != "11804"
        or label_contract.get("near_miss_min_count") != 2
        or label_contract.get("near_miss_count_unit") != "deduplicated_valid_record"
        or label_contract.get("future_label_window_is_feature") is not False
    ):
        errors.append("label_contract must reproduce ADR-0006/ADR-0007 and isolate the future window")

    for code in ("60292", "60294"):
        item = next((row for row in mappings if str(row.get("code")) == code), None)
        if not item:
            continue
        conditions = " ".join(item.get("applicability_conditions", []))
        denominator = str(item.get("exposure_denominator", ""))
        if "equipped" not in conditions or "lane_change" not in denominator:
            errors.append(
                f"blind-spot event {code} requires an equipped cohort and lane_change denominator"
            )

    decisions = contract.get("decisions", [])
    if {item.get("id") for item in decisions} != {f"D{i}" for i in range(1, 7)}:
        errors.append("decisions must cover D1-D6 exactly once")
    ambiguities = contract.get("ambiguity_decisions", [])
    if {item.get("id") for item in ambiguities} != {f"M{i}" for i in range(1, 9)}:
        errors.append("ambiguity_decisions must cover M1-M8 exactly once")
    cross_rules = contract.get("cross_event_rules", [])
    if len(cross_rules) != 10:
        errors.append("cross_event_rules must contain ten entries")
    if {item.get("id") for item in cross_rules} != {f"CR{i:02d}" for i in range(1, 11)}:
        errors.append("cross_event_rules must cover CR01-CR10 exactly once")
    for rule in cross_rules:
        if rule.get("window_seconds") not in ALLOWED_WINDOWS:
            errors.append(
                f"cross-event rule {rule.get('id')} must use a 60/120/600 second window"
            )

    chains = contract.get("risk_chains", [])
    if len(chains) != 5:
        errors.append("risk_chains must contain five entries")
    for chain in chains:
        if chain.get("merge_window_seconds") not in ALLOWED_WINDOWS:
            errors.append(
                f"risk chain {chain.get('id')} must use a 60/120/600 second merge window"
            )

    required_reserved = {"C25", "C26", "C27", "C28", "C29", "C30", "C31", "C32"}
    for ident in sorted(required_reserved):
        item = class_by_id.get(ident)
        if not item or item.get("enabled") is not False:
            errors.append(f"conditional/reserved class {ident} must exist and be disabled")

    return errors


def validate_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return validate_contract(json.load(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "contract",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("classification_v1.json"),
    )
    args = parser.parse_args()
    errors = validate_file(args.contract)
    payload = {
        "contract": str(args.contract),
        "status": "pass" if not errors else "fail",
        "error_count": len(errors),
        "errors": errors,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
