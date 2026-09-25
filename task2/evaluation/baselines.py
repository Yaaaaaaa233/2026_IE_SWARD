"""Data-free baseline implementations for task 2.

B0 is intentionally a weak, fully transparent floor.  It records whether an
eligible risk code appeared and does not learn weights, bins, or thresholds.
Q1 converts an already out-of-fold task-1 probability into a safety score.
Neither function reads competition data.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


SCORING_ROLES = {"behavior", "context_warning"}
DEVICE_QUALITY_CODES = {"41006", "41021"}


def probability_to_safety_score(risk_probability: float) -> float:
    """Return Q1 = 100 * (1 - p), rejecting invalid probabilities."""

    value = float(risk_probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("risk_probability must be finite and within [0, 1]")
    return 100.0 * (1.0 - value)


def transparent_presence_baseline(
    event_counts: Mapping[str, int | float],
    eligible_codes: Iterable[str],
    classification_contract: Mapping[str, Any],
    *,
    insufficient_evidence_prior: float = 50.0,
    min_observed_dimensions: int = 3,
    historical_outcome_cap: float | None = None,
) -> dict[str, Any]:
    """Compute the B0 equal-presence baseline.

    A code is eligible only after its exposure and equipment gate has passed.
    Within each observed public dimension, every eligible behavior/context code
    receives equal weight and only its presence is used.  This makes B0 a
    reproducible lower baseline rather than a candidate replacement for R1.
    """

    prior = float(insufficient_evidence_prior)
    if not 0.0 <= prior <= 100.0:
        raise ValueError("insufficient_evidence_prior must be within [0, 100]")
    if min_observed_dimensions < 1:
        raise ValueError("min_observed_dimensions must be positive")

    counts: dict[str, float] = {}
    for code, raw_value in event_counts.items():
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"event count for {code} must be finite and nonnegative")
        counts[str(code)] = value
    eligible = {str(code) for code in eligible_codes}

    dimensions = {
        item["id"]: item["name"] for item in classification_contract["public_dimensions"]
    }
    scorable_by_dimension: dict[str, set[str]] = {key: set() for key in dimensions}
    for item in classification_contract["event_mappings"]:
        code = str(item["code"])
        if item["scoring_role"] in SCORING_ROLES and code in eligible:
            scorable_by_dimension[item["public_dimension"]].add(code)

    dimension_scores: dict[str, float | None] = {}
    dimension_evidence: dict[str, dict[str, Any]] = {}
    for dimension_id, dimension_name in dimensions.items():
        codes = sorted(scorable_by_dimension[dimension_id])
        if not codes:
            dimension_scores[dimension_name] = None
            dimension_evidence[dimension_name] = {
                "eligible_code_count": 0,
                "active_code_count": 0,
                "active_codes": [],
            }
            continue
        active = [code for code in codes if counts.get(code, 0.0) > 0.0]
        score = 100.0 * (1.0 - len(active) / len(codes))
        dimension_scores[dimension_name] = score
        dimension_evidence[dimension_name] = {
            "eligible_code_count": len(codes),
            "active_code_count": len(active),
            "active_codes": active,
        }

    observed = [score for score in dimension_scores.values() if score is not None]
    if len(observed) < min_observed_dimensions:
        safety_score = prior
        observation_status = "insufficient_evidence"
    else:
        safety_score = sum(observed) / len(observed)
        observation_status = "observed"

    historical_outcome = counts.get("11803", 0.0) >= 1 or counts.get("11804", 0.0) >= 2
    cap_applied = False
    if historical_outcome and historical_outcome_cap is not None:
        cap = float(historical_outcome_cap)
        if not 0.0 <= cap <= 100.0:
            raise ValueError("historical_outcome_cap must be within [0, 100]")
        if safety_score > cap:
            safety_score = cap
            cap_applied = True

    device_anomalies = sorted(
        code for code in DEVICE_QUALITY_CODES if counts.get(code, 0.0) > 0.0
    )
    if observation_status != "observed" or device_anomalies:
        confidence = "low"
    else:
        confidence = "standard"

    return {
        "baseline": "B0_equal_presence_v1",
        "safety_score": safety_score,
        "dimension_scores": dimension_scores,
        "dimension_evidence": dimension_evidence,
        "observation_status": observation_status,
        "evidence_confidence": confidence,
        "device_quality_flags": device_anomalies,
        "historical_outcome_triggered": historical_outcome,
        "historical_outcome_cap_applied": cap_applied,
    }
