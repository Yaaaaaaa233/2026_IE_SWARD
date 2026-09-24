"""Apply deterministic operational adjustments to R1 base deductions.

This module deliberately treats R1 as an upstream black box.  It consumes a
base deduction for each event and never recreates R1 bins, weights, smoothing,
or group aggregation.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping


REVIEW_STATUSES = {"valid", "pending", "exempt"}


def _number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be finite and at least {minimum}")
    return result


def _date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD") from exc


def _datetime(value: Any, field: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field} must use ISO-8601 format") from exc


def classify_time_band(value: str | datetime) -> str:
    """Use the official task-2 time bands from the 0923 clarification."""

    timestamp = _datetime(value, "event_time") if isinstance(value, str) else value
    hour = timestamp.hour
    if hour >= 21 or hour < 6:
        return "night"
    if hour < 9:
        return "morning"
    if hour < 18:
        return "day"
    return "dusk"


def _resolve_event(
    event: Mapping[str, Any],
    as_of: date,
    pending_days: int,
    time_coefficients: Mapping[str, float],
) -> dict[str, Any]:
    event_id = str(event.get("event_id", "")).strip()
    if not event_id:
        raise ValueError("event_id must be nonempty")
    timestamp = _datetime(event.get("event_time"), "event_time")
    if timestamp.date() > as_of:
        raise ValueError(f"event {event_id} occurs after as_of_date")
    deduction = _number(event.get("base_deduction"), "base_deduction")
    status = str(event.get("review_status", "")).strip()
    if status not in REVIEW_STATUSES:
        raise ValueError(f"event {event_id} has an invalid review_status")
    age_days = (as_of - timestamp.date()).days
    if status == "exempt":
        resolved_status = "exempt"
        included = False
    elif status == "pending" and age_days < pending_days:
        resolved_status = "pending"
        included = False
    elif status == "pending":
        resolved_status = "auto_valid_after_deadline"
        included = True
    else:
        resolved_status = "valid"
        included = True
    band = classify_time_band(timestamp)
    coefficient = _number(time_coefficients[band], f"time coefficient {band}")
    return {
        "event_id": event_id,
        "event_date": timestamp.date(),
        "time_band": band,
        "time_coefficient": coefficient,
        "base_deduction": deduction,
        "adjusted_deduction": deduction * coefficient if included else 0.0,
        "resolved_status": resolved_status,
        "included": included,
    }


def run_vehicle(
    vehicle: Mapping[str, Any],
    as_of_date: str | date,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce a deterministic score history for one vehicle."""

    gpsno = str(vehicle.get("gpsno", "")).strip()
    if not gpsno:
        raise ValueError("gpsno must be nonempty")
    as_of = _date(as_of_date, "as_of_date") if isinstance(as_of_date, str) else as_of_date
    initial_score = _number(vehicle.get("initial_score"), "initial_score")
    if initial_score > 100.0:
        raise ValueError("initial_score must be within [0, 100]")
    initial_score_is_previous_valid = vehicle.get("initial_score_is_previous_valid") is True

    dynamics = policy["dynamics"]
    half_life = _number(dynamics["half_life_exposure_days"], "half_life_exposure_days", minimum=1e-12)
    decay = 0.5 ** (1.0 / half_life)
    pending_days = int(dynamics["pending_review_days"])
    if pending_days < 0:
        raise ValueError("pending_review_days must be nonnegative")
    persistence_window = int(dynamics["persistence_window_days"])
    if persistence_window < 1:
        raise ValueError("persistence_window_days must be positive")
    persistence_step = _number(dynamics["persistence_step"], "persistence_step")
    persistence_cap = _number(dynamics["persistence_multiplier_cap"], "persistence_multiplier_cap", minimum=1.0)
    prior_score = _number(dynamics["low_exposure_prior_score"], "low_exposure_prior_score")
    if prior_score > 100.0:
        raise ValueError("low_exposure_prior_score must be within [0, 100]")
    prior_strength = _number(dynamics["prior_strength_exposure"], "prior_strength_exposure", minimum=1e-12)
    minimum_exposure = _number(dynamics["minimum_observed_exposure"], "minimum_observed_exposure")
    recovery_cap = _number(dynamics["recovery_cap_per_exposure_day"], "recovery_cap_per_exposure_day")
    coefficients = dynamics["time_coefficients"]
    if set(coefficients) != {"night", "morning", "day", "dusk"}:
        raise ValueError("time_coefficients must define night, morning, day and dusk")

    days = list(vehicle.get("days", []))
    day_dates = [_date(item.get("date"), "day.date") for item in days]
    if len(set(day_dates)) != len(day_dates):
        raise ValueError(f"vehicle {gpsno} has duplicate day rows")
    if day_dates != sorted(day_dates):
        raise ValueError(f"vehicle {gpsno} day rows must be sorted")
    if day_dates and day_dates[-1] > as_of:
        raise ValueError(f"vehicle {gpsno} has a day after as_of_date")

    resolved_events = [
        _resolve_event(event, as_of, pending_days, coefficients)
        for event in vehicle.get("events", [])
    ]
    event_ids = [event["event_id"] for event in resolved_events]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError(f"vehicle {gpsno} has duplicate event_id values")
    events_by_date: dict[date, list[dict[str, Any]]] = {}
    for event in resolved_events:
        events_by_date.setdefault(event["event_date"], []).append(event)
    known_days = set(day_dates)
    orphan_dates = sorted(set(events_by_date) - known_days)
    if orphan_dates:
        raise ValueError(f"vehicle {gpsno} has events without a matching day row")

    cumulative_exposure = 0.0
    risk_state = 0.0
    published_score = initial_score
    active_risk_days: list[date] = []
    history: list[dict[str, Any]] = []
    for item, current_date in zip(days, day_dates):
        exposure = _number(item.get("exposure"), "day.exposure")
        daily_events = events_by_date.get(current_date, [])
        included_events = [event for event in daily_events if event["included"]]
        daily_base_risk = sum(event["adjusted_deduction"] for event in included_events)
        if exposure == 0.0 and daily_base_risk > 0.0:
            raise ValueError(
                f"vehicle {gpsno} has included risk deductions on a zero-exposure day"
            )

        previous_score = published_score
        if exposure == 0.0:
            if not history and not initial_score_is_previous_valid:
                published_score = prior_score
                observation_status = "no_exposure_conservative"
            else:
                observation_status = "stopped_frozen"
            multiplier = 1.0
            adjusted_daily_risk = 0.0
            raw_safety_score = published_score
        else:
            cumulative_exposure += exposure
            window_start = current_date - timedelta(days=persistence_window - 1)
            active_risk_days = [
                value for value in active_risk_days if value >= window_start
            ]
            if daily_base_risk > 0.0:
                active_risk_days.append(current_date)
            multiplier = min(
                persistence_cap,
                1.0 + persistence_step * max(0, len(active_risk_days) - 1),
            )
            adjusted_daily_risk = min(100.0, daily_base_risk * multiplier)
            risk_state = decay * risk_state + adjusted_daily_risk
            unshrunk_score = max(0.0, initial_score - risk_state)
            reliability = cumulative_exposure / (cumulative_exposure + prior_strength)
            raw_safety_score = reliability * unshrunk_score + (1.0 - reliability) * prior_score
            candidate_score = min(100.0, max(0.0, raw_safety_score))
            if adjusted_daily_risk > 0.0:
                candidate_score = min(candidate_score, previous_score)
            elif candidate_score > previous_score:
                candidate_score = min(candidate_score, previous_score + recovery_cap)
            published_score = candidate_score
            observation_status = (
                "observed" if cumulative_exposure >= minimum_exposure else "low_exposure"
            )

        history.append(
            {
                "date": current_date.isoformat(),
                "exposure": exposure,
                "cumulative_exposure": cumulative_exposure,
                "base_risk_points": daily_base_risk,
                "persistence_multiplier": multiplier,
                "adjusted_daily_risk": adjusted_daily_risk,
                "ewma_risk": risk_state,
                "raw_safety_score": raw_safety_score,
                "safety_score": published_score,
                "score_change": published_score - previous_score,
                "observation_status": observation_status,
                "included_event_ids": [event["event_id"] for event in included_events],
                "pending_event_ids": [
                    event["event_id"]
                    for event in daily_events
                    if event["resolved_status"] == "pending"
                ],
                "exempt_event_ids": [
                    event["event_id"]
                    for event in daily_events
                    if event["resolved_status"] == "exempt"
                ],
            }
        )

    if not history:
        if not initial_score_is_previous_valid:
            published_score = prior_score
        observation_status = "no_observation"
    else:
        observation_status = history[-1]["observation_status"]
    return {
        "gpsno": gpsno,
        "as_of_date": as_of.isoformat(),
        "safety_score": published_score,
        "score_change": published_score - initial_score,
        "observation_status": observation_status,
        "history": history,
        "event_resolution": [
            {
                key: value.isoformat() if isinstance(value, date) else value
                for key, value in event.items()
            }
            for event in resolved_events
        ],
        "policy_version": policy["policy_version"],
        "upstream_score_version": str(vehicle.get("upstream_score_version", "")),
    }


def run_payload(payload: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    as_of = payload.get("as_of_date")
    vehicles = list(payload.get("vehicles", []))
    results = [run_vehicle(vehicle, as_of, policy) for vehicle in vehicles]
    ids = [item["gpsno"] for item in results]
    if len(set(ids)) != len(ids):
        raise ValueError("payload contains duplicate gpsno values")
    return {
        "policy_version": policy["policy_version"],
        "as_of_date": str(as_of),
        "vehicle_count": len(results),
        "vehicles": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).with_name("operational_policy_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    result = run_payload(payload, policy)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
