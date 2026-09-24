"""R1 output writers: scores rows for T2-R4, adapter payload for T2-R2 and
the recomputation path used by the traceability acceptance.

Constraints implemented here are the published contracts of the other two
lines (``evaluation_contract_v1.json`` columns and
``task2/operations/r1-adapter-contract.md``); violations raise before any
file leaves this module.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from .engine import DeductionRuleset, VehicleScoring, aggregate_group_deduction

#: rows consumed by task2/evaluation/evaluate_scores.py
SCORES_COLUMNS = (
    "gpsno",
    "y",
    "safety_score",
    "fold",
    "rule_recomputed_score",
)

ADAPTER_REVIEW_STATUSES = ("valid", "pending", "exempt")
MIN_EVALUABLE_CLASSES = 3  # publishing-layer convention from T2-R4 hard checks


# --------------------------------------------------------------------------- #
# T2-R4 interface: score rows + independent recomputation
# --------------------------------------------------------------------------- #
def recompute_score_from_ledger(
    scoring: VehicleScoring,
    ruleset: DeductionRuleset,
) -> float:
    """Recompute the score from class-level ledger entries only.

    This deliberately re-runs group aggregation instead of reading the stored
    group totals, so ``published == recomputed`` is a real check that the
    ledger fully explains the score.
    """

    by_group: dict[str, dict[str, float]] = {}
    for cls in sorted(scoring.class_points):
        by_group.setdefault(ruleset.class_group[cls], {})[cls] = scoring.class_points[cls]
    total = sum(
        aggregate_group_deduction(members, fraction=ruleset.group_agg_fraction)
        for _, members in sorted(by_group.items())
    )
    return max(0.0, min(100.0, 100.0 - total))


def build_scores_rows(
    scorings: Sequence[VehicleScoring],
    ruleset: DeductionRuleset,
    labels: Mapping[str, int],
    folds: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Build evaluate_scores-compatible rows with recomputation attached."""

    rows: list[dict[str, Any]] = []
    for scoring in scorings:
        gpsno = scoring.gpsno
        if gpsno not in labels:
            raise ValueError(f"missing label for {gpsno}")
        if gpsno not in folds:
            raise ValueError(f"missing fold for {gpsno}")
        rows.append(
            {
                "gpsno": gpsno,
                "y": int(labels[gpsno]),
                "safety_score": scoring.safety_score,
                "fold": str(folds[gpsno]),
                "rule_recomputed_score": recompute_score_from_ledger(scoring, ruleset),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# T2-R2 interface: adapter payload per r1-adapter-contract.md
# --------------------------------------------------------------------------- #
def build_adapter_payload(
    scorings: Sequence[VehicleScoring],
    ruleset: DeductionRuleset,
    segments_by_gpsno: Mapping[str, Sequence[Mapping[str, Any]]],
    exposure_days: Mapping[str, Sequence[Mapping[str, Any]]],
    as_of_date: str,
    *,
    upstream_score_version: str,
    review_status_by_segment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the R1 -> R2 adapter payload.

    ``exposure_days`` maps gpsno to ``[{date, exposure}, ...]`` in the unified
    exposure unit of ``ruleset.denominator_kind``.  Event-level deductions are
    allocated as equal shares of their class's points (recorded in the R1
    charter as the allocation convention); statuses default to ``valid``.
    """

    if not upstream_score_version:
        raise ValueError("upstream_score_version must be non-empty")
    as_of = date.fromisoformat(str(as_of_date))
    review_status_by_segment = review_status_by_segment or {}

    vehicles: list[dict[str, Any]] = []
    for scoring in scorings:
        gpsno = scoring.gpsno
        days = _normalise_days(exposure_days[gpsno])
        exposure_by_date = {day["date"]: float(day["exposure"]) for day in days}

        events: list[dict[str, Any]] = []
        for ledger_entry in scoring.ledger:
            cls = ledger_entry["catalog_class"]
            points = scoring.class_points[cls]
            segment_ids = ledger_entry["segment_ids"]
            if not segment_ids:
                continue
            share = points / len(segment_ids)
            for segment_id in segment_ids:
                segment = _find_segment(segments_by_gpsno[gpsno], segment_id)
                moment = _as_datetime(segment["start_time"])
                status = review_status_by_segment.get(segment_id, "valid")
                if status not in ADAPTER_REVIEW_STATUSES:
                    raise ValueError(f"invalid review status for {segment_id}: {status}")
                if moment.date() > as_of:
                    raise ValueError(
                        f"event {segment_id} is later than as_of_date {as_of_date}"
                    )
                event_date = moment.date()
                if event_date not in exposure_by_date:
                    raise ValueError(
                        f"event {segment_id} falls outside the day records of {gpsno}"
                    )
                if status == "valid" and exposure_by_date[event_date] <= 0:
                    raise ValueError(
                        f"valid deduction {segment_id} lands on a zero-exposure day"
                    )
                events.append(
                    {
                        "event_id": segment_id,
                        "event_time": moment.isoformat(),
                        "base_deduction": share,
                        "review_status": status,
                    }
                )
        events.sort(key=lambda item: (item["event_time"], item["event_id"]))
        vehicles.append(
            {
                "gpsno": gpsno,
                "initial_score": scoring.safety_score,
                "initial_score_is_previous_valid": False,
                "upstream_score_version": upstream_score_version,
                "days": [{"date": day["date"].isoformat(), "exposure": day["exposure"]} for day in days],
                "events": events,
            }
        )
    return {"as_of_date": as_of.isoformat(), "vehicles": vehicles}


def validate_adapter_payload(payload: Mapping[str, Any]) -> list[str]:
    """Re-check the adapter contract; returns human-readable violations."""

    errors: list[str] = []
    seen_vehicles: set[str] = set()
    for vehicle in payload.get("vehicles", []):
        gpsno = str(vehicle.get("gpsno", ""))
        if not gpsno:
            errors.append("vehicle without gpsno")
            continue
        if gpsno in seen_vehicles:
            errors.append(f"duplicate gpsno: {gpsno}")
        seen_vehicles.add(gpsno)
        if not str(vehicle.get("upstream_score_version", "")):
            errors.append(f"{gpsno}: upstream_score_version must be non-empty")

        days = vehicle.get("days", [])
        dates = [str(day.get("date", "")) for day in days]
        if len(set(dates)) != len(dates):
            errors.append(f"{gpsno}: day records must be unique")
        if dates != sorted(dates):
            errors.append(f"{gpsno}: day records must be increasing")

        exposure_by_date = {str(day.get("date")): float(day.get("exposure", 0.0)) for day in days}
        seen_events: set[str] = set()
        for event in vehicle.get("events", []):
            event_id = str(event.get("event_id", ""))
            if event_id in seen_events:
                errors.append(f"{gpsno}: duplicate event_id {event_id}")
            seen_events.add(event_id)
            if event.get("review_status") not in ADAPTER_REVIEW_STATUSES:
                errors.append(f"{gpsno}: bad review status for {event_id}")
            event_date = _as_datetime(event["event_time"]).date().isoformat()
            if event_date not in exposure_by_date:
                errors.append(f"{gpsno}: event {event_id} outside day records")
            elif event.get("review_status") == "valid" and exposure_by_date[event_date] <= 0:
                errors.append(f"{gpsno}: valid deduction {event_id} on zero-exposure day")
            if _as_datetime(event["event_time"]).date().isoformat() > str(payload.get("as_of_date")):
                errors.append(f"{gpsno}: event {event_id} later than as_of_date")
    return errors


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #
def _normalise_days(days: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalised = [
        {"date": date.fromisoformat(str(day["date"])), "exposure": float(day["exposure"])}
        for day in days
    ]
    normalised.sort(key=lambda day: day["date"])
    if len({day["date"] for day in normalised}) != len(normalised):
        raise ValueError("day records must be unique per vehicle")
    return normalised


def _find_segment(segments: Iterable[Mapping[str, Any]], segment_id: str) -> Mapping[str, Any]:
    for segment in segments:
        if segment["segment_id"] == segment_id:
            return segment
    raise ValueError(f"segment not found for vehicle: {segment_id}")


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
