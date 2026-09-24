"""R1 deduction pipeline: segmentation -> exposure normalisation -> fold-internal
binning -> EB smoothing -> PAVA monotonicity -> rule compression -> diminishing
group aggregation (roadmap D7 stage list, D8-D11 parameters).

Every function here is deterministic and data-free: numbers only appear once
``fit_ruleset`` is fed real training rows in the data stage.  Invariants kept
by construction and asserted in tests:

* monotonicity -- removing risky evidence never lowers the safety score;
* recomputability -- the published score is reproducible from the ledger;
* value domain -- scores stay within [0, 100].
"""

from __future__ import annotations

import bisect
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from .contract_loader import DEDUCTIBLE_ROLES, OUTCOME_CODES

_SPEED_NEIGHBOR_CHAIN = "SPEED_NEIGHBOR"


# --------------------------------------------------------------------------- #
# Stage 1: risk-chain segmentation (T2-R0 charter section 8, CR01-CR10)
# --------------------------------------------------------------------------- #
def segment_events(
    events: Iterable[Mapping[str, Any]],
    lookups: Mapping[str, Any],
    *,
    speed_neighbor_window_seconds: float = 60.0,
) -> list[dict[str, Any]]:
    """Merge raw events into risk-chain segments.

    ``events`` rows carry ``gpsno``, ``code``, ``time`` (datetime or ISO
    string) and optional ``severity`` (default 1.0).  Chained codes merge when
    consecutive events of the same vehicle and chain are within the contract's
    merge window; codes outside every chain (the scene-speed family) use the
    charter's same-road-process neighbour window (CR06).  Outcome-gate codes
    (11803/11804) are never segmented here.  The representative event of a
    segment is the strongest evidence (max severity, earliest on ties).
    """

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw in events:
        code = str(raw["code"])
        if code in OUTCOME_CODES:
            continue
        chain = lookups["chain_by_code"].get(code)
        if chain is not None:
            chain_id, window = chain["chain_id"], float(chain["window_seconds"])
        else:
            chain_id, window = _SPEED_NEIGHBOR_CHAIN, float(speed_neighbor_window_seconds)
        moment = _as_datetime(raw["time"])
        buckets.setdefault((str(raw["gpsno"]), chain_id), []).append(
            {
                "code": code,
                "time": moment,
                "severity": float(raw.get("severity", 1.0)),
                "window": window,
            }
        )

    segments: list[dict[str, Any]] = []
    for (gpsno, chain_id), members in buckets.items():
        members.sort(key=lambda item: (item["time"], item["code"]))
        clusters: list[list[dict[str, Any]]] = []
        for member in members:
            if clusters and (
                member["time"] - clusters[-1][-1]["time"]
            ).total_seconds() <= member["window"]:
                clusters[-1].append(member)
            else:
                clusters.append([member])
        for index, cluster in enumerate(clusters):
            representative = max(cluster, key=lambda item: (item["severity"], -item["time"].timestamp()))
            segments.append(
                {
                    "segment_id": f"{gpsno}-{chain_id}-{index:06d}",
                    "gpsno": gpsno,
                    "chain_id": chain_id,
                    "representative_code": representative["code"],
                    "catalog_class": lookups["by_code"][representative["code"]]["catalog_class"],
                    "merged_codes": sorted({item["code"] for item in cluster}),
                    "start_time": cluster[0]["time"],
                    "end_time": cluster[-1]["time"],
                    "duration_seconds": (cluster[-1]["time"] - cluster[0]["time"]).total_seconds(),
                    "max_severity": max(item["severity"] for item in cluster),
                    "member_count": len(cluster),
                }
            )
    segments.sort(key=lambda seg: (seg["gpsno"], seg["start_time"], seg["segment_id"]))
    return segments


# --------------------------------------------------------------------------- #
# Stage 3 helper: empirical-Bayes smoothed rate (in-fold prior only)
# --------------------------------------------------------------------------- #
def eb_smooth_rate(
    count: float,
    exposure: float,
    prior_rate: float,
    prior_strength: float,
) -> float:
    """Shrink a raw rate toward the fold-internal prior rate."""

    if exposure < 0 or count < 0:
        raise ValueError("count and exposure must be nonnegative")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    return (count + prior_strength * prior_rate) / (exposure + prior_strength)


def fold_prior_rate(pairs: Sequence[tuple[float, float]]) -> float:
    """Aggregate in-fold prior rate from (count, exposure) pairs."""

    total_count = sum(count for count, _ in pairs)
    total_exposure = sum(exposure for _, exposure in pairs)
    if total_exposure <= 0:
        return 0.0
    return total_count / total_exposure


# --------------------------------------------------------------------------- #
# Stage 4-6: quantile bins -> PAVA monotonic risks -> compression
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BinSpec:
    """Ordered rate bins with PAVA-monotone, compressed risks."""

    boundaries: tuple[float, ...]  # upper edge of each bin except the last
    risks: tuple[float, ...]       # non-decreasing by construction

    def bin_index(self, rate: float) -> int:
        return bisect.bisect_right(self.boundaries, rate)


def _pava_monotone(values: Sequence[float], weights: Sequence[int]) -> list[float]:
    """Weighted pool-adjacent-violators; returns one pooled value per bin."""

    blocks: list[list[float]] = []  # [weighted_sum, weight]
    for value, weight in zip(values, weights):
        blocks.append([value * weight, float(weight)])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            top = blocks.pop()
            prev = blocks.pop()
            blocks.append([prev[0] + top[0], prev[1] + top[1]])
    return [weighted_sum / weight for weighted_sum, weight in blocks]


def fit_bins(samples: Sequence[tuple[float, float]], n_bins: int = 4) -> BinSpec:
    """Fit ordered bins on (rate, future_label) training samples.

    Quantile edges are learned on rates; each bin's risk is its mean label;
    PAVA enforces non-decreasing risk with the rate; adjacent equal-risk bins
    are compressed.  Fewer samples or ties naturally produce fewer bins.
    """

    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if not samples:
        return BinSpec(boundaries=(), risks=(0.0,))
    rates = sorted(rate for rate, _ in samples)
    if len(samples) < 2 or rates[0] == rates[-1]:
        return BinSpec(boundaries=(), risks=(statistics.fmean(label for _, label in samples),))

    candidate_edges = sorted(
        {rates[min(len(rates) - 1, int(round(k * len(rates) / n_bins)))] for k in range(1, n_bins)}
    )
    bin_labels: list[list[float]] = [[] for _ in range(len(candidate_edges) + 1)]
    for rate, label in samples:
        bin_labels[bisect.bisect_right(candidate_edges, rate)].append(label)

    occupied = [index for index, labels in enumerate(bin_labels) if labels]
    raw_risks = [statistics.fmean(bin_labels[index]) for index in occupied]
    counts = [len(bin_labels[index]) for index in occupied]
    monotone = _pava_monotone(raw_risks, counts)
    compressed: list[tuple[float, float]] = []  # (boundary-so-far, risk)
    for slot, risk in zip(occupied, monotone):
        boundary = candidate_edges[slot] if slot < len(candidate_edges) else float("inf")
        if compressed and compressed[-1][1] == risk:
            compressed[-1] = (boundary, risk)
        else:
            compressed.append((boundary, risk))
    boundaries = tuple(item[0] for item in compressed[:-1])
    return BinSpec(boundaries=boundaries, risks=tuple(item[1] for item in compressed))


# --------------------------------------------------------------------------- #
# Stage 7: diminishing group aggregation (D10)
# --------------------------------------------------------------------------- #
def aggregate_group_deduction(
    class_points: Mapping[str, float],
    *,
    fraction: float = 0.25,
) -> float:
    """Max single class plus ``fraction`` x mean of the remaining classes."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    if not class_points:
        return 0.0
    values = sorted(class_points.values(), reverse=True)
    if len(values) == 1:
        return float(values[0])
    return float(values[0] + fraction * statistics.fmean(values[1:]))


# --------------------------------------------------------------------------- #
# Ruleset: everything learned in the data stage, frozen into one object
# --------------------------------------------------------------------------- #
@dataclass
class DeductionRuleset:
    rule_version: str
    denominator_kind: str                     # D9; "proxy_total_distance" first
    bins_by_class: dict[str, BinSpec]
    prior_rate_by_class: dict[str, float]
    class_group: dict[str, str]               # catalog class -> semantic group
    class_budget: dict[str, float]            # points ceiling per class
    group_agg_fraction: float = 0.25
    min_exposure: float = 0.0
    eb_prior_strength: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_version": self.rule_version,
            "denominator_kind": self.denominator_kind,
            "bins_by_class": {
                cls: {"boundaries": list(spec.boundaries), "risks": list(spec.risks)}
                for cls, spec in sorted(self.bins_by_class.items())
            },
            "prior_rate_by_class": dict(sorted(self.prior_rate_by_class.items())),
            "class_group": dict(sorted(self.class_group.items())),
            "class_budget": dict(sorted(self.class_budget.items())),
            "group_agg_fraction": self.group_agg_fraction,
            "min_exposure": self.min_exposure,
            "eb_prior_strength": self.eb_prior_strength,
        }


@dataclass
class VehicleScoring:
    gpsno: str
    safety_score: float
    evaluable_class_count: int
    insufficient_evidence: bool
    class_points: dict[str, float] = field(default_factory=dict)
    group_deductions: dict[str, float] = field(default_factory=dict)
    ledger: list[dict[str, Any]] = field(default_factory=list)


def fit_ruleset(
    train_rows: Sequence[Mapping[str, Any]],
    lookups: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> DeductionRuleset:
    """Learn one fold's rules from training vehicle rows.

    ``train_rows`` carry ``gpsno``, ``exposure`` (D9 proxy units),
    ``segments`` (from :func:`segment_events`) and ``label`` (future outcome,
    D8 future-label learning).  Bins, priors and budgets are learned on these
    rows only -- never on validation vehicles.
    """

    n_bins = int(policy["n_bins"])
    fraction = float(policy["group_agg_fraction"])
    min_exposure = float(policy["min_exposure"])
    prior_strength = float(policy["eb_prior_strength"])
    budgets: Mapping[str, float] = policy["group_budgets"]

    counts: dict[str, list[tuple[float, float]]] = {}
    rate_samples: dict[str, list[tuple[float, float]]] = {}
    class_group: dict[str, str] = {}
    for row in train_rows:
        exposure = float(row["exposure"])
        if exposure <= min_exposure:
            continue
        per_class: dict[str, int] = {}
        for segment in row["segments"]:
            code = str(segment["representative_code"])
            if lookups["role_by_code"][code] not in DEDUCTIBLE_ROLES:
                continue
            cls = str(segment["catalog_class"])
            class_group[cls] = lookups["group_by_code"][code]
            per_class[cls] = per_class.get(cls, 0) + 1
        for cls, count in per_class.items():
            counts.setdefault(cls, []).append((float(count), exposure))
            rate_samples.setdefault(cls, []).append((float(count) / exposure, float(row["label"])))

    bins_by_class: dict[str, BinSpec] = {}
    prior_by_class: dict[str, float] = {}
    for cls in sorted(counts):
        prior_by_class[cls] = fold_prior_rate(counts[cls])
        bins_by_class[cls] = fit_bins(rate_samples[cls], n_bins=n_bins)

    group_classes: dict[str, list[str]] = {}
    for cls in bins_by_class:
        group_classes.setdefault(class_group[cls], []).append(cls)
    class_budget: dict[str, float] = {}
    for group, members in group_classes.items():
        share = float(budgets.get(group, 0.0)) / len(members)
        for cls in members:
            class_budget[cls] = share

    return DeductionRuleset(
        rule_version=str(policy["rule_version"]),
        denominator_kind=str(policy["denominator_kind"]),
        bins_by_class=bins_by_class,
        prior_rate_by_class=prior_by_class,
        class_group=class_group,
        class_budget=class_budget,
        group_agg_fraction=fraction,
        min_exposure=min_exposure,
        eb_prior_strength=prior_strength,
    )


def score_vehicle(
    vehicle: Mapping[str, Any],
    ruleset: DeductionRuleset,
    lookups: Mapping[str, Any],
) -> VehicleScoring:
    """Score one vehicle under a frozen ruleset; emits a full deduction ledger."""

    gpsno = str(vehicle["gpsno"])
    exposure = float(vehicle["exposure"])
    per_class: dict[str, tuple[int, list[Mapping[str, Any]]]] = {}
    for segment in vehicle["segments"]:
        code = str(segment["representative_code"])
        if lookups["role_by_code"][code] not in DEDUCTIBLE_ROLES:
            continue
        if str(segment["catalog_class"]) not in ruleset.bins_by_class:
            continue
        cls = str(segment["catalog_class"])
        count, members = per_class.get(cls, (0, []))
        per_class[cls] = (count + 1, members + [segment])

    class_points: dict[str, float] = {}
    ledger: list[dict[str, Any]] = []
    if exposure > ruleset.min_exposure:
        for cls in sorted(per_class):
            count, members = per_class[cls]
            rate = eb_smooth_rate(
                float(count),
                exposure,
                ruleset.prior_rate_by_class.get(cls, 0.0),
                ruleset.eb_prior_strength,
            )
            spec = ruleset.bins_by_class[cls]
            bin_index = spec.bin_index(rate)
            risk = spec.risks[bin_index]
            max_risk = max(spec.risks)
            points = ruleset.class_budget[cls] * (risk / max_risk if max_risk > 0 else 0.0)
            class_points[cls] = points
            ledger.append(
                {
                    "gpsno": gpsno,
                    "catalog_class": cls,
                    "evidence_count": count,
                    "segment_ids": [item["segment_id"] for item in members],
                    "rate": rate,
                    "bin_index": bin_index,
                    "bin_risk": risk,
                    "points": points,
                    "rule_version": ruleset.rule_version,
                }
            )

    by_group: dict[str, dict[str, float]] = {}
    for cls, points in class_points.items():
        by_group.setdefault(ruleset.class_group[cls], {})[cls] = points
    group_deductions = {
        group: aggregate_group_deduction(members, fraction=ruleset.group_agg_fraction)
        for group, members in sorted(by_group.items())
    }
    total = sum(group_deductions.values())
    safety_score = max(0.0, min(100.0, 100.0 - total))
    return VehicleScoring(
        gpsno=gpsno,
        safety_score=safety_score,
        evaluable_class_count=len(class_points),
        insufficient_evidence=len(class_points) < 3,
        class_points=class_points,
        group_deductions=group_deductions,
        ledger=ledger,
    )


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))
