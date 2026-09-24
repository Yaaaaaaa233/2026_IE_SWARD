"""R1 deduction engine: contract-driven, data-free skeleton.

All numeric weights are learned in the data stage; this module only fixes
pipeline stages, interfaces and invariants (roadmap D7-D11).
"""

from .contract_loader import build_lookups, load_contract
from .engine import (
    DeductionRuleset,
    aggregate_group_deduction,
    eb_smooth_rate,
    fit_bins,
    fit_ruleset,
    score_vehicle,
    segment_events,
)
from .writers import (
    build_adapter_payload,
    build_scores_rows,
    recompute_score_from_ledger,
    validate_adapter_payload,
)

__all__ = [
    "DeductionRuleset",
    "aggregate_group_deduction",
    "build_adapter_payload",
    "build_lookups",
    "build_scores_rows",
    "eb_smooth_rate",
    "fit_bins",
    "fit_ruleset",
    "load_contract",
    "recompute_score_from_ledger",
    "score_vehicle",
    "segment_events",
    "validate_adapter_payload",
]
