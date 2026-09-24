"""Read-only views over the T2-R0 classification contract.

The R1 engine never re-classifies events: ownership, scoring role, exposure
denominator and risk-chain membership all come from
``task2/classification/classification_v1.json``.  Changing any of those
semantics requires a contract version bump on the R0 side, not an edit here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "classification" / "classification_v1.json"
)

#: Roles that may receive behavioural deductions (T2-R0 charter section 3).
DEDUCTIBLE_ROLES = ("behavior", "context_warning")
OUTCOME_CODES = ("11803", "11804")
EVIDENCE_QUALITY_CODES = ("41006", "41021")


def load_contract(path: Path | str | None = None) -> dict[str, Any]:
    """Load and structurally sanity-check the classification contract."""

    source = Path(path) if path is not None else DEFAULT_CONTRACT_PATH
    contract = json.loads(source.read_text(encoding="utf-8"))
    for key in ("event_mappings", "risk_chains", "semantic_groups", "public_dimensions"):
        if key not in contract:
            raise ValueError(f"classification contract missing key: {key}")
    return contract


def build_lookups(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Derive code-level lookup tables used by the deduction engine.

    Returns a plain dict (JSON-serialisable diagnostics) with:

    * ``by_code``: the full event mapping row per event code;
    * ``chain_by_code``: risk-chain id and merge window per chained code;
    * ``speed_neighbor_codes``: codes outside any chain that use the
      charter's 60-second same-road-process neighbour window (CR06);
    * ``group_by_code`` / ``role_by_code`` / ``denominator_by_code``.
    """

    by_code: dict[str, Mapping[str, Any]] = {}
    for mapping in contract["event_mappings"]:
        code = str(mapping["code"])
        if code in by_code:
            raise ValueError(f"duplicate event code in contract: {code}")
        by_code[code] = mapping

    chain_by_code: dict[str, dict[str, Any]] = {}
    for chain in contract["risk_chains"]:
        window = float(chain["merge_window_seconds"])
        for code in chain["codes"]:
            code = str(code)
            if code not in by_code:
                raise ValueError(f"risk chain references unknown code: {code}")
            chain_by_code[code] = {"chain_id": chain["id"], "window_seconds": window}

    speed_neighbor_codes = sorted(
        code
        for code in by_code
        if code not in chain_by_code and code not in OUTCOME_CODES
    )

    return {
        "by_code": by_code,
        "chain_by_code": chain_by_code,
        "speed_neighbor_codes": speed_neighbor_codes,
        "group_by_code": {
            code: str(row["semantic_group"]) for code, row in by_code.items()
        },
        "role_by_code": {
            code: str(row["scoring_role"]) for code, row in by_code.items()
        },
        "denominator_by_code": {
            code: str(row["exposure_denominator"]) for code, row in by_code.items()
        },
    }
