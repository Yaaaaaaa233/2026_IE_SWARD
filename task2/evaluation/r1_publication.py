"""Publish R1 rule scores under the R4 low-evidence discipline.

R1 owns segmentation, fitted bins, weights and ledger aggregation.  This
adapter only applies the already-published R4 rule that an insufficiently
observed vehicle cannot leave the pipeline with a fabricated high score.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from task2.deduction.engine import DeductionRuleset, VehicleScoring
from task2.deduction.writers import recompute_score_from_ledger


def build_r1_publication_rows(
    scorings: Sequence[VehicleScoring],
    ruleset: DeductionRuleset,
    labels: Mapping[str, int],
    folds: Mapping[str, str],
    *,
    low_evidence_prior: float = 50.0,
) -> list[dict[str, Any]]:
    """Build R4 rows while preserving the R1 raw score and ledger check.

    Normal vehicles publish the exact R1 score.  When R1 marks a vehicle as
    insufficiently observed, the public score and its independently
    reproducible value both become the declared conservative prior; the raw
    ledger score remains visible in ``raw_rule_score`` for audit.
    """

    prior = float(low_evidence_prior)
    if not math.isfinite(prior) or not 0.0 <= prior <= 100.0:
        raise ValueError("low_evidence_prior must be finite and within [0, 100]")
    ids = [str(scoring.gpsno) for scoring in scorings]
    if len(set(ids)) != len(ids):
        raise ValueError("scorings contain duplicate gpsno values")
    if set(ids) != {str(value) for value in labels}:
        raise ValueError("labels must cover exactly the scored vehicles")
    if set(ids) != {str(value) for value in folds}:
        raise ValueError("folds must cover exactly the scored vehicles")

    rows: list[dict[str, Any]] = []
    for scoring in scorings:
        gpsno = str(scoring.gpsno)
        label = int(labels[gpsno])
        if label not in {0, 1}:
            raise ValueError(f"label for {gpsno} must be 0 or 1")
        fold = str(folds[gpsno]).strip()
        if not fold:
            raise ValueError(f"fold for {gpsno} must be nonempty")
        raw_rule_score = recompute_score_from_ledger(scoring, ruleset)
        if abs(raw_rule_score - float(scoring.safety_score)) > 1e-9:
            raise ValueError(f"R1 ledger does not reproduce the raw score for {gpsno}")
        if scoring.insufficient_evidence:
            published_score = prior
            observation_status = "insufficient_evidence"
            publication_policy = "low_evidence_prior"
        else:
            published_score = raw_rule_score
            observation_status = "observed"
            publication_policy = "raw_rule_score"
        rows.append(
            {
                "gpsno": gpsno,
                "y": label,
                "fold": fold,
                "safety_score": published_score,
                "rule_recomputed_score": published_score,
                "raw_rule_score": raw_rule_score,
                "observation_status": observation_status,
                "publication_policy": publication_policy,
                "rule_version": ruleset.rule_version,
            }
        )
    return rows
