"""Fusion mechanics for T2-R3 (D18-D20).

Conventions fixed here (see fusion-charter-v1.md section 2):

* ``alpha_q0`` is the weight of the R1 rule score Q0; ``alpha_q0 = 1.0`` is
  the pure-rule arm (roadmap "α=0" reading A), ``0 < alpha_q0 < 1`` are the
  fusion arms, and ``0.0`` is the pure-model reference arm Q1 only;
* Q1 comes from ``probability_to_safety_score`` (100 x (1 - p)) upstream, so
  the task-1 probability column is recovered as ``1 - Q1 / 100`` and is only
  attached for ``alpha_q0 < 1`` -- evaluation declares it accordingly;
* the same T2-R4 contract and manifest gate every arm; the task-1 probability
  is never used to pick the winner (circularity red line).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from task2.evaluation.evaluate_scores import evaluate_scores

DEFAULT_GRID: tuple[float, ...] = (1.0, 0.7, 0.5, 0.3, 0.0)


def fuse_scores(
    q0_scores: Mapping[str, float],
    q1_scores: Mapping[str, float],
    alpha_q0: float,
) -> dict[str, float]:
    """Blend Q0 and Q1 safety scores per vehicle."""

    if not 0.0 <= alpha_q0 <= 1.0:
        raise ValueError("alpha_q0 must be within [0, 1]")
    if set(q0_scores) != set(q1_scores):
        raise ValueError("Q0 and Q1 must cover the same vehicles")
    fused: dict[str, float] = {}
    for gpsno, q0 in q0_scores.items():
        q1 = q1_scores[gpsno]
        for name, value in (("Q0", q0), ("Q1", q1)):
            if not 0.0 <= float(value) <= 100.0:
                raise ValueError(f"{name} score out of [0, 100] for {gpsno}")
        blended = alpha_q0 * float(q0) + (1.0 - alpha_q0) * float(q1)
        if not 0.0 <= blended <= 100.0:
            raise ValueError(f"fused score out of [0, 100] for {gpsno}")
        fused[str(gpsno)] = blended
    return fused


def run_alpha_grid(
    q0_scores: Mapping[str, float],
    q1_scores: Mapping[str, float],
    labels: Mapping[str, int],
    folds: Mapping[str, str],
    evaluation_contract: Mapping[str, Any],
    manifest_base: Mapping[str, Any],
    alphas: Sequence[float] = DEFAULT_GRID,
) -> dict[str, Any]:
    """Run every alpha arm under the same T2-R4 evaluation contract.

    Selection is mechanical and registered, not a judgement: among arms whose
    hard checks pass **and** primary gate holds, the highest ROC AUC wins;
    ties prefer the larger ``alpha_q0`` (more rule weight -- the charter's
    interpretability tie-break).  Arms may be rejected by the evaluation
    itself; those results are kept verbatim for the experiment report.
    """

    runs: list[dict[str, Any]] = []
    for alpha in alphas:
        fused = fuse_scores(q0_scores, q1_scores, float(alpha))
        rows: list[dict[str, Any]] = []
        uses_probability = alpha < 1.0
        for gpsno in sorted(fused):
            row: dict[str, Any] = {
                "gpsno": gpsno,
                "y": int(labels[gpsno]),
                "safety_score": fused[gpsno],
                "fold": str(folds[gpsno]),
            }
            if uses_probability:
                row["task1_risk_probability"] = 1.0 - float(q1_scores[gpsno]) / 100.0
            rows.append(row)
        manifest = dict(manifest_base)
        manifest["score_version"] = f"{manifest_base['score_version']}-w0{alpha:g}"
        manifest["uses_task1_probability"] = uses_probability
        manifest["task1_predictions_are_oof"] = uses_probability
        manifest["target_count"] = len(rows)
        report = evaluate_scores(rows, evaluation_contract, manifest)
        runs.append({"alpha_q0": float(alpha), "row_count": len(rows), "report": report})

    eligible = [
        run
        for run in runs
        if run["report"].get("status") == "pass" and run["report"].get("primary_gate_pass")
    ]
    selection: dict[str, Any] | None = None
    if eligible:
        best = max(
            eligible,
            key=lambda run: (run["report"]["roc_auc"] if run["report"]["roc_auc"] is not None else -1.0, run["alpha_q0"]),
        )
        selection = {
            "alpha_q0": best["alpha_q0"],
            "roc_auc": best["report"]["roc_auc"],
            "criterion": "primary_gate_then_roc_auc_tiebreak_larger_alpha_q0",
        }
    return {"runs": runs, "selection": selection}


def rank_disagreements(
    q0_scores: Mapping[str, float],
    q1_scores: Mapping[str, float],
    fused_scores: Mapping[str, float],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Rank vehicles by risk (low score = rank 1) across the three scores.

    The output feeds the D18 human-review ammunition: the top vehicles whose
    Q0 / Q1 / fused ranks disagree most, with each rank attached.
    """

    if not top_k >= 1:
        raise ValueError("top_k must be positive")
    if not (set(q0_scores) == set(q1_scores) == set(fused_scores)):
        raise ValueError("all three score maps must cover the same vehicles")

    def ranks(scores: Mapping[str, float]) -> dict[str, int]:
        order = sorted(scores, key=lambda gpsno: (scores[gpsno], str(gpsno)))
        return {gpsno: index + 1 for index, gpsno in enumerate(order)}

    rank_q0 = ranks(q0_scores)
    rank_q1 = ranks(q1_scores)
    rank_fused = ranks(fused_scores)

    entries: list[dict[str, Any]] = []
    for gpsno in sorted(q0_scores):
        triple = (rank_q0[gpsno], rank_q1[gpsno], rank_fused[gpsno])
        entries.append(
            {
                "gpsno": str(gpsno),
                "rank_q0": triple[0],
                "rank_q1": triple[1],
                "rank_fused": triple[2],
                "max_rank_gap": max(triple) - min(triple),
            }
        )
    entries.sort(key=lambda item: (-item["max_rank_gap"], item["gpsno"]))
    return {"vehicle_count": len(entries), "top_disagreements": entries[:top_k]}


def dimension_talking_point(
    dimension_name: str,
    peer_percentile: float,
) -> str:
    """Format the D20 driver-facing sentence; never exposes model internals."""

    if not dimension_name.strip():
        raise ValueError("dimension_name must be non-empty")
    if not 0.0 < peer_percentile < 100.0:
        raise ValueError("peer_percentile must be within (0, 100)")
    return f"您的{dimension_name.strip()}风险高于{peer_percentile:.0f}%的同行司机"
