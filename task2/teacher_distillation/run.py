"""CLI for the independent D0 task-1 probability distillation route."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .model import (
    build_feature_specs,
    cross_fit_distillation,
    fit_ruleset,
    recompute_score,
    score_vehicle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--teacher-scores", type=Path, required=True)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument(
        "--classification",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "classification" / "classification_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/task2_teacher_distillation"))
    parser.add_argument("--teacher-model-version")
    parser.add_argument("--rule-version", default="D0-task1-distilled-v1")
    parser.add_argument("--exposure-column", default="exposure_km_20d")
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--label-column", default="y")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--group-agg-fraction", type=float, default=0.25)
    parser.add_argument("--min-exposure", type=float, default=0.0)
    parser.add_argument("--low-evidence-prior", type=float, default=50.0)
    parser.add_argument(
        "--confirm-oof",
        action="store_true",
        help="Explicitly attest that the supplied probabilities are OOF when calib_flag is absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = json.loads(args.classification.read_text(encoding="utf-8"))
    specs = build_feature_specs(contract)
    features = pd.read_csv(args.features, dtype={"gpsno": str})
    teacher = pd.read_csv(args.teacher_scores, dtype={"gpsno": str})
    _require_columns(features, ["gpsno", args.exposure_column, *[spec.column for spec in specs]])
    _require_columns(teacher, ["gpsno", "prob"])
    _unique_ids(features, "features")
    _unique_ids(teacher, "teacher scores")
    if set(features.gpsno) != set(teacher.gpsno):
        raise ValueError("feature and teacher vehicle sets must match exactly")

    model_version = _teacher_version(teacher, args.teacher_model_version)
    oof_declared = args.confirm_oof or (
        "calib_flag" in teacher.columns
        and teacher.calib_flag.astype(str).str.contains("oof", case=False).all()
    )
    if not oof_declared:
        raise ValueError("teacher probabilities are not identified as OOF; supply verified data or --confirm-oof")

    folds = _load_folds(features, args)
    labels = _load_labels(args) if args.labels else None
    records = []
    for row in features.to_dict(orient="records"):
        records.append(
            {
                "gpsno": str(row["gpsno"]),
                "exposure": row[args.exposure_column],
                **{spec.column: row[spec.column] for spec in specs},
            }
        )
    teacher_map = dict(zip(teacher.gpsno.astype(str), teacher.prob.astype(float)))
    common: dict[str, Any] = {
        "teacher_model_version": model_version,
        "teacher_scores_are_oof": True,
        "rule_version": args.rule_version,
        "n_bins": args.n_bins,
        "ridge": args.ridge,
        "group_agg_fraction": args.group_agg_fraction,
        "min_exposure": args.min_exposure,
        "low_evidence_prior": args.low_evidence_prior,
    }
    crossfit = cross_fit_distillation(
        records, teacher_map, folds, specs, actual_labels=labels, **common
    )
    final_rules = fit_ruleset(records, teacher_map, specs, **common)
    final_rows = []
    for row in records:
        gpsno = row["gpsno"]
        scored = score_vehicle(row, final_rules, teacher_probability=teacher_map[gpsno])
        final_rows.append(
            {
                "gpsno": gpsno,
                "teacher_probability": teacher_map[gpsno],
                "safety_score": scored.safety_score,
                "rule_recomputed_score": recompute_score(scored, final_rules),
                "observation_status": scored.observation_status,
                "top_reasons": json.dumps(scored.top_reasons, ensure_ascii=False),
                "rule_version": final_rules.rule_version,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "distilled_rules.json").write_text(
        json.dumps(final_rules.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "crossfit_metrics.json").write_text(
        json.dumps(crossfit.metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(crossfit.rows).assign(
        top_reasons=lambda frame: frame.top_reasons.map(lambda value: json.dumps(value, ensure_ascii=False))
    ).to_csv(args.output_dir / "crossfit_scores.csv", index=False)
    pd.DataFrame(final_rows).to_csv(args.output_dir / "delivery_scores.csv", index=False)
    _write_rule_table(args.output_dir / "deduction_rules.csv", final_rules)
    print(
        json.dumps(
            {
                "status": "ok",
                "vehicles": len(records),
                "features": len(specs),
                "groups": len(final_rules.group_weights),
                "teacher_model_version": model_version,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


def _load_folds(features: pd.DataFrame, args: argparse.Namespace) -> dict[str, str]:
    if args.folds:
        data = pd.read_csv(args.folds, dtype={"gpsno": str})
        _require_columns(data, ["gpsno", args.fold_column])
        _unique_ids(data, "folds")
        if set(data.gpsno) != set(features.gpsno):
            raise ValueError("fold and feature vehicle sets must match exactly")
        return dict(zip(data.gpsno.astype(str), data[args.fold_column].astype(str)))
    if args.fold_column in features.columns:
        return dict(zip(features.gpsno.astype(str), features[args.fold_column].astype(str)))
    if args.n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    return {
        str(gpsno): f"d0f{int.from_bytes(hashlib.sha256(str(gpsno).encode()).digest()[:8], 'big') % args.n_folds}"
        for gpsno in features.gpsno
    }


def _load_labels(args: argparse.Namespace) -> dict[str, int]:
    data = pd.read_csv(args.labels, dtype={"gpsno": str})
    _require_columns(data, ["gpsno", args.label_column])
    _unique_ids(data, "labels")
    return dict(zip(data.gpsno.astype(str), data[args.label_column].astype(int)))


def _teacher_version(teacher: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if "model_version" in teacher.columns:
            actual = set(teacher.model_version.astype(str))
            if actual != {explicit}:
                raise ValueError("teacher model_version column does not match the requested version")
        return explicit
    if "model_version" not in teacher.columns:
        raise ValueError("teacher model version must be supplied by column or argument")
    versions = set(teacher.model_version.astype(str))
    if len(versions) != 1:
        raise ValueError("teacher scores must contain exactly one model version")
    return versions.pop()


def _write_rule_table(path: Path, ruleset: Any) -> None:
    names = {spec.key: spec for spec in ruleset.feature_specs}
    rows = []
    for key, curve in sorted(ruleset.curves.items()):
        spec = names[key]
        for index, severity in enumerate(curve.severity):
            rows.append(
                {
                    "catalog_class": key,
                    "name": spec.name,
                    "semantic_group": spec.semantic_group,
                    "public_dimension": spec.public_dimension,
                    "bin": index,
                    "rate_upper": curve.boundaries[index] if index < len(curve.boundaries) else None,
                    "monotone_severity": severity,
                    "teacher_probability_mean": curve.teacher_means[index],
                    "support": curve.support[index],
                    "group_weight": ruleset.group_weights.get(spec.semantic_group, 0.0),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _unique_ids(frame: pd.DataFrame, label: str) -> None:
    if frame.gpsno.isna().any() or frame.gpsno.astype(str).str.strip().eq("").any():
        raise ValueError(f"{label} contain empty gpsno")
    if frame.gpsno.astype(str).duplicated().any():
        raise ValueError(f"{label} contain duplicate gpsno")


if __name__ == "__main__":
    main()
