#!/usr/bin/env python3
"""Audit the locked FEAT-009 F3 proxy input for FEAT-011 S0."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

LABEL_VERSION = "label_v2_record_count_20260923"
SPLIT_VERSION = "split_v2_record_strat5_seed42"
FEATURE_WINDOW = "[2026-06-01,2026-06-21)"
LABEL_WINDOW = "2026-06-21/2026-07-31"
AS_OF = "2026-06-21T00:00:00+08:00"
META = {
    "sample_id", "gpsno", "y", "fold", "label_version", "split_version",
    "label_window", "horizon_days", "as_of", "window_start", "window_end",
    "lookback_days", "source_version", "feature_version", "feature_version_f1",
    "feature_version_f2", "feature_version_f3",
}
NIGHT_SUMMARY = {
    "f3_night_deep_rate", "f3_night_day_rate", "f3_night_degradation",
    "f3_night_exposure_share",
}
BLOCKED_EXACT = {
    "monthly_avg_mileage", "monthly_avg_hours", "monthly_avg_stops", "highway_ratio",
    "morning_ratio", "dusk_ratio", "night_hours_ratio", "night_mileage_ratio",
    "energy_type", "cohort_fallback", "f2_prior_incident_x_night",
    "f2_prior_incident_x_highway",
}
FUTURE_RE = re.compile(r"(^|_)(future|next|post|target|after_as_of)(_|$)", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def make_views(columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    features = [column for column in columns if column not in META]
    detailed_night = [column for column in features
                      if column.startswith("f3_night_") and column not in NIGHT_SUMMARY]
    slim = [column for column in features if column not in set(detailed_night)]
    return features, slim, detailed_night


def validate_column_views(full: list[str], slim: list[str], removed: list[str]) -> None:
    full_set, slim_set, removed_set = set(full), set(slim), set(removed)
    _require(len(full) == len(full_set), "full feature view contains duplicate columns")
    _require(len(slim) == len(slim_set), "slim feature view contains duplicate columns")
    _require(len(removed) == len(removed_set), "removed-column list contains duplicates")
    _require(slim_set.isdisjoint(removed_set), "slim and removed column sets overlap")
    _require(full_set == slim_set | removed_set, "full/slim column conservation failed")
    _require(removed_set == {c for c in full if c.startswith("f3_night_") and c not in NIGHT_SUMMARY},
             "removed columns differ from the preregistered detailed-night rule")
    _require(NIGHT_SUMMARY.issubset(full_set), "one or more required night summary columns are missing")
    _require(NIGHT_SUMMARY.issubset(slim_set), "a night summary column was removed from the slim view")


def audit_frame(frame: pd.DataFrame, manifest: dict[str, Any], input_sha256: str) -> dict[str, Any]:
    _require(frame.columns.is_unique, "input contains duplicate column names")
    _require(manifest.get("protocol") == LABEL_VERSION, "label protocol differs from FEAT-011 contract")
    _require(manifest.get("split_version") == SPLIT_VERSION, "split version differs from FEAT-011 contract")
    _require(manifest.get("as_of") == AS_OF, "as_of differs from the FEAT-009 cutoff")
    _require(manifest.get("feature_window") == FEATURE_WINDOW, "feature window differs from the FEAT-011 contract")
    _require(manifest.get("label_window") == LABEL_WINDOW, "label window differs from the FEAT-011 contract")
    _require(manifest.get("rows") == len(frame), "row count differs from the locked manifest")
    output_meta = manifest.get("outputs", {}).get("f3_model_input.csv", {})
    _require(output_meta.get("sha256") == input_sha256, "F3 input hash differs from the locked manifest")
    manifest_columns = output_meta.get("columns")
    if isinstance(manifest_columns, list):
        _require(manifest_columns == list(frame.columns), "F3 column order differs from the locked manifest")
    else:
        _require(manifest_columns == len(frame.columns), "F3 column count differs from the locked manifest")

    required_meta = {"sample_id", "gpsno", "y", "fold", "label_version", "split_version",
                     "label_window", "horizon_days"}
    _require(required_meta.issubset(frame.columns), "required label/split metadata is missing")
    for key in ("sample_id", "gpsno"):
        _require(frame[key].notna().all() and not frame[key].duplicated().any(),
                 f"{key} is null or duplicated")
    labels = pd.to_numeric(frame.y, errors="coerce")
    folds = pd.to_numeric(frame.fold, errors="coerce")
    _require(labels.notna().all() and labels.isin([0, 1]).all(),
             "labels are not complete binary values")
    _require(folds.notna().all() and folds.isin([0, 1, 2, 3, 4]).all(),
             "folds are not the frozen five-fold IDs")
    _require(frame.label_version.astype(str).eq(LABEL_VERSION).all(), "row label version differs")
    _require(frame.split_version.astype(str).eq(SPLIT_VERSION).all(), "row split version differs")
    _require(frame.label_window.astype(str).eq(LABEL_WINDOW).all(), "row label window differs")
    _require(pd.to_numeric(frame.horizon_days, errors="coerce").eq(40).all(), "row horizon is not 40 days")
    _require(int(labels.sum()) == manifest.get("positives"), "positive count differs from the locked manifest")
    fold_classes = pd.DataFrame({"_fold": folds.astype(int), "_y": labels.astype(int)}).groupby("_fold")._y.nunique()
    _require(len(fold_classes) == 5 and fold_classes.eq(2).all(), "one or more folds lack both classes")

    features, slim, detailed_night = make_views(list(frame.columns))
    blocked = [column for column in features
               if column in BLOCKED_EXACT or column.startswith("f3_profile_")
               or "_cohort_" in column or FUTURE_RE.search(column)]
    _require(not blocked, f"blocked/future-derived columns remain: {blocked[:8]}")
    validate_column_views(features, slim, detailed_night)
    return {
        "status": "pass", "protocol": LABEL_VERSION, "split_version": SPLIT_VERSION,
        "feature_window": FEATURE_WINDOW, "label_window": LABEL_WINDOW,
        "as_of": AS_OF, "rows": int(len(frame)), "positives": int(labels.sum()),
        "folds": int(folds.nunique()), "full_feature_count": len(features),
        "slim_feature_count": len(slim), "removed_detailed_night_count": len(detailed_night),
        "full_features_sha256": hashlib.sha256("\n".join(features).encode()).hexdigest(),
        "slim_features_sha256": hashlib.sha256("\n".join(slim).encode()).hexdigest(),
        "removed_detailed_night_sha256": hashlib.sha256("\n".join(detailed_night).encode()).hexdigest(),
        "input_sha256": input_sha256,
        "checks": ["manifest and input hash", "row identity uniqueness", "label/split/time binding",
                   "five-fold class presence", "blocked/future column exclusion", "column conservation"],
    }


def _family(column: str) -> str:
    if column in META:
        return "metadata"
    pieces = column.split("_")
    if column.startswith("f3_night_"):
        return "f3_night"
    if pieces[0] == "f3" and len(pieces) > 2:
        return "_".join(pieces[:2])
    return pieces[0]


def build_ledger(frame: pd.DataFrame, visibility: pd.DataFrame) -> pd.DataFrame:
    provenance_fields = {"source", "source_end", "decision", "reason", "parents", "fit_scope"}
    _require({"feature", *provenance_fields}.issubset(visibility.columns),
             "Y0 visibility ledger lacks required provenance fields")
    sources: dict[str, dict[str, str]] = {}
    for _, row in visibility.iterrows():
        name = str(row.feature)
        item = {key: "" if pd.isna(row[key]) else str(row[key])
                for key in sorted(provenance_fields)}
        if name not in sources:
            sources[name] = item
        elif sources[name] != item:
            sources[name] = {"source": "ambiguous_in_visibility_ledger", "source_end": "", "decision": "review"}

    features, slim, detailed = make_views(list(frame.columns))
    validate_column_views(features, slim, detailed)
    slim_set, detailed_set = set(slim), set(detailed)
    rows = []
    for column in frame.columns:
        meta = column in META
        provenance = sources.get(column, {})
        rows.append({
            "column": column,
            "role": "metadata" if meta else "predictor",
            "family": _family(column),
            "visibility_source": provenance.get("source", "not_mapped_in_y0_ledger"),
            "source_end": provenance.get("source_end", ""),
            "visibility_decision": provenance.get("decision", "not_mapped"),
            "visibility_reason": provenance.get("reason", ""),
            "visibility_parents": provenance.get("parents", ""),
            "visibility_fit_scope": provenance.get("fit_scope", ""),
            "in_full_f3": not meta,
            "in_slim_f3": column in slim_set,
            "removed_by_plan": column in detailed_set,
            "slim_view_reason": ("metadata" if meta else "remove detailed f3_night column"
                                 if column in detailed_set else "retain"),
        })
    return pd.DataFrame(rows)


def write_family_figure(ledger: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = (ledger[ledger.role.eq("predictor")]
              .groupby(["family", "in_slim_f3"], dropna=False).size().unstack(fill_value=0))
    if False not in counts.columns:
        counts[False] = 0
    if True not in counts.columns:
        counts[True] = 0
    counts = counts.sort_values(by=[True, False], ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.31 * len(counts))))
    y = range(len(counts))
    ax.barh(list(y), counts[True], color="#397a92", label="Retained in slim view")
    ax.barh(list(y), counts[False], left=counts[True], color="#d7a15c", label="Removed detailed night columns")
    ax.set_yticks(list(y), counts.index.astype(str))
    ax.set_xlabel("Feature columns (schema count; no performance data)")
    ax.set_title("FEAT-011 S0: F3 feature-family ledger")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.94, bottom=0.12)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_self_tests() -> dict[str, Any]:
    columns = ["sample_id", "gpsno", "y", "fold", "label_version", "split_version",
               "label_window", "horizon_days", "base_speed_mean", *sorted(NIGHT_SUMMARY),
               "f3_night_collision_warn_deep_rate", "f3_night_collision_warn_day_rate"]
    rows = []
    for i in range(50):
        rows.append({"sample_id": f"S{i:03d}", "gpsno": f"G{i:03d}", "y": int(i % 5 == 0),
                     "fold": i // 10, "label_version": LABEL_VERSION, "split_version": SPLIT_VERSION,
                     "label_window": LABEL_WINDOW, "horizon_days": 40,
                     "base_speed_mean": float(i), **{name: 0.1 for name in NIGHT_SUMMARY},
                     "f3_night_collision_warn_deep_rate": 0.2,
                     "f3_night_collision_warn_day_rate": 0.1})
    valid = pd.DataFrame(rows, columns=columns)

    def manifest_for(frame: pd.DataFrame) -> dict[str, Any]:
        return {"protocol": LABEL_VERSION, "split_version": SPLIT_VERSION, "as_of": AS_OF,
                "feature_window": FEATURE_WINDOW, "label_window": LABEL_WINDOW, "rows": len(frame),
                "positives": int(frame.y.sum()),
                "outputs": {"f3_model_input.csv": {"sha256": sha256_frame(frame), "columns": len(frame.columns)}}}

    checks = 0
    audit_frame(valid, manifest_for(valid), sha256_frame(valid)); checks += 1
    for label, mutate in (
        ("future profile field", lambda d: d.assign(f3_profile_monthly_avg=1.0)),
        ("duplicate vehicle", lambda d: d.assign(gpsno=["G001"] + d.gpsno.astype(str).tolist()[1:])),
        ("changed label version", lambda d: d.assign(label_version="old_label")),
        ("changed split binding", lambda d: d.assign(split_version="old_split")),
        ("fractional label", lambda d: d.assign(y=[0.5] + d.y.astype(float).tolist()[1:])),
        ("fractional fold", lambda d: d.assign(fold=[0.5] + d.fold.astype(float).tolist()[1:])),
    ):
        bad = mutate(valid.copy())
        try:
            audit_frame(bad, manifest_for(bad), sha256_frame(bad))
        except ValueError:
            checks += 1
        else:
            raise AssertionError(f"S0 audit accepted bad case: {label}")
    full, slim, removed = make_views(columns)
    try:
        validate_column_views(full, slim, removed[:-1])
    except ValueError:
        checks += 1
    else:
        raise AssertionError("column-difference drift was not rejected")
    return {"status": "pass", "cases_passed": checks, "cases_expected": 8,
            "cases": ["valid synthetic input", "future profile field", "duplicate vehicle",
                      "changed label version", "changed split binding", "fractional label",
                      "fractional fold", "column-difference drift"]}


def execute_audit(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    frame = pd.read_csv(args.input, dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    visibility = pd.read_csv(args.visibility, low_memory=False)
    digest = sha256_file(args.input)
    result = audit_frame(frame, manifest, digest)
    result.update({
        "run_id": output.name,
        "input_ref": "outputs/feat-009/v2/y1/f3_model_input.csv",
        "input_manifest_ref": "outputs/feat-009/v2/y1/input_manifest.json",
        "manifest_sha256": sha256_file(args.manifest),
        "visibility_ledger_sha256": sha256_file(args.visibility),
        "visibility_mapped_feature_columns": int(sum(
            1 for c in make_views(list(frame.columns))[0] if c in set(visibility.feature.astype(str))
        )),
        "visibility_unmapped_feature_columns": int(sum(
            1 for c in make_views(list(frame.columns))[0] if c not in set(visibility.feature.astype(str))
        )),
        "human_review": "pending",
        "synthetic_bad_case_checks": run_self_tests(),
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
    })
    ledger = build_ledger(frame, visibility)
    ledger.to_csv(output / "column_ledger.csv", index=False, lineterminator="\n")
    (output / "input_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_family_figure(ledger, output / "feature_family_counts.png")
    (output / "acceptance.md").write_text(
        "# FEAT-011 S0 acceptance\n\n"
        "- Question: does the locked FEAT-009 v2 F3 table satisfy the accepted version, visibility, and fixed column-view contract?\n"
        "- Implementation/data contract: `pass` for the checks listed in `input_check.json`; source review is still pending.\n"
        "- Experimental effect: `not_run`; S0 does not train a model or calculate a new score.\n"
        "- Evidence: `partial`; machine ledger/check and family chart exist, and synthetic visual samples await Ye An's review.\n"
        "- Environment: `pass` for this local read-only audit; outputs remain in the ignored controlled directory.\n"
        "- Scope: locked 20+40 proxy input only. No OOF was generated.\n"
        "- Next: review the synthetic time-window and 2x2 samples, then complete the outstanding EVAL-001/EVAL-003/FEAT-009 responsibility reviews before S1 preregistration.\n",
        encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--visibility", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(run_self_tests(), ensure_ascii=False, indent=2))
    if not any((args.input, args.manifest, args.visibility, args.output_dir)):
        return
    if not all((args.input, args.manifest, args.visibility, args.output_dir)):
        parser.error("real audit requires --input, --manifest, --visibility, and --output-dir")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty S0 run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = execute_audit(args, output)
    except Exception as exc:
        failure = {"status": "fail", "run_id": output.name,
                   "error_type": type(exc).__name__, "error": str(exc)}
        (output / "failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
        raise
    print(json.dumps({k: result[k] for k in ("status", "run_id", "rows", "full_feature_count",
                                             "slim_feature_count", "removed_detailed_night_count",
                                             "visibility_mapped_feature_columns",
                                             "visibility_unmapped_feature_columns", "human_review")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
