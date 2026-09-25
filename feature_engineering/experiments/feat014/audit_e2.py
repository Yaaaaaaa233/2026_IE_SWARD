#!/usr/bin/env python3
"""Independently recalculate FEAT-014 E2 summaries from locked OOF files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from feature_engineering.experiments.feat014 import run as e  # noqa: E402


def rank(y: np.ndarray, p: np.ndarray, ids: np.ndarray) -> set[int]:
    return set(np.lexsort((ids.astype(str), -p))[: min(100, len(y))].tolist())


def changed(y: np.ndarray, candidate: set[int], reference: set[int]) -> dict:
    plus, minus = candidate - reference, reference - candidate
    return {
        "reference_false_positives_removed": int(sum(y[i] == 0 for i in minus)),
        "candidate_false_positives_added": int(sum(y[i] == 0 for i in plus)),
        "new_true_positives_in_top100": int(sum(y[i] == 1 for i in plus)),
        "true_positives_dropped_from_top100": int(sum(y[i] == 1 for i in minus)),
        "top100_overlap": int(len(candidate & reference)),
    }


def resolve_oof(run_dir: Path, current_batch: str, reference: str,
                frame: pd.DataFrame) -> tuple[np.ndarray, Path]:
    if "/" in reference:
        batch_name, version_name = reference.split("/", 1)
    else:
        batch_name, version_name = current_batch, reference
    path = run_dir / "batches" / batch_name / version_name / "oof.csv"
    if not path.is_file():
        raise ValueError(f"focus OOF does not exist: {reference}")
    return e.read_oof(path, frame, "probability"), path


def audit(run_dir: Path, batch: str, focus: list[str]) -> dict:
    lock, frame, _, _, _, y, folds, p_rf, p_f3 = e.load_run_inputs(run_dir)
    directory = run_dir / "batches" / batch
    doc = json.loads((directory / "e2_diagnosis.json").read_text(encoding="utf-8"))
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if doc.get("input_lock_sha256") != e.sha(run_dir / "input_lock.json"):
        errors.append("diagnosis is bound to a different E0 input lock")
    if doc.get("focus_versions") != focus:
        errors.append("diagnosis focus differs from requested versions")
    ids = frame.sample_id.astype(str).to_numpy()
    preds = {}
    selected = {}
    for item in plan["versions"]:
        name = item["version"]
        oof_path = directory / name / "oof.csv"
        p = e.read_oof(oof_path, frame, "probability")
        preds[name] = p
        selected[name] = rank(y, p, ids)
        if doc.get("oof_sha256", {}).get(name) != e.sha(oof_path):
            errors.append(f"diagnosis OOF fingerprint mismatch: {name}")
        row = doc.get("metrics", {}).get(name, {})
        expected = {
            "auc": float(roc_auc_score(y, p)),
            "ap": float(average_precision_score(y, p)),
            "recall_at_100": float(y[list(selected[name])].sum() / y.sum()),
            "brier": float(brier_score_loss(y, p)),
            "delta_auc_vs_rf": e.compute_pair(y, p, p_rf),
            "delta_auc_vs_f3": e.compute_pair(y, p, p_f3),
        }
        for key in ("auc", "ap", "recall_at_100", "brier"):
            if key not in row or not np.isclose(row[key], expected[key], atol=1e-12, rtol=0):
                errors.append(f"E2 metric mismatch: {name}/{key}")
        for key in ("delta_auc_vs_rf", "delta_auc_vs_f3"):
            saved = row.get(key, {})
            if (not np.isclose(saved.get("point", np.nan), expected[key]["point"], atol=1e-12, rtol=0)
                    or not np.allclose(saved.get("ci95", [np.nan, np.nan]), expected[key]["ci95"], atol=1e-12, rtol=0)):
                errors.append(f"E2 paired interval mismatch: {name}/{key}")
        if row.get("delta_recall_vs_rf") is None or row.get("delta_recall_vs_f3") is None:
            errors.append(f"E2 anchor recall comparison missing: {name}")
        else:
            recall = expected["recall_at_100"]
            rr = float(y[list(rank(y, p_rf, ids))].sum() / y.sum())
            rf3 = float(y[list(rank(y, p_f3, ids))].sum() / y.sum())
            if not np.isclose(row["delta_recall_vs_rf"], recall - rr, atol=1e-12, rtol=0): errors.append(f"E2 RF recall delta mismatch: {name}")
            if not np.isclose(row["delta_recall_vs_f3"], recall - rf3, atol=1e-12, rtol=0): errors.append(f"E2 F3 recall delta mismatch: {name}")
        for key, reference in (("delta_brier_vs_rf", p_rf), ("delta_brier_vs_f3", p_f3)):
            expect = float(brier_score_loss(y, p) - brier_score_loss(y, reference))
            if row.get(key) is None or not np.isclose(row[key], expect, atol=1e-12, rtol=0): errors.append(f"E2 Brier delta mismatch: {name}/{key}")
    base_top = rank(y, p_f3, ids)
    expected_changes = {name: changed(y, selected[name], base_top) for name in selected}
    if expected_changes != doc.get("top100_vs_f3"):
        errors.append("Top100 error changes do not reproduce")
    names = sorted(preds)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            saved = doc.get("pairwise_auc", {}).get(f"{left}_minus_{right}", {})
            expected = e.compute_pair(y, preds[left], preds[right])
            if (not np.isclose(saved.get("point", np.nan), expected["point"], atol=1e-12, rtol=0)
                    or not np.allclose(saved.get("ci95", [np.nan, np.nan]), expected["ci95"], atol=1e-12, rtol=0)):
                errors.append(f"within-batch paired interval mismatch: {left}_minus_{right}")
    expected_direct = {}
    for item in plan["versions"]:
        name = item["version"]
        raw_refs = item.get("comparison", {}).get("compare_to", "")
        refs = [x.strip() for x in raw_refs.split(" and ") if x.strip()]
        if not refs:
            continue
        expected_direct[name] = {}
        for reference in refs:
            try:
                parent_p, parent_path = resolve_oof(run_dir, batch, reference, frame)
            except Exception as exc:
                errors.append(f"registered comparison OOF unavailable: {name}/{reference}: {exc}")
                continue
            parent_top = rank(y, parent_p, ids)
            expected_direct[name][reference] = {
                "delta_auc_candidate_minus_reference": e.compute_pair(y, preds[name], parent_p),
                "delta_recall_at_100_candidate_minus_reference": float(
                    y[list(selected[name])].sum() / y.sum() - y[list(parent_top)].sum() / y.sum()),
                "delta_brier_candidate_minus_reference": float(
                    brier_score_loss(y, preds[name]) - brier_score_loss(y, parent_p)),
                "top100_change": changed(y, selected[name], parent_top),
                "reference_oof_sha256": e.sha(parent_path),
            }
    actual_direct = doc.get("direct_comparisons", {})
    if set(expected_direct) != set(actual_direct):
        errors.append("registered direct comparison set mismatch")
    for name, refs in expected_direct.items():
        for reference, expected in refs.items():
            actual = actual_direct.get(name, {}).get(reference, {})
            if expected["top100_change"] != actual.get("top100_change"):
                errors.append(f"direct comparison Top100 mismatch: {name}/{reference}")
            if expected["reference_oof_sha256"] != actual.get("reference_oof_sha256"):
                errors.append(f"direct comparison OOF fingerprint mismatch: {name}/{reference}")
            if not np.isclose(expected["delta_recall_at_100_candidate_minus_reference"],
                              actual.get("delta_recall_at_100_candidate_minus_reference", np.nan), atol=1e-12, rtol=0):
                errors.append(f"direct comparison recall mismatch: {name}/{reference}")
            if not np.isclose(expected["delta_brier_candidate_minus_reference"],
                              actual.get("delta_brier_candidate_minus_reference", np.nan), atol=1e-12, rtol=0):
                errors.append(f"direct comparison Brier mismatch: {name}/{reference}")
            actual_auc = actual.get("delta_auc_candidate_minus_reference", {})
            expected_auc = expected["delta_auc_candidate_minus_reference"]
            if (not np.isclose(expected_auc["point"], actual_auc.get("point", np.nan), atol=1e-12, rtol=0)
                    or not np.allclose(expected_auc["ci95"], actual_auc.get("ci95", [np.nan, np.nan]), atol=1e-12, rtol=0)):
                errors.append(f"direct comparison AUC mismatch: {name}/{reference}")
    expected_fusion_refs = {}
    for item in plan["versions"]:
        if item["family"] != "fusion":
            continue
        for member in item["params"]["members"]:
            reference = f"{member['batch']}/{member['version']}"
            if reference not in expected_fusion_refs:
                p, _ = resolve_oof(run_dir, batch, reference, frame)
                expected_fusion_refs[reference] = p
        expected_fusion_refs[f"{batch}/{item['version']}"] = preds[item["version"]]
    saved_fusion = doc.get("fusion_comparison", {})
    if set(saved_fusion) != set(expected_fusion_refs):
        errors.append("fusion member/output comparison set mismatch")
    for reference, p in expected_fusion_refs.items():
        row = saved_fusion.get(reference, {})
        expected_auc = float(roc_auc_score(y, p))
        expected_recall = float(y[list(rank(y, p, ids))].sum() / y.sum())
        expected_brier = float(brier_score_loss(y, p))
        expected_delta = e.compute_pair(y, p, p_f3)
        for key, value in (("auc", expected_auc), ("recall_at_100", expected_recall), ("brier", expected_brier)):
            if not np.isclose(row.get(key, np.nan), value, atol=1e-12, rtol=0): errors.append(f"fusion comparison metric mismatch: {reference}/{key}")
        saved_delta = row.get("delta_auc_vs_f3", {})
        if (not np.isclose(saved_delta.get("point", np.nan), expected_delta["point"], atol=1e-12, rtol=0)
                or not np.allclose(saved_delta.get("ci95", [np.nan, np.nan]), expected_delta["ci95"], atol=1e-12, rtol=0)):
            errors.append(f"fusion comparison paired interval mismatch: {reference}")
    focus_predictions, focus_selected, focus_paths = {}, {}, {}
    if len(focus) != 2:
        errors.append("focus must contain exactly two OOF references")
    for reference in focus:
        try:
            p, path = resolve_oof(run_dir, batch, reference, frame)
            focus_predictions[reference] = p
            focus_selected[reference] = rank(y, p, ids)
            focus_paths[reference] = path
            if doc.get("focus_oof_sha256", {}).get(reference) != e.sha(path):
                errors.append(f"focus OOF fingerprint mismatch: {reference}")
        except Exception as exc:
            errors.append(f"focus OOF unavailable: {reference}: {exc}")
    if len(focus_predictions) != 2:
        pair = {}
    else:
        a, b = focus
        both, only_a, only_b = focus_selected[a] & focus_selected[b], focus_selected[a] - focus_selected[b], focus_selected[b] - focus_selected[a]
        neither = set(range(len(y))) - (focus_selected[a] | focus_selected[b])
        pair = {
            "both_true_positives": int(sum(y[i] == 1 for i in both)),
            "first_only_true_positives": int(sum(y[i] == 1 for i in only_a)),
            "second_only_true_positives": int(sum(y[i] == 1 for i in only_b)),
            "neither_true_positives": int(sum(y[i] == 1 for i in neither)),
            "first_false_positives": int(sum(y[i] == 0 for i in focus_selected[a])),
            "second_false_positives": int(sum(y[i] == 0 for i in focus_selected[b])),
            "top100_intersection": int(len(focus_selected[a] & focus_selected[b])),
            "probability_pearson_correlation": float(np.corrcoef(focus_predictions[a], focus_predictions[b])[0, 1]),
            "delta_auc_first_minus_second": e.compute_pair(y, focus_predictions[a], focus_predictions[b]),
            "fold_ids": sorted(map(int, np.unique(folds))),
        }
        saved_pair = doc.get("pair_complementarity", {})
        for key, value in pair.items():
            if key == "delta_auc_first_minus_second":
                saved_delta = saved_pair.get(key, {})
                if (not np.isclose(saved_delta.get("point", np.nan), value["point"], atol=1e-12, rtol=0)
                        or not np.allclose(saved_delta.get("ci95", [np.nan, np.nan]), value["ci95"], atol=1e-12, rtol=0)):
                    errors.append(f"pair comparison mismatch: {key}")
                continue
            if isinstance(value, float):
                if not np.isclose(saved_pair.get(key, np.nan), value, atol=1e-12, rtol=0): errors.append(f"pair comparison mismatch: {key}")
            elif saved_pair.get(key) != value:
                errors.append(f"pair comparison mismatch: {key}")
    return {
        "task": "FEAT-014", "stage": "E2", "batch": batch,
        "status": "fail" if errors else "pass", "errors": errors,
        "versions_recalculated": sorted(preds),
        "focus_versions": focus,
        "input_lock_sha256": e.sha(run_dir / "input_lock.json"),
        "human_visual_review": "pending; machine audit does not represent user chart acceptance",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--focus", nargs=2, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit(args.run_dir.expanduser().resolve(), args.batch, args.focus)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.write:
        e.json_dump(args.run_dir / "batches" / args.batch / "independent_audit_e2.json", result)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
