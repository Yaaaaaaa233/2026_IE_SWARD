#!/usr/bin/env python3
"""Independent FEAT-014 E3 audit for registered seeds, mean predictions, and replay."""
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
from feature_engineering.experiments.feat014.fusion import logit_mean  # noqa: E402
from feature_engineering.experiments.feat014.finalize import FUSION_ROUTE, REPLAY, SINGLE_ROUTE  # noqa: E402


def resolve(run_dir: Path, reference: str, frame):
    batch, version = reference.split("/", 1)
    directory = run_dir / "batches" / batch / version
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    path = directory / "oof.csv"
    if e.sha(path) != result["oof_sha256"]:
        raise ValueError(f"OOF hash mismatch: {reference}")
    return e.read_oof(path, frame, "probability"), result, path


def recalc(y, p, ids, p_rf, p_f3):
    top = np.lexsort((ids, -p))[:min(100, len(y))]
    rf_top = np.lexsort((ids, -p_rf))[:min(100, len(y))]
    f3_top = np.lexsort((ids, -p_f3))[:min(100, len(y))]
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "recall_at_100": float(y[top].sum() / y.sum()),
        "brier": float(brier_score_loss(y, p)),
        "delta_recall_vs_rf": float(y[top].sum() / y.sum() - y[rf_top].sum() / y.sum()),
        "delta_recall_vs_f3": float(y[top].sum() / y.sum() - y[f3_top].sum() / y.sum()),
        "delta_brier_vs_rf": float(brier_score_loss(y, p) - brier_score_loss(y, p_rf)),
        "delta_brier_vs_f3": float(brier_score_loss(y, p) - brier_score_loss(y, p_f3)),
        "delta_auc_vs_rf": e.compute_pair(y, p, p_rf),
        "delta_auc_vs_f3": e.compute_pair(y, p, p_f3),
    }


def compare_metrics(actual: dict, expected: dict, label: str, errors: list[str]) -> None:
    scalar = ("auc", "ap", "recall_at_100", "brier", "delta_recall_vs_rf", "delta_recall_vs_f3",
              "delta_brier_vs_rf", "delta_brier_vs_f3")
    for key in scalar:
        if key not in actual or not np.isclose(actual[key], expected[key], atol=1e-12, rtol=0):
            errors.append(f"E3 metric mismatch: {label}/{key}")
    for key in ("delta_auc_vs_rf", "delta_auc_vs_f3"):
        saved = actual.get(key, {})
        expected_pair = expected[key]
        if (not np.isclose(saved.get("point", np.nan), expected_pair["point"], atol=1e-12, rtol=0)
                or not np.allclose(saved.get("ci95", [np.nan, np.nan]), expected_pair["ci95"], atol=1e-12, rtol=0)):
            errors.append(f"E3 paired AUC mismatch: {label}/{key}")


def verify_candidate_gates(report: dict, frame, y, p_rf, p_f3, batch_versions, errors: list[str]) -> None:
    ids = frame.sample_id.astype(str).to_numpy()
    refs = report.get("candidate_set", [])
    gates = report.get("candidate_gates", {})
    if set(gates) != set(refs):
        errors.append("E3 candidate gate references do not match candidate set")
    for reference in refs:
        if reference not in batch_versions:
            errors.append(f"E3 candidate OOF missing: {reference}")
            continue
        m = recalc(y, batch_versions[reference]["p"], ids, p_rf, p_f3)
        saved_metrics = report.get("version_summaries", {}).get(reference, {})
        compare_metrics(saved_metrics, m, f"candidate/{reference}", errors)
        low = m["delta_auc_vs_f3"]["ci95"][0]
        actual = gates.get(reference, {})
        expected = {
            "auc_point_positive": m["delta_auc_vs_f3"]["point"] > 0,
            "auc_interval_excludes_zero": low > 0,
            "meaningful_lower_bound_0_01": low >= .01,
            "recall_at_100_within_minus_0_02": m["delta_recall_vs_f3"] >= -.02,
            "brier_within_plus_0_005": m["delta_brier_vs_f3"] <= .005,
        }
        if actual != expected:
            errors.append(f"E3 candidate gate calculation mismatch: {reference}")


def audit(run_dir: Path) -> dict:
    lock, frame, _, _, _, y, folds, p_rf, p_f3 = e.load_run_inputs(run_dir)
    report_path = run_dir / "e3_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if report.get("input_lock_sha256") != e.sha(run_dir / "input_lock.json"):
        errors.append("E3 report is bound to a different E0 input lock")
    registration_path = run_dir / "e3_registration.json"
    if report.get("registration_sha256") != e.sha(registration_path):
        errors.append("E3 registration fingerprint mismatch")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("status") != "locked_before_any_stability_fit":
        errors.append("E3 stability plan was not locked before fitting")
    registered_plan_refs = registration.get("plans", {})
    for batch_name, expected_hash in registered_plan_refs.items():
        plan_path = run_dir / "batches" / batch_name / "plan.json"
        if not plan_path.is_file() or e.sha(plan_path) != expected_hash:
            errors.append(f"E3 preregistered plan fingerprint mismatch: {batch_name}")
    feedback = registration.get("feedback_refs", {})
    feedback_paths = {
        "B001_plan_sha256": run_dir / "batches/B001/plan.json",
        "B002_plan_sha256": run_dir / "batches/B002/plan.json",
        "B003_plan_sha256": run_dir / "batches/B003/plan.json",
        "B003_e2_diagnosis_sha256": run_dir / "batches/B003/e2_diagnosis.json",
        "B003_e2_audit_sha256": run_dir / "batches/B003/independent_audit_e2.json",
    }
    for key, path in feedback_paths.items():
        if key not in feedback or not path.is_file() or e.sha(path) != feedback[key]:
            errors.append(f"E3 stability feedback fingerprint mismatch: {key}")
    expected_versions = set()
    batch_versions = {}
    for batch in sorted((run_dir / "batches").glob("B*")):
        plan_path = batch / "plan.json"
        if not plan_path.is_file():
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("input_lock_sha256") != e.sha(run_dir / "input_lock.json"):
            errors.append(f"batch input-lock mismatch: {batch.name}")
        batch_audit_path = batch / "independent_audit.json"
        if not batch_audit_path.is_file() or json.loads(batch_audit_path.read_text(encoding="utf-8")).get("status") != "pass":
            errors.append(f"batch E1 audit is missing or failed: {batch.name}")
        for item in plan["versions"]:
            reference = f"{batch.name}/{item['version']}"
            expected_versions.add(reference)
            p, result, path = resolve(run_dir, reference, frame)
            batch_versions[reference] = {"p": p, "result": result, "path": path}
    actual_versions = {row["reference"] for row in report.get("ledger", [])}
    if expected_versions != actual_versions:
        errors.append("E3 complete-version ledger set mismatch")
    if len(actual_versions) != report.get("version_count") or len(actual_versions) > 30:
        errors.append("E3 version count or budget mismatch")
    if len(actual_versions) != registration.get("versions_total_after_registration"):
        errors.append("E3 total version count differs from preregistered budget ledger")

    ids = frame.sample_id.astype(str).to_numpy()
    routes = {"single_v005": SINGLE_ROUTE, "fusion_v014": FUSION_ROUTE}
    recomputed_routes = {}
    for route_name, references in routes.items():
        saved_route = report.get("seed_stability", {}).get(route_name, {})
        saved_runs = saved_route.get("runs", {})
        if set(saved_runs) != set(references):
            errors.append(f"E3 seed references mismatch: {route_name}")
            continue
        aucs = []
        for role, reference in references.items():
            if reference not in batch_versions:
                errors.append(f"E3 seed OOF missing: {route_name}/{role}")
                continue
            p = batch_versions[reference]["p"]
            metrics = recalc(y, p, ids, p_rf, p_f3)
            if saved_runs[role].get("reference") != reference:
                errors.append(f"E3 seed reference label mismatch: {route_name}/{role}")
            if saved_runs[role].get("oof_sha256") != e.sha(batch_versions[reference]["path"]):
                errors.append(f"E3 seed OOF fingerprint mismatch: {route_name}/{role}")
            compare_metrics(saved_runs[role].get("metrics", {}), metrics, f"{route_name}/{role}", errors)
            if role != "seed_mean":
                aucs.append(metrics["auc"])
        if len(aucs) == 3:
            seed_range = float(max(aucs) - min(aucs))
            for key, value in (("seed_auc_mean", float(np.mean(aucs))),
                               ("seed_auc_min", float(min(aucs))),
                               ("seed_auc_max", float(max(aucs))),
                               ("seed_auc_range", seed_range),
                               ("seed_auc_population_sd", float(np.std(aucs, ddof=0)))):
                if not np.isclose(saved_route.get(key, np.nan), value, atol=1e-12, rtol=0):
                    errors.append(f"E3 seed statistic mismatch: {route_name}/{key}")
        recomputed_routes[route_name] = saved_runs

    if len(recomputed_routes) == len(routes):
        expected_stable_route = min(
            recomputed_routes,
            key=lambda name: report["seed_stability"][name]["seed_auc_range"],
        )
        if report.get("lower_seed_variance_route") != expected_stable_route:
            errors.append("E3 lower-seed-variance route selection mismatch")

    ledger = report.get("ledger", [])
    if ledger:
        expected_best = max(ledger, key=lambda row: row["auc"])["reference"]
        eligible = [row for row in ledger if row["family"] != "fusion" and row["fit_seconds"] <= 15]
        expected_low_cost = max(eligible, key=lambda row: row["auc"])["reference"] if eligible else None
        stable_route = report.get("lower_seed_variance_route")
        stable_mean = report.get("seed_stability", {}).get(stable_route, {}).get("runs", {}).get("seed_mean", {}).get("reference")
        expected_candidates = list(dict.fromkeys([expected_best, stable_mean, expected_low_cost]))
        if report.get("best_exploration_point") != expected_best:
            errors.append("E3 best exploration point selection mismatch")
        if report.get("low_cost_candidate") != expected_low_cost:
            errors.append("E3 low-cost candidate selection mismatch")
        if report.get("candidate_set") != expected_candidates:
            errors.append("E3 candidate set selection mismatch")
        verify_candidate_gates(report, frame, y, p_rf, p_f3, batch_versions, errors)

    # Verify every fixed-weight OOF combination directly from its preregistered members.
    for batch in ("B003", "B006", "B007", "B008", "B009"):
        plan = json.loads((run_dir / "batches" / batch / "plan.json").read_text(encoding="utf-8"))
        for item in plan["versions"]:
            expected_members = []
            for member in item["params"]["members"]:
                reference = f"{member['batch']}/{member['version']}"
                if reference not in batch_versions:
                    errors.append(f"fusion member missing: {batch}/{item['version']}/{reference}")
                    continue
                expected_members.append(batch_versions[reference]["p"])
            if len(expected_members) != len(item["params"]["members"]):
                continue
            fused = logit_mean(expected_members, item["params"]["weights"], item["params"].get("logit_clip", 1e-6))
            actual = batch_versions[f"{batch}/{item['version']}"]["p"]
            diff = float(np.max(np.abs(fused - actual)))
            if diff > 1e-12:
                errors.append(f"fixed OOF combination mismatch: {batch}/{item['version']} ({diff})")

    # Replay comparison is recalculated from the saved row-aligned OOF arrays.
    p_ref = batch_versions[REPLAY[0]]["p"]
    p_replay = batch_versions[REPLAY[1]]["p"]
    max_diff = float(np.max(np.abs(p_ref - p_replay)))
    saved_replay = report.get("replay", {})
    if saved_replay.get("reference_oof_sha256") != e.sha(batch_versions[REPLAY[0]]["path"]): errors.append("replay reference hash mismatch")
    if saved_replay.get("replay_oof_sha256") != e.sha(batch_versions[REPLAY[1]]["path"]): errors.append("replay OOF hash mismatch")
    if not np.isclose(saved_replay.get("max_abs_probability_difference", np.nan), max_diff, atol=1e-12, rtol=0):
        errors.append("replay difference does not reproduce")

    for name in ("D0_iteration_curve.png", "D1_seed_stability.png", "D2_anchor_gates.png", "D3_ablation_summary.png"):
        if not (run_dir / "figures" / name).is_file(): errors.append(f"missing E3 visualization: {name}")
    return {
        "task": "FEAT-014", "stage": "E3", "status": "fail" if errors else "pass",
        "errors": errors, "version_count": len(actual_versions),
        "candidate_set": report.get("candidate_set", []),
        "seed_routes_recalculated": sorted(recomputed_routes),
        "input_lock_sha256": e.sha(run_dir / "input_lock.json"),
        "human_visual_review": "pending; machine audit does not represent user acceptance",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = audit(args.run_dir.expanduser().resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.write:
        e.json_dump(args.run_dir / "independent_audit_e3.json", result)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
