#!/usr/bin/env python3
"""Build pre-registered fixed-weight combinations from aligned FEAT-014 OOFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from feature_engineering.experiments.feat014 import run as e  # noqa: E402


def logit_mean(probabilities: list[np.ndarray], weights: list[float], clip: float = 1e-6) -> np.ndarray:
    if not probabilities or len(probabilities) != len(weights):
        raise ValueError("each probability member requires one weight")
    w = np.asarray(weights, dtype=float)
    if not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("fusion weights must be finite, nonnegative, and sum to one")
    if not np.isfinite(clip) or not 0 < clip < 0.5:
        raise ValueError("logit clip must be in (0, 0.5)")
    arrays = [np.asarray(p, dtype=float) for p in probabilities]
    if any(p.shape != arrays[0].shape for p in arrays):
        raise ValueError("fusion members have different row counts")
    if any(not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() for p in arrays):
        raise ValueError("fusion member probabilities must be finite and in [0,1]")
    logits = [np.log(np.clip(p, clip, 1 - clip) / (1 - np.clip(p, clip, 1 - clip))) for p in arrays]
    z = sum(weight * logit for weight, logit in zip(w, logits, strict=True))
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def load_member(run_dir: Path, reference: dict[str, str], frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    batch, version = reference["batch"], reference["version"]
    directory = run_dir / "batches" / batch / version
    result_path, oof_path = directory / "result.json", directory / "oof.csv"
    if not result_path.is_file() or not oof_path.is_file():
        raise FileNotFoundError(f"fusion member is incomplete: {batch}/{version}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if e.sha(oof_path) != result.get("oof_sha256"):
        raise ValueError(f"fusion member OOF hash mismatch: {batch}/{version}")
    return e.read_oof(oof_path, frame, "probability"), result


def run_batch(run_dir: Path, batch_name: str) -> dict[str, Any]:
    lock, frame, _, _, _, y, folds, p_rf, p_f3 = e.load_run_inputs(run_dir)
    batch_dir = run_dir / "batches" / batch_name
    plan_path = batch_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("input_lock_sha256") != e.sha(run_dir / "input_lock.json"):
        raise ValueError("fusion plan is not bound to the E0 input lock")
    if plan.get("status") != "locked_before_candidate_oof":
        raise ValueError("fusion plan was not preregistered")
    metrics_path = batch_dir / "metrics.json"
    if metrics_path.exists() and json.loads(metrics_path.read_text()).get("status") == "complete":
        raise FileExistsError("fusion batch is already complete")
    ids = frame.sample_id.astype(str).to_numpy()
    gps = frame.gpsno.astype(str).to_numpy()
    results = {}
    for item in plan["versions"]:
        name = item["version"]
        version_dir = batch_dir / name
        if (version_dir / "result.json").exists():
            saved = json.loads((version_dir / "result.json").read_text(encoding="utf-8"))
            if e.sha(version_dir / "oof.csv") != saved["oof_sha256"]:
                raise ValueError(f"existing fusion OOF changed: {name}")
            results[name] = saved
            continue
        version_dir.mkdir(parents=True, exist_ok=True)
        params = item["params"]
        members, member_results = [], []
        member_records = []
        for reference in params["members"]:
            p, result = load_member(run_dir, reference, frame)
            members.append(p)
            member_results.append(result)
            member_records.append({
                **reference,
                "oof_sha256": result["oof_sha256"],
                "folds_sha256": hashlib.sha256("\n".join(map(str, frame.fold.astype(int))).encode()).hexdigest(),
            })
        started = time.monotonic()
        prediction = logit_mean(members, params["weights"], params.get("logit_clip", 1e-6))
        elapsed = time.monotonic() - started
        oof_path = version_dir / "oof.csv"
        pd.DataFrame({"sample_id": ids, "gpsno": gps, "y": y, "fold": folds,
                      "probability": prediction}).to_csv(oof_path, index=False, lineterminator="\n")
        metrics = e.score_version(y, prediction, ids, folds, p_rf, p_f3, p_f3)
        fold_audit = []
        for outer in sorted(np.unique(folds)):
            train_ids = frame.loc[folds != outer, "sample_id"].astype(str).tolist()
            fold_audit.append({
                "outer_fold": int(outer), "train_rows": int((folds != outer).sum()),
                "test_rows": int((folds == outer).sum()), "train_positive": int(y[folds != outer].sum()),
                "test_positive": int(y[folds == outer].sum()),
                "train_sample_ids_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
                "selected_columns": len(members), "selected_C": None, "inner_auc_by_C": {},
                "preprocessor_fit_scope": "no fitted estimator; fixed-weight combination of each member's matching outer-fold OOF prediction",
                "test_train_overlap": False,
            })
        columns = [f"OOF::{m['batch']}/{m['version']}" for m in params["members"]]
        output = {
            "task": "FEAT-014", "batch": batch_name, "version": name,
            "hypothesis": item["hypothesis"], "parent": "fixed_f3_ebm_a",
            "family": "fusion", "view": "fixed_logit_oof_combination",
            "feature_count": len(columns), "column_names": columns,
            "columns_sha256": hashlib.sha256("\n".join(columns).encode()).hexdigest(),
            "params": params, "seed": 42, "fit_seconds": elapsed,
            "metrics": metrics, "fold_audit": fold_audit,
            "members": member_records, "oof_sha256": e.sha(oof_path),
            "implementation_sha256": e.sha(Path(__file__).resolve()),
            "status": "complete", "attempt": 1,
            "code_commit": lock["code_commit"],
        }
        e.json_dump(version_dir / "result.json", output)
        results[name] = output
    doc = {
        "task": "FEAT-014", "batch": batch_name,
        "status": "complete" if len(results) == len(plan["versions"]) else "partial",
        "input_lock_sha256": e.sha(run_dir / "input_lock.json"),
        "versions": results, "failures": [],
        "history_reference": "Fixed RF@F0v2 and F3 EBM-A; fixed logit fusion members are registered before combination.",
        "selection_bias": "Fusion members are selected from repeated development OOF; the result is exploratory and is not independent confirmation.",
    }
    e.json_dump(metrics_path, doc)
    pd.DataFrame([{"version": n, "family": "fusion", "view": "fixed_logit_oof_combination",
                   "feature_count": d["feature_count"], "auc": d["metrics"]["auc"],
                   "delta_vs_f3": d["metrics"]["delta_vs_f3_ebm_a"]["point"],
                   "ci_low": d["metrics"]["delta_vs_f3_ebm_a"]["ci95"][0],
                   "ci_high": d["metrics"]["delta_vs_f3_ebm_a"]["ci95"][1],
                   "delta_vs_rf": d["metrics"]["delta_vs_rf_f0v2"]["point"],
                   "recall_at_100": d["metrics"]["recall_at_100"],
                   "brier": d["metrics"]["brier"], "seconds": d["fit_seconds"]}
                  for n, d in results.items()]).to_csv(batch_dir / "leaderboard.csv", index=False, lineterminator="\n")
    print(json.dumps({"batch": batch_name, "versions": len(results), "failures": 0,
                      "output": str(batch_dir), "status": doc["status"]}, ensure_ascii=False, indent=2))
    return doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()
    run_batch(args.run_dir.expanduser().resolve(), args.batch)


if __name__ == "__main__":
    main()
