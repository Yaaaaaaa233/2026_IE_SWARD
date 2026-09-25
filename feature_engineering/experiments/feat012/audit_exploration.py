#!/usr/bin/env python3
"""Independently recalculate FEAT-012/013 stored OOF metrics and fusion formulas."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oof(path: Path, frame: pd.DataFrame) -> np.ndarray:
    d = pd.read_csv(path, dtype={"sample_id": "string", "gpsno": "string"})
    for key in ("sample_id", "gpsno"):
        if not d[key].astype(str).equals(frame[key].astype(str)):
            raise ValueError(f"{path.name}: {key} order or values differ")
    for key in ("y", "fold"):
        if not d[key].astype(int).equals(frame[key].astype(int)):
            raise ValueError(f"{path.name}: {key} differs")
    pcol = "probability" if "probability" in d else next(c for c in d if c.startswith("p_"))
    p = d[pcol].astype(float).to_numpy()
    if len(p) != len(frame) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"{path.name}: invalid OOF probabilities")
    return p


def paired(y: np.ndarray, p: np.ndarray, anchor: np.ndarray) -> tuple[float, float, float]:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(42)
    draws = np.empty(2000)
    for i in range(len(draws)):
        ix = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        draws[i] = roc_auc_score(y[ix], p[ix]) - roc_auc_score(y[ix], anchor[ix])
    return (float(roc_auc_score(y, p) - roc_auc_score(y, anchor)),
            float(np.quantile(draws, .025)), float(np.quantile(draws, .975)))


def expected_fusion(name: str, item: dict, run: Path, previous: Path | None,
                    frame: pd.DataFrame, cache: dict[str, np.ndarray]) -> np.ndarray | None:
    if item.get("kind") == "equal_logit_fusion" or item.get("family") == "equal_logit_fusion":
        members, weights = item["members"], item.get("weights", [1.0] * len(item["members"]))
    elif item.get("kind") == "fusion":
        members, weights = item["members"], item["weights"]
    elif item.get("kind") == "seed_ensemble":
        members, weights = item["members"], item.get("weights", [1.0] * len(item["members"]))
    else:
        return None
    values = []
    for member in members:
        key = member
        if key in cache:
            values.append(cache[key]); continue
        if key.endswith("_s42") and previous is not None:
            base = key[:-4]
            value = oof(previous / f"oof_{base}.csv", frame)
        else:
            source = run / f"oof_{key}.csv"
            if not source.exists():
                if key == "full_ebm_a" and previous is not None:
                    source = previous / "oof_full_ebm_a.csv"
                else:
                    raise FileNotFoundError(f"missing fusion member OOF: {key}")
            value = oof(source, frame)
        cache[key] = value
        values.append(value)
    w = np.asarray(weights, dtype=float); w /= w.sum()
    logits = [np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)) for p in values]
    z = sum(weight * logit for weight, logit in zip(w, logits))
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--previous-run", type=Path)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    run = args.run.expanduser().resolve()
    frame = pd.read_csv(args.input.expanduser().resolve(), dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)
    prereg = json.loads((run / "preregistration.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if sha(args.input.expanduser().resolve()) != prereg["input_sha256"]:
        raise ValueError("input hash differs from preregistration")
    if len(summary["versions"]) != prereg.get("total_max_versions", prereg.get("max_report_versions")):
        raise ValueError("stored version count differs from preregistered version count")
    previous = args.previous_run.expanduser().resolve() if args.previous_run else None
    anchor_path = (previous / "oof_full_ebm_a.csv") if summary["task"] == "FEAT-013" else run / "oof_full_ebm_a.csv"
    anchor = oof(anchor_path, frame)
    y, folds, ids = frame.y.astype(int).to_numpy(), frame.fold.astype(int).to_numpy(), frame.sample_id.astype(str).to_numpy()
    cache: dict[str, np.ndarray] = {"full_ebm_a": anchor}
    errors = []
    details = {}
    for name, item in summary["versions"].items():
        p = oof(run / f"oof_{name}.csv", frame)
        cache[name] = p
        if name == "full_ebm_a":
            anchor = p
        expected = item["metrics"]
        auc = float(roc_auc_score(y, p)); apv = float(average_precision_score(y, p)); brier = float(brier_score_loss(y, p))
        order = np.lexsort((ids, -p))[:min(100, len(y))]
        recall = float(y[order].sum() / y.sum())
        delta, lo, hi = paired(y, p, anchor)
        calculated = {"auc": auc, "ap": apv, "brier": brier, "recall_at_100": recall,
                      "delta_auc_vs_full_ebm_a": delta, "paired_ci_95": [lo, hi]}
        for key, value in calculated.items():
            target = expected[key]
            if isinstance(value, list):
                okay = np.allclose(value, target, atol=1e-12, rtol=0)
            else:
                okay = abs(value - float(target)) <= 1e-12
            if not okay:
                errors.append(f"{name}: {key} stored={target} recomputed={value}")
        folds_auc = {str(f): float(roc_auc_score(y[folds == f], p[folds == f])) for f in sorted(np.unique(folds))}
        if not np.allclose(list(folds_auc.values()), list(expected["fold_auc"].values()), atol=1e-12, rtol=0):
            errors.append(f"{name}: fold AUC mismatch")
        fused = expected_fusion(name, item, run, previous, frame, cache)
        fusion_error = None
        if fused is not None:
            fusion_error = float(np.max(np.abs(fused - p)))
            if fusion_error > 1e-12:
                errors.append(f"{name}: fusion formula max absolute error {fusion_error}")
        details[name] = {**calculated, "max_fusion_abs_error": fusion_error}
    for filename in ["metrics.json"] + [f"oof_{n}.csv" for n in summary["versions"]]:
        expected_hash = manifest.get("output_sha256", {}).get(filename)
        if expected_hash and sha(run / filename) != expected_hash:
            errors.append(f"manifest output hash mismatch: {filename}")
    result = {"run": str(run), "task": summary["task"], "rows": len(frame), "versions_checked": len(details),
             "checks": ["input hash", "sample/gpsno/label/fold alignment", "probability domain", "AUC/AP/Recall@100/Brier", "fold AUC", "paired bootstrap interval", "registered fusion formulas", "OOF/metrics output hashes"],
             "status": "fail" if errors else "pass", "errors": errors, "versions": details,
             "note": "Independent metric recomputation; this does not remove repeated-selection bias or validate official future-period performance."}
    print(json.dumps({k: v for k, v in result.items() if k != "versions"}, ensure_ascii=False, indent=2))
    if args.write:
        target = run / "independent_audit.json"
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"audit saved: {target}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
