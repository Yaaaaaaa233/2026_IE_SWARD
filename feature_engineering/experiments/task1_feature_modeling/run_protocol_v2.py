#!/usr/bin/env python3
"""Run the preregistered FEAT-009 protocol-v2 nested comparison in controlled outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from modeling import CANDIDATES, nested_oof

REPO = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").exists())
VERSION = "feat009-protocol-v2-candidates-20260924-r1"
META = {"sample_id", "gpsno", "y", "fold", "label_version", "split_version", "label_window", "horizon_days"}
BLOCKED_PREFIXES = ("f3_profile_",)
BLOCKED_EXACT = {
    "monthly_avg_mileage", "monthly_avg_hours", "monthly_avg_stops", "highway_ratio",
    "morning_ratio", "dusk_ratio", "night_hours_ratio", "night_mileage_ratio", "energy_type",
    "cohort_fallback", "f2_prior_incident_x_night", "f2_prior_incident_x_highway",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def recall_at_k(y: np.ndarray, p: np.ndarray, ids: np.ndarray, k: int = 100) -> float:
    idx = np.lexsort((ids.astype(str), -p))[: min(k, len(y))]
    return float(y[idx].sum() / y.sum())


def paired_auc(y: np.ndarray, a: np.ndarray, b: np.ndarray, *, n: int = 2000, seed: int = 42) -> dict:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    boot = np.empty(n)
    for i in range(n):
        ix = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        boot[i] = roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix])
    return {"delta_auc": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)),
            "bootstrap_n": n, "bootstrap_seed": seed}


def make_report(out: Path, y: np.ndarray, ids: np.ndarray, rf: np.ndarray,
                predictions: dict[str, np.ndarray], diagnostics: dict) -> dict:
    comparison = {}
    rf_auc = float(roc_auc_score(y, rf))
    rf_brier = float(brier_score_loss(y, rf))
    rf_recall = recall_at_k(y, rf, ids)
    for name, p in predictions.items():
        delta = paired_auc(y, p, rf)
        comparison[name] = {
            **delta, "auc": float(roc_auc_score(y, p)), "rf_auc": rf_auc,
            "recall_at_100": recall_at_k(y, p, ids), "rf_recall_at_100": rf_recall,
            "recall_at_100_delta": recall_at_k(y, p, ids) - rf_recall,
            "brier": float(brier_score_loss(y, p)), "rf_brier": rf_brier,
            "brier_delta": float(brier_score_loss(y, p) - rf_brier),
        }
        comparison[name]["exceeds_baseline"] = bool(delta["delta_auc"] > 0 and delta["ci_low"] > 0)
        comparison[name]["meaningful_gain"] = bool(delta["ci_low"] >= 0.01)
        comparison[name]["auxiliary_pass"] = bool(comparison[name]["recall_at_100_delta"] >= -0.02 and comparison[name]["brier_delta"] <= 0.005)
        comparison[name]["overall_pass"] = bool(comparison[name]["exceeds_baseline"] and comparison[name]["auxiliary_pass"])
    summary = {
        "run_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "label_v2_record_count_20260923", "split_version": "split_v2_record_strat5_seed42",
        "feature_window": "[2026-06-01,2026-06-21)", "label_window": "[2026-06-21,2026-07-31)",
        "timezone": "Asia/Shanghai", "rows": int(len(y)), "positives": int(y.sum()),
        "main_comparator": "new_rf_anchor", "main_candidates": ["F1", "F3"],
        "criteria": {"paired_bootstrap": "vehicle-stratified percentile, 2000, 95%, seed 42",
                     "exceeds_baseline": "delta AUC > 0 and 95% CI lower bound > 0",
                     "meaningful_gain": "95% CI lower bound >= 0.01",
                     "recall_at_100_delta_min": -0.02, "brier_delta_max": 0.005},
        "baseline": {"auc": rf_auc, "recall_at_100": rf_recall, "brier": rf_brier},
        "comparisons": comparison,
        "diagnostics": {k: [{kk: vv for kk, vv in d.items() if kk not in {"fit_scopes", "selected_features"}} for d in v]
                        for k, v in diagnostics.items()},
        "decision_note": "Development-period proxy OOF; not independent official test performance. Human review pending.",
    }
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diag_rows = []
    for name, rows in diagnostics.items():
        for row in rows:
            diag_rows.append({"dataset": name, "outer_fold": row["outer_fold"], "train_rows": row["train_rows"],
                              "test_rows": row["test_rows"], "test_positive": row["test_positive"],
                              "selected_candidate": row["selected_candidate"], "outer_auc": row["outer_auc"],
                              **{f"inner_{k}": v for k, v in row["inner_auc"].items()}})
    pd.DataFrame(diag_rows).to_csv(out / "fold_selection.csv", index=False, lineterminator="\n")
    for name, p in predictions.items():
        pd.DataFrame({"sample_id": ids, "y": y, "p_" + name.lower(): p}).to_csv(out / f"oof_{name.lower()}.csv", index=False, lineterminator="\n")
    return summary


def render(out: Path, metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(metrics["comparisons"])
    comps = [metrics["comparisons"][n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    yloc = np.arange(len(names))
    centers = np.array([c["delta_auc"] for c in comps])
    lo = np.array([c["ci_low"] for c in comps])
    hi = np.array([c["ci_high"] for c in comps])
    ax.errorbar(centers, yloc, xerr=[centers-lo, hi-centers], fmt="o", capsize=4, color="#23658a")
    ax.axvline(0, color="#444", lw=1)
    ax.axvline(.01, color="#d88923", lw=1, ls="--")
    ax.set_yticks(yloc, names)
    ax.set_xlabel("OOF ΔAUC vs new RF (95% vehicle-paired bootstrap CI)")
    ax.set_title(f"Protocol-v2 proxy comparison | N={metrics['rows']}, positives={metrics['positives']}\n0 = no gain; +0.01 = meaningful-gain reference")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "delta_auc.png", dpi=160)
    plt.close(fig)
    folds = pd.read_csv(out / "fold_selection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ds in zip(axes, ("F1", "F3")):
        part = folds[folds.dataset == ds]
        ax.bar(part.outer_fold.astype(str), part.outer_auc, color="#649578")
        ax.set_title(f"{ds}: outer-fold diagnostic")
        ax.set_xlabel("held-out fold")
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUC (not independent per-fold result)")
    fig.suptitle("Nested selection diagnostics; one pooled OOF comparison is primary")
    fig.tight_layout()
    fig.savefig(out / "fold_auc.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(REPO / "outputs/feat-009/v2/y1"))
    ap.add_argument("--rf-oof", default=str(REPO / "outputs/feat-009/v2/y2/new_rf_anchor/task1_baseline_oof_predictions.csv"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--confirm-preregistration", action="store_true")
    args = ap.parse_args()
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    inputs = Path(args.inputs)
    prereg = REPO / "outputs/feat-009/v2/y2/pre_registration.json"
    if not args.confirm_preregistration or not prereg.is_file():
        raise RuntimeError("candidate run requires a saved pre_registration.json and --confirm-preregistration")
    reg = json.loads(prereg.read_text(encoding="utf-8"))
    if reg.get("status") not in {"locked_before_candidate_oof", "locked_before_valid_candidate_oof"} or \
       reg.get("candidate_pool") != [c["name"] for c in CANDIDATES]:
        raise RuntimeError("pre-registration does not match the frozen candidate pool")
    tables = {name: pd.read_csv(inputs / f"{name.lower()}_model_input.csv", dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)
              for name in ("F0", "F1", "F3")}
    rf_path = Path(args.rf_oof).expanduser().resolve()
    for key, path in (("f0_model_input.csv", inputs / "f0_model_input.csv"),
                      ("f1_model_input.csv", inputs / "f1_model_input.csv"),
                      ("f3_model_input.csv", inputs / "f3_model_input.csv"),
                      ("new_rf_oof", rf_path)):
        if reg.get("source_inputs_sha256", {}).get(key) != sha(path):
            raise RuntimeError(f"pre-registration input fingerprint differs: {key}")
    rf = pd.read_csv(rf_path, dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)
    rf_col = "p_random_forest"
    if rf_col not in rf:
        raise ValueError(f"new RF OOF lacks {rf_col}")
    ref = tables["F0"]
    for name, t in tables.items():
        if t.sample_id.duplicated().any() or t.gpsno.duplicated().any():
            raise ValueError(f"{name}: duplicated vehicle/sample")
        if not t.sample_id.astype(str).equals(ref.sample_id.astype(str)):
            raise ValueError(f"{name}: order differs from frozen F0")
        for c in ("y", "fold", "label_version", "split_version"):
            if not t[c].astype(str).equals(ref[c].astype(str)):
                raise ValueError(f"{name}: {c} differs from F0")
        blocked = [c for c in t.columns if c in BLOCKED_EXACT or c.startswith(BLOCKED_PREFIXES) or "_cohort_" in c]
        if blocked:
            raise ValueError(f"{name}: future portrait/dependent features remain: {blocked}")
    if not rf.sample_id.astype(str).equals(ref.sample_id.astype(str)):
        raise ValueError("RF OOF sample order differs from frozen inputs")
    for c in ("y", "fold"):
        if not rf[c].astype(str).equals(ref[c].astype(str)):
            raise ValueError(f"RF OOF {c} differs from v2 inputs")
    y = ref.y.astype(int).to_numpy()
    folds = ref.fold.astype(int).to_numpy()
    ids = ref.sample_id.astype(str).to_numpy()
    groups = ref[["sample_id", "gpsno"]].copy()
    baseline = rf[rf_col].astype(float).to_numpy()
    predictions, diagnostics = {}, {}
    for name in ("F1", "F3"):
        table = tables[name]
        x = table[[c for c in table.columns if c not in META]].copy()
        started = time.monotonic()
        predictions[name], diagnostics[name] = nested_oof(name, x, y, folds, groups, cohort_sources=[], cohort_values=None)
        diagnostics[name] = [{**d, "runtime_seconds": None} for d in diagnostics[name]]
        print(f"completed {name} in {time.monotonic()-started:.1f}s", flush=True)
    metrics = make_report(out, y, ids, baseline, predictions, diagnostics)
    render(out, metrics)
    packages = {}
    for name in ("interpret", "lightgbm", "numpy", "pandas", "scikit-learn", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    files = sorted(p for p in out.iterdir() if p.is_file())
    manifest = {"run_version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "code_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "worktree_status_sha256": hashlib.sha256(subprocess.run(["git", "-C", str(REPO), "diff", "--binary"], capture_output=True).stdout +
                     subprocess.run(["git", "-C", str(REPO), "diff", "--binary", "--cached"], capture_output=True).stdout).hexdigest(),
                "python": platform.python_version(), "packages": packages, "seed": 42,
                "inputs": {"f0": sha(inputs / "f0_model_input.csv"), "f1": sha(inputs / "f1_model_input.csv"),
                           "f3": sha(inputs / "f3_model_input.csv"), "rf_oof": sha(rf_path), "pre_registration": sha(prereg)},
                "outputs": {p.name: sha(p) for p in files}, "human_review": "pending"}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "comparisons": metrics["comparisons"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
