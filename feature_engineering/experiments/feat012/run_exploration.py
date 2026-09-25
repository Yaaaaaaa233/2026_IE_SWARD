#!/usr/bin/env python3
"""Bounded, task-local iterative exploration on locked protocol-v2 F3 inputs."""
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
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
MODELING = ROOT / "feature_engineering/experiments/task1_feature_modeling"
if str(MODELING) not in sys.path:
    sys.path.insert(0, str(MODELING))
from modeling import fit_transformer  # noqa: E402

META_COLUMNS = {
    "sample_id", "gpsno", "y", "fold", "as_of", "window_start", "window_end",
    "lookback_days", "horizon_days", "label_window", "label_status", "label_version",
    "split_version", "source_version", "feature_version", "feature_version_f1",
    "feature_version_f2", "feature_version_f3", "cohort_fallback",
}
BLOCKED_EXACT = {
    "monthly_avg_mileage", "monthly_avg_hours", "monthly_avg_stops", "highway_ratio",
    "morning_ratio", "dusk_ratio", "night_hours_ratio", "night_mileage_ratio", "energy_type",
    "cohort_fallback", "f2_prior_incident_x_night", "f2_prior_incident_x_highway",
}
EXPECTED_PROTOCOL = "label_v2_record_count_20260923"
EXPECTED_SPLIT = "split_v2_record_strat5_seed42"
BASE_ARMS: tuple[dict[str, Any], ...] = (
    {"name": "full_ebm_a", "view": "full", "family": "ebm", "params": {"interactions": 0, "max_bins": 64, "min_samples_leaf": 10}},
    {"name": "full_ebm_c", "view": "full", "family": "ebm", "params": {"interactions": 0, "max_bins": 32, "min_samples_leaf": 20}},
    {"name": "full_ebm_d", "view": "full", "family": "ebm", "params": {"interactions": 0, "max_bins": 24, "min_samples_leaf": 30}},
    {"name": "full_rf", "view": "full", "family": "rf", "params": {"n_estimators": 500, "max_features": 0.5, "min_samples_leaf": 4, "class_weight": "balanced_subsample"}},
    {"name": "full_histgb", "view": "full", "family": "histgb", "params": {"max_iter": 120, "learning_rate": 0.05, "max_leaf_nodes": 7, "min_samples_leaf": 20, "l2_regularization": 2.0}},
    {"name": "slim_ebm_a", "view": "slim_night", "family": "ebm", "params": {"interactions": 0, "max_bins": 64, "min_samples_leaf": 10}},
    {"name": "slim_ebm_c", "view": "slim_night", "family": "ebm", "params": {"interactions": 0, "max_bins": 32, "min_samples_leaf": 20}},
    {"name": "history_ebm_a", "view": "history", "family": "ebm", "params": {"interactions": 0, "max_bins": 64, "min_samples_leaf": 10}},
    {"name": "history_ebm_c", "view": "history", "family": "ebm", "params": {"interactions": 0, "max_bins": 32, "min_samples_leaf": 20}},
    {"name": "no_traj_ebm_a", "view": "no_traj", "family": "ebm", "params": {"interactions": 0, "max_bins": 64, "min_samples_leaf": 10}},
    {"name": "no_accident_ebm_a", "view": "no_accident", "family": "ebm", "params": {"interactions": 0, "max_bins": 64, "min_samples_leaf": 10}},
    {"name": "no_legacy_ebm_a", "view": "no_legacy", "family": "ebm", "params": {"interactions": 0, "max_bins": 64, "min_samples_leaf": 10}},
    {"name": "event_signal_ebm_a", "view": "event_signal", "family": "ebm", "params": {"interactions": 0, "max_bins": 64, "min_samples_leaf": 10}},
)
HISTORY_FAMILIES = {"evt", "f1", "f2", "f2r2", "accident", "f3_hist", "f3_chain", "f3_event", "f3_fatigue", "f3_score"}
EVENT_SIGNAL_FAMILIES = {"evt", "accident", "f3_hist", "f3_chain", "f3_event", "f3_fatigue", "f3_score"}
LEGACY_FAMILIES = {"f1", "f2", "f2r2"}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def load_locked_inputs(input_path: Path, manifest_path: Path, ledger_path: Path):
    input_path, manifest_path, ledger_path = input_path.resolve(), manifest_path.resolve(), ledger_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(input_path, dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)
    ledger = pd.read_csv(ledger_path).fillna("")
    expected = manifest.get("outputs", {}).get("f3_model_input.csv", {}).get("sha256")
    if not expected or sha_file(input_path) != expected:
        raise ValueError("F3 input SHA-256 does not match the locked manifest")
    if manifest.get("protocol") != EXPECTED_PROTOCOL or manifest.get("split_version") != EXPECTED_SPLIT:
        raise ValueError("label or split version differs from the official v2 lock")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any() or frame.gpsno.isna().any() or frame.gpsno.duplicated().any():
        raise ValueError("sample_id/gpsno must be present and unique")
    for col in ("y", "fold", "label_version", "split_version"):
        if col not in frame:
            raise ValueError(f"locked input lacks {col}")
    y = pd.to_numeric(frame.y, errors="raise").astype(int).to_numpy()
    folds = pd.to_numeric(frame.fold, errors="raise").astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1} or set(np.unique(folds)) != {0, 1, 2, 3, 4}:
        raise ValueError("expected binary labels and frozen folds 0..4")
    if not frame.label_version.astype(str).eq(EXPECTED_PROTOCOL).all() or not frame.split_version.astype(str).eq(EXPECTED_SPLIT).all():
        raise ValueError("row-level protocol differs from the locked manifest")
    pred_rows = ledger[(ledger.role == "predictor") & (ledger.visibility_decision == "retain_candidate") & (ledger.in_full_f3 == True)]
    features = pred_rows.column.astype(str).tolist()
    if not features or len(features) != len(set(features)):
        raise ValueError("accepted predictor ledger is empty or duplicated")
    data_columns = [c for c in frame.columns if c not in META_COLUMNS]
    if features != data_columns:
        missing, extra = sorted(set(data_columns) - set(features)), sorted(set(features) - set(data_columns))
        raise ValueError(f"ledger/input predictor mismatch; missing={missing[:10]}, extra={extra[:10]}")
    blocked = [c for c in features if c in BLOCKED_EXACT or "_cohort_" in c or any(k in c.lower() for k in ("future", "target", "after_as_of"))]
    if blocked:
        raise ValueError(f"blocked or future-derived features found: {blocked}")
    slim = ledger[(ledger.role == "predictor") & (ledger.visibility_decision == "retain_candidate") & (ledger.in_slim_f3 == True)].column.astype(str).tolist()
    family = pred_rows.set_index("column").family.astype(str).to_dict()
    if set(slim) - set(features):
        raise ValueError("slim view includes a feature outside full F3")
    views = {
        "full": features,
        "slim_night": slim,
        "history": [c for c in features if family[c] in HISTORY_FAMILIES],
        "no_traj": [c for c in features if family[c] not in {"f3_traj", "traj"}],
        "no_accident": [c for c in features if family[c] != "accident"],
        "no_legacy": [c for c in features if family[c] not in LEGACY_FAMILIES],
        "event_signal": [c for c in features if family[c] in EVENT_SIGNAL_FAMILIES],
    }
    if any(not cols for cols in views.values()):
        raise ValueError("one or more registered feature views are empty")
    groups = frame[["sample_id", "gpsno"]].copy()
    return frame, manifest, ledger, views, groups, y, folds, features


def prepare(args) -> None:
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    frame, manifest, ledger, views, _, y, folds, features = load_locked_inputs(Path(args.input), Path(args.manifest), Path(args.ledger))
    reference = Path(args.reference_oof).expanduser().resolve()
    prereg = {
        "task": "FEAT-012", "status": "locked_before_candidate_oof", "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "task-local development exploration; repeated use of the same 500 vehicles; not a global baseline or independent test",
        "protocol": EXPECTED_PROTOCOL, "split_version": EXPECTED_SPLIT,
        "feature_window": "[2026-06-01,2026-06-21)", "label_window": "[2026-06-21,2026-07-31)",
        "input_sha256": sha_file(Path(args.input).expanduser().resolve()),
        "manifest_sha256": sha_file(Path(args.manifest).expanduser().resolve()),
        "ledger_sha256": sha_file(Path(args.ledger).expanduser().resolve()),
        "reference_oof_sha256": sha_file(reference),
        "rows": int(len(frame)), "positive_count": int(y.sum()), "fold_ids": sorted(map(int, np.unique(folds))),
        "feature_views": {name: {"columns": len(cols), "sha256": sha_bytes("\n".join(cols).encode())} for name, cols in views.items()},
        "base_arms": list(BASE_ARMS), "round_1_max_versions": len(BASE_ARMS),
        "round_2_rule": "For each feature view, retain the highest pooled OOF AUC arm (stable name tie-break); rank distinct views by that score; fuse the top four views pairwise, equal-weight on clipped logit scale. This is adaptive and exploratory.",
        "round_2_max_versions": 6, "total_max_versions": len(BASE_ARMS) + 6,
        "metrics": {"primary": "pooled OOF AUC", "paired_uncertainty": "vehicle-stratified paired bootstrap percentile 95%, 2000 draws, seed 42", "secondary": ["average precision", "Recall@100", "Brier", "five outer-fold AUC diagnostics"]},
        "model_seed": 42, "preprocessing": "per outer fold only; fit_transformer on outer training rows; cohort-derived sources disabled",
        "candidate_scope": "existing protocol-v2 F3 predictors only; no raw feature generation; no 61-day final-window training",
        "human_review": "pending",
    }
    (out / "preregistration.json").write_text(json.dumps(prereg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"locked": True, "output": str(out), "rows": len(frame), "predictors": len(features), "views": {k: len(v) for k, v in views.items()}, "versions_cap": prereg["total_max_versions"]}, ensure_ascii=False, indent=2))


def make_estimator(spec: dict[str, Any], seed: int):
    family, p = spec["family"], spec["params"]
    if family == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(**p, random_state=seed, n_jobs=1)
    if family == "rf":
        return RandomForestClassifier(**p, random_state=seed, n_jobs=1)
    if family == "histgb":
        return HistGradientBoostingClassifier(**p, random_state=seed)
    raise ValueError(f"unknown model family: {family}")


def fit_oof(name: str, spec: dict[str, Any], view_columns: list[str], frame: pd.DataFrame,
            groups: pd.DataFrame, y: np.ndarray, folds: np.ndarray, seed: int = 42) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    x = frame[view_columns]
    pred = np.full(len(y), np.nan, dtype=float)
    audit = []
    started = time.monotonic()
    for fold in sorted(np.unique(folds)):
        tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if set(np.unique(y[tr])) != {0, 1} or set(np.unique(y[te])) != {0, 1}:
            raise ValueError(f"fold {fold} lacks both classes in train or validation")
        transformer = fit_transformer(x.iloc[tr], groups.iloc[tr], cohort_sources=[])
        xtr = transformer.transform(x.iloc[tr], groups.iloc[tr])
        xte = transformer.transform(x.iloc[te], groups.iloc[te])
        model = make_estimator(spec, seed)
        model.fit(xtr, y[tr])
        class_index = list(model.classes_).index(1)
        pred[te] = model.predict_proba(xte)[:, class_index]
        audit.append({"fold": int(fold), "train_rows": int(len(tr)), "test_rows": int(len(te)),
                      "train_positives": int(y[tr].sum()), "test_positives": int(y[te].sum()),
                      "fit_sample_ids_sha256": sha_bytes("\n".join(sorted(groups.iloc[tr].sample_id.astype(str))).encode()),
                      "selected_columns": len(transformer.selected), "auc": float(roc_auc_score(y[te], pred[te]))})
    if not np.isfinite(pred).all() or ((pred < 0) | (pred > 1)).any():
        raise RuntimeError(f"{name}: invalid or incomplete OOF predictions")
    return pred, audit, time.monotonic() - started


def recall_at_100(y: np.ndarray, p: np.ndarray, ids: np.ndarray) -> float:
    order = np.lexsort((ids.astype(str), -p))[: min(100, len(y))]
    return float(y[order].sum() / y.sum())


def paired_auc(y: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int = 42, n: int = 2000) -> dict[str, float]:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    values = np.empty(n)
    for i in range(n):
        idx = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        values[i] = roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
    delta = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    return {"delta_auc": delta, "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975))}


def score(y: np.ndarray, p: np.ndarray, anchor: np.ndarray, ids: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    paired = paired_auc(y, p, anchor)
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "recall_at_100": recall_at_100(y, p, ids), "brier": float(brier_score_loss(y, p)),
            "delta_auc_vs_full_ebm_a": paired["delta_auc"], "paired_ci_95": [paired["ci_low"], paired["ci_high"]],
            "fold_auc": {str(f): float(roc_auc_score(y[folds == f], p[folds == f])) for f in sorted(np.unique(folds))}}


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def render(out: Path, result: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(result["versions"])
    names.sort(key=lambda n: result["versions"][n]["metrics"]["delta_auc_vs_full_ebm_a"])
    fig, ax = plt.subplots(figsize=(11, max(5, .31 * len(names))))
    ypos = np.arange(len(names))
    centers = np.array([result["versions"][n]["metrics"]["delta_auc_vs_full_ebm_a"] for n in names])
    lows = np.array([result["versions"][n]["metrics"]["paired_ci_95"][0] for n in names])
    highs = np.array([result["versions"][n]["metrics"]["paired_ci_95"][1] for n in names])
    for round_number, color, label in ((1, "#376b8c", "Round 1 model/view"), (2, "#cf8128", "Round 2 adaptive fusion")):
        idx = np.array([result["versions"][n]["round"] == round_number for n in names])
        if idx.any():
            ax.errorbar(centers[idx], ypos[idx], xerr=[centers[idx]-lows[idx], highs[idx]-centers[idx]],
                        fmt="o", capsize=3, color=color, label=label)
    ax.axvline(0, color="#333333", lw=1, label="no AUC change")
    ax.axvline(.01, color="#a86424", linestyle="--", lw=1, label="+0.01 reference")
    ax.set_yticks(ypos, names)
    ax.set_xlabel("ΔAUC vs full EBM-A (paired 95% bootstrap interval)")
    ax.set_title(f"FEAT-012 | N={result['rows']}, positives={result['positives']} | feature [Jun 1, Jun 21), label [Jun 21, Jul 31)\nDevelopment proxy; intervals do not adjust for iterative selection; run {result['run_id']}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "leaderboard.png", dpi=160)
    plt.close(fig)
    fusions = [n for n in names if result["versions"][n]["round"] == 2]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if fusions:
        delta = [result["versions"][n]["metrics"]["delta_auc_vs_full_ebm_a"] for n in fusions]
        lows = [result["versions"][n]["metrics"]["paired_ci_95"][0] for n in fusions]
        highs = [result["versions"][n]["metrics"]["paired_ci_95"][1] for n in fusions]
        ypos = np.arange(len(fusions))
        ax.errorbar(delta, ypos, xerr=[np.array(delta)-lows, np.array(highs)-delta], fmt="o", capsize=4, color="#cf8128")
        ax.set_yticks(ypos, fusions)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlabel("Fusion ΔAUC vs full EBM-A (paired 95% bootstrap interval)")
    ax.set_title(f"Round 2: adaptive equal-logit fusions | N={result['rows']}, positives={result['positives']}\nProxy window [Jun 21, Jul 31); run {result['run_id']}; exploratory only")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "fusion_intervals.png", dpi=160)
    plt.close(fig)


def run(args) -> None:
    out = Path(args.output).expanduser().resolve()
    prereg_path = out / "preregistration.json"
    if not prereg_path.is_file():
        raise FileNotFoundError("run --prepare first; no locked preregistration found")
    existing = [p.name for p in out.iterdir() if p.name != "preregistration.json"]
    if existing:
        raise FileExistsError(f"refusing to overwrite run output: {existing}")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    frame, manifest, ledger, views, groups, y, folds, features = load_locked_inputs(Path(args.input), Path(args.manifest), Path(args.ledger))
    if sha_file(Path(args.input).expanduser().resolve()) != prereg["input_sha256"]:
        raise ValueError("input changed after preregistration")
    ref_path = Path(args.reference_oof).expanduser().resolve()
    if sha_file(ref_path) != prereg["reference_oof_sha256"]:
        raise ValueError("reference OOF changed after preregistration")
    ref = pd.read_csv(ref_path, dtype={"sample_id": "string"})
    pcol = "p_f3"
    if pcol not in ref or not ref.sample_id.astype(str).equals(frame.sample_id.astype(str)) or not ref.y.astype(int).equals(frame.y.astype(int)):
        raise ValueError("FEAT-009 F3 reference OOF rows or labels do not match locked F3 input")
    reference = ref[pcol].astype(float).to_numpy()
    if not np.isfinite(reference).all():
        raise ValueError("reference OOF contains invalid scores")
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    for spec in BASE_ARMS:
        name = spec["name"]
        if spec["view"] not in views:
            raise ValueError(f"missing registered view {spec['view']}")
        pred, audit, seconds = fit_oof(name, spec, views[spec["view"]], frame, groups, y, folds)
        predictions[name] = pred
        results[name] = {"round": 1, "view": spec["view"], "family": spec["family"], "params": spec["params"],
                         "feature_count": len(views[spec["view"]]), "runtime_seconds": seconds,
                         "metrics": score(y, pred, predictions.get("full_ebm_a", pred), frame.sample_id.to_numpy(), folds),
                         "fold_audit": audit}
        if name == "full_ebm_a":
            for other_name in list(results):
                results[other_name]["metrics"] = score(y, predictions[other_name], pred, frame.sample_id.to_numpy(), folds)
        pd.DataFrame({"sample_id": frame.sample_id.astype(str), "gpsno": frame.gpsno.astype(str), "y": y,
                      "fold": folds, "probability": pred}).to_csv(out / f"oof_{name}.csv", index=False, lineterminator="\n")
        print(f"round 1 {name}: AUC={results[name]['metrics']['auc']:.6f}; {seconds:.1f}s", flush=True)
    ref_auc = float(roc_auc_score(y, reference))
    ref_comparison = paired_auc(y, predictions["full_ebm_a"], reference)
    # Round 2 is a preregistered adaptive rule: one strongest arm per distinct view, then all pairs.
    best_by_view: dict[str, str] = {}
    for name, item in results.items():
        view = item["view"]
        if view not in best_by_view or (item["metrics"]["auc"], name) > (results[best_by_view[view]]["metrics"]["auc"], best_by_view[view]):
            best_by_view[view] = name
    ranked_views = sorted(best_by_view, key=lambda v: (-results[best_by_view[v]]["metrics"]["auc"], v))[:4]
    round2_members = [best_by_view[v] for v in ranked_views]
    for i, a_name in enumerate(round2_members):
        for b_name in round2_members[i + 1:]:
            name = f"fuse_{a_name}__{b_name}"
            fused = sigmoid((logit(predictions[a_name]) + logit(predictions[b_name])) / 2)
            predictions[name] = fused
            results[name] = {"round": 2, "view": f"{results[a_name]['view']}+{results[b_name]['view']}",
                             "family": "equal_logit_fusion", "members": [a_name, b_name], "runtime_seconds": 0.0,
                             "metrics": score(y, fused, predictions["full_ebm_a"], frame.sample_id.to_numpy(), folds)}
            pd.DataFrame({"sample_id": frame.sample_id.astype(str), "gpsno": frame.gpsno.astype(str), "y": y,
                          "fold": folds, "probability": fused}).to_csv(out / f"oof_{name}.csv", index=False, lineterminator="\n")
            print(f"round 2 {name}: AUC={results[name]['metrics']['auc']:.6f}", flush=True)
    # Recompute all comparisons against the fixed internal anchor after it exists.
    anchor = predictions["full_ebm_a"]
    for name, item in results.items():
        item["metrics"] = score(y, predictions[name], anchor, frame.sample_id.to_numpy(), folds)
    summary = {
        "task": "FEAT-012", "run_id": out.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exploration_complete", "protocol": EXPECTED_PROTOCOL, "split_version": EXPECTED_SPLIT,
        "rows": len(frame), "positives": int(y.sum()), "feature_count_full": len(features), "anchor_auc": float(roc_auc_score(y, anchor)),
        "feat009_reference_auc": ref_auc, "anchor_vs_feat009_reference": ref_comparison,
        "round2_selected_views": ranked_views, "round2_members": round2_members,
        "versions": results,
        "conclusion_boundary": "Same vehicles and frozen proxy folds were reused across multiple adaptive rounds. OOF-based selection is optimistic; no independent official test evidence, cross-line winner, global baseline change, or final 61-day model claim follows.",
    }
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render(out, summary)
    lines = ["# FEAT-012 受控探索结果", "", f"- Run: {out.name}", f"- 协议：{EXPECTED_PROTOCOL}；固定折分：{EXPECTED_SPLIT}",
             f"- 样本：{len(frame)}；代理正类：{int(y.sum())}", "- 定位：开发期探索；重复查看同一批车辆的 OOF；不构成独立测试或跨线胜负。", "",
             "| 版本 | 轮次 | 特征视角 | 列数 | AUC | AP | Recall@100 | Brier | 相对 EBM-A ΔAUC [95% CI] |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in sorted(results.items(), key=lambda kv: (-kv[1]["metrics"]["auc"], kv[0])):
        m = item["metrics"]
        ci = m["paired_ci_95"]
        lines.append(f"| {name} | {item['round']} | {item['view']} | {item.get('feature_count', '融合')} | {m['auc']:.5f} | {m['ap']:.5f} | {m['recall_at_100']:.4f} | {m['brier']:.5f} | {m['delta_auc_vs_full_ebm_a']:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}] |")
    lines += ["", f"FEAT-009 已有 F3 OOF 与本轮 full EBM-A 配对差值：{ref_comparison['delta_auc']:+.5f} [{ref_comparison['ci_low']:+.5f}, {ref_comparison['ci_high']:+.5f}]。", "",
              "第一轮检验现有特征视角与固定算法配方；第二轮按已登记规则从第一轮选不同视角，再做等权 logit 两两融合。区间只描述固定车辆配对抽样下的变动，未校正多轮选择。", "",
              "![候选版本总览](leaderboard.png)", "", "![第二轮融合配对区间](fusion_intervals.png)", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "interpret", "matplotlib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "loaded-from-local-cache-or-unregistered"
    outputs = sorted(p for p in out.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest_out = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "code_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                    "code_sha256": sha_file(Path(__file__)), "input_sha256": sha_file(Path(args.input).expanduser().resolve()),
                    "manifest_sha256": sha_file(Path(args.manifest).expanduser().resolve()), "ledger_sha256": sha_file(Path(args.ledger).expanduser().resolve()),
                    "reference_oof_sha256": sha_file(ref_path), "python": platform.python_version(), "packages": packages,
                    "output_sha256": {p.name: sha_file(p) for p in outputs}, "versions_run": len(results), "version_cap": prereg["total_max_versions"],
                    "human_review": "pending"}
    (out / "manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "anchor_auc": summary["anchor_auc"], "reference_delta": ref_comparison,
                      "round2_views": ranked_views, "versions": len(results)}, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("prepare", "run"), required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--reference-oof", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    prepare(args) if args.phase == "prepare" else run(args)


if __name__ == "__main__":
    main()
