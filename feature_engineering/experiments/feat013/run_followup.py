#!/usr/bin/env python3
"""Run the locked FEAT-013 seed-stability and fusion follow-up."""
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
from sklearn.metrics import roc_auc_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
sys.path.insert(0, str(ROOT / "feature_engineering/experiments/feat012"))
import run_exploration as e  # noqa: E402

SEEDS = (42, 7, 2026)
SEED_GROUPS = {
    "full_ebm_c": ("full", {"name": "full_ebm_c", "family": "ebm", "params": {"interactions": 0, "max_bins": 32, "min_samples_leaf": 20}}),
    "slim_ebm_c": ("slim_night", {"name": "slim_ebm_c", "family": "ebm", "params": {"interactions": 0, "max_bins": 32, "min_samples_leaf": 20}}),
    "history_ebm_a": ("history", {"name": "history_ebm_a", "family": "ebm", "params": {"interactions": 0, "max_bins": 64, "min_samples_leaf": 10}}),
}
TUNED_ARMS = (
    {"name": "full_ebm_c_bag28", "view": "full", "family": "ebm", "params": {"interactions": 0, "max_bins": 32, "min_samples_leaf": 20, "outer_bags": 28}},
    {"name": "slim_ebm_c_bag28", "view": "slim_night", "family": "ebm", "params": {"interactions": 0, "max_bins": 32, "min_samples_leaf": 20, "outer_bags": 28}},
    {"name": "full_ebm_c_lr03", "view": "full", "family": "ebm", "params": {"interactions": 0, "max_bins": 32, "min_samples_leaf": 20, "learning_rate": 0.03}},
)
COMPOSITES = (
    "full_ebm_c_s3mean", "slim_ebm_c_s3mean", "history_ebm_a_s3mean",
    "full_history_s3_equal", "full_history_s3_70_30", "full_history_s3_30_70",
    "full_slim_s3_equal", "full_slim_history_s3_equal",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_previous(prev: Path, name: str, frame: pd.DataFrame) -> np.ndarray:
    manifest = json.loads((prev / "manifest.json").read_text(encoding="utf-8"))
    path = prev / f"oof_{name}.csv"
    if manifest.get("output_sha256", {}).get(path.name) != sha(path):
        raise ValueError(f"FEAT-012 artifact hash mismatch: {path.name}")
    x = pd.read_csv(path, dtype={"sample_id": "string", "gpsno": "string"})
    if not x.sample_id.astype(str).equals(frame.sample_id.astype(str)) or \
       not x.gpsno.astype(str).equals(frame.gpsno.astype(str)) or \
       not x.y.astype(int).equals(frame.y.astype(int)) or \
       not x.fold.astype(int).equals(frame.fold.astype(int)):
        raise ValueError(f"FEAT-012 OOF rows, labels or folds differ: {name}")
    return x.probability.astype(float).to_numpy()


def load_sources(args):
    frame, manifest, ledger, views, groups, y, folds, features = e.load_locked_inputs(
        Path(args.input), Path(args.manifest), Path(args.ledger))
    prev = Path(args.previous_run).expanduser().resolve()
    previous_manifest_path = prev / "manifest.json"
    previous_metrics_path = prev / "metrics.json"
    previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    if previous_manifest.get("input_sha256") != sha(Path(args.input).expanduser().resolve()):
        raise ValueError("FEAT-012 input hash differs from current locked input")
    if previous_manifest.get("output_sha256", {}).get("metrics.json") != sha(previous_metrics_path):
        raise ValueError("FEAT-012 metrics hash differs from prior run manifest")
    return frame, manifest, ledger, views, groups, y, folds, features, prev, previous_manifest


def prepare(args):
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty run directory: {out}")
    frame, manifest, ledger, views, _, y, folds, features, prev, pmanifest = load_sources(args)
    out.mkdir(parents=True, exist_ok=True)
    prereg = {
        "task": "FEAT-013", "status": "locked_before_candidate_oof", "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True, "protocol": e.EXPECTED_PROTOCOL, "split_version": e.EXPECTED_SPLIT,
        "input_sha256": sha(Path(args.input).expanduser().resolve()),
        "manifest_sha256": sha(Path(args.manifest).expanduser().resolve()),
        "ledger_sha256": sha(Path(args.ledger).expanduser().resolve()),
        "previous_run_manifest_sha256": sha(prev / "manifest.json"),
        "previous_run_metrics_sha256": sha(prev / "metrics.json"),
        "reused_oof_sha256": {f"oof_{n}.csv": pmanifest["output_sha256"][f"oof_{n}.csv"] for n in ("full_ebm_a", "full_ebm_c", "slim_ebm_c", "history_ebm_a")},
        "rows": len(frame), "positives": int(y.sum()), "full_feature_count": len(features),
        "feature_view_sha256": {k: hashlib.sha256("\n".join(v).encode()).hexdigest() for k, v in views.items()},
        "seeds": list(SEEDS), "seed_aggregation": "equal-weight mean of clipped logits",
        "tuned_arms": list(TUNED_ARMS), "composites": list(COMPOSITES),
        "new_seed_oof_count": 6, "tuned_oof_count": len(TUNED_ARMS), "derived_composite_count": len(COMPOSITES),
        "max_report_versions": 6 + len(TUNED_ARMS) + len(COMPOSITES),
        "primary_reference": "FEAT-012 full_ebm_a OOF; all other comparisons use the same rows and folds",
        "metrics": "pooled AUC, paired vehicle-stratified bootstrap 2000 seed 42 95% percentile CI, AP, Recall@100, Brier, outer-fold AUC",
        "selection_bias": "post-hoc continuation after 19 FEAT-012 versions; no interval adjusts for repeated selection",
    }
    (out / "preregistration.json").write_text(json.dumps(prereg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"locked": True, "output": str(out), "rows": len(frame), "versions_cap": prereg["max_report_versions"], "new_seed_fits": 6, "tuned_fits": len(TUNED_ARMS), "composites": len(COMPOSITES)}, ensure_ascii=False, indent=2))


def write_oof(out: Path, name: str, frame: pd.DataFrame, y: np.ndarray, folds: np.ndarray, pred: np.ndarray):
    pd.DataFrame({"sample_id": frame.sample_id.astype(str), "gpsno": frame.gpsno.astype(str), "y": y,
                  "fold": folds, "probability": pred}).to_csv(out / f"oof_{name}.csv", index=False, lineterminator="\n")


def combine(predictions: dict[str, np.ndarray], names: list[str], weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    return e.sigmoid(sum(weight * e.logit(predictions[name]) for name, weight in zip(names, w)))


def render(out: Path, summary: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(summary["versions"])
    names.sort(key=lambda n: summary["versions"][n]["metrics"]["delta_auc_vs_full_ebm_a"])
    fig, ax = plt.subplots(figsize=(11, max(5, .32 * len(names))))
    yp = np.arange(len(names))
    centers = np.array([summary["versions"][n]["metrics"]["delta_auc_vs_full_ebm_a"] for n in names])
    lows = np.array([summary["versions"][n]["metrics"]["paired_ci_95"][0] for n in names])
    highs = np.array([summary["versions"][n]["metrics"]["paired_ci_95"][1] for n in names])
    for kind, color, label in (("seed_fit", "#649578", "single-seed fit"), ("tuned_fit", "#376b8c", "EBM configuration"),
                               ("seed_ensemble", "#8b69a6", "three-seed mean"), ("fusion", "#cf8128", "fixed fusion")):
        idx = np.array([summary["versions"][n]["kind"] == kind for n in names])
        if idx.any():
            ax.errorbar(centers[idx], yp[idx], xerr=[centers[idx]-lows[idx], highs[idx]-centers[idx]],
                        fmt="o", capsize=3, color=color, label=label)
    ax.axvline(0, color="#333", lw=1, label="no AUC change")
    ax.axvline(.01, color="#a86424", linestyle="--", lw=1, label="+0.01 reference")
    ax.set_yticks(yp, names)
    ax.set_xlabel("ΔAUC vs FEAT-012 full EBM-A (paired 95% bootstrap interval)")
    ax.set_title(f"FEAT-013 | N={summary['rows']}, positives={summary['positives']} | feature [Jun 1, Jun 21), label [Jun 21, Jul 31)\nSame development proxy; intervals do not adjust for iterative selection; run {summary['run_id']}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "leaderboard.png", dpi=160)
    plt.close(fig)
    names = [n for n, x in summary["versions"].items() if x["kind"] in {"seed_ensemble", "fusion"}]
    fig, ax = plt.subplots(figsize=(10, max(4.5, .3 * len(names))))
    vals = [summary["versions"][n]["metrics"]["delta_auc_vs_full_ebm_a"] for n in names]
    lows = [summary["versions"][n]["metrics"]["paired_ci_95"][0] for n in names]
    highs = [summary["versions"][n]["metrics"]["paired_ci_95"][1] for n in names]
    yp = np.arange(len(names))
    ax.errorbar(vals, yp, xerr=[np.array(vals)-lows, np.array(highs)-vals], fmt="o", capsize=3, color="#cf8128")
    ax.set_yticks(yp, names)
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("ΔAUC vs FEAT-012 full EBM-A (paired 95% bootstrap interval)")
    ax.set_title(f"Seed stability and fixed fusions | N={summary['rows']}, positives={summary['positives']}\nProxy development window; repeated-selection intervals are optimistic")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "fusion_intervals.png", dpi=160)
    plt.close(fig)


def run(args):
    out = Path(args.output).expanduser().resolve()
    prereg_path = out / "preregistration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if [p.name for p in out.iterdir() if p.name != "preregistration.json"]:
        raise FileExistsError("refusing to overwrite FEAT-013 outputs")
    frame, manifest, ledger, views, groups, y, folds, features, prev, pmanifest = load_sources(args)
    if sha(Path(args.input).expanduser().resolve()) != prereg["input_sha256"] or sha(prev / "manifest.json") != prereg["previous_run_manifest_sha256"]:
        raise ValueError("source input or prior run changed after preregistration")
    ids = frame.sample_id.astype(str).to_numpy()
    predictions: dict[str, np.ndarray] = {}
    versions: dict[str, dict] = {}
    anchor = load_previous(prev, "full_ebm_a", frame)
    predictions["full_ebm_a"] = anchor
    for base_name, (view, spec) in SEED_GROUPS.items():
        predictions[base_name + "_s42"] = load_previous(prev, base_name, frame)
        for seed in (7, 2026):
            name = f"{base_name}_s{seed}"
            pred, audit, seconds = e.fit_oof(name, spec, views[view], frame, groups, y, folds, seed=seed)
            predictions[name] = pred
            versions[name] = {"kind": "seed_fit", "family": "ebm", "view": view, "seed": seed,
                              "feature_count": len(views[view]), "runtime_seconds": seconds, "fold_audit": audit,
                              "metrics": e.score(y, pred, anchor, ids, folds)}
            write_oof(out, name, frame, y, folds, pred)
            print(f"seed fit {name}: AUC={versions[name]['metrics']['auc']:.6f}; {seconds:.1f}s", flush=True)
    for spec in TUNED_ARMS:
        name = spec["name"]
        pred, audit, seconds = e.fit_oof(name, spec, views[spec["view"]], frame, groups, y, folds, seed=42)
        predictions[name] = pred
        versions[name] = {"kind": "tuned_fit", "family": "ebm", "view": spec["view"], "params": spec["params"],
                          "feature_count": len(views[spec["view"]]), "runtime_seconds": seconds, "fold_audit": audit,
                          "metrics": e.score(y, pred, anchor, ids, folds)}
        write_oof(out, name, frame, y, folds, pred)
        print(f"tuned fit {name}: AUC={versions[name]['metrics']['auc']:.6f}; {seconds:.1f}s", flush=True)
    full3 = combine(predictions, ["full_ebm_c_s42", "full_ebm_c_s7", "full_ebm_c_s2026"], [1, 1, 1])
    slim3 = combine(predictions, ["slim_ebm_c_s42", "slim_ebm_c_s7", "slim_ebm_c_s2026"], [1, 1, 1])
    hist3 = combine(predictions, ["history_ebm_a_s42", "history_ebm_a_s7", "history_ebm_a_s2026"], [1, 1, 1])
    compositions = {
        "full_ebm_c_s3mean": (full3, "seed_ensemble", ["full_ebm_c_s42", "full_ebm_c_s7", "full_ebm_c_s2026"], [1, 1, 1]),
        "slim_ebm_c_s3mean": (slim3, "seed_ensemble", ["slim_ebm_c_s42", "slim_ebm_c_s7", "slim_ebm_c_s2026"], [1, 1, 1]),
        "history_ebm_a_s3mean": (hist3, "seed_ensemble", ["history_ebm_a_s42", "history_ebm_a_s7", "history_ebm_a_s2026"], [1, 1, 1]),
        "full_history_s3_equal": (combine({"f": full3, "h": hist3}, ["f", "h"], [1, 1]), "fusion", ["full_ebm_c_s3mean", "history_ebm_a_s3mean"], [1, 1]),
        "full_history_s3_70_30": (combine({"f": full3, "h": hist3}, ["f", "h"], [.7, .3]), "fusion", ["full_ebm_c_s3mean", "history_ebm_a_s3mean"], [.7, .3]),
        "full_history_s3_30_70": (combine({"f": full3, "h": hist3}, ["f", "h"], [.3, .7]), "fusion", ["full_ebm_c_s3mean", "history_ebm_a_s3mean"], [.3, .7]),
        "full_slim_s3_equal": (combine({"f": full3, "s": slim3}, ["f", "s"], [1, 1]), "fusion", ["full_ebm_c_s3mean", "slim_ebm_c_s3mean"], [1, 1]),
        "full_slim_history_s3_equal": (combine({"f": full3, "s": slim3, "h": hist3}, ["f", "s", "h"], [1, 1, 1]), "fusion", ["full_ebm_c_s3mean", "slim_ebm_c_s3mean", "history_ebm_a_s3mean"], [1, 1, 1]),
    }
    for name, (pred, kind, members, weights) in compositions.items():
        predictions[name] = pred
        versions[name] = {"kind": kind, "members": members, "weights": weights,
                          "metrics": e.score(y, pred, anchor, ids, folds)}
        write_oof(out, name, frame, y, folds, pred)
    if len(versions) != prereg["max_report_versions"]:
        raise RuntimeError(f"version count differs from preregistration: {len(versions)}")
    summary = {"task": "FEAT-013", "run_id": out.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "protocol": e.EXPECTED_PROTOCOL, "split_version": e.EXPECTED_SPLIT,
               "rows": len(frame), "positives": int(y.sum()), "anchor_auc": float(roc_auc_score(y, anchor)),
               "versions": versions,
               "conclusion_boundary": "Post-hoc continuation after FEAT-012. Multi-seed and weight choices use the same vehicles/folds already examined; results and intervals are exploratory, not independent evidence.",
               "review": "pending"}
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render(out, summary)
    rows = ["# FEAT-013 受控探索结果", "", f"- Run: {out.name}", f"- 协议：{e.EXPECTED_PROTOCOL}；固定折分：{e.EXPECTED_SPLIT}",
            f"- 样本：{len(frame)}；代理正类：{int(y.sum())}", "- 定位：FEAT-012 后续探索，复用相同代理车辆及折分，结果有自适应选择偏差。", "",
            "| 版本 | 类型 | AUC | AP | Recall@100 | Brier | ΔAUC vs EBM-A [95% CI] |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, item in sorted(versions.items(), key=lambda kv: (-kv[1]["metrics"]["auc"], kv[0])):
        m = item["metrics"]; lo, hi = m["paired_ci_95"]
        rows.append(f"| {name} | {item['kind']} | {m['auc']:.5f} | {m['ap']:.5f} | {m['recall_at_100']:.4f} | {m['brier']:.5f} | {m['delta_auc_vs_full_ebm_a']:+.5f} [{lo:+.5f}, {hi:+.5f}] |")
    rows += ["", "图中区间只表示固定车辆配对 bootstrap 的抽样波动，不校正模型择优、多轮探索或融合权重选择。", "", "![全部版本](leaderboard.png)", "", "![配对区间](fusion_intervals.png)", ""]
    (out / "report.md").write_text("\n".join(rows), encoding="utf-8")
    package_versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "interpret", "matplotlib"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "loaded-from-local-cache-or-unregistered"
    output_files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest_out = {"code_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                    "code_sha256": sha(Path(__file__)), "feat012_code_sha256": sha(ROOT / "feature_engineering/experiments/feat012/run_exploration.py"),
                    "input_sha256": sha(Path(args.input).expanduser().resolve()), "previous_manifest_sha256": sha(prev / "manifest.json"),
                    "python": platform.python_version(), "packages": package_versions,
                    "output_sha256": {p.name: sha(p) for p in output_files}, "versions_run": len(versions),
                    "human_review": "pending"}
    (out / "manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "versions": len(versions), "anchor_auc": summary["anchor_auc"],
                      "best": sorted(((n, round(v["metrics"]["auc"], 6)) for n, v in versions.items()), key=lambda x: (-x[1], x[0]))[:5]}, ensure_ascii=False, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("prepare", "run"), required=True)
    ap.add_argument("--input", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--ledger", required=True); ap.add_argument("--previous-run", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.phase == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
