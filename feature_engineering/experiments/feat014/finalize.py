#!/usr/bin/env python3
"""Build the controlled FEAT-014 E3 stability and handoff report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from feature_engineering.experiments.feat014 import run as e  # noqa: E402
from feature_engineering.experiments.feat014.fusion import logit_mean  # noqa: E402


SINGLE_ROUTE = {"seed_42": "B001/V005", "seed_7": "B004/V015", "seed_2026": "B005/V018", "seed_mean": "B008/V023"}
FUSION_ROUTE = {"seed_42": "B003/V014", "seed_7": "B006/V021", "seed_2026": "B007/V022", "seed_mean": "B009/V024"}
REPLAY = ("B001/V005", "B010/V025")


def resolve(run_dir: Path, reference: str, frame: pd.DataFrame) -> tuple[np.ndarray, dict, Path]:
    batch, version = reference.split("/", 1)
    version_dir = run_dir / "batches" / batch / version
    result_path, oof_path = version_dir / "result.json", version_dir / "oof.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if e.sha(oof_path) != result["oof_sha256"]:
        raise ValueError(f"OOF fingerprint mismatch: {reference}")
    p = e.read_oof(oof_path, frame, "probability")
    return p, result, oof_path


def summarize(y: np.ndarray, p: np.ndarray, ids: np.ndarray, p_rf: np.ndarray, p_f3: np.ndarray) -> dict:
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


def render(run_dir: Path, report: dict, ledger: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = run_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    # D0: all outcomes plus the running best envelope; colors separate fitted models and fixed OOF combinations.
    names = [row["reference"] for row in ledger]
    values = np.asarray([row["auc"] for row in ledger])
    best = np.maximum.accumulate(values)
    colors = ["#cf8128" if row["family"] == "fusion" else "#376b8c" for row in ledger]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(np.arange(len(names)), values, c=colors, s=34, zorder=3, label="Version OOF AUC")
    ax.plot(np.arange(len(names)), best, color="#333", lw=1.4, label="Best observed so far")
    ax.axhline(report["fixed_anchors"]["f3_auc"], color="#8c4a2f", ls="--", lw=1, label="Fixed F3 EBM-A")
    ax.set_xticks(np.arange(len(names)), names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Pooled OOF AUC")
    ax.set_title("FEAT-014 D0 | all registered versions and best-so-far envelope")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "D0_iteration_curve.png", dpi=160)
    plt.close(fig)

    # D1: three seeds and the registered mean prediction for both routes.
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, route) in enumerate(report["seed_stability"].items()):
        runs = route["runs"]
        seed_names = ["seed_42", "seed_7", "seed_2026"]
        aucs = [runs[k]["metrics"]["auc"] for k in seed_names]
        ax.scatter([i - .1, i, i + .1], aucs, s=55, color="#376b8c" if name == "single_v005" else "#cf8128")
        avg = runs["seed_mean"]["metrics"]["auc"]
        ax.scatter(i, avg, s=90, marker="D", facecolors="none", edgecolors="#333", zorder=4)
        ax.plot([i - .12, i + .12], [min(aucs), min(aucs)], color="#555", lw=.8)
        ax.plot([i - .12, i + .12], [max(aucs), max(aucs)], color="#555", lw=.8)
        ax.text(i, min(aucs) - .002, f"range={max(aucs)-min(aucs):.4f}", ha="center", fontsize=8)
    ax.axhline(report["fixed_anchors"]["f3_auc"], color="#8c4a2f", ls="--", lw=1, label="Fixed F3 EBM-A")
    ax.set_xticks([0, 1], ["Single V005", "Fusion V014"])
    ax.set_ylabel("Pooled OOF AUC")
    ax.set_title("FEAT-014 D1 | seed-to-seed range and three-seed mean prediction")
    ax.grid(axis="y", alpha=.2)
    ax.legend(["Seed result", "Seed-mean OOF", "Fixed F3 EBM-A"], fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "D1_seed_stability.png", dpi=160)
    plt.close(fig)

    # D2: fixed-anchor primary and auxiliary gates for the final comparison set.
    candidate_refs = report["candidate_set"]
    rows = [(ref, report["version_summaries"][ref]) for ref in candidate_refs]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for i, (ref, m) in enumerate(rows):
        point = m["delta_auc_vs_f3"]["point"]
        low, high = m["delta_auc_vs_f3"]["ci95"]
        color = "#cf8128" if "B003/" in ref or "B009/" in ref else "#376b8c"
        axes[0].errorbar(point, i, xerr=[[point-low], [high-point]], fmt="o", color=color, capsize=3)
    axes[0].axvline(0, color="#333", lw=.8)
    axes[0].axvline(.01, color="#9c5730", ls="--", lw=.8)
    axes[0].set_yticks(np.arange(len(rows)), [r for r, _ in rows])
    axes[0].set_xlabel("Δ AUC vs F3 EBM-A")
    axes[0].set_title("Primary gate")
    recall = [m["delta_recall_vs_f3"] for _, m in rows]
    axes[1].barh([r for r, _ in rows], recall, color="#376b8c")
    axes[1].axvline(-.02, color="#9c5730", ls="--", lw=.8)
    axes[1].set_xlabel("Δ Recall@100")
    axes[1].set_title("Auxiliary gate (floor −0.02)")
    brier = [m["delta_brier_vs_f3"] for _, m in rows]
    axes[2].barh([r for r, _ in rows], brier, color="#cf8128")
    axes[2].axvline(.005, color="#9c5730", ls="--", lw=.8)
    axes[2].set_xlabel("Δ Brier (lower is better)")
    axes[2].set_title("Auxiliary gate (ceiling +0.005)")
    fig.suptitle("FEAT-014 D2 | candidate gains and fixed auxiliary limits", y=1.02)
    fig.tight_layout()
    fig.savefig(figdir / "D2_anchor_gates.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # D3: selected feature/model ablations from B002, all relative to the preregistered parent.
    b002 = json.loads((run_dir / "batches/B002/e2_diagnosis.json").read_text(encoding="utf-8"))
    wanted = [
        ("G1 removed", "V007", "B001/V006"),
        ("compact interactions 0", "V008", "B001/V005"),
        ("compact interactions 2", "V009", "B001/V005"),
        ("interactions 2 vs 0", "V009", "B002/V008"),
        ("history+G1 interactions 5", "V010", "B001/V006"),
        ("history+G1 minimum leaf 20", "V011", "B001/V006"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (label, version, parent) in enumerate(wanted):
        ci = b002["direct_comparisons"][version][parent]["delta_auc_candidate_minus_reference"]
        point, low, high = ci["point"], ci["ci95"][0], ci["ci95"][1]
        ax.errorbar(point, i, xerr=[[point-low], [high-point]], fmt="o", capsize=3, color="#376b8c")
    ax.axvline(0, color="#333", lw=.8)
    ax.set_yticks(np.arange(len(wanted)), [x[0] for x in wanted])
    ax.set_xlabel("Δ pooled OOF AUC vs preregistered parent (95% paired vehicle bootstrap)")
    ax.set_title("FEAT-014 D3 | feature and interaction ablations")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(figdir / "D3_ablation_summary.png", dpi=160)
    plt.close(fig)


def finalize(run_dir: Path) -> dict[str, Any]:
    lock, frame, _, _, _, y, folds, p_rf, p_f3 = e.load_run_inputs(run_dir)
    ids = frame.sample_id.astype(str).to_numpy()
    baseline = {"f3_auc": float(roc_auc_score(y, p_f3)), "rf_auc": float(roc_auc_score(y, p_rf)),
                "f3_recall_at_100": float(y[np.lexsort((ids, -p_f3))[:100]].sum() / y.sum()),
                "f3_brier": float(brier_score_loss(y, p_f3))}

    ledger: list[dict] = []
    audit_status = {}
    batch_dirs = sorted((run_dir / "batches").glob("B*"))
    for batch_dir in batch_dirs:
        plan_path = batch_dir / "plan.json"
        if not plan_path.is_file():
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("input_lock_sha256") != e.sha(run_dir / "input_lock.json"):
            raise ValueError(f"batch plan input lock differs: {batch_dir.name}")
        audit_path = batch_dir / "independent_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {"status": "missing"}
        audit_status[batch_dir.name] = audit.get("status", "missing")
        for item in plan["versions"]:
            reference = f"{batch_dir.name}/{item['version']}"
            _, result, oof_path = resolve(run_dir, reference, frame)
            metric = result["metrics"]
            if metric.get("auc") is None:
                raise ValueError(f"version lacks a complete metric row: {reference}")
            ledger.append({
                "reference": reference, "family": result["family"], "view": result["view"],
                "feature_count": int(result["feature_count"]), "auc": float(metric["auc"]),
                "delta_vs_f3": metric["delta_vs_f3_ebm_a"]["point"],
                "delta_vs_f3_ci95": metric["delta_vs_f3_ebm_a"]["ci95"],
                "delta_vs_rf": metric["delta_vs_rf_f0v2"]["point"],
                "recall_at_100": float(metric["recall_at_100"]), "brier": float(metric["brier"]),
                "fit_seconds": float(result["fit_seconds"]), "oof_sha256": e.sha(oof_path),
                "batch_audit_status": audit.get("status", "missing"),
            })
    ledger.sort(key=lambda row: int(row["reference"].split("V")[-1]))
    if len(ledger) > 30:
        raise ValueError("FEAT-014 version budget exceeded")
    if any(row["batch_audit_status"] != "pass" for row in ledger):
        raise ValueError("an E1 batch lacks a passing independent audit")

    routes = {"single_v005": SINGLE_ROUTE, "fusion_v014": FUSION_ROUTE}
    stability = {}
    version_summaries = {}
    for route_name, refs in routes.items():
        items = {}
        seed_aucs = []
        for key, reference in refs.items():
            p, result, path = resolve(run_dir, reference, frame)
            metric = summarize(y, p, ids, p_rf, p_f3)
            items[key] = {"reference": reference, "oof_sha256": e.sha(path), "metrics": metric,
                          "family": result["family"], "view": result["view"],
                          "fit_seconds": float(result["fit_seconds"])}
            version_summaries[reference] = metric
            if key != "seed_mean":
                seed_aucs.append(metric["auc"])
        stability[route_name] = {
            "runs": items,
            "seed_auc_mean": float(np.mean(seed_aucs)),
            "seed_auc_min": float(np.min(seed_aucs)),
            "seed_auc_max": float(np.max(seed_aucs)),
            "seed_auc_range": float(np.max(seed_aucs) - np.min(seed_aucs)),
            "seed_auc_population_sd": float(np.std(seed_aucs, ddof=0)),
            "mean_prediction_auc": items["seed_mean"]["metrics"]["auc"],
        }

    # Include a cheap, non-seed-averaged baseline and the highest development point for D2 comparison.
    all_results = {row["reference"]: row for row in ledger}
    best_ref = max(ledger, key=lambda row: row["auc"])["reference"]
    eligible = [row for row in ledger if row["family"] != "fusion" and row["fit_seconds"] <= 15]
    low_cost_ref = max(eligible, key=lambda row: row["auc"])["reference"]
    stable_route = min(stability, key=lambda name: stability[name]["seed_auc_range"])
    stable_mean_ref = stability[stable_route]["runs"]["seed_mean"]["reference"]
    candidate_set = list(dict.fromkeys([best_ref, stable_mean_ref, low_cost_ref]))
    candidate_gates = {}
    for reference in candidate_set:
        if reference not in version_summaries:
            p, _, _ = resolve(run_dir, reference, frame)
            version_summaries[reference] = summarize(y, p, ids, p_rf, p_f3)
        m = version_summaries[reference]
        auc_point = m["delta_auc_vs_f3"]["point"]
        auc_low = m["delta_auc_vs_f3"]["ci95"][0]
        candidate_gates[reference] = {
            "auc_point_positive": bool(auc_point > 0),
            "auc_interval_excludes_zero": bool(auc_low > 0),
            "meaningful_lower_bound_0_01": bool(auc_low >= .01),
            "recall_at_100_within_minus_0_02": bool(m["delta_recall_vs_f3"] >= -.02),
            "brier_within_plus_0_005": bool(m["delta_brier_vs_f3"] <= .005),
        }

    origin_p, _, origin_path = resolve(run_dir, REPLAY[0], frame)
    replay_p, _, replay_path = resolve(run_dir, REPLAY[1], frame)
    max_abs = float(np.max(np.abs(origin_p - replay_p)))
    replay = {"reference": REPLAY[0], "replay": REPLAY[1],
              "reference_oof_sha256": e.sha(origin_path), "replay_oof_sha256": e.sha(replay_path),
              "max_abs_probability_difference": max_abs,
              "rmse_probability_difference": float(np.sqrt(np.mean((origin_p - replay_p) ** 2))),
              "exact_oof_match": bool(max_abs == 0.0),
              "threshold": 1e-12, "status": "pass" if max_abs <= 1e-12 else "investigate"}

    registration = json.loads((run_dir / "e3_registration.json").read_text(encoding="utf-8"))
    final_window_handoff = {
        "status": "not_run_outside_development_proxy",
        "training_window": "2026-06-01 through 2026-07-31 inclusive (61 days), subject to source-availability audit",
        "prediction_window": "2026-08-01 through 2026-09-10 inclusive (40 days); labels remain official-only",
        "checklist": [
            "After FEAT-014 human review, select a candidate without treating proxy OOF as official-period evidence.",
            "Rebuild the selected feature view with all available 61 training days and verify every feature cutoff and source timestamp.",
            "Fit the selected configuration on all eligible training vehicles; do not use prediction-period labels or post-cutoff information.",
            "Run one inference smoke test, then validate row count, sample identity, finite probabilities, output schema, UTF-8 without BOM, and LF line endings.",
            "Record the full-window code/config/data fingerprints and compare the final artifact against the locked FEAT-009 v2 F3 EBM-A fallback before delivery.",
        ],
        "rollback_pointer": "FEAT-009 v2 fixed F3 EBM-A configuration and artifact; retain the current run's E0 lock and all version OOFs unchanged.",
    }
    report = {
        "task": "FEAT-014", "stage": "E3", "status": "complete_for_review",
        "input_lock_sha256": e.sha(run_dir / "input_lock.json"),
        "registration_sha256": e.sha(run_dir / "e3_registration.json"),
        "version_count": len(ledger), "version_budget": 30,
        "fixed_anchors": baseline, "batch_e1_audits": audit_status,
        "seed_stability": stability, "replay": replay,
        "best_exploration_point": best_ref,
        "lower_seed_variance_route": stable_route,
        "low_cost_candidate": low_cost_ref,
        "candidate_set": candidate_set,
        "candidate_gates": candidate_gates,
        "version_summaries": version_summaries,
        "ledger": ledger,
        "final_window_handoff": final_window_handoff,
        "reproduction": {
            "environment": "/private/tmp/feat014-runtime (Python 3.12.13; package versions in the locked input_lock.json and environment_reconstruction.json)",
            "finalize_command": "MPLCONFIGDIR=/private/tmp/mpl-feat014 /private/tmp/feat014-runtime/bin/python feature_engineering/experiments/feat014/finalize.py --run-dir outputs/feat-014/20260926-e0-r2",
            "audit_command": "MPLCONFIGDIR=/private/tmp/mpl-feat014 /private/tmp/feat014-runtime/bin/python feature_engineering/experiments/feat014/audit_e3.py --run-dir outputs/feat-014/20260926-e0-r2 --write",
        },
        "human_visual_review": "pending; report status is not user acceptance",
        "interpretation_boundary": "All comparisons reuse the same development proxy labels. Seed spread describes model randomness only; it is not an independent test-set confidence interval and does not remove adaptive-selection bias.",
    }
    render(run_dir, report, ledger)
    e.json_dump(run_dir / "e3_report.json", report)
    rows = []
    for route, contents in stability.items():
        for key, item in contents["runs"].items():
            rows.append({"route": route, "seed_role": key, "reference": item["reference"],
                         "auc": item["metrics"]["auc"], "delta_vs_f3": item["metrics"]["delta_auc_vs_f3"]["point"],
                         "ci_low": item["metrics"]["delta_auc_vs_f3"]["ci95"][0],
                         "ci_high": item["metrics"]["delta_auc_vs_f3"]["ci95"][1],
                         "recall_at_100": item["metrics"]["recall_at_100"], "brier": item["metrics"]["brier"]})
    pd.DataFrame(rows).to_csv(run_dir / "e3_seed_stability.csv", index=False, lineterminator="\n")
    (run_dir / "report.md").write_text(render_markdown(run_dir, report), encoding="utf-8")
    print(json.dumps({"stage": "E3", "status": report["status"], "versions": len(ledger),
                      "best_exploration_point": best_ref, "lower_seed_variance_route": stable_route,
                      "replay": replay["status"], "output": str(run_dir)}, ensure_ascii=False, indent=2))
    return report


def render_markdown(run_dir: Path, report: dict) -> str:
    lines = [
        "# FEAT-014 E3 稳定性与候选交接",
        "",
        "- 状态：机器汇总完成，等待叶安实际复核；不是 `accepted`。",
        f"- 受控版本数：{report['version_count']} / {report['version_budget']}。",
        f"- 固定 F3 EBM-A AUC：{report['fixed_anchors']['f3_auc']:.6f}；RF@F0v2 AUC：{report['fixed_anchors']['rf_auc']:.6f}。",
        f"- 最高探索点：`{report['best_exploration_point']}`；较小种子跨度路线：`{report['lower_seed_variance_route']}`；低成本候选：`{report['low_cost_candidate']}`。",
        f"- 同种子复跑最大逐车概率差：`{report['replay']['max_abs_probability_difference']:.3g}`（阈值 `1e-12`，状态 `{report['replay']['status']}`）。",
        "- 所有成对区间和指标仍来自同一开发代理标签；种子跨度只描述模型随机性，不能代替独立未来测试，也不校正自适应选型偏差。",
        "",
        "## 三种子表现",
        "",
        "| 路线 | seed 42 AUC | seed 7 AUC | seed 2026 AUC | AUC 跨度 | 三种子均值 OOF AUC | Recall@100 | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, route in report["seed_stability"].items():
        runs = route["runs"]
        lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.4f} | {:.6f} |".format(
            name, runs["seed_42"]["metrics"]["auc"], runs["seed_7"]["metrics"]["auc"],
            runs["seed_2026"]["metrics"]["auc"], route["seed_auc_range"],
            runs["seed_mean"]["metrics"]["auc"], runs["seed_mean"]["metrics"]["recall_at_100"],
            runs["seed_mean"]["metrics"]["brier"]))
    lines += ["", "## 收口候选与固定门槛", "",
              "| 候选 | ΔAUC 对 F3 | 95%区间 | Recall@100 门 | Brier 门 | 主门（区间下界>0） |", "|---|---:|---:|---|---|---|"]
    for reference in report["candidate_set"]:
        m = report["version_summaries"][reference]
        gates = report["candidate_gates"][reference]
        lo, hi = m["delta_auc_vs_f3"]["ci95"]
        lines.append("| {} | {:+.6f} | [{:+.6f}, {:+.6f}] | {} | {} | {} |".format(
            reference, m["delta_auc_vs_f3"]["point"], lo, hi,
            "pass" if gates["recall_at_100_within_minus_0_02"] else "fail",
            "pass" if gates["brier_within_plus_0_005"] else "fail",
            "pass" if gates["auc_interval_excludes_zero"] else "fail"))
    lines += ["", "## 版本曲线", "", "![D0 全版本曲线](figures/D0_iteration_curve.png)", "",
              "![D1 多种子稳定性](figures/D1_seed_stability.png)", "",
              "![D2 固定锚点主辅门](figures/D2_anchor_gates.png)", "",
              "![D3 特征与交互消融](figures/D3_ablation_summary.png)", "",
              "## 官方最终窗口交接", "",
              "本报告未执行官方最终窗口重训或预测。人工复核并选定候选后，按 2026-06-01 至 2026-07-31 全 61 天训练、预测 2026-08-01 至 2026-09-10；检查输入来源和截止时点，完成一次推理冒烟测试，并核对行数、身份、概率、schema、UTF-8 无 BOM 与 LF。若新候选无法复现或交付检查失败，回退到 FEAT-009 v2 固定 F3 EBM-A 配置及产物。", "",
              "## 复现入口", "",
              f"- 环境：`{report['reproduction']['environment']}`。",
              f"- E3 汇总：`{report['reproduction']['finalize_command']}`。",
              f"- 独立审计：`{report['reproduction']['audit_command']}`。", "",
              "## 复核边界", "",
              "本报告的 OOF、分数、预测指纹和逐车统计只保留在受控运行目录。图表用于理解候选差异，不能替代人工确认，也不证明官方未来期成绩。固定 RF 与 F3 对照继续分开列示；最终提交窗口仍需按官方 61 天训练数据构建。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    finalize(args.run_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
