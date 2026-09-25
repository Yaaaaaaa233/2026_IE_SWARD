#!/usr/bin/env python3
"""Create controlled E2 comparisons from completed FEAT-014 OOF predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from feature_engineering.experiments.feat014 import run as e  # noqa: E402


def top_k(y: np.ndarray, p: np.ndarray, ids: np.ndarray, k: int = 100) -> set[int]:
    return set(np.lexsort((ids.astype(str), -p))[: min(k, len(y))].tolist())


def top100_change(y: np.ndarray, candidate: set[int], reference: set[int]) -> dict:
    added, removed = candidate - reference, reference - candidate
    return {
        "reference_false_positives_removed": int(sum(y[i] == 0 for i in removed)),
        "candidate_false_positives_added": int(sum(y[i] == 0 for i in added)),
        "new_true_positives_in_top100": int(sum(y[i] == 1 for i in added)),
        "true_positives_dropped_from_top100": int(sum(y[i] == 1 for i in removed)),
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


def metric_row(y: np.ndarray, p: np.ndarray, ids: np.ndarray, p_rf: np.ndarray, p_f3: np.ndarray,
               fit_seconds: float) -> dict:
    auc = float(roc_auc_score(y, p))
    rf = e.compute_pair(y, p, p_rf)
    f3 = e.compute_pair(y, p, p_f3)
    return {
        "auc": auc,
        "ap": float(average_precision_score(y, p)),
        "recall_at_100": float(y[list(top_k(y, p, ids))].sum() / y.sum()),
        "brier": float(brier_score_loss(y, p)),
        "fit_seconds": float(fit_seconds),
        "delta_auc_vs_rf": rf,
        "delta_auc_vs_f3": f3,
        "delta_recall_vs_rf": float(y[list(top_k(y, p, ids))].sum() / y.sum() - y[list(top_k(y, p_rf, ids))].sum() / y.sum()),
        "delta_recall_vs_f3": float(y[list(top_k(y, p, ids))].sum() / y.sum() - y[list(top_k(y, p_f3, ids))].sum() / y.sum()),
        "delta_brier_vs_rf": float(brier_score_loss(y, p) - brier_score_loss(y, p_rf)),
        "delta_brier_vs_f3": float(brier_score_loss(y, p) - brier_score_loss(y, p_f3)),
    }


def render(batch_dir: Path, data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(data["metrics"], key=lambda n: (data["metrics"][n]["delta_auc_vs_f3"]["point"], n))
    fig, ax = plt.subplots(figsize=(10, max(4.5, .58 * len(names))))
    for i, name in enumerate(names):
        row = data["metrics"][name]
        point = row["delta_auc_vs_f3"]["point"]
        low, high = row["delta_auc_vs_f3"]["ci95"]
        color = "#cf8128" if row["family"] == "ebm" else "#376b8c"
        ax.errorbar(point, i, xerr=[[point - low], [high - point]], fmt="o", capsize=3, color=color)
    ax.axvline(0, color="#333", lw=1, label="No change vs fixed F3 EBM-A")
    ax.axvline(.01, color="#9c5730", ls="--", lw=1, label="+0.01 reference")
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_xlabel("Δ pooled OOF AUC vs fixed F3 EBM-A (95% paired vehicle bootstrap)")
    ax.set_title(f"FEAT-014 {batch_dir.name} | candidate comparison\nDevelopment proxy; adaptive-selection intervals are not independent confirmation")
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(batch_dir / "B1_auc_intervals.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        row = data["metrics"][name]
        point = row["delta_auc_vs_f3"]["point"]
        ax.scatter(row["fit_seconds"] / 60, point, s=50)
        ax.annotate(name, (row["fit_seconds"] / 60, point), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#333", lw=1)
    ax.axhline(.01, color="#9c5730", ls="--", lw=1)
    ax.set_xlabel("Five-fold fit time (minutes)")
    ax.set_ylabel("Δ pooled OOF AUC vs fixed F3 EBM-A")
    ax.set_title(f"FEAT-014 {batch_dir.name} | effect and fit cost")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(batch_dir / "B2_cost_effect.png", dpi=160)
    plt.close(fig)

    names = data["focus_versions"]
    a, b = names
    pair = data["pair_complementarity"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ["Both top 100", f"{a} only", f"{b} only", "Neither"]
    values = [pair["both_true_positives"], pair["first_only_true_positives"],
              pair["second_only_true_positives"], pair["neither_true_positives"]]
    axes[0].bar(labels, values, color=["#376b8c", "#83a8bf", "#cf8128", "#d8dde2"])
    axes[0].set_ylabel("Positive vehicles")
    axes[0].set_title("Positive capture in each model's Top 100")
    axes[0].tick_params(axis="x", rotation=22)
    fp_values = [pair["first_false_positives"], pair["second_false_positives"]]
    axes[1].bar(names, fp_values, color=["#376b8c", "#cf8128"])
    axes[1].set_ylabel("False positives in Top 100")
    axes[1].set_title("False positive count")
    fig.suptitle("FEAT-014 E2 | model error complementarity", y=1.02)
    fig.tight_layout()
    fig.savefig(batch_dir / "C0_error_complementarity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    changes = data["top100_vs_f3"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(changes))
    gained = [changes[n]["new_true_positives_in_top100"] for n in changes]
    lost = [-changes[n]["true_positives_dropped_from_top100"] for n in changes]
    ax.bar(x, gained, label="New positives in Top 100", color="#376b8c")
    ax.bar(x, lost, label="Positives dropped from Top 100", color="#cf8128")
    ax.axhline(0, color="#333", lw=.8)
    ax.set_xticks(x, list(changes))
    ax.set_ylabel("Vehicles relative to fixed F3 EBM-A")
    ax.set_title("FEAT-014 E2 | Top 100 positive capture changes")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(batch_dir / "C1_top100_change.png", dpi=160)
    plt.close(fig)

    fusion = data.get("fusion_comparison", {})
    if fusion:
        names = list(fusion)
        colors = ["#cf8128" if fusion[name]["is_fusion"] else "#376b8c" for name in names]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        centers = [fusion[name]["delta_auc_vs_f3"]["point"] for name in names]
        lows = [fusion[name]["delta_auc_vs_f3"]["ci95"][0] for name in names]
        highs = [fusion[name]["delta_auc_vs_f3"]["ci95"][1] for name in names]
        for i, color in enumerate(colors):
            point = centers[i]
            axes[0].errorbar(point, i, xerr=[[point-lows[i]], [highs[i]-point]], fmt="o", color=color, capsize=3)
        axes[0].axvline(0, color="#333", lw=.8)
        axes[0].set_yticks(np.arange(len(names)), names)
        axes[0].set_xlabel("Δ AUC vs fixed F3 EBM-A")
        axes[0].set_title("Pooled OOF AUC")
        axes[1].barh(names, [fusion[name]["recall_at_100"] for name in names], color=colors)
        axes[1].set_xlabel("Recall@100")
        axes[1].set_title("Top 100 positive capture")
        axes[2].barh(names, [fusion[name]["brier"] for name in names], color=colors)
        axes[2].set_xlabel("Brier (lower is better)")
        axes[2].set_title("Probability error")
        fig.suptitle("FEAT-014 E2 | registered fusion members and outputs", y=1.02)
        fig.tight_layout()
        fig.savefig(batch_dir / "C2_fusion_comparison.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def diagnose(run_dir: Path, batch_name: str, focus: list[str]) -> dict:
    lock, frame, _, _, _, y, folds, p_rf, p_f3 = e.load_run_inputs(run_dir)
    batch_dir = run_dir / "batches" / batch_name
    plan = json.loads((batch_dir / "plan.json").read_text(encoding="utf-8"))
    metrics_doc = json.loads((batch_dir / "metrics.json").read_text(encoding="utf-8"))
    if not set(item["version"] for item in plan["versions"]).issubset(metrics_doc.get("versions", {})):
        raise ValueError("E2 requires OOF results for every registered version")
    ids = frame.sample_id.astype(str).to_numpy()
    probs = {}
    metrics = {}
    top = {}
    for item in plan["versions"]:
        name = item["version"]
        result = json.loads((batch_dir / name / "result.json").read_text(encoding="utf-8"))
        pred_path = batch_dir / name / "oof.csv"
        if e.sha(pred_path) != result["oof_sha256"]:
            raise ValueError(f"OOF hash mismatch for {name}")
        p = e.read_oof(pred_path, frame, "probability")
        probs[name] = p
        top[name] = top_k(y, p, ids)
        metrics[name] = {
            **metric_row(y, p, ids, p_rf, p_f3, result["fit_seconds"]),
            "family": result["family"], "view": result["view"],
            "feature_count": result["feature_count"],
        }
    reference_top = top_k(y, p_f3, ids)
    changes = {n: top100_change(y, top[n], reference_top) for n in probs}
    if len(focus) != 2:
        raise ValueError("focus must identify exactly two completed versions")
    focus_probs, focus_paths, focus_top = {}, {}, {}
    for name in focus:
        p, path = resolve_oof(run_dir, batch_name, name, frame)
        focus_probs[name], focus_paths[name] = p, path
        focus_top[name] = top_k(y, p, ids)
    aa, bb = focus
    both = focus_top[aa] & focus_top[bb]
    only_a, only_b = focus_top[aa] - focus_top[bb], focus_top[bb] - focus_top[aa]
    neither = set(range(len(y))) - (focus_top[aa] | focus_top[bb])
    pair = {
        "both_true_positives": int(sum(y[i] == 1 for i in both)),
        "first_only_true_positives": int(sum(y[i] == 1 for i in only_a)),
        "second_only_true_positives": int(sum(y[i] == 1 for i in only_b)),
        "neither_true_positives": int(sum(y[i] == 1 for i in neither)),
        "first_false_positives": int(sum(y[i] == 0 for i in focus_top[aa])),
        "second_false_positives": int(sum(y[i] == 0 for i in focus_top[bb])),
        "top100_intersection": int(len(both)),
        "probability_pearson_correlation": float(np.corrcoef(focus_probs[aa], focus_probs[bb])[0, 1]),
        "delta_auc_first_minus_second": e.compute_pair(y, focus_probs[aa], focus_probs[bb]),
        "fold_ids": sorted(map(int, np.unique(folds))),
    }
    pairwise_auc = {}
    names = sorted(probs)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            pairwise_auc[f"{left}_minus_{right}"] = e.compute_pair(y, probs[left], probs[right])
    fusion_comparison = {}
    for item in plan["versions"]:
        if item["family"] != "fusion":
            continue
        for member in item["params"]["members"]:
            reference = f"{member['batch']}/{member['version']}"
            if reference in fusion_comparison:
                continue
            p, _ = resolve_oof(run_dir, batch_name, reference, frame)
            result = json.loads((run_dir / "batches" / member["batch"] / member["version"] / "result.json").read_text(encoding="utf-8"))
            fusion_comparison[reference] = {
                **metric_row(y, p, ids, p_rf, p_f3, result["fit_seconds"]),
                "family": result["family"], "is_fusion": False,
            }
        fusion_comparison[f"{batch_name}/{item['version']}"] = {
            **metrics[item["version"]], "is_fusion": True,
        }
    direct_comparisons = {}
    for item in plan["versions"]:
        name = item["version"]
        raw_refs = item.get("comparison", {}).get("compare_to", "")
        refs = [x.strip() for x in raw_refs.split(" and ") if x.strip()]
        if not refs:
            continue
        direct_comparisons[name] = {}
        for reference in refs:
            parent_p, parent_path = resolve_oof(run_dir, batch_name, reference, frame)
            parent_top = top_k(y, parent_p, ids)
            direct_comparisons[name][reference] = {
                "delta_auc_candidate_minus_reference": e.compute_pair(y, probs[name], parent_p),
                "delta_recall_at_100_candidate_minus_reference": float(
                    y[list(top[name])].sum() / y.sum() - y[list(parent_top)].sum() / y.sum()),
                "delta_brier_candidate_minus_reference": float(
                    brier_score_loss(y, probs[name]) - brier_score_loss(y, parent_p)),
                "top100_change": top100_change(y, top[name], parent_top),
                "reference_oof_sha256": e.sha(parent_path),
            }
    report = {
        "task": "FEAT-014", "stage": "E2", "batch": batch_name,
        "input_lock_sha256": e.sha(run_dir / "input_lock.json"),
        "focus_versions": focus,
        "status": "pass",
        "metrics": metrics,
        "top100_vs_f3": changes,
        "pair_complementarity": pair,
        "pairwise_auc": pairwise_auc,
        "fusion_comparison": fusion_comparison,
        "direct_comparisons": direct_comparisons,
        "interpretation_boundary": "Repeated development OOF reuse is exploratory. Paired bootstrap intervals do not remove adaptive selection bias.",
        "oof_sha256": {name: e.sha(batch_dir / name / "oof.csv") for name in probs},
        "focus_oof_sha256": {name: e.sha(path) for name, path in focus_paths.items()},
        "diagnose_script_sha256": e.sha(Path(__file__).resolve()),
    }
    e.json_dump(batch_dir / "e2_diagnosis.json", report)
    pd.DataFrame([{"version": name, **row} for name, row in metrics.items()]).to_csv(
        batch_dir / "e2_comparison.csv", index=False, lineterminator="\n")
    render(batch_dir, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--focus", nargs=2, required=True,
                        help="two versions; use BATCH/VERSION to compare across batches")
    args = parser.parse_args()
    result = diagnose(args.run_dir.expanduser().resolve(), args.batch, args.focus)
    print(json.dumps({"stage": "E2", "status": result["status"], "batch": args.batch,
                      "focus": args.focus, "output": str(args.run_dir / "batches" / args.batch)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
