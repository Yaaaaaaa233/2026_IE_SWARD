#!/usr/bin/env python3
"""Controlled E0 lock and adaptive feature/model batches for FEAT-014."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
FEAT012 = ROOT / "feature_engineering/experiments/feat012"
if str(FEAT012) not in sys.path:
    sys.path.insert(0, str(FEAT012))
import run_exploration as prior  # noqa: E402

FAMILY = "feat014_g1"
G1_COLUMNS = [
    "feat014_g1_segments_per_1000km",
    "feat014_g1_segments_per_drive_hour",
    "feat014_g1_event_day_share",
    "feat014_g1_daily_rate_cv",
    "feat014_g1_active_day_share",
]
META = prior.META_COLUMNS
CODE_FILES = [
    ROOT / "feature_engineering/experiments/feat014/run.py",
    ROOT / "feature_engineering/experiments/feat014/audit.py",
    ROOT / "feature_engineering/experiments/feat012/run_exploration.py",
    ROOT / "feature_engineering/experiments/task1_feature_modeling/modeling.py",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_oof(path: Path, frame: pd.DataFrame, pcol: str | None = None) -> np.ndarray:
    oof = pd.read_csv(path, dtype={"sample_id": "string", "gpsno": "string"}, low_memory=False)
    if "sample_id" not in oof or oof.sample_id.isna().any() or oof.sample_id.duplicated().any():
        raise ValueError(f"OOF lacks a unique sample_id: {path.name}")
    if not oof.sample_id.astype(str).equals(frame.sample_id.astype(str)):
        raise ValueError(f"OOF sample order/identity differs: {path.name}")
    for key in ("gpsno", "y", "fold"):
        if key in oof and not oof[key].astype(str).equals(frame[key].astype(str)):
            raise ValueError(f"OOF {key} differs: {path.name}")
    if pcol is None:
        pcol = "probability" if "probability" in oof else next((c for c in oof if c.startswith("p_")), None)
    if pcol is None or pcol not in oof:
        raise ValueError(f"OOF probability column is missing: {path.name}")
    p = pd.to_numeric(oof[pcol], errors="coerce").to_numpy(float)
    if len(p) != len(frame) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"OOF probabilities are invalid: {path.name}")
    return p


def extract_g1_features(daily: pd.DataFrame, samples: pd.DataFrame, *, cutoff: str,
                        lookback_days: int = 20) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate vehicle-day rows strictly inside [cutoff-lookback, cutoff)."""
    required = {"gpsno", "date", "evt_segments30", "evt_days_flag", "traj_km", "traj_hours"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"vehicle_day lacks required fields: {missing}")
    if daily["gpsno"].isna().any() or daily["date"].isna().any():
        raise ValueError("vehicle_day has null vehicle/date keys")
    work = daily.copy()
    work["gpsno"] = work.gpsno.astype(str)
    work["date"] = pd.to_datetime(work.date, errors="raise").dt.normalize()
    if work.duplicated(["gpsno", "date"]).any():
        raise ValueError("vehicle_day has duplicate vehicle/date rows")
    for col in ("evt_segments30", "evt_days_flag", "traj_km", "traj_hours"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
        if work[col].isna().any():
            raise ValueError(f"vehicle_day has missing {col}; cannot treat missing as zero exposure")
        if (work[col].dropna() < 0).any():
            raise ValueError(f"vehicle_day has negative {col}")
    if not work.evt_days_flag.isin([0, 1]).all():
        raise ValueError("vehicle_day.evt_days_flag must be binary")
    ids = samples[["sample_id", "gpsno"]].copy()
    ids["gpsno"] = ids.gpsno.astype(str)
    if ids.sample_id.isna().any() or ids.sample_id.duplicated().any() or ids.gpsno.duplicated().any():
        raise ValueError("sample input keys are not unique")
    unknown = sorted(set(work.gpsno) - set(ids.gpsno))
    if unknown:
        raise ValueError("vehicle_day contains vehicles outside the locked cohort")
    end = pd.Timestamp(cutoff)
    start = end - pd.Timedelta(days=lookback_days)
    in_window = work["date"].ge(start) & work["date"].lt(end)
    x = work.loc[in_window].copy()
    x["_active"] = (x.traj_hours.fillna(0).gt(0) | x.traj_km.fillna(0).gt(0)).astype(int)
    x["_daily_rate"] = np.divide(x.evt_segments30, x.traj_hours,
                                 out=np.full(len(x), np.nan, dtype=float),
                                 where=x.traj_hours.to_numpy(float) > 0)
    grouped = x.groupby("gpsno", sort=False, observed=True)
    summary = grouped.agg(
        _days=("date", "nunique"), _segments=("evt_segments30", "sum"),
        _event_days=("evt_days_flag", "sum"), _km=("traj_km", "sum"),
        _hours=("traj_hours", "sum"), _active=("_active", "sum"),
    ).reset_index()
    rates = (x.loc[x.traj_hours.gt(0)].groupby("gpsno", sort=False, observed=True)
             ["_daily_rate"].agg(["mean", "std"]).reset_index())
    summary = summary.merge(rates, on="gpsno", how="left", validate="one_to_one")
    summary[G1_COLUMNS[0]] = np.divide(summary["_segments"] * 1000, summary["_km"],
                                       out=np.full(len(summary), np.nan), where=summary["_km"].to_numpy(float) > 0)
    summary[G1_COLUMNS[1]] = np.divide(summary["_segments"], summary["_hours"],
                                       out=np.full(len(summary), np.nan), where=summary["_hours"].to_numpy(float) > 0)
    summary[G1_COLUMNS[2]] = np.divide(summary["_event_days"], summary["_days"],
                                       out=np.full(len(summary), np.nan), where=summary["_days"].to_numpy(float) > 0)
    summary[G1_COLUMNS[3]] = np.divide(summary["std"], summary["mean"],
                                       out=np.full(len(summary), np.nan), where=summary["mean"].to_numpy(float) > 0)
    summary[G1_COLUMNS[4]] = np.divide(summary["_active"], summary["_days"],
                                       out=np.full(len(summary), np.nan), where=summary["_days"].to_numpy(float) > 0)
    keep = summary[["gpsno", *G1_COLUMNS]].copy()
    if keep.gpsno.duplicated().any() or len(set(keep.gpsno) - set(ids.gpsno)):
        raise ValueError("G1 aggregation keys are not one-to-one with the locked cohort")
    out = ids.merge(keep, on="gpsno", how="left", validate="one_to_one", sort=False)
    if not out.sample_id.astype(str).equals(ids.sample_id.astype(str)) or len(out) != len(samples):
        raise ValueError("G1 join changed sample order or row count")
    no_observation = ~out[G1_COLUMNS[0]].notna()
    out.loc[no_observation, G1_COLUMNS[4]] = np.nan
    audit = {
        "source_rows": int(len(daily)), "window_rows": int(len(x)),
        "excluded_at_or_after_cutoff_rows": int((~work.date.lt(end)).sum()),
        "excluded_before_window_rows": int((work.date.lt(start)).sum()),
        "window_start_inclusive": start.date().isoformat(), "cutoff_exclusive": end.isoformat(),
        "sample_rows": int(len(out)), "vehicles_with_window_observation": int(out[G1_COLUMNS[0]].notna().sum()),
        "features": G1_COLUMNS,
        "semantics": {
            G1_COLUMNS[0]: "sum(evt_segments30)/sum(traj_km)*1000; NaN when exposure is zero",
            G1_COLUMNS[1]: "sum(evt_segments30)/sum(traj_hours); NaN when exposure is zero",
            G1_COLUMNS[2]: "sum(evt_days_flag)/observed vehicle-days",
            G1_COLUMNS[3]: "CV of daily evt_segments30/traj_hours among positive-hour days",
            G1_COLUMNS[4]: "active vehicle-days / observed vehicle-days; missing when no observation",
        },
    }
    return out, audit


def render_e0(out: Path, frame: pd.DataFrame, ledger: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.6))
    bars = [
        ("Dev features [20d]", "2026-06-01", "2026-06-21", "#376b8c"),
        ("Dev proxy labels [40d]", "2026-06-21", "2026-07-31", "#8aaec2"),
        ("Final history [61d]", "2026-06-01", "2026-08-01", "#cf8128"),
        ("Official prediction [40d]", "2026-08-01", "2026-09-10", "#e8b66f"),
    ]
    for row, (label, start, end, color) in enumerate(bars[::-1]):
        left, right = pd.Timestamp(start), pd.Timestamp(end)
        ax.barh(row, (right-left).days, left=left, height=.48, color=color, align="center")
        ax.text(left + (right-left)/2, row, f"{start} to {end}", ha="center", va="center", fontsize=8,
                color="#17212b" if row in (0, 1) else "#202020")
    for date, label in (("2026-06-21", "Proxy as-of"), ("2026-08-01", "Official start")):
        x = pd.Timestamp(date); ax.axvline(x, color="#444", linestyle="--", lw=.9)
        ax.annotate(label, (x, 3.35), xytext=(4, 0), textcoords="offset points", fontsize=8, ha="left")
    ax.set_yticks(range(4), [x[0] for x in bars[::-1]])
    ax.set_xlim(pd.Timestamp("2026-05-28"), pd.Timestamp("2026-09-15"))
    ax.set_ylim(-.55, 3.7); ax.set_title("FEAT-014 E0 | Time visibility and fixed evaluation windows")
    ax.grid(axis="x", alpha=.2); fig.tight_layout()
    fig.tight_layout(); fig.savefig(out / "A0_time_windows.png", dpi=160); plt.close(fig)

    kept = ledger[(ledger.role == "predictor") & ledger.visibility_decision.eq("retain_candidate") & ledger.in_full_f3.eq(True)]
    counts = kept.groupby("family").column.nunique().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(5, .29 * len(counts))))
    ax.barh(counts.index.astype(str), counts.values, color="#376b8c")
    ax.set_xlabel("Features in the locked full F3 view")
    ax.set_title("FEAT-014 E0 | Feature-family inventory (names only)")
    ax.grid(axis="x", alpha=.2); fig.tight_layout(); fig.savefig(out / "A1_feature_families.png", dpi=160); plt.close(fig)


def run_audit(run_dir: Path, input_path: Path, prior_run: Path, is_013: bool = False) -> dict[str, Any]:
    script = FEAT012 / "audit_exploration.py"
    command = [sys.executable, str(script), "--run", str(run_dir), "--input", str(input_path)]
    if is_013:
        command += ["--previous-run", str(prior_run)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def prepare(args) -> None:
    out = Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {out}")
    frame, source_manifest, ledger, views, _, y, folds, features = prior.load_locked_inputs(
        Path(args.input), Path(args.manifest), Path(args.ledger))
    out.mkdir(parents=True, exist_ok=True)
    ref_f3_path, ref_rf_path = Path(args.f3_oof).resolve(), Path(args.rf_oof).resolve()
    p_f3 = read_oof(ref_f3_path, frame, "p_f3")
    p_rf = read_oof(ref_rf_path, frame, "p_random_forest")
    # FEAT-012 full EBM-A is the reproducible F3 EBM-A anchor.
    p_ebm = read_oof(Path(args.feat012) / "oof_full_ebm_a.csv", frame)
    if not np.allclose(p_f3, p_ebm, atol=1e-10, rtol=0):
        raise ValueError("FEAT-012 full EBM-A no longer reproduces the frozen FEAT-009 F3 OOF")
    audits = {
        "FEAT-012": run_audit(Path(args.feat012), Path(args.input).resolve(), Path(args.feat012)),
        "FEAT-013": run_audit(Path(args.feat013), Path(args.input).resolve(), Path(args.feat012), is_013=True),
    }
    if any(x.get("status") != "pass" for x in audits.values()):
        raise ValueError("historical OOF independent audit failed")
    daily_path = Path(args.vehicle_day).resolve()
    daily = pd.read_csv(daily_path, dtype={"gpsno": "string"}, low_memory=False)
    g1, g1_audit = extract_g1_features(daily, frame[["sample_id", "gpsno"]], cutoff="2026-06-21")
    g1_path = out / "features_g1_vehicle_day.csv"
    g1.to_csv(g1_path, index=False, lineterminator="\n")
    g1_meta = {**g1_audit, "source_path": str(daily_path), "source_sha256": sha(daily_path),
               "output_sha256": sha(g1_path), "source_version_values": sorted(map(str, daily.source_version.dropna().unique()))
               if "source_version" in daily else []}

    input_paths = {name: str(Path(value).expanduser().resolve()) for name, value in {
        "f3_model_input": args.input, "input_manifest": args.manifest, "column_ledger": args.ledger,
        "rf_oof": args.rf_oof, "f3_oof": args.f3_oof, "vehicle_day": args.vehicle_day,
    }.items()}
    input_hashes = {name: sha(Path(value)) for name, value in input_paths.items()}
    history: list[dict[str, Any]] = []
    for run_name, run_path, expected_task in (("FEAT-012", Path(args.feat012), "FEAT-012"),
                                               ("FEAT-013", Path(args.feat013), "FEAT-013")):
        summary = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
        if summary.get("task") != expected_task:
            raise ValueError(f"unexpected historical metrics task in {run_name}")
        for name, item in summary["versions"].items():
            pred_path = run_path / f"oof_{name}.csv"
            pred = read_oof(pred_path, frame)
            m = item["metrics"]
            ids = frame.sample_id.astype(str).to_numpy()
            top = np.lexsort((ids, -pred))[:min(100, len(frame))]
            history.append({"task": run_name, "version": name, "kind": item.get("kind", item.get("family", "model")),
                            "view": item.get("view", "composite"), "family": item.get("family", "unknown"),
                            "feature_count": item.get("feature_count"), "auc": float(m["auc"]),
                            "delta_vs_f3_anchor": float(m.get("delta_auc_vs_full_ebm_a", 0.0)),
                            "recall_at_100": float(m["recall_at_100"]), "brier": float(m["brier"]),
                            "top100_true_positive": int(y[top].sum()), "top100_false_positive": int((1-y[top]).sum()),
                            "oof_sha256": sha(pred_path)})
    ledger_path = out / "historical_ledger.csv"
    pd.DataFrame(history).sort_values(["task", "version"]).to_csv(ledger_path, index=False, lineterminator="\n")
    versions12 = len(json.loads((Path(args.feat012) / "metrics.json").read_text())["versions"])
    versions13 = len(json.loads((Path(args.feat013) / "metrics.json").read_text())["versions"])
    lock = {
        "task": "FEAT-014", "stage": "E0", "status": "locked", "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "protocol": source_manifest["protocol"], "split_version": source_manifest["split_version"],
        "feature_window": source_manifest["feature_window"], "label_window": source_manifest["label_window"],
        "rows": int(len(frame)), "positives": int(y.sum()), "fold_ids": sorted(map(int, np.unique(folds))),
        "input_paths": input_paths, "input_sha256": input_hashes,
        "execution_code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in CODE_FILES},
        "feature_count": int(len(features)), "feature_views": {k: {"count": len(v), "columns_sha256": hashlib.sha256("\n".join(v).encode()).hexdigest()} for k,v in views.items()},
        "anchors": {"RF@F0v2": {"path": str(ref_rf_path), "sha256": sha(ref_rf_path), "prediction_sha256": hashlib.sha256(p_rf.tobytes()).hexdigest()},
                    "F3_EBM_A": {"path": str(ref_f3_path), "sha256": sha(ref_f3_path), "prediction_sha256": hashlib.sha256(p_f3.tobytes()).hexdigest(),
                                 "feat012_full_ebm_a_max_abs_diff": float(np.max(np.abs(p_f3-p_ebm))) }},
        "historical_runs": {"FEAT-012": {"path": str(Path(args.feat012).resolve()), "versions": versions12, "audit": audits["FEAT-012"]},
                            "FEAT-013": {"path": str(Path(args.feat013).resolve()), "versions": versions13, "audit": audits["FEAT-013"]}},
        "g1_vehicle_day": g1_meta,
        "fixed_rules": {"top_k": min(100, len(frame)), "tie_break": "sample_id ascending", "bootstrap_draws": 2000,
                        "bootstrap_seed": 42, "ci": 0.95, "delta_auc_positive_gate": 0.0,
                        "delta_auc_meaningful_gate": 0.01, "recall_at_100_tolerance": -0.02, "brier_tolerance": 0.005},
        "runtime": {"python": platform.python_version(), "packages": {}},
        "human_review": "pending",
    }
    for package in ("numpy", "pandas", "scikit-learn", "interpret", "matplotlib", "lightgbm"):
        try: lock["runtime"]["packages"][package] = version(package)
        except PackageNotFoundError: lock["runtime"]["packages"][package] = "not_installed"
    json_dump(out / "input_lock.json", lock)
    b001 = [
        {"version":"V001","hypothesis":"正则化后全 F3 可否由稀疏线性信号概括","view":"full","family":"logistic","params":{"penalty":"l2","C_candidates":[0.01,0.1,1.0,10.0],"inner_folds":3,"scaler":"StandardScaler","max_iter":3000}},
        {"version":"V002","hypothesis":"精简输入能否帮助 RF","view":"slim_night","family":"rf","params":{"n_estimators":500,"max_features":0.5,"min_samples_leaf":4,"class_weight":"balanced_subsample"}},
        {"version":"V003","hypothesis":"精简输入能否帮助 HGB","view":"slim_night","family":"histgb","params":{"max_iter":120,"learning_rate":0.05,"max_leaf_nodes":7,"min_samples_leaf":20,"l2_regularization":2.0}},
        {"version":"V004","hypothesis":"全 F3 是否需要少量二阶交互","view":"full","family":"ebm","params":{"interactions":5,"max_bins":32,"min_samples_leaf":20}},
        {"version":"V005","hypothesis":"精简输入与交互是否相互补充","view":"slim_night","family":"ebm","params":{"interactions":5,"max_bins":32,"min_samples_leaf":20}},
        {"version":"V006","hypothesis":"车辆日面板的暴露率／日间波动是否补充历史事件特征","view":"history_plus_g1","family":"ebm","params":{"interactions":0,"max_bins":64,"min_samples_leaf":10}},
    ]
    batch = {"task":"FEAT-014","batch":"B001","status":"locked_before_candidate_oof","input_lock_sha256":sha(out / "input_lock.json"),
             "created_at_utc":datetime.now(timezone.utc).isoformat(),"seed":42,"versions":b001,
             "selection":"No FEAT-014 B001 candidate outcome existed before this batch registration.",
             "fallback":"If G1 source audit or EBM interaction runtime fails, retain the failed attempt and run unaffected registered arms; do not silently replace its slot."}
    json_dump(out / "batches/B001/plan.json", batch)
    render_e0(out, frame, ledger)
    report = ["# FEAT-014 E0 输入锁定与历史证据审计", "", "- 状态：机器检查通过；真实人工图表复核待进行。",
              f"- 协议：`{lock['protocol']}`；折分：`{lock['split_version']}`；特征窗 `{lock['feature_window']}`；标签窗 `{lock['label_window']}`。",
              f"- 输入车辆行数：{lock['rows']}；F3 可用特征数：{lock['feature_count']}。",
              f"- 历史 OOF 版本：FEAT-012 {versions12} 版，FEAT-013 {versions13} 版；两份独立审计均通过。",
              f"- F3 EBM-A 对 FEAT-009 锁定 F3 OOF 最大绝对差：`{lock['anchors']['F3_EBM_A']['feat012_full_ebm_a_max_abs_diff']:.3g}`。",
              f"- G1 日表：窗口内 {g1_audit['window_rows']} 条日记录；截止后 {g1_audit['excluded_at_or_after_cutoff_rows']} 条未参与特征；车辆行数守恒至 {g1_audit['sample_rows']}。",
              "- 对照：RF@F0v2 与 FEAT-009 F3 EBM-A 均单独锁定；探索最好版只作为滚动父版。", "",
              "## 人类读图入口", "", "![时间窗与可见性](A0_time_windows.png)", "", "![锁定特征族](A1_feature_families.png)", "",
              "具体预测、指标、指纹和逐车错误统计仅保留在受控目录。图表及来源人工验收状态仍为待复核。", ""]
    (out / "e0_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"stage":"E0","status":"pass","output":str(out),"rows":len(frame),
                      "feat012_versions":versions12,"feat013_versions":versions13,"history_versions":len(history),
                      "g1_window_rows":g1_audit["window_rows"],"g1_post_cutoff_rows_excluded":g1_audit["excluded_at_or_after_cutoff_rows"],
                      "charts":["A0_time_windows.png","A1_feature_families.png"],"human_review":"pending"},ensure_ascii=False,indent=2),flush=True)


def make_estimator(spec: dict[str, Any], seed: int):
    p = spec["params"]
    family = spec["family"]
    if family == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(**p, random_state=seed, n_jobs=1)
    if family == "rf":
        return RandomForestClassifier(**p, random_state=seed, n_jobs=1)
    if family == "histgb":
        return HistGradientBoostingClassifier(**p, random_state=seed)
    if family == "logistic":
        return LogisticRegression(penalty="l2", C=float(spec["selected_C"]), max_iter=p["max_iter"], solver="lbfgs")
    raise ValueError(f"unsupported FEAT-014 family: {family}")


def fold_transform(x: pd.DataFrame, groups: pd.DataFrame, train: np.ndarray, test: np.ndarray):
    from modeling import fit_transformer
    transformer = fit_transformer(x.iloc[train], groups.iloc[train], cohort_sources=[])
    a = transformer.transform(x.iloc[train], groups.iloc[train])
    b = transformer.transform(x.iloc[test], groups.iloc[test])
    return transformer, a, b


def fit_version(item: dict[str, Any], frame: pd.DataFrame, x: pd.DataFrame,
                y: np.ndarray, folds: np.ndarray, seed: int = 42):
    groups = frame[["sample_id", "gpsno"]].copy()
    predictions = np.full(len(y), np.nan)
    records = []
    runtime_start = time.monotonic()
    for outer in sorted(np.unique(folds)):
        tr, te = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        chosen_c = None
        inner_scores = {}
        if item["family"] == "logistic":
            params = item["params"]
            inner = StratifiedKFold(n_splits=params["inner_folds"], shuffle=True, random_state=seed)
            fold_scores = {str(c): [] for c in params["C_candidates"]}
            for it_rel, iv_rel in inner.split(np.zeros(len(tr)), y[tr]):
                it, iv = tr[it_rel], tr[iv_rel]
                _, a, b = fold_transform(x, groups, it, iv)
                scaler = StandardScaler()
                a, b = scaler.fit_transform(a), scaler.transform(b)
                for c in params["C_candidates"]:
                    model = LogisticRegression(penalty="l2", C=float(c), max_iter=params["max_iter"], solver="lbfgs")
                    model.fit(a, y[it]); fold_scores[str(c)].append(float(roc_auc_score(y[iv], model.predict_proba(b)[:, 1])))
            inner_scores = {c: float(np.mean(v)) for c,v in fold_scores.items()}
            chosen_c = float(next(c for c in item["params"]["C_candidates"]
                                  if inner_scores[str(c)] == max(inner_scores.values())))
            model_spec = {**item, "selected_C": chosen_c}
        else:
            model_spec = item
        transform, a, b = fold_transform(x, groups, tr, te)
        scaler = None
        if item["family"] == "logistic":
            scaler = StandardScaler(); a, b = scaler.fit_transform(a), scaler.transform(b)
        model = make_estimator(model_spec, seed)
        fit_ids = groups.iloc[tr].sample_id.astype(str).tolist()
        model.fit(a, y[tr]); predictions[te] = model.predict_proba(b)[:, list(model.classes_).index(1)]
        records.append({"outer_fold":int(outer),"train_rows":int(len(tr)),"test_rows":int(len(te)),
                        "train_positive":int(y[tr].sum()),"test_positive":int(y[te].sum()),
                        "train_sample_ids_sha256":hashlib.sha256("\n".join(sorted(fit_ids)).encode()).hexdigest(),
                        "selected_columns":len(transform.selected),"selected_C":chosen_c,"inner_auc_by_C":inner_scores,
                        "preprocessor_fit_scope":"outer_train_only; inner train only during C selection" if item["family"]=="logistic" else "outer_train_only",
                        "test_train_overlap":False})
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise RuntimeError("OOF prediction coverage or probability range invalid")
    return predictions, records, time.monotonic()-runtime_start


def compute_pair(y: np.ndarray, p: np.ndarray, reference: np.ndarray, *, draws: int = 2000, seed: int = 42):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    delta = np.empty(draws, dtype=float)
    for i in range(draws):
        idx = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        delta[i] = roc_auc_score(y[idx], p[idx]) - roc_auc_score(y[idx], reference[idx])
    return {"point":float(roc_auc_score(y,p)-roc_auc_score(y,reference)),
            "ci95":[float(np.quantile(delta,.025)),float(np.quantile(delta,.975))]}


def score_version(y: np.ndarray, p: np.ndarray, ids: np.ndarray, folds: np.ndarray,
                  p_rf: np.ndarray, p_f3: np.ndarray, parent: np.ndarray | None = None):
    order = np.lexsort((ids,-p))[:min(100,len(y))]
    result = {"auc":float(roc_auc_score(y,p)),"ap":float(average_precision_score(y,p)),
              "recall_at_100":float(y[order].sum()/y.sum()),"brier":float(brier_score_loss(y,p)),
              "top100_true_positive":int(y[order].sum()),"top100_false_positive":int((1-y[order]).sum()),
              "fold_auc":{str(f):float(roc_auc_score(y[folds==f],p[folds==f])) for f in sorted(np.unique(folds))},
              "delta_vs_rf_f0v2":compute_pair(y,p,p_rf),"delta_vs_f3_ebm_a":compute_pair(y,p,p_f3)}
    if parent is not None:
        result["delta_vs_parent_auc"] = float(result["auc"]-roc_auc_score(y,parent))
        po=np.lexsort((ids,-parent))[:min(100,len(y))]
        candidate_set=set(order); parent_set=set(po)
        result["top100_errors_vs_parent"]={"corrected_to_positive":int(sum(y[i]==1 for i in candidate_set-parent_set)),
                                            "new_positive_misses":int(sum(y[i]==1 for i in parent_set-candidate_set)),
                                            "changed_rank_median":float(np.median(np.argsort(np.argsort(-p)) - np.argsort(np.argsort(-parent))))}
    return result


def render_batch(batch_dir: Path, results: dict[str,dict[str,Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=sorted(results,key=lambda n:(results[n]["metrics"]["delta_vs_f3_ebm_a"]["point"],n))
    fig,ax=plt.subplots(figsize=(10,max(4.5,.58*len(names))))
    y=np.arange(len(names)); d=np.array([results[n]["metrics"]["delta_vs_f3_ebm_a"]["point"] for n in names])
    lo=np.array([results[n]["metrics"]["delta_vs_f3_ebm_a"]["ci95"][0] for n in names])
    hi=np.array([results[n]["metrics"]["delta_vs_f3_ebm_a"]["ci95"][1] for n in names])
    colors=["#cf8128" if results[n]["family"]=="ebm" else "#376b8c" for n in names]
    for i,n in enumerate(names): ax.errorbar(d[i],y[i],xerr=[[d[i]-lo[i]],[hi[i]-d[i]]],fmt="o",capsize=3,color=colors[i])
    ax.axvline(0,color="#333",lw=1,label="No change vs F3 EBM-A");ax.axvline(.01,color="#9c5730",ls="--",lw=1,label="+0.01 reference")
    ax.set_yticks(y,names);ax.set_xlabel("Δ pooled OOF AUC vs fixed F3 EBM-A (95% paired vehicle bootstrap)")
    ax.set_title("FEAT-014 B001 | candidate comparison\nDevelopment proxy; iterative-selection intervals are not independent confirmation")
    ax.grid(axis="x",alpha=.2);ax.legend(loc="lower right",fontsize=8);fig.tight_layout()
    fig.savefig(batch_dir/"B1_auc_intervals.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for n in names:
        row=results[n]; metric=row["metrics"]["delta_vs_f3_ebm_a"]["point"]
        ax.scatter(row["fit_seconds"]/60,metric,s=50)
        ax.annotate(n,(row["fit_seconds"]/60,metric),xytext=(4,3),textcoords="offset points",fontsize=8)
    ax.axhline(0,color="#333",lw=1);ax.axhline(.01,color="#9c5730",ls="--",lw=1)
    ax.set_xlabel("Five-fold fit time (minutes)");ax.set_ylabel("Δ pooled OOF AUC vs fixed F3 EBM-A")
    ax.set_title("FEAT-014 B002 | effect and fit cost");ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(batch_dir/"B2_cost_effect.png",dpi=160);plt.close(fig)


def load_run_inputs(run_dir: Path):
    lock = json.loads((run_dir / "input_lock.json").read_text(encoding="utf-8"))
    files = lock["input_paths"]
    frame, manifest, ledger, views, _, y, folds, features = prior.load_locked_inputs(
        Path(files["f3_model_input"]),Path(files["input_manifest"]),Path(files["column_ledger"]))
    for name,path in files.items():
        if sha(Path(path)) != lock["input_sha256"][name]:
            raise ValueError(f"locked E0 input changed after registration: {name}")
    for rel,expected in lock["execution_code_sha256"].items():
        if sha(ROOT/rel)!=expected: raise ValueError(f"locked execution code changed after E0: {rel}")
    p_rf=read_oof(Path(files["rf_oof"]),frame,"p_random_forest")
    p_f3=read_oof(Path(files["f3_oof"]),frame,"p_f3")
    return lock,frame,manifest,ledger,views,y,folds,p_rf,p_f3


def load_view(item: dict[str, Any], frame: pd.DataFrame, views: dict[str,list[str]], run_dir: Path):
    name=item["view"]
    if name=="history_plus_g1":
        x=frame[views["history"]].copy()
        g1path=run_dir/"features_g1_vehicle_day.csv"
        if sha(g1path)!=json.loads((run_dir/"input_lock.json").read_text())["g1_vehicle_day"]["output_sha256"]:
            raise ValueError("locked G1 sidecar changed")
        g1=pd.read_csv(g1path,dtype={"sample_id":"string"})
        if not g1.sample_id.astype(str).equals(frame.sample_id.astype(str)):
            raise ValueError("G1 sidecar sample identity changed")
        return pd.concat([x.reset_index(drop=True),g1[G1_COLUMNS].reset_index(drop=True)],axis=1)
    if name not in views: raise ValueError(f"unknown feature view: {name}")
    return frame[views[name]].copy()


def run_batch(args):
    run_dir=Path(args.run_dir).expanduser().resolve()
    lock,frame,manifest,ledger,views,y,folds,p_rf,p_f3=load_run_inputs(run_dir)
    plan_path=run_dir/"batches"/args.batch/"plan.json"
    plan=json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("input_lock_sha256")!=sha(run_dir/"input_lock.json"):
        raise ValueError("batch plan was made against a different E0 input lock")
    batch_dir=run_dir/"batches"/args.batch
    metrics_path=batch_dir/"metrics.json"
    if metrics_path.exists() and json.loads(metrics_path.read_text()).get("status")=="complete":
        raise FileExistsError("batch is already complete")
    batch_dir.mkdir(parents=True,exist_ok=True)
    results={}; ids=frame.sample_id.astype(str).to_numpy(); pred_cache={}
    failures=[json.loads(p.read_text(encoding="utf-8")) for p in sorted(batch_dir.glob("*/attempts/attempt-*/failure.json"))]
    for item in plan["versions"]:
        version_dir=batch_dir/item["version"]
        if (version_dir/"result.json").is_file():
            prior_result=json.loads((version_dir/"result.json").read_text(encoding="utf-8"))
            saved_pred=read_oof(version_dir/"oof.csv",frame,"probability")
            if sha(version_dir/"oof.csv")!=prior_result["oof_sha256"]: raise ValueError(f"completed version OOF changed: {item['version']}")
            results[item["version"]]=prior_result;pred_cache[item["version"]]=saved_pred
    parent_path=Path(args.parent_oof).resolve() if args.parent_oof else None
    parent=read_oof(parent_path,frame) if parent_path else p_f3
    for item in plan["versions"]:
        name=item["version"]
        version_dir=batch_dir/name
        if (version_dir/"result.json").exists():
            continue
        version_dir.mkdir(parents=True,exist_ok=True)
        attempts=sorted((version_dir/"attempts").glob("attempt-*")) if (version_dir/"attempts").exists() else []
        if len(attempts)>=3: raise RuntimeError(f"retry limit reached for {name}; inspect recorded attempts")
        total_failures=sum(1 for p in batch_dir.glob("*/attempts/attempt-*/failure.json"))
        if total_failures>=10: raise RuntimeError("batch failure retry budget (10) exhausted")
        attempt_no=len(attempts)+1; attempt_dir=version_dir/"attempts"/f"attempt-{attempt_no:03d}"
        attempt_dir.mkdir(parents=True,exist_ok=False)
        start=time.monotonic()
        try:
            x=load_view(item,frame,views,run_dir)
            if x.columns.duplicated().any() or any(c in META for c in x): raise ValueError(f"invalid predictor columns for {name}")
            pred,fold_audit,seconds=fit_version(item,frame,x,y,folds,seed=int(plan["seed"]))
            item_parent=pred_cache.get(item.get("parent"),parent)
            metrics=score_version(y,pred,ids,folds,p_rf,p_f3,item_parent)
            oof_path=attempt_dir/"oof.csv"
            pd.DataFrame({"sample_id":ids,"gpsno":frame.gpsno.astype(str),"y":y,"fold":folds,"probability":pred}).to_csv(oof_path,index=False,lineterminator="\n")
            output={"task":"FEAT-014","batch":args.batch,"version":name,"hypothesis":item["hypothesis"],
                    "parent":item.get("parent") or "fixed_f3_ebm_a","family":item["family"],"view":item["view"],"feature_count":len(x.columns),
                    "columns_sha256":hashlib.sha256("\n".join(x.columns.astype(str)).encode()).hexdigest(),
                    "column_names":x.columns.astype(str).tolist(),
                    "params":item["params"],"seed":int(plan["seed"]),"fit_seconds":seconds,
                    "metrics":metrics,"fold_audit":fold_audit,"oof_sha256":sha(oof_path),"status":"complete",
                    "attempt":attempt_no,"code_commit":lock["code_commit"]}
            json_dump(attempt_dir/"result.json",output)
            # Promote only a fully written attempt to the immutable version result.
            final_oof=version_dir/"oof.csv"; tmp_oof=version_dir/"oof.csv.tmp"
            tmp_oof.write_bytes(oof_path.read_bytes()); tmp_oof.replace(final_oof)
            output["oof_sha256"]=sha(final_oof); json_dump(version_dir/"result.json",output)
            pred_cache[name]=pred; results[name]=output
            print(f"{args.batch} {name}: complete ({seconds:.1f}s)",flush=True)
        except Exception as exc:
            failure={"task":"FEAT-014","batch":args.batch,"version":name,"attempt":attempt_no,
                     "elapsed_seconds":time.monotonic()-start,"error_type":type(exc).__name__,"error":str(exc),"status":"failed"}
            json_dump(attempt_dir/"failure.json",failure);failures.append(failure)
            print(f"{args.batch} {name}: failed ({type(exc).__name__}); retained {attempt_dir}",flush=True)
    metrics_doc={"task":"FEAT-014","batch":args.batch,"status":"partial" if failures or len(results)<len(plan["versions"]) else "complete",
                 "input_lock_sha256":sha(run_dir/"input_lock.json"),"versions":results,"failures":failures,
                 "history_reference":"Fixed RF@F0v2 and F3 EBM-A; parent is only for iteration ranking.",
                 "selection_bias":"Repeated development OOF reuse is exploratory; intervals do not adjust for adaptive selection."}
    json_dump(batch_dir/"metrics.json",metrics_doc)
    pd.DataFrame([{"version":n,"family":d["family"],"view":d["view"],"feature_count":d["feature_count"],
                   "auc":d["metrics"]["auc"],"delta_vs_f3":d["metrics"]["delta_vs_f3_ebm_a"]["point"],
                   "ci_low":d["metrics"]["delta_vs_f3_ebm_a"]["ci95"][0],"ci_high":d["metrics"]["delta_vs_f3_ebm_a"]["ci95"][1],
                   "delta_vs_rf":d["metrics"]["delta_vs_rf_f0v2"]["point"],
                   "rf_ci_low":d["metrics"]["delta_vs_rf_f0v2"]["ci95"][0],"rf_ci_high":d["metrics"]["delta_vs_rf_f0v2"]["ci95"][1],
                   "recall_at_100":d["metrics"]["recall_at_100"],"brier":d["metrics"]["brier"],"seconds":d["fit_seconds"]}
                  for n,d in results.items()]).to_csv(batch_dir/"leaderboard.csv",index=False,lineterminator="\n")
    if results: render_batch(batch_dir,results)
    print(json.dumps({"batch":args.batch,"versions":len(results),"failures":len(failures),"output":str(batch_dir),"status":metrics_doc["status"]},ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare",help="lock E0 inputs and audit prior OOF")
    for flag in ("input","manifest","ledger","rf-oof","f3-oof","feat012","feat013","vehicle-day","output"):
        p.add_argument("--"+flag,required=True)
    b=sub.add_parser("batch",help="run a preregistered adaptive batch")
    b.add_argument("--run-dir",required=True); b.add_argument("--batch",required=True)
    b.add_argument("--parent-oof")
    args=ap.parse_args()
    if args.command=="prepare": prepare(args)
    else: run_batch(args)


if __name__=="__main__": main()
