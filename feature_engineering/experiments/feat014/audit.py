#!/usr/bin/env python3
"""Independent E0/E1 checks for FEAT-014 controlled runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from feature_engineering.experiments.feat014 import run as e


def audit_e0(run_dir: Path) -> dict:
    lock=json.loads((run_dir/"input_lock.json").read_text(encoding="utf-8"))
    errors=[]
    for name,path in lock["input_paths"].items():
        if e.sha(Path(path))!=lock["input_sha256"][name]: errors.append(f"changed locked input: {name}")
    inp=Path(lock["input_paths"]["f3_model_input"])
    frame,manifest,ledger,views,_,y,folds,features=e.prior.load_locked_inputs(
        inp,Path(lock["input_paths"]["input_manifest"]),Path(lock["input_paths"]["column_ledger"]))
    if len(frame)!=lock["rows"] or int(y.sum())!=lock["positives"]: errors.append("input row/label conservation failed")
    for name in ("FEAT-012","FEAT-013"):
        if lock["historical_runs"][name]["audit"].get("status")!="pass": errors.append(f"prior audit failed: {name}")
    hist=pd.read_csv(run_dir/"historical_ledger.csv")
    expected=sum(lock["historical_runs"][name]["versions"] for name in ("FEAT-012","FEAT-013"))
    if len(hist)!=expected: errors.append("historical version count mismatch")
    for row in hist.itertuples(index=False):
        p=Path(lock["historical_runs"][row.task]["path"])/f"oof_{row.version}.csv"
        if not p.is_file() or e.sha(p)!=row.oof_sha256: errors.append(f"historical OOF changed: {row.task}/{row.version}")
        else:
            try: e.read_oof(p,frame)
            except Exception as exc: errors.append(f"historical OOF row binding failed: {row.task}/{row.version}: {exc}")
    g1path=run_dir/"features_g1_vehicle_day.csv"
    if not g1path.is_file() or e.sha(g1path)!=lock["g1_vehicle_day"]["output_sha256"]:
        errors.append("G1 sidecar hash mismatch")
    else:
        daily=pd.read_csv(lock["input_paths"]["vehicle_day"],dtype={"gpsno":"string"},low_memory=False)
        g1,stats=e.extract_g1_features(daily,frame[["sample_id","gpsno"]],cutoff="2026-06-21")
        saved=pd.read_csv(g1path,dtype={"sample_id":"string","gpsno":"string"})
        try: pd.testing.assert_frame_equal(saved,g1,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
        except AssertionError as exc: errors.append(f"G1 feature reconstruction failed: {exc}")
        if stats["excluded_at_or_after_cutoff_rows"]!=lock["g1_vehicle_day"]["excluded_at_or_after_cutoff_rows"]:
            errors.append("G1 cutoff audit mismatch")
    for image in ("A0_time_windows.png","A1_feature_families.png"):
        if not (run_dir/image).is_file(): errors.append(f"missing E0 chart: {image}")
    batch=json.loads((run_dir/"batches/B001/plan.json").read_text(encoding="utf-8"))
    if batch.get("input_lock_sha256")!=e.sha(run_dir/"input_lock.json"): errors.append("B001 plan does not bind the E0 lock")
    return {"stage":"E0","status":"fail" if errors else "pass","rows":len(frame),"historical_versions":len(hist),
            "g1_features":e.G1_COLUMNS,"charts":["A0_time_windows.png","A1_feature_families.png"],"errors":errors,
            "human_visual_review":"pending; charts are evidence artifacts, not automated human approval"}


def audit_e1(run_dir: Path, batch_name: str) -> dict:
    batch=run_dir/"batches"/batch_name
    lock,frame,manifest,ledger,views,y,folds,p_rf,p_f3=e.load_run_inputs(run_dir)
    plan=json.loads((batch/"plan.json").read_text(encoding="utf-8"))
    metrics_doc=json.loads((batch/"metrics.json").read_text(encoding="utf-8"))
    errors=[]; details=[]; ids=frame.sample_id.astype(str).to_numpy()
    for item in plan["versions"]:
        name=item["version"]; version_dir=batch/name
        result=json.loads((version_dir/"result.json").read_text(encoding="utf-8"))
        oof_path=version_dir/"oof.csv"
        if e.sha(oof_path)!=result["oof_sha256"]: errors.append(f"OOF hash mismatch: {name}")
        p=e.read_oof(oof_path,frame,"probability")
        parent_name=item.get("parent") or "fixed_f3_ebm_a"
        if parent_name=="fixed_f3_ebm_a": pp=p_f3
        else: pp=e.read_oof(batch/parent_name/"oof.csv",frame,"probability")
        expected=e.score_version(y,p,ids,folds,p_rf,p_f3,pp)
        for key in ("auc","ap","recall_at_100","brier"):
            if not np.isclose(expected[key],result["metrics"][key],atol=1e-12,rtol=0): errors.append(f"{name} metric mismatch: {key}")
        for ref in ("delta_vs_rf_f0v2","delta_vs_f3_ebm_a"):
            a,b=expected[ref],result["metrics"][ref]
            if not np.isclose(a["point"],b["point"],atol=1e-12,rtol=0) or not np.allclose(a["ci95"],b["ci95"],atol=1e-12,rtol=0): errors.append(f"{name} paired interval mismatch: {ref}")
        if expected["fold_auc"]!=result["metrics"]["fold_auc"]: errors.append(f"{name} fold AUC mismatch")
        if not np.isclose(expected["delta_vs_parent_auc"],result["metrics"]["delta_vs_parent_auc"],atol=1e-12,rtol=0): errors.append(f"{name} parent delta mismatch")
        if expected["top100_errors_vs_parent"]!=result["metrics"]["top100_errors_vs_parent"]: errors.append(f"{name} top100 error comparison mismatch")
        fold_audit=result["fold_audit"]
        if sorted(r["outer_fold"] for r in fold_audit)!=[0,1,2,3,4]: errors.append(f"{name} lacks five outer folds")
        for row in fold_audit:
            outer=int(row["outer_fold"]); train_ids=frame.loc[folds!=outer,"sample_id"].astype(str).tolist()
            expected_hash=hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest()
            if row["train_sample_ids_sha256"]!=expected_hash or row["test_train_overlap"]: errors.append(f"{name} fold fit-scope mismatch: {outer}")
        if result["columns_sha256"]!=hashlib.sha256("\n".join(result.get("column_names",[])).encode()).hexdigest():
            errors.append(f"{name} predictor names hash mismatch")
        details.append({"version":name,"status":"pass" if not any(name in x for x in errors) else "fail"})
    if set(metrics_doc["versions"])!=set(x["version"] for x in details): errors.append("batch summary version set differs")
    return {"stage":"E1","batch":batch_name,"status":"fail" if errors else "pass","versions":details,"errors":errors,
            "note":"Independent OOF, metric, pairing, and outer-fit-scope checks; adaptive-selection bias is not removed."}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run-dir",required=True,type=Path);ap.add_argument("--stage",choices=["E0","E1"],required=True);ap.add_argument("--batch",default="B001");ap.add_argument("--write",action="store_true")
    args=ap.parse_args();run=args.run_dir.expanduser().resolve()
    result=audit_e0(run) if args.stage=="E0" else audit_e1(run,args.batch)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.write:
        name="independent_audit_e0.json" if args.stage=="E0" else f"batches/{args.batch}/independent_audit.json"
        e.json_dump(run/name,result)
    if result["status"]!="pass": raise SystemExit(1)


if __name__=="__main__": main()
