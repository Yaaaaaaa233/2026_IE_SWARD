#!/usr/bin/env python3
"""Independently verify the saved FEAT-009 v2 OOF run and its preregistration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

REPO=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").exists())
BLOCKED={"monthly_avg_mileage","monthly_avg_hours","monthly_avg_stops","highway_ratio","morning_ratio","dusk_ratio","night_hours_ratio","night_mileage_ratio","energy_type","cohort_fallback","f2_prior_incident_x_night","f2_prior_incident_x_highway"}

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def recall(y,p,ids):
    ix=np.lexsort((ids.astype(str),-p))[:min(100,len(y))]
    return float(y[ix].sum()/y.sum())
def delta_ci(y,a,b,n=2000,seed=42):
    pos,neg=np.flatnonzero(y==1),np.flatnonzero(y==0); rng=np.random.default_rng(seed); vals=np.empty(n)
    for i in range(n):
        ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        vals[i]=roc_auc_score(y[ix],a[ix])-roc_auc_score(y[ix],b[ix])
    return float(roc_auc_score(y,a)-roc_auc_score(y,b)),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--root',default='outputs/feat-009/v2'); args=ap.parse_args()
    run=Path(args.run_dir); root=Path(args.root); regp=root/'y2/pre_registration.json'; reg=json.loads(regp.read_text()); manifest=json.loads((run/'manifest.json').read_text()); metrics=json.loads((run/'metrics.json').read_text())
    errors=[]; inputs=root/'y1'; rfpath=root/'y2/new_rf_anchor/task1_baseline_oof_predictions.csv'
    filemap={'f0_model_input.csv':inputs/'f0_model_input.csv','f1_model_input.csv':inputs/'f1_model_input.csv','f3_model_input.csv':inputs/'f3_model_input.csv','new_rf_oof':rfpath}
    for key,path in filemap.items():
        expected=reg['source_inputs_sha256'].get(key)
        if not path.is_file() or sha(path)!=expected: errors.append(f'preregistered input fingerprint mismatch: {key}')
    if sha(regp)!=manifest.get('pre_registration_sha256'): errors.append('pre-registration fingerprint differs from run manifest')
    tables={n:pd.read_csv(inputs/f'{n.lower()}_model_input.csv',dtype={'sample_id':'string'},low_memory=False) for n in ('F0','F1','F3')}
    ref=tables['F0']; y=ref.y.to_numpy(int); folds=ref.fold.to_numpy(int); ids=ref.sample_id.astype(str).to_numpy()
    rf=pd.read_csv(rfpath,dtype={'sample_id':'string'}); baseline=rf.p_random_forest.to_numpy(float)
    if not rf.sample_id.astype(str).equals(ref.sample_id.astype(str)) or not rf.y.astype(int).equals(ref.y.astype(int)) or not rf.fold.astype(int).equals(ref.fold.astype(int)):
        errors.append('RF OOF keys/labels/folds differ from frozen v2 input')
    candidates={}
    for name,t in tables.items():
        cols=[c for c in t if c not in {'sample_id','gpsno','y','fold','label_version','split_version','label_window','horizon_days'}]
        blocked=[c for c in cols if c in BLOCKED or c.startswith('f3_profile_') or '_cohort_' in c]
        if blocked: errors.append(f'{name} contains forbidden portrait descendants: {blocked}')
        if not t.sample_id.astype(str).equals(ref.sample_id.astype(str)) or not t.y.astype(int).equals(ref.y.astype(int)) or not t.fold.astype(int).equals(ref.fold.astype(int)):
            errors.append(f'{name} keys/labels/folds differ from F0')
        if name in ('F1','F3'):
            o=pd.read_csv(run/f'oof_{name.lower()}.csv',dtype={'sample_id':'string'})
            col=f'p_{name.lower()}'
            if not o.sample_id.astype(str).equals(ref.sample_id.astype(str)) or not o.y.astype(int).equals(ref.y.astype(int)):
                errors.append(f'{name} OOF keys/labels differ from frozen input')
            p=o[col].to_numpy(float)
            if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): errors.append(f'{name} has invalid predictions')
            candidates[name]=p
    recomputed={}
    for name,p in candidates.items():
        d,lo,hi=delta_ci(y,p,baseline)
        item=metrics['comparisons'][name]
        values={'delta_auc':d,'ci_low':lo,'ci_high':hi,'auc':float(roc_auc_score(y,p)),
                'recall_at_100_delta':recall(y,p,ids)-recall(y,baseline,ids),
                'brier_delta':float(brier_score_loss(y,p)-brier_score_loss(y,baseline))}
        recomputed[name]=values
        for k,v in values.items():
            if abs(item[k]-v)>1e-12: errors.append(f'{name} metric does not reproduce: {k}')
        if not np.array_equal(np.bincount(folds,minlength=5),np.bincount(ref.fold.to_numpy(int),minlength=5)):
            errors.append('outer folds do not cover the fixed sample')
    status='pass' if not errors else 'fail'
    audit={'machine_status':status,'human_review':'pending','run_id':run.name,'registration_id':reg.get('registration_id'),
           'checks':{'input_fingerprints':not any('fingerprint mismatch' in x for x in errors),'same_keys_labels_folds':not any('keys/labels/folds' in x for x in errors),
                     'forbidden_features_absent':not any('forbidden' in x for x in errors),'predictions_valid':not any('invalid predictions' in x for x in errors),
                     'metrics_recomputed':not any('metric does not reproduce' in x for x in errors)},
           'recomputed_metrics':recomputed,'errors':errors,'review_note':'Proxy evidence only; public label review and human plot/data audit remain outstanding.'}
    (run/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(1)
if __name__=='__main__': main()
