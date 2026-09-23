#!/usr/bin/env python3
"""Post-hoc, explicitly exploratory regularization check; never replaces Y3 primary OOF."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from modeling import fit_transformer, make_candidate, fit_predict
from run_protocol_v2 import paired_auc, recall_at_k

REPO=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").exists())
SPEC={"name":"EBM-C-regularized","kind":"ebm","interactions":0,"max_bins":32,"min_samples_leaf":20}
META={"sample_id","gpsno","y","fold","label_version","split_version","label_window","horizon_days"}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',default='outputs/feat-009/v2/y1'); ap.add_argument('--rf-oof',default='outputs/feat-009/v2/y2/new_rf_anchor/task1_baseline_oof_predictions.csv'); ap.add_argument('--primary-oof',default='outputs/feat-009/v2/y3/20260924-r2/oof_f3.csv'); ap.add_argument('--output',required=True); args=ap.parse_args()
    inp=Path(args.inputs); out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    f3p=inp/'f3_model_input.csv'; rfpath=Path(args.rf_oof); mainpath=Path(args.primary_oof)
    reg={'registration_id':'FEAT-009-EXPLORATORY-20260924-EBM-C','registered_at':'2026-09-24','status':'locked_before_exploration',
         'question':'Does stronger smoothing of the F3 Explainable Boosting model preserve the proxy ranking gain while reducing variance?',
         'protocol':'Reuse the same fixed protocol-v2 outer folds for a post-hoc robustness/mechanism check; this is not an independent validation and cannot replace or upgrade Y3.',
         'model':SPEC,'fit_scope':'Fixed EBM-C per outer fold; all input filtering, imputation and encoding fit on that fold training set. No candidate selection.',
         'comparator':'new protocol-v2 RF anchor and primary F3 EBM-selected OOF, both already observed; exploratory only.',
         'input_sha256':{'f3':sha(f3p),'rf_oof':sha(rfpath),'primary_f3_oof':sha(mainpath)},
         'seeds':{'model':42,'paired_bootstrap':42},'metrics':['AUC','paired 95% bootstrap delta AUC (2000)','Recall@100','Brier'],
         'decision_boundary':'No formal pass/fail or adoption claim. Results remain post-hoc because this hypothesis was chosen after Y3.'}
    (out/'registration.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n')
    frame=pd.read_csv(f3p,dtype={'sample_id':'string','gpsno':'string'},low_memory=False)
    rf=pd.read_csv(rfpath,dtype={'sample_id':'string'}); primary=pd.read_csv(mainpath,dtype={'sample_id':'string'})
    if not frame.sample_id.astype(str).equals(rf.sample_id.astype(str)) or not frame.sample_id.astype(str).equals(primary.sample_id.astype(str)): raise ValueError('sample order differs')
    y=frame.y.to_numpy(int); fold=frame.fold.to_numpy(int); ids=frame.sample_id.astype(str).to_numpy(); groups=frame[['sample_id','gpsno']].copy(); x=frame[[c for c in frame if c not in META]].copy()
    pred=np.full(len(y),np.nan); details=[]; start=time.monotonic()
    for k in range(5):
        tr=np.flatnonzero(fold!=k); te=np.flatnonzero(fold==k)
        transform=fit_transformer(x.iloc[tr],groups.iloc[tr],cohort_sources=[])
        a=transform.transform(x.iloc[tr],groups.iloc[tr]); b=transform.transform(x.iloc[te],groups.iloc[te])
        model=make_candidate(SPEC,seed=42); pp,scope=fit_predict(model,a,y[tr],b,groups.iloc[tr].sample_id.astype(str).tolist(),SPEC['name'])
        pred[te]=pp
        details.append({'fold':k,'n_train':len(tr),'n_test':len(te),'n_positive':int(y[te].sum()),'auc':float(roc_auc_score(y[te],pp)),'selected_features':transform.selected,'fit_scope':scope,'preprocessor_train_sha256':transform.training_fingerprint})
    if not np.isfinite(pred).all(): raise RuntimeError('incomplete exploratory OOF')
    p_rf=rf.p_random_forest.to_numpy(float); p_main=primary.p_f3.to_numpy(float)
    def metrics(p,base_name,base):
        ci=paired_auc(y,p,base,n=2000,seed=42)
        return {'comparator':base_name,**ci,'auc':float(roc_auc_score(y,p)),'recall_at_100':recall_at_k(y,p,ids),'recall_at_100_delta':recall_at_k(y,p,ids)-recall_at_k(y,base,ids),'brier':float(brier_score_loss(y,p)),'brier_delta':float(brier_score_loss(y,p)-brier_score_loss(y,base))}
    result={'run_id':out.name,'status':'exploratory_posthoc','n':len(y),'n_positive':int(y.sum()),'model':SPEC,'runtime_seconds':time.monotonic()-start,
            'versus_rf':metrics(pred,'new_rf_anchor',p_rf),'versus_primary_f3':metrics(pred,'F3_selected_primary',p_main),
            'folds':details,'interpretation_limit':'Same vehicles and fixed folds were already used to choose this follow-up question. Descriptive only; no formal acceptance claim.'}
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    pd.DataFrame({'sample_id':ids,'gpsno':frame.gpsno.astype(str),'y':y,'fold':fold,'p_ebm_c':pred}).to_csv(out/'oof.csv',index=False,lineterminator='\n')
    (out/'manifest.json').write_text(json.dumps({'created_at_utc':datetime.now(timezone.utc).isoformat(),'code_commit':subprocess.run(['git','-C',str(REPO),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),'python':platform.python_version(),'seed':42,'registration_sha256':sha(out/'registration.json'),'inputs':reg['input_sha256'],'outputs':{'oof.csv':sha(out/'oof.csv'),'metrics.json':sha(out/'metrics.json')},'human_review':'pending'},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'run_id':out.name,'versus_rf':result['versus_rf'],'versus_primary_f3':result['versus_primary_f3'],'runtime_seconds':result['runtime_seconds']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
