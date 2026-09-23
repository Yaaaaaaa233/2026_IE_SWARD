#!/usr/bin/env python3
"""Create same-run Y4 proxy diagnostics from saved v2 OOF files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rf-oof", default="outputs/feat-009/v2/y2/new_rf_anchor/task1_baseline_oof_predictions.csv")
    ap.add_argument("--f0", default="outputs/feat-009/v2/y1/f0_model_input.csv")
    args = ap.parse_args()
    run = Path(args.run_dir); rf_path = Path(args.rf_oof); f0_path = Path(args.f0)
    out = run.parent.parent / "y4"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rf = pd.read_csv(rf_path, dtype={"sample_id":"string"})[["sample_id","p_random_forest"]]
    oofs = {name: pd.read_csv(run / f"oof_{name.lower()}.csv", dtype={"sample_id":"string"})
            for name in ("F1","F3")}
    frame = oofs["F3"].merge(oofs["F1"], on=["sample_id","y"], validate="1:1").merge(rf,on="sample_id",validate="1:1")
    if len(frame) != 500 or frame.sample_id.duplicated().any() or frame.y.isna().any():
        raise ValueError("OOF join is incomplete or duplicated")
    ids=frame.sample_id.astype(str).to_numpy(); y=frame.y.to_numpy(int)
    p3=frame.p_f3.to_numpy(float); p1=frame.p_f1.to_numpy(float); pr=frame.p_random_forest.to_numpy(float)
    def ranked(p): return np.lexsort((ids,-p))
    top={name:set(ranked(p)[:100]) for name,p in (("F1",p1),("F3",p3),("RF",pr))}
    overlaps={f"{a}_x_{b}":len(top[a]&top[b]) for a,b in (("F1","F3"),("F3","RF"),("F1","RF"))}
    summary={"run_id":run.name,"rows":len(frame),"positives":int(y.sum()),"top_100_overlap":overlaps,
             "top_100_hits":{name:int(y[list(ix)].sum()) for name,ix in top.items()},
             "notes":"Descriptive diagnostics on the same development proxy OOF; not independent validation.",
             "coverage_by_prediction_group":{}}
    pred_group=np.where(np.isin(np.arange(len(y)),list(top["F3"])),"F3 top 100","outside F3 top 100")
    f0=pd.read_csv(f0_path,dtype={"sample_id":"string"},low_memory=False)
    coverage=[c for c in ("coverage_days","coverage_imu_observed_seconds","coverage_imu_traj_overlap_ratio") if c in f0]
    if coverage:
        cov=frame[["sample_id"]].merge(f0[["sample_id",*coverage]],on="sample_id",validate="1:1")
        for col in coverage:
            values=pd.to_numeric(cov[col],errors="coerce")
            summary["coverage_by_prediction_group"][col]={}
            for group in ("F3 top 100","outside F3 top 100"):
                v=values[pred_group==group]
                summary["coverage_by_prediction_group"][col][group]={"n_observed":int(v.notna().sum()),"n_group":int((pred_group==group).sum()),"n_missing":int(v.isna().sum()),"median":float(v.median()) if v.notna().any() else None}
    (out/"diagnosis.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    pd.DataFrame([{"model_pair":k,"vehicles_in_both_top_100":v,"top_k":100} for k,v in overlaps.items()]).to_csv(out/"top100_overlap.csv",index=False,lineterminator="\n")
    # Percentile ranks are easier to interpret than raw probabilities across model families.
    ranks={name:pd.Series(p).rank(method="average",ascending=False,pct=True).to_numpy() for name,p in (("F1",p1),("F3",p3),("RF",pr))}
    fig,axes=__import__("matplotlib.pyplot",fromlist=["subplots"]).subplots(1,2,figsize=(10,4.6))
    for ax,other,name in ((axes[0],"RF","F3 vs RF"),(axes[1],"F1","F3 vs F1")):
        ax.scatter(ranks[other],ranks["F3"],c=np.where(y==1,"#c35c52","#6d93aa"),s=15,alpha=.65,linewidths=0)
        ax.plot([0,1],[0,1],color="#444",lw=1,ls="--")
        ax.set_xlabel(f"{other} risk rank percentile"); ax.set_ylabel("F3 risk rank percentile")
        ax.set_title(name); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=.15)
    fig.suptitle(f"Risk rank relationship | same proxy OOF run {run.name}; red = positive\nLower percentile means ranked as higher risk")
    fig.tight_layout(); fig.savefig(out/"figures/rank_relationship.png",dpi=160); __import__("matplotlib.pyplot",fromlist=["close"]).close(fig)
    fig,ax=__import__("matplotlib.pyplot",fromlist=["subplots"]).subplots(figsize=(7,4))
    pairs=list(overlaps)
    ax.bar(pairs,[overlaps[k] for k in pairs],color="#789f87")
    ax.axhline(100,color="#555",ls=":",lw=1)
    for i,k in enumerate(pairs): ax.text(i,overlaps[k]+2,str(overlaps[k]),ha="center")
    ax.set_ylim(0,110); ax.set_ylabel("Vehicles shared in the top 100")
    ax.set_title("Top-100 overlap | ties resolved by sample_id")
    fig.tight_layout(); fig.savefig(out/"figures/top100_overlap.png",dpi=160); __import__("matplotlib.pyplot",fromlist=["close"]).close(fig)


if __name__ == "__main__": main()
