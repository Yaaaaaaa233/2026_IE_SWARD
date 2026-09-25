#!/usr/bin/env python3
"""Create clearly marked synthetic chart-layout previews for FEAT-014."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True,type=Path);args=ap.parse_args()
    out=args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"refusing to overwrite preview directory: {out}")
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(14026)
    watermark="SYNTHETIC LAYOUT • NO EXPERIMENT DATA"
    names=["V001 LR","V002 RF","V003 HGB","V004 EBM interaction","V005 slim EBM","V006 history+G1"]
    means=np.array([-.012,-.003,.002,.007,.011,.005]);err=np.array([.018,.014,.013,.012,.016,.015])
    y=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(10,5))
    ax.errorbar(means,y,xerr=err,fmt="o",capsize=4,color="#376b8c")
    ax.axvline(0,color="#333");ax.axvline(.01,color="#9c5730",ls="--",label="+0.01 reference")
    ax.set_yticks(y,names);ax.set_xlabel("Δ pooled OOF AUC vs fixed F3 EBM-A (95% paired interval)")
    ax.set_title("FEAT-014 B1 • synthetic layout preview\n"+watermark);ax.legend();ax.grid(axis="x",alpha=.2)
    fig.tight_layout();fig.savefig(out/"B1_auc_intervals_synthetic.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    cost=np.array([2.4,1.8,1.1,4.2,2.6,1.5]);ax.scatter(cost,means,s=55,color="#cf8128")
    for x,m,n in zip(cost,means,names):ax.annotate(n,(x,m),xytext=(4,4),textcoords="offset points",fontsize=8)
    ax.axhline(0,color="#333");ax.axhline(.01,color="#9c5730",ls="--")
    ax.set_xlabel("Five-fold fit time (minutes)");ax.set_ylabel("Δ pooled OOF AUC vs F3 EBM-A")
    ax.set_title("FEAT-014 B2 • synthetic layout preview\n"+watermark);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/"B2_cost_effect_synthetic.png",dpi=150);plt.close(fig)

    groups=["event", "exposure", "night", "history", "IMU"]
    err_types=["False positive", "Missed positive"]
    heat=rng.integers(0,13,size=(len(err_types),len(groups)))
    fig,ax=plt.subplots(figsize=(8,3.7));im=ax.imshow(heat,cmap="Blues",aspect="auto")
    ax.set_xticks(range(len(groups)),groups);ax.set_yticks(range(2),err_types)
    for i in range(2):
        for j in range(len(groups)):ax.text(j,i,str(int(heat[i,j])),ha="center",va="center",fontsize=9)
    ax.set_title("FEAT-014 C0 • synthetic error-group heatmap\n"+watermark);fig.colorbar(im,ax=ax,label="synthetic case count")
    fig.tight_layout();fig.savefig(out/"C0_error_heatmap_synthetic.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));labels=["Fixed anchor","Candidate"]
    corrected=[0,8];new_misses=[0,3];x=np.arange(2)
    ax.bar(x,corrected,label="Corrected positive misses",color="#5b9b78");ax.bar(x,new_misses,bottom=corrected,label="New positive misses",color="#c96b59")
    ax.set_xticks(x,labels);ax.set_ylabel("Synthetic Top-100 cases");ax.set_title("FEAT-014 C1 • synthetic Top-100 change preview\n"+watermark);ax.legend()
    fig.tight_layout();fig.savefig(out/"C1_top100_changes_synthetic.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));single=[.80,.82,.83];fusion=[.81,.84,.845];x=np.arange(3);w=.35
    ax.bar(x-w/2,single,w,label="Single models",color="#376b8c");ax.bar(x+w/2,fusion,w,label="Fixed fusion",color="#cf8128")
    ax.set_xticks(x,["full+history","slim+history","event+night"]);ax.set_ylim(.76,.87);ax.set_ylabel("Synthetic pooled OOF AUC")
    ax.set_title("FEAT-014 C2 • synthetic fusion comparison\n"+watermark);ax.legend()
    fig.tight_layout();fig.savefig(out/"C2_fusion_synthetic.png",dpi=150);plt.close(fig)

    versions=np.arange(1,19);scores=.82+np.cumsum(rng.normal(.001,.003,size=18))
    fig,ax=plt.subplots(figsize=(9,4));ax.plot(versions,scores,marker="o",label="Version point score");ax.plot(versions,np.maximum.accumulate(scores),ls="--",label="Best seen so far")
    ax.axhline(.82,color="#333",label="Fixed anchor");ax.set_xlabel("Registered version");ax.set_ylabel("Synthetic pooled OOF AUC")
    ax.set_title("FEAT-014 D0 • synthetic iteration curve\n"+watermark);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/"D0_iteration_curve_synthetic.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));seed_means=np.array([.842,.848,.845]);seed_lo=np.array([.827,.832,.831]);seed_hi=np.array([.858,.863,.859]);xx=np.arange(3)
    ax.errorbar(seed_means,xx,xerr=[seed_means-seed_lo,seed_hi-seed_means],fmt="o",capsize=4,color="#8b69a6")
    ax.set_yticks(xx,["Candidate A","Candidate B","Candidate C"]);ax.set_xlabel("Synthetic pooled OOF AUC (seed spread)")
    ax.set_title("FEAT-014 D1 • synthetic seed-stability preview\n"+watermark);ax.grid(axis="x",alpha=.2)
    fig.tight_layout();fig.savefig(out/"D1_seed_stability_synthetic.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));names2=["Fixed RF","Fixed F3","Candidate"];deltas=[0,.01,.014];los=[0,-.002,.002];his=[0,.021,.027]
    for i,(m,l,h) in enumerate(zip(deltas,los,his)):ax.errorbar(m,i,xerr=[[m-l],[h-m]],fmt="o",capsize=4,color="#376b8c")
    ax.axvline(0,color="#333");ax.axvline(.01,color="#9c5730",ls="--",label="+0.01 meaning reference")
    ax.set_yticks(range(3),names2);ax.set_xlabel("Synthetic Δ pooled OOF AUC");ax.set_title("FEAT-014 D2 • fixed-anchor decision preview\n"+watermark);ax.legend()
    fig.tight_layout();fig.savefig(out/"D2_anchor_comparison_synthetic.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));feature_groups=["event","night","trajectory","history","G1"]
    full=np.array([.85,.85,.85,.85,.85]);ablated=np.array([.84,.852,.847,.842,.856]);xx=np.arange(len(feature_groups));w=.35
    ax.bar(xx-w/2,full,w,label="Full candidate",color="#376b8c");ax.bar(xx+w/2,ablated,w,label="Ablation",color="#cf8128")
    ax.set_xticks(xx,feature_groups);ax.set_ylim(.80,.87);ax.set_ylabel("Synthetic pooled OOF AUC")
    ax.set_title("FEAT-014 D3 • synthetic feature ablation preview\n"+watermark);ax.legend()
    fig.tight_layout();fig.savefig(out/"D3_ablation_synthetic.png",dpi=150);plt.close(fig)
    print(f"synthetic previews saved: {out}; files={len(list(out.glob('*.png')))}")


if __name__=="__main__":main()
