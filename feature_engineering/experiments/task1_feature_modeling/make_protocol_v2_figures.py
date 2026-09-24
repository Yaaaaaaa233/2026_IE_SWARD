#!/usr/bin/env python3
"""Render controlled Y0/Y1 acceptance figures from protocol-v2 audit outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/feat-009/v2")
    args = ap.parse_args()
    root = Path(args.root)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    y0, y1 = root / "y0", root / "y1"
    (y0 / "figures").mkdir(parents=True, exist_ok=True)
    (y1 / "figures").mkdir(parents=True, exist_ok=True)
    visibility = pd.read_csv(y0 / "visibility.csv")
    counts = visibility.decision.value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    order = [x for x in ("retain_candidate", "exclude_future", "exclude_unverified", "exclude_dependency") if x in counts]
    colors = ["#4a8b68", "#bd5b53", "#e6a23c", "#8b6da8"][:len(order)]
    axes[0].barh(order, [counts[x] for x in order], color=colors)
    axes[0].set_xlabel("Candidate fields")
    axes[0].set_title("Y0 visibility decisions")
    for y, x in enumerate([counts[x] for x in order]): axes[0].text(x + 0.5, y, str(x), va="center")
    ax = axes[1]
    nodes = {
        "portrait": (0.05, .68, "Fleet portrait\nthrough Jul 31", "#f3c4bf"),
        "dynamic": (.42, .68, "Dynamic fields\nexcluded", "#e8a37d"),
        "desc": (.72, .68, "Profile deviation /\ncohort descendants", "#cfb2d5"),
        "history": (.05, .23, "Per-vehicle history\n[Jun 1, Jun 21)", "#b9d8c5"),
        "candidates": (.72, .23, "F0 / F1 / F3\ntrain-fold selection", "#b5cadc"),
    }
    for _, (x, y, label, color) in nodes.items():
        box = FancyBboxPatch((x, y), .23, .17, boxstyle="round,pad=0.015", facecolor=color, edgecolor="#444", transform=ax.transAxes)
        ax.add_patch(box); ax.text(x + .115, y + .085, label, ha="center", va="center", fontsize=9, transform=ax.transAxes)
    for a, b, color in [((.28,.765),(.42,.765),"#b44"),((.65,.765),(.72,.765),"#b44"),((.28,.315),(.72,.315),"#39765b")]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=13,color=color,lw=1.5,transform=ax.transAxes))
    ax.text(.52,.88,"Future-source path excluded",ha="center",color="#9b3636",fontsize=9,transform=ax.transAxes)
    ax.text(.50,.16,"Verified historical window; screening is fitted on training partitions",ha="center",color="#285d43",fontsize=9,transform=ax.transAxes)
    ax.set_axis_off(); ax.set_title("Source and cutoff flow | Asia/Shanghai")
    fig.suptitle("Y0 audit | as_of 2026-06-21 | proxy labels begin at cutoff", fontsize=12)
    fig.tight_layout()
    fig.savefig(y0 / "figures/visibility_map.png", dpi=160)
    plt.close(fig)

    tables = {name: pd.read_csv(y1 / f"{name}_model_input.csv", low_memory=False) for name in ("f0", "f1", "f3")}
    y = tables["f0"].y.to_numpy(int)
    fold = tables["f0"].fold.to_numpy(int)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax=axes[0]
    ax.barh(["Feature window", "Proxy label window", "Fleet portrait period"], [20,40,61], left=[0,20,0], color=["#5f9878","#6e91b0","#c58563"])
    ax.axvline(20,color="#333",ls="--",lw=1)
    ax.set_xticks([0,20,40,61],["Jun 1", "Jun 21", "Jul 11", "Jul 31"])
    ax.set_title("Time alignment (days)"); ax.set_xlabel("2026 | right edges excluded")
    ax=axes[1]
    for i,(name,t) in enumerate(tables.items()):
        cols=[c for c in t.columns if c not in {"sample_id","gpsno","y","fold","label_version","split_version","label_window","horizon_days"}]
        ax.bar(name,len(cols),color=["#758ea8","#8cb59a","#c2906c"][i])
        ax.text(i,len(cols)+2,str(len(cols)),ha="center")
    ax.set_ylim(0,max(len(t.columns) for t in tables.values())+30); ax.set_ylabel("Candidate feature columns")
    ax.set_title("Visible candidate set after exclusions")
    ax=axes[2]
    for cls,color,label in [(0,"#6f93ad","Negative"),(1,"#c26c62","Positive")]:
        vals=[int(((fold==k)&(y==cls)).sum()) for k in range(5)]
        ax.bar(np.arange(5)+(-.18 if cls==0 else .18),vals,width=.36,color=color,label=label)
        for k,v in enumerate(vals): ax.text(k+(-.18 if cls==0 else .18),v+.7,str(v),ha="center",fontsize=8)
    ax.set_xticks(range(5),[f"F{k}" for k in range(5)]); ax.set_title("Frozen split-v2 class counts")
    ax.set_ylabel("Vehicles per fold"); ax.legend(frameon=False,fontsize=8)
    fig.suptitle(f"Y1 rebuilt inputs | n={len(y)} | positives={int(y.sum())} | label_v2_record_count_20260923",fontsize=11)
    fig.tight_layout()
    fig.savefig(y1 / "figures/input_audit.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9,4.5))
    for name, color in zip(("f0","f1","f3"),("#758ea8","#8cb59a","#c2906c")):
        t=tables[name]; cols=[c for c in t if c not in {"sample_id","gpsno","y","fold","label_version","split_version","label_window","horizon_days"}]
        miss=t[cols].isna().mean().sort_values(ascending=False)
        ax.plot(np.arange(1,len(miss)+1),miss.to_numpy(),label=name.upper(),color=color,lw=1.5)
    ax.set_xlabel("Features, sorted by missing fraction within each input")
    ax.set_ylabel("Missing fraction"); ax.set_ylim(0,1); ax.grid(alpha=.2); ax.legend(frameon=False)
    ax.set_title("Feature missingness profile | all 500 vehicles retained")
    fig.tight_layout(); fig.savefig(y1 / "figures/missingness.png",dpi=160); plt.close(fig)


if __name__ == "__main__": main()
