#!/usr/bin/env python3
"""Create clearly synthetic FEAT-011 S0 visual acceptance examples."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(ax, xy, width, height, text, face, *, edge="#52616b", fontsize=10):
    patch = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.025,rounding_size=0.04",
                           linewidth=1, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(xy[0] + width/2, xy[1] + height/2, text, ha="center", va="center", fontsize=fontsize)


def make_timeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 8.1); ax.axis("off")
    ax.set_title("FEAT-011 S0 synthetic preview: time windows and field provenance", fontsize=14, pad=12)
    x0, x1, x2, x3 = 8, 35, 90, 95
    ax.text(50,7.72,"SYNTHETIC EXAMPLE — no real records, counts, or model results",ha="center",fontsize=10,color="#a12b2b",weight="bold")
    ax.annotate("", xy=(x3, 7.0), xytext=(x0, 7.0), arrowprops={"arrowstyle":"->", "lw":1.5})
    for x, label, align in [(x0,"6/1","center"),(x1,"6/21 cutoff","center"),
                            (x2,"7/31 proxy end","right"),(x3,"8/1","left")]:
        ax.plot([x,x],[6.88,7.13],color="#263238",lw=1)
        label_x = x-1.0 if x == x2 else x+1.0 if x == x3 else x
        ax.text(label_x,7.28,label,ha=align,fontsize=9)
    ax.add_patch(plt.Rectangle((x0,6.25),x1-x0,0.34,color="#75a9c5",alpha=.85))
    ax.text((x0+x1)/2,6.42,"20-day proxy feature window",ha="center",va="center",fontsize=9,color="white",weight="bold")
    ax.add_patch(plt.Rectangle((x1,6.25),x2-x1,0.34,color="#d6a253",alpha=.9))
    ax.text((x1+x2)/2,6.42,"40-day proxy label window",ha="center",va="center",fontsize=9,color="white",weight="bold")
    ax.add_patch(plt.Rectangle((x0,5.72),x3-x0,.32,color="#94b7a2",alpha=.9))
    ax.text((x0+x3)/2,5.88,"Final submission history: 6/1–7/31 (61 calendar days); prediction starts 8/1",ha="center",va="center",fontsize=9,color="white",weight="bold")

    headers = [(4, "Source family"), (34, "Time relative to cutoff"), (65, "Treatment in model input")]
    widths = [29, 30, 31]
    for (x, label), width in zip(headers, widths):
        box(ax,(x,4.85),width,.52,label,"#e8edf0",fontsize=9)
    rows = [
        ("Behavior/event history", "records before 6/21", "candidate history features\nverify source and visibility", "#e4eef3"),
        ("GPS / IMU", "observations before 6/21", "candidate sensor features\ncheck coverage and meaning", "#e4eef3"),
        ("Proxy outcome records", "6/21 through proxy end", "target y only\nnever enter X", "#f5ead7"),
        ("Portrait generated after cutoff", "after 6/21", "blocked from predictors", "#f2dddd"),
    ]
    for row_index, (source, time_text, treatment, color) in enumerate(rows):
        y = 4.05 - row_index*.78
        for (x, _), width, text in zip(headers, widths, (source, time_text, treatment)):
            box(ax,(x,y),width,.62,text,color,fontsize=8.5)
    ax.text(50,.56,"Night definitions stay attached to the existing columns; this experiment removes selected research deep-night columns only.",
            ha="center",fontsize=9,color="#39464e")
    ax.text(50,.2,"Official night: 21:00–06:00   |   Research deep-night columns: 23:00–05:00   |   Proxy and final windows are separate contracts.",
            ha="center",fontsize=8.5,color="#39464e")
    fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches="tight")
    plt.close(fig)


def make_views(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.set_title("FEAT-011 S0 synthetic preview: fixed feature views", fontsize=14, pad=12)
    ax.text(5.5,5.65,"SYNTHETIC EXAMPLE — illustrates the 2×2 comparison, not results",ha="center",fontsize=10,color="#a12b2b",weight="bold")
    box(ax,(.55,4.55),3.7,.65,"Feature view", "#e8edf0",fontsize=10)
    box(ax,(4.55,4.55),2.35,.65,"EBM-A", "#e8edf0",fontsize=10)
    box(ax,(7.2,4.55),2.35,.65,"EBM-C", "#e8edf0",fontsize=10)
    box(ax,(.55,3.05),3.7,1.05,"Full F3\n(base + night summaries + detail)","#d8e9f0",fontsize=9)
    box(ax,(4.55,3.05),2.35,1.05,"Arm A\nmain comparator", "#dce7f5",fontsize=10)
    box(ax,(7.2,3.05),2.35,1.05,"Arm C\nmodel diagnostic", "#dce7f5",fontsize=10)
    box(ax,(.55,1.55),3.7,1.05,"Slim F3\n(base + 4 night summaries)","#e1eee4",fontsize=9)
    box(ax,(4.55,1.55),2.35,1.05,"Arm B\nfeature diagnostic", "#e5eee2",fontsize=10)
    box(ax,(7.2,1.55),2.35,1.05,"Arm D\nfeature/model diagnostic", "#e5eee2",fontsize=10)
    box(ax,(3.1,.2),4.8,.68,"E = fixed equal logit average of OOF arms A and D", "#f5e7d2",fontsize=9)
    ax.text(5.5,.98,"The two feature views share one locked input; no adaptive model or weight search is shown.",
            ha="center",fontsize=9,color="#39464e")
    fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    make_timeline(out/"s0_timeline_synthetic.png")
    make_views(out/"s0_feature_views_synthetic.png")
    (out/"README.txt").write_text(
        "Synthetic FEAT-011 S0 visual acceptance samples. All dates/window labels follow the accepted plan; "
        "feature counts, sample values, and model results are illustrative or omitted. These images are not real-data evidence.\n",
        encoding="utf-8")
    print(f"created 2 synthetic previews under {out}")


if __name__=="__main__":
    main()
