#!/usr/bin/env python3
"""Render FEAT-010 human-review previews from synthetic examples only."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def _save(fig, path: Path) -> None:
    fig.text(0.5, 0.015, "Synthetic example only | No competition records, predictions, metrics, or private data",
             ha="center", color="#8b3a3a", fontsize=9, weight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, width, height, text, *, face="#eef4f8", edge="#42677e", fontsize=10):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=1.1, facecolor=face, edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, wrap=True)


def render_a0(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.suptitle("FEAT-010 A0 | Label rule and time boundary", fontsize=15)
    rows = [
        ("Synthetic car A", 0, 1, "negative"),
        ("Synthetic car B", 0, 2, "positive"),
        ("Synthetic car C", 1, 0, "positive"),
    ]
    ax.axis("off")
    table = ax.table(
        cellText=[[name, str(acc), str(near), label] for name, acc, near, label in rows],
        colLabels=["Example", "Accident records", "Valid near-miss records", "Rule result"],
        cellLoc="center", colLoc="center", loc="upper left", bbox=[0.02, 0.45, 0.96, 0.40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for r in (2, 3):
        table[(r, 3)].set_facecolor("#dcefe2")
    table[(1, 3)].set_facecolor("#f4e8ca")
    ax.text(0.03, 0.29, "Positive = accident ≥ 1 OR valid near-miss records ≥ 2",
            transform=ax.transAxes, fontsize=11, weight="bold")
    ax.text(0.03, 0.16,
            "Real review: trace source event time, valid-record de-duplication, and vehicle binding.",
            transform=ax.transAxes, fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    _save(fig, out / "a0-label-rule.png")


def render_a1(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.3))
    fig.suptitle("FEAT-010 A1 | Proxy windows and final 61-day input stay separate", fontsize=15)
    ax.set_xlim(-1, 62)
    ax.set_ylim(-0.5, 2.8)
    ax.set_yticks([0, 1, 2], ["Proxy labels", "Proxy features", "Final input"])
    ax.set_xticks([0, 20, 61], ["Jun 1", "Jun 21", "Aug 1"])
    ax.grid(axis="x", alpha=0.2)
    ax.barh(0, 40, left=20, height=0.42, color="#e7a56b")
    ax.text(40, 0, "40-day labels [Jun 21, Jul 31)", ha="center", va="center", fontsize=9)
    ax.barh(1, 20, left=0, height=0.42, color="#6da5c4")
    ax.text(10, 1, "20 days", ha="center", va="center", fontsize=9, color="white", weight="bold")
    ax.barh(2, 61, left=0, height=0.42, color="#7da88b")
    ax.text(30.5, 2, "61-day history [Jun 1, Aug 1)", ha="center", va="center", fontsize=9,
            color="white", weight="bold")
    ax.axvline(20, color="#555555", linestyle="--", linewidth=1)
    ax.axvline(61, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("Days from Jun 1; window ends are exclusive")
    ax.text(0.5, -0.23,
            "Do not calculate proxy AUC by attaching final 61-day features to these proxy labels.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#8b4e35")
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    _save(fig, out / "a1-window-separation.png")


def render_b0(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.suptitle("FEAT-010 B0 | One controlled comparison of six direct observation totals", fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.07, 0.31), 0.20, 0.45,
         "Same F3 input\nSame labels and 5 folds\nSame fixed EBM-A", face="#edf0f6")
    _box(ax, (0.39, 0.60), 0.23, 0.28,
         "Arm 1: keep\n6 direct observation totals", face="#e9f1f6")
    _box(ax, (0.39, 0.19), 0.23, 0.28,
         "Arm 2: remove\nthose 6 direct totals", face="#f7eee5", edge="#a36c40")
    _box(ax, (0.73, 0.31), 0.20, 0.45,
         "Paired OOF comparison\nΔAUC + 95% interval\ncoverage-group diagnostics", face="#edf4ec", edge="#557b58")
    ax.annotate("", xy=(0.37, 0.73), xytext=(0.27, 0.61), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.annotate("", xy=(0.37, 0.33), xytext=(0.27, 0.45), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.annotate("", xy=(0.71, 0.70), xytext=(0.62, 0.73), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.annotate("", xy=(0.71, 0.38), xytext=(0.62, 0.33), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.text(0.5, 0.06,
            "Keep driving exposure (km / hours), event rates, and quality ratios. This tests only the six named totals.",
            ha="center", va="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    _save(fig, out / "b0-ablation-flow.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/feat-010/synthetic")
    args = parser.parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11})
    render_a0(out)
    render_a1(out)
    render_b0(out)
    print(out)


if __name__ == "__main__":
    main()
