"""Create a synthetic-only FEAT-009 S1 acceptance-chart preview.

All numeric values below are fabricated for layout and reading checks. This file
must never be used to infer experiment results.
"""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "feat009-matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures" / "feat-009-s1-acceptance-preview.png"
font = "Arial Unicode MS"
try:
    font_manager.findfont(font, fallback_to_default=False)
except Exception:
    font = "DejaVu Sans"
plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "font.size": 10})

fig = plt.figure(figsize=(15, 9), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95])
fig.suptitle("FEAT-009 S1 验收图样｜合成示例，不是实验结果\n合成样例：500 辆 / 正类 126 / 配对 bootstrap 2,000 次（数字编造）",
             fontsize=14, weight="bold")

# Panel 1: paired bootstrap gate illustration.
ax = fig.add_subplot(gs[0, 0])
labels = ["F3 自适应模型 − V5（主判定）", "F3 自适应模型 − RF（辅助对照）"]
point = np.array([0.012, 0.023])
lo = np.array([-0.002, 0.004])
hi = np.array([0.027, 0.044])
y = np.arange(2)[::-1]
ax.axvline(0, color="#b42318", lw=1.6, label="0：是否超过基线")
ax.axvline(0.01, color="#7a5af8", lw=1.4, ls="--", label="+0.01：实际意义增益")
for idx in range(2):
    color = "#c47c00" if lo[idx] <= 0 else "#16804a"
    ax.errorbar(point[idx], y[idx], xerr=[[point[idx]-lo[idx]], [hi[idx]-point[idx]]],
                fmt="o", color=color, capsize=5, lw=2)
    ax.text(0.055, y[idx], f"{point[idx]:+.3f} [{lo[idx]:+.3f}, {hi[idx]:+.3f}]",
            va="center", fontsize=9)
ax.set_yticks(y, labels)
ax.set_xlim(-0.03, 0.11)
ax.set_xlabel("配对 ΔAUC（95% bootstrap 区间）")
ax.set_title("A｜效果：先看区间是否越过 0")
ax.grid(axis="x", alpha=.22)
ax.legend(loc="center left", bbox_to_anchor=(.01, .5), fontsize=8, frameon=True)
ax.text(0.01, -0.26, "辅助容差示例：ΔRecall@100 +0.005（下限 −0.02）；ΔBrier −0.002（上限 +0.005）。\n示例解读：主判定区间跨 0，因此不算超过 V5。",
        transform=ax.transAxes, fontsize=9, color="#344054")

# Panel 2: fold selection table is explicitly diagnostic.
ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
ax.set_title("B｜逐折选择：看波动，不单独判胜负", loc="left", pad=9)
col_labels = ["外折", "验证集 n / 正类", "折内选中", "内层均值 AUC", "外折 AUC"]
rows = [
    ["0", "100 / 24", "EBM-A", ".741", ".758"],
    ["1", "100 / 27", "LGBM-A", ".766", ".739"],
    ["2", "100 / 26", "EBM-B", ".752", ".801"],
    ["3", "100 / 25", "EBM-A", ".733", ".720"],
    ["4", "100 / 24", "LGBM-B", ".770", ".773"],
]
tab = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center",
               colWidths=[.09, .22, .19, .22, .20])
tab.auto_set_font_size(False); tab.set_fontsize(8.5); tab.scale(1, 1.6)
for (r, c), cell in tab.get_celld().items():
    if r == 0:
        cell.set_facecolor("#e9efff"); cell.set_text_props(weight="bold", color="#1d2939")
    elif r > 0 and c == 0:
        cell.set_facecolor("#f2f4f7")
ax.text(.01, .02, "折内 AUC 用于选型；外折 AUC 仅展示每折起伏，主结论看全体 OOF 配对区间。",
        transform=ax.transAxes, fontsize=8.7, color="#344054")

# Panel 3: use comparable, same-run adaptive pipelines only.
ax = fig.add_subplot(gs[1, :])
run_min = [42, 51]
oof_auc = [.774, .801]
peak_mem = [1.2, 1.6]
for label, xval, yval, mem, color in [
    ("F1 自适应流程（合成）", run_min[0], oof_auc[0], peak_mem[0], "#175cd3"),
    ("F3 自适应流程（合成）", run_min[1], oof_auc[1], peak_mem[1], "#039855"),
]:
    ax.scatter(xval, yval, s=110, color=color, edgecolor="white", linewidth=1.2, zorder=3)
    ax.annotate(f"{label}\n峰值内存 {mem:.1f} GiB｜模型拟合 65 次", (xval, yval), xytext=(10, 10),
                textcoords="offset points", fontsize=9, color=color)
ax.set_xlim(30, 65); ax.set_ylim(.72, .84)
ax.set_xlabel("同环境／同车辆与折分下完成五折全流程的 wall time（分钟）")
ax.set_ylabel("全体 OOF AUC（示例）")
ax.set_title("C｜成本与结果：只比较本次同环境运行的 F1 / F3 流程", loc="left")
ax.grid(alpha=.22)
ax.text(.01, -.23, "不把历史 V5 用时放进此图；V5 只在 A 图作效果参照。运行编号、版本和时间窗由正式报告补齐。",
        transform=ax.transAxes, fontsize=9, color="#344054")

fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(OUT)
