#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_figure1.py — Figure 1:年龄过渡带曲线(child 段低迷、成年段高位、16-18 无数据)

读 results/age_analysis.csv(必需)与 results_wavlm/age_analysis.csv(可选),
画 macro PCC vs 年龄段的单栏图(3.5 in,IEEE 单栏宽)。

用法:
  python plot_figure1.py --data-dir ./data                       # 单 backbone(wav2vec2)
  python plot_figure1.py --data-dir ./data --with-wavlm          # 双 backbone(需先跑 05 --out-suffix wavlm)
  python plot_figure1.py --data-dir ./data --model b2            # 指定模型(默认 b2)
输出: figure1.pdf(投稿用矢量图)+ figure1.png(300dpi 预览)
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 年龄段:标签、数值中点(用于连续 x 轴)
BINS = [("<=10", 8.0), ("11-12", 11.5), ("13-15", 14.0), ("16-18", 17.0), ("19-25", 22.0), (">25", 30.0)]
GAP_REGION = (15, 19)   # 无数据过渡带
STYLE = {  # backbone -> (颜色, 标记, 线型),色盲友好
    "wav2vec2": ("#0072B2", "o", "-"),
    "WavLM": ("#D55E00", "s", "--"),
}


def load(fp, model):
    """读 age_analysis.csv,返回 {bin_label: (pcc, n)}(仅指定模型)。"""
    out = {}
    with open(fp, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["model"] == model and row["pcc"] not in ("nan", ""):
                out[row["bin"]] = (float(row["pcc"]), int(row["n"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--model", default="b2")
    ap.add_argument("--with-wavlm", action="store_true", help="叠加 results_wavlm 的线")
    ap.add_argument("--out", default="figure1")
    args = ap.parse_args()

    series = {}
    fp = os.path.join(args.data_dir, "results", "age_analysis.csv")
    assert os.path.exists(fp), f"缺少 {fp}(先跑 05_age_analysis.py)"
    series["wav2vec2"] = load(fp, args.model)
    if args.with_wavlm:
        fp2 = os.path.join(args.data_dir, "results_wavlm", "age_analysis.csv")
        assert os.path.exists(fp2), f"缺少 {fp2}(先跑 05_age_analysis.py --out-suffix wavlm)"
        series["WavLM"] = load(fp2, args.model)

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                         "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in"})
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # 16-18 无数据过渡带
    ax.axvspan(GAP_REGION[0], GAP_REGION[1], color="0.92", zorder=0)
    ax.text(np.mean(GAP_REGION), 0.06, "no data\n(16-18)", ha="center", va="bottom",
            fontsize=6.5, color="0.45")

    for si, (name, data) in enumerate(series.items()):
        lab_off = (0, 5) if si == 0 else (0, -13)  # 多线时数值标签上下错开,避免重叠
        pts = [(x, data[lab][0], data[lab][1]) for lab, x in BINS if lab in data]
        c, mk, ls = STYLE.get(name, ("#009E73", "^", "-."))
        child_pts = [p for p in pts if p[0] < GAP_REGION[0]]
        adult_pts = [p for p in pts if p[0] > GAP_REGION[1]]
        # 组内实线;跨 16-18 无数据带用同色虚线(视觉上不假装有数据)
        for seg in (child_pts, adult_pts):
            if len(seg) > 1:
                ax.plot([p[0] for p in seg], [p[1] for p in seg], ls, color=c, linewidth=1.4, zorder=2)
        if child_pts and adult_pts:
            ax.plot([child_pts[-1][0], adult_pts[0][0]], [child_pts[-1][1], adult_pts[0][1]],
                    ":", color=c, linewidth=1.0, alpha=0.55, zorder=1)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, color=c, markersize=4.5,
                linestyle="none", zorder=3, label=name)
        for x, y, n in pts:
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=lab_off,
                        ha="center", fontsize=6.5, color=c)
            if si == 0:
                ax.annotate(f"n={n}", (x, 0.025), ha="center", fontsize=5.5, color="0.55")

    ax.set_xlabel("Age group (bin center)", fontsize=8)
    ax.set_ylabel("Macro PCC (test)", fontsize=8)
    ax.set_xlim(6, 32)
    ax.set_ylim(0, 0.95)
    ax.set_xticks([x for _, x in BINS])
    ax.set_xticklabels(["<=10", "11-12", "13-15", "16-18", "19-25", ">25"], fontsize=7)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=300)
    print(f"✓ {args.out}.pdf + {args.out}.png(模型 {args.model};backbone: {', '.join(series)})")
    for name, data in series.items():
        print(f"  {name}: " + ", ".join(f"{lab}={data[lab][0]:.3f}" for lab, _ in BINS if lab in data))


if __name__ == "__main__":
    main()
