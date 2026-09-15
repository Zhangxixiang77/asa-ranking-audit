#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_age_analysis.py — 现象刻画：age-bin 排序可靠性曲线 + matched 归因分析（零重训）

直接读取已有的 manifest 与 results/*_preds.jsonl，产出论文首图的数据：
  A) age-bin 分析：每个年龄段内各模型的 PCC / MAE / 样本数（看排序可靠性随年龄怎么变化）
  B) matched 分析：按真实分数分层，在同一分数段内比较 child vs adult 的 PCC
     —— 回应"gap 只是两组分数分布不同造成的"这一质疑

用法：
  python 05_age_analysis.py --data-dir ./data
  python 05_age_analysis.py --data-dir ./data --models b2,ours,ours_rank
输出：results/age_analysis.md（可直接贴给 Kimi）+ age_analysis.csv
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy import stats


def load_preds(rd, name):
    fp = os.path.join(rd, f"{name}_preds.jsonl")
    if not os.path.exists(fp):
        return None
    d = {}
    for l in open(fp, encoding="utf-8"):
        r = json.loads(l)
        d[r["utt"]] = (np.array(r["y"]), np.array(r["pred"]))
    return d


def macro_pcc(y, p):
    vals = []
    for i in range(y.shape[1]):
        if y[:, i].std() > 1e-9 and p[:, i].std() > 1e-9:
            vals.append(stats.pearsonr(y[:, i], p[:, i])[0])
    return float(np.mean(vals)) if vals else float("nan")


def table_for_bins(items, bin_of, models):
    out = {}
    for utt, age, sg in items:
        out.setdefault(bin_of(age), []).append(utt)
    res = {}
    for b, utts in out.items():
        res[b] = {}
        for m, preds in models.items():
            ys = [preds[u][0] for u in utts if u in preds]
            ps = [preds[u][1] for u in utts if u in preds]
            if len(ys) < 8:
                res[b][m] = (float("nan"), float("nan"), len(ys))
                continue
            Y, P = np.stack(ys), np.stack(ps)
            res[b][m] = (macro_pcc(Y, P), float(np.mean(np.abs(Y - P))), len(ys))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--models", default=None, help="逗号分隔，默认全部可用模型")
    ap.add_argument("--out-suffix", default="",
                    help="读取 results_<suffix>/（与 03 的 --out-suffix 对应）")
    args = ap.parse_args()
    rd = os.path.join(args.data_dir, "results" + (f"_{args.out_suffix}" if args.out_suffix else ""))

    mani = {}
    for l in open(os.path.join(args.data_dir, "manifest.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        mani[r["utt"]] = r

    names = args.models.split(",") if args.models else \
        [n[:-12] for n in sorted(os.listdir(rd)) if n.endswith("_preds.jsonl")]
    models = {}
    for n in names:
        p = load_preds(rd, n)
        if p:
            models[n] = p
    assert models, "没有找到任何 *_preds.jsonl，先跑 03"

    utts = [u for u in next(iter(models.values())) if u in mani and mani[u]["age"] is not None]
    items = [(u, float(mani[u]["age"]), mani[u].get("subgroup", "?")) for u in utts]

    L = ["# Age-bin 排序可靠性分析 + matched 归因分析\n"]
    L.append(f"- 模型: {', '.join(models)}；test 样本 {len(items)} 条\n")

    # ---------- A) age-bin ----------
    edges = [0, 10, 12, 15, 18, 25, 200]
    labels = ["<=10", "11-12", "13-15", "16-18", "19-25", ">25"]
    def bin_of(age):
        for i in range(len(edges) - 1):
            if edges[i] < age <= edges[i + 1]:
                return labels[i]
        return labels[-1]
    res = table_for_bins(items, bin_of, models)
    L.append("## A) 各年龄段 PCC（macro）/ MAE / n")
    L.append("| 年龄段 | " + " | ".join(f"{m} PCC" for m in models) + " | " +
             " | ".join(f"{m} MAE" for m in models) + " | n |")
    L.append("|---" * (2 * len(models) + 2) + "|")
    csv_rows = [["bin", "model", "pcc", "mae", "n"]]
    for b in labels:
        if b not in res:
            continue
        pccs, maes, nn = [], [], res[b][next(iter(models))][2]
        for m in models:
            pcc, mae, n = res[b][m]
            pccs.append(f"{pcc:.3f}" if pcc == pcc else "nan")
            maes.append(f"{mae:.3f}" if mae == mae else "nan")
            csv_rows.append([b, m, f"{pcc:.4f}", f"{mae:.4f}", n])
        L.append(f"| {b} | " + " | ".join(pccs) + " | " + " | ".join(maes) + f" | {nn} |")
    L.append("\n> 若 PCC 随年龄单调爬升，首图即此表的可视化（PCC vs 年龄，按模型分线）。")

    # ---------- B) matched 分析 ----------
    L.append("\n## B) matched 分析：同分数段内 child vs adult（排除分数分布混杂）")
    ref = next(iter(models.values()))
    y_macro = {u: float(np.mean(ref[u][0])) for u, _, _ in items}
    adult_scores = sorted(v for (u, age, sg), v in zip(items, y_macro.values()) if sg == "adult")
    q1, q2 = (np.percentile(adult_scores, 33.3), np.percentile(adult_scores, 66.6))
    def seg_of(score):
        return "低分段" if score <= q1 else ("中分段" if score <= q2 else "高分段")
    L.append(f"- 分层依据：adult 的 macro 均分三分位（<={q1:.2f} / <={q2:.2f} / 以上），对全样本统一适用\n")
    L.append("| 分数段 | 组别 | " + " | ".join(f"{m} PCC" for m in models) + " | n |")
    L.append("|---" * (len(models) + 3) + "|")
    for seg in ("低分段", "中分段", "高分段"):
        for sg in ("child", "adult"):
            sub = [u for (u, age, s) in items if s == sg and seg_of(y_macro[u]) == seg]
            cells = []
            for m, preds in models.items():
                ys = [preds[u][0] for u in sub if u in preds]
                ps = [preds[u][1] for u in sub if u in preds]
                if len(ys) < 8:
                    cells.append("n<8")
                    continue
                cells.append(f"{macro_pcc(np.stack(ys), np.stack(ps)):.3f}")
            L.append(f"| {seg} | {sg} | " + " | ".join(cells) + f" | {len(sub)} |")
    L.append("\n> 若每个分数段内 child PCC 仍明显低于 adult，则 gap 不是分数分布差异驱动的——归因成立。")

    out_md = os.path.join(rd, "age_analysis.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(rd, "age_analysis.csv"), "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    print("\n".join(L))
    print(f"\n✓ 写入 {out_md}（贴给 Kimi）与 age_analysis.csv（画图用）")


if __name__ == "__main__":
    main()
