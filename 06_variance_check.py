#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_variance_check.py — range restriction 防线:方差报告 + 方差匹配子采样 + 逐对排序一致率(零重训)

回应审稿意见:"child PCC 更低,可能只是儿童真实分数方差更小的数学必然(range restriction)"
  A) SD 报告:逐维真实分数 / 预测分数的 child vs adult 标准差 —— 先看敌人有多大
  B) 方差匹配子采样:把成人 test 子采样到与儿童相同的真实分数 SD 再算 PCC(重复 --reps 次)
     —— 若方差对齐后 adult PCC 仍远高于 child,range restriction 解释即被排除
  C) 逐对排序一致率(pairwise concordance):对边际方差更鲁棒的排序指标
     —— 真值差 >= --min-gap 的句对上,预测方向正确的比例,utterance 级 bootstrap CI

用法:
  python 06_variance_check.py --data-dir ./data
  python 06_variance_check.py --data-dir ./data --models b2
  python 06_variance_check.py --data-dir ./data --out-suffix wavlm
输出: results[_suffix]/variance_check.md(贴给 Kimi) + variance_check.csv
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy import stats

DIMS = ["accuracy", "completeness", "fluency", "prosodic"]  # 与 01/03 一致(此前误序,仅影响逐维标签)


def load_preds(rd, name):
    fp = os.path.join(rd, f"{name}_preds.jsonl")
    if not os.path.exists(fp):
        return None
    d = {}
    for l in open(fp, encoding="utf-8"):
        r = json.loads(l)
        d[r["utt"]] = (np.array(r["y"], dtype=float), np.array(r["pred"], dtype=float))
    return d


def macro_pcc(Y, P):
    vals = []
    for i in range(Y.shape[1]):
        if Y[:, i].std() > 1e-9 and P[:, i].std() > 1e-9:
            vals.append(stats.pearsonr(Y[:, i], P[:, i])[0])
    return float(np.mean(vals)) if vals else float("nan")


def match_window(adult_y, target_sd, min_n=30):
    """在成人分数里找以中位数为中心的窗口,使窗口内 SD 尽量接近 target_sd。返回索引。"""
    med = np.median(adult_y)
    best = None
    lo, hi = 1e-6, float(adult_y.max() - adult_y.min()) / 2 + 1e-6
    for _ in range(60):
        h = (lo + hi) / 2
        sel = np.abs(adult_y - med) <= h
        n = int(sel.sum())
        sd = float(adult_y[sel].std(ddof=1)) if n > 2 else 0.0
        if sd > target_sd:
            hi = h
        else:
            lo = h
        if best is None or abs(sd - target_sd) < abs(best[2] - target_sd):
            best = (sel.copy(), n, sd)
    sel, n, sd = best
    if n < min_n:  # 分数太离散,窗口装不下 min_n 人:退化为取最接近中位数的 min_n 个
        idx = np.argsort(np.abs(adult_y - med))[:min_n]
        sel = np.zeros(len(adult_y), bool)
        sel[idx] = True
    return np.where(sel)[0]


def concordance(y, p, min_gap, n_pairs, rng):
    """逐对排序一致率(蒙特卡洛版):随机抽 n_pairs 个句对估计,不构造 n×n 矩阵。
    数学上与全量配对等价(无偏估计),速度/内存优化两个数量级。"""
    n = len(y)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    dy = y[i[keep]] - y[j[keep]]
    dp = p[i[keep]] - p[j[keep]]
    mask = np.abs(dy) >= min_gap
    if int(mask.sum()) < 50:
        return float("nan"), int(mask.sum())
    score = ((dy * dp) > 0).astype(float) + 0.5 * (dp == 0)
    return float(score[mask].mean()), int(mask.sum())


def macro_concordance(Y, P, min_gap, n_pairs, rng):
    vals, npairs = [], 0
    for i in range(Y.shape[1]):
        c, n = concordance(Y[:, i], P[:, i], min_gap, n_pairs, rng)
        if c == c:
            vals.append(c)
            npairs += n
    return (float(np.mean(vals)) if vals else float("nan")), npairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--models", default=None, help="逗号分隔,默认全部可用模型")
    ap.add_argument("--out-suffix", default="", help="读 results_<suffix>/")
    ap.add_argument("--reps", type=int, default=1000, help="方差匹配子采样重复次数")
    ap.add_argument("--boot", type=int, default=200, help="concordance 的 bootstrap 次数")
    ap.add_argument("--min-gap", type=float, default=1.0, help="concordance 的最小真值差")
    ap.add_argument("--pairs", type=int, default=200000, help="concordance 每次估计的蒙特卡洛句对数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rd = os.path.join(args.data_dir, "results" + (f"_{args.out_suffix}" if args.out_suffix else ""))

    mani = {}
    for l in open(os.path.join(args.data_dir, "manifest.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        mani[r["utt"]] = r

    names = args.models.split(",") if args.models else \
        [n[:-12] for n in sorted(os.listdir(rd)) if n.endswith("_preds.jsonl")]
    models = {n: p for n in names if (p := load_preds(rd, n))}
    assert models, "没有找到任何 *_preds.jsonl,先跑 03"

    utts = [u for u in next(iter(models.values())) if u in mani and mani[u].get("subgroup") in ("child", "adult")]
    child = [u for u in utts if mani[u]["subgroup"] == "child"]
    adult = [u for u in utts if mani[u]["subgroup"] == "adult"]
    assert len(child) >= 30 and len(adult) >= 30, f"子群样本过少: child={len(child)} adult={len(adult)}"

    L = ["# Range restriction 防线:方差报告 + 方差匹配 + 逐对排序一致率\n"]
    L.append(f"- 模型: {', '.join(models)};test child {len(child)} / adult {len(adult)} 条;min-gap={args.min_gap}\n")
    csv_rows = [["section", "model", "dim", "metric", "child", "adult", "note"]]

    for mi, (m, preds) in enumerate(models.items(), 1):
        print(f"[{mi}/{len(models)}] {m}: A) SD 报告...", flush=True)
        Yc = np.stack([preds[u][0] for u in child]); Pc = np.stack([preds[u][1] for u in child])
        Ya = np.stack([preds[u][0] for u in adult]); Pa = np.stack([preds[u][1] for u in adult])
        nd = Yc.shape[1]
        dims = DIMS[:nd] if nd <= len(DIMS) else [f"dim{i}" for i in range(nd)]

        # ---------- A) SD 报告 ----------
        L.append(f"\n## A) 逐维分数 SD(模型 {m})")
        L.append("| 维度 | child y-SD | adult y-SD | SD 比(a/c) | child pred-SD | adult pred-SD |")
        L.append("|---|---|---|---|---|---|")
        for i, d in enumerate(dims):
            sc, sa = Yc[:, i].std(ddof=1), Ya[:, i].std(ddof=1)
            pc, pa = Pc[:, i].std(ddof=1), Pa[:, i].std(ddof=1)
            L.append(f"| {d} | {sc:.3f} | {sa:.3f} | {sa / max(sc, 1e-9):.2f} | {pc:.3f} | {pa:.3f} |")
            csv_rows.append(["A_sd", m, d, "sd", f"{sc:.4f}", f"{sa:.4f}", f"pred {pc:.4f}/{pa:.4f}"])
        L.append("> SD 比 > 1 越多,range restriction 嫌疑越大,越需要 B/C 两道防线。")

        # ---------- B) 方差匹配子采样 ----------
        print(f"[{mi}/{len(models)}] {m}: B) 方差匹配 ×{args.reps}...", flush=True)
        pcc_child = macro_pcc(Yc, Pc)
        pcc_adult_raw = macro_pcc(Ya, Pa)
        windows = [match_window(Ya[:, i], Yc[:, i].std(ddof=1)) for i in range(nd)]
        matched = []
        for _ in range(args.reps):
            vals = []
            for i in range(nd):
                w = windows[i]
                k = min(len(child), len(w))
                idx = rng.choice(w, size=k, replace=True)  # bootstrap:窗口小于 child 时仍有重采样变异
                ys, ps = Ya[idx, i], Pa[idx, i]
                if ys.std() > 1e-9 and ps.std() > 1e-9:
                    vals.append(stats.pearsonr(ys, ps)[0])
            if vals:
                matched.append(float(np.mean(vals)))
        matched = np.array(matched)
        m_mean, m_lo, m_hi = float(matched.mean()), float(np.percentile(matched, 2.5)), float(np.percentile(matched, 97.5))
        gap_raw = pcc_adult_raw - pcc_child
        gap_mat = m_mean - pcc_child
        explained = (1 - gap_mat / gap_raw) * 100 if abs(gap_raw) > 1e-9 else float("nan")
        L.append(f"\n## B) 方差匹配子采样(模型 {m},{args.reps} 次重复)")
        L.append("| 指标 | 值 |")
        L.append("|---|---|")
        L.append(f"| child macro PCC(全样本) | {pcc_child:.3f} |")
        L.append(f"| adult macro PCC(全样本) | {pcc_adult_raw:.3f} |")
        L.append(f"| adult macro PCC(方差匹配后) | {m_mean:.3f} [{m_lo:.3f}, {m_hi:.3f}] |")
        L.append(f"| 原始 gap | {gap_raw:.3f} |")
        L.append(f"| 方差对齐后 gap | {gap_mat:.3f}(被方差解释的部分: {explained:.0f}%) |")
        csv_rows.append(["B_vmatch", m, "macro", "pcc", f"{pcc_child:.4f}",
                         f"{m_mean:.4f} [{m_lo:.4f},{m_hi:.4f}]", f"raw adult {pcc_adult_raw:.4f}"])
        if m_lo > pcc_child + 0.1:
            L.append(f"> ✅ 判决:方差对齐后 adult PCC 区间下限({m_lo:.3f})仍高于 child + 0.1,"
                     "range restriction 解释不成立。")
        else:
            L.append(f"> ⚠️ 判决:方差对齐后 adult 区间下限({m_lo:.3f})未超过 child + 0.1,"
                     "gap 中有相当部分可被方差解释——论文表述必须回炉。")

        # ---------- C) 逐对排序一致率 ----------
        print(f"[{mi}/{len(models)}] {m}: C) concordance bootstrap ×{args.boot}...", flush=True)

        def boot_conc(Y, P):
            vals = []
            for _ in range(args.boot):
                idx = rng.integers(0, len(Y), len(Y))
                c, _ = macro_concordance(Y[idx], P[idx], args.min_gap, args.pairs, rng)
                if c == c:
                    vals.append(c)
            v = np.array(vals)
            return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), v

        cc_mean, cc_lo, cc_hi, cc_dist = boot_conc(Yc, Pc)
        ca_mean, ca_lo, ca_hi, ca_dist = boot_conc(Ya, Pa)
        ncp = macro_concordance(Yc, Pc, args.min_gap, args.pairs, rng)[1] + \
              macro_concordance(Ya, Pa, args.min_gap, args.pairs, rng)[1]
        print(f"[{mi}/{len(models)}] {m}: 完成。", flush=True)
        diff = ca_dist - cc_dist
        L.append(f"\n## C) 逐对排序一致率(模型 {m},真值差 >= {args.min_gap},有效句对 {ncp})")
        L.append("| 组别 | concordance [95% CI] |")
        L.append("|---|---|")
        L.append(f"| child | {cc_mean:.3f} [{cc_lo:.3f}, {cc_hi:.3f}] |")
        L.append(f"| adult | {ca_mean:.3f} [{ca_lo:.3f}, {ca_hi:.3f}] |")
        L.append(f"| gap(a - c) | {float(diff.mean()):.3f} [{float(np.percentile(diff, 2.5)):.3f}, {float(np.percentile(diff, 97.5)):.3f}] |")
        csv_rows.append(["C_conc", m, "macro", "concordance",
                         f"{cc_mean:.4f} [{cc_lo:.4f},{cc_hi:.4f}]",
                         f"{ca_mean:.4f} [{ca_lo:.4f},{ca_hi:.4f}]", f"pairs {ncp}"])
        L.append("> 0.5 为随机水平。该指标不依赖边际方差,是 PCC 的鲁棒性旁证。")

    out_md = os.path.join(rd, "variance_check.md")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(rd, "variance_check.csv"), "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(csv_rows)
    print("\n".join(L))
    print(f"\n✓ 写入 {out_md}(贴给 Kimi)与 variance_check.csv")


if __name__ == "__main__":
    main()
