#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_paired_ft_bootstrap.py — B2-FT vs 冻结 B2 的配对 bootstrap:Δ整体、Δchild、Δadult、Δgap 的 95% CI

用途:坐实"微调提升所有人但不改变 gap"的统计表述(审稿人必问:-0.463 vs -0.460 是真不变还是噪声)。
读 manifest.jsonl + results/<a>_preds.jsonl 与 <b>_preds.jsonl,utterance 级分层配对重采样。

用法:
  python 08_paired_ft_bootstrap.py --data-dir ./data --a b2 --b b2ft
  python 08_paired_ft_bootstrap.py --data-dir ./data --a b2 --b b2ft --out-suffix wavlm
"""
import argparse
import json
import os

import numpy as np
from scipy import stats


def load_preds(rd, name):
    d = {}
    for l in open(os.path.join(rd, f"{name}_preds.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        d[r["utt"]] = (np.array(r["y"], dtype=float), np.array(r["pred"], dtype=float))
    return d


def macro_pcc(d, us):
    Y = np.stack([d[u][0] for u in us]); P = np.stack([d[u][1] for u in us])
    v = [stats.pearsonr(Y[:, i], P[:, i])[0] for i in range(Y.shape[1])
         if Y[:, i].std() > 1e-9 and P[:, i].std() > 1e-9]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--a", default="b2", help="基线模型名(默认 b2)")
    ap.add_argument("--b", default="b2ft", help="对比模型名(默认 b2ft)")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rd = os.path.join(args.data_dir, "results" + (f"_{args.out_suffix}" if args.out_suffix else ""))

    mani = {}
    for l in open(os.path.join(args.data_dir, "manifest.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        mani[r["utt"]] = r
    A, B = load_preds(rd, args.a), load_preds(rd, args.b)
    utts = [u for u in A if u in B and u in mani and mani[u].get("subgroup") in ("child", "adult")]
    child = [u for u in utts if mani[u]["subgroup"] == "child"]
    adult = [u for u in utts if mani[u]["subgroup"] == "adult"]
    assert len(child) >= 30 and len(adult) >= 30, f"样本过少: child={len(child)} adult={len(adult)}"
    print(f"[data] 配对样本 {len(utts)}(child {len(child)} / adult {len(adult)});{args.a} vs {args.b}")

    rows = {"d_overall": [], "d_child": [], "d_adult": [], "d_gap": []}
    for _ in range(args.reps):
        cu = [child[i] for i in rng.integers(0, len(child), len(child))]
        au = [adult[i] for i in rng.integers(0, len(adult), len(adult))]
        for k, us in (("overall", cu + au), ("child", cu), ("adult", au)):
            rows[f"d_{k}"].append(macro_pcc(B, us) - macro_pcc(A, us))
        rows["d_gap"].append((macro_pcc(B, cu) - macro_pcc(B, au)) - (macro_pcc(A, cu) - macro_pcc(A, au)))

    print(f"\n===== 配对 bootstrap({args.reps} 次,B − A)=====")
    print("| 指标 | Δ [95% CI] | 判决 |")
    print("|---|---|---|")
    for k, label in (("d_overall", "Δ overall PCC"), ("d_child", "Δ child PCC"),
                     ("d_adult", "Δ adult PCC"), ("d_gap", "Δ gap PCC")):
        v = np.array(rows[k]); v = v[~np.isnan(v)]
        m, lo, hi = float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        sig = "显著" if (lo > 0 or hi < 0) else "不显著(跨零)"
        print(f"| {label} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | {sig} |")
    print("\n判读:Δ overall/child/adult 显著为正 + Δ gap 不显著 ⇒"
          " \"提升所有人,唯独不改变差距\"成立,可写进论文 §3.3/3.4 与摘要句 4。")


if __name__ == "__main__":
    main()
