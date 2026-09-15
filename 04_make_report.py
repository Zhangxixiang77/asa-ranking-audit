#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_make_report.py — 把 results/ 汇总成一份可直接贴给 Kimi 的判断报告（SUMMARY.md）

Pivot 后的三个价值判据（排序可靠性版本）：
  Q1 问题成立吗：B2 的 child−adult PCC gap 是否大且逐维一致（MAE/bias 是否同时"看似公平"）
  Q2 方法有效吗：Ours-Rank 相对 B2 / B2-Rank 是否显著提升 PCC 并缩小 gap
  Q3 有 shortcut / 廉价解吗：B0 是否弱、B6 后校准是否救不了排序

用法：python 04_make_report.py --data-dir ./data
"""
import argparse
import json
import os

DIMS = ("accuracy", "completeness", "fluency", "prosodic")


def fmt(x, nd=4, sign=False):
    if x is None or (isinstance(x, float) and x != x):
        return "nan"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def fmt_ci(b):
    return f"{fmt(b['est'], 4, True)} [{fmt(b['lo'], 4, True)}, {fmt(b['hi'], 4, True)}]"


def sig(b):
    return "显著" if (b["lo"] > 0) or (b["hi"] < 0) else "不显著"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-suffix", default="",
                    help="读取 results_<suffix>/（与 03 的 --out-suffix 对应）")
    args = ap.parse_args()
    rd = os.path.join(args.data_dir, "results" + (f"_{args.out_suffix}" if args.out_suffix else ""))

    models = [n[:-13] for n in sorted(os.listdir(rd)) if n.endswith("_metrics.json")]
    M = {n: json.load(open(os.path.join(rd, f"{n}_metrics.json"))) for n in models}
    P = json.load(open(os.path.join(rd, "pairwise.json")))
    cfg = json.load(open(os.path.join(args.data_dir, "split.json")))

    L = []
    L.append("# ASA 子群校准 · 关键结果汇总（贴给 Kimi 判断价值用）\n")
    L.append(f"- 数据划分: {cfg['counts']}；child/adult 阈值 ≤ {cfg['child_max_age']}")
    L.append("- 口径: macro 对退化维度（如 completeness 常数）取 nanmean；gap = child − adult\n")

    L.append("## 主结果表（macro 平均）")
    L.append("| 模型 | overall PCC | overall MAE | child PCC | adult PCC | **gap PCC** | gap MAE | gap bias |")
    L.append("|---|---|---|---|---|---|---|---|")
    for n in models:
        m = M[n]
        row = [f"**{n.upper()}**",
               fmt(m["overall"]["macro"]["pcc"]), fmt(m["overall"]["macro"]["mae"])]
        if "gap" in m:
            row += [fmt(m["child"]["macro"]["pcc"]), fmt(m["adult"]["macro"]["pcc"]),
                    fmt(m["gap"]["macro"]["pcc"], 4, True),
                    fmt(m["gap"]["macro"]["mae"], 4, True),
                    fmt(m["gap"]["macro"]["bias"], 4, True)]
        else:
            row += ["-"] * 5
        L.append("| " + " | ".join(row) + " |")

    L.append("\n## 逐维 PCC（B2 vs 各模型，看方向一致性）")
    L.append("| 维度 | 组别 | " + " | ".join(n.upper() for n in models) + " |")
    L.append("|---|---" + "|---" * len(models) + "|")
    for d in DIMS:
        for sg in ("child", "adult"):
            cells = []
            for n in models:
                v = M[n].get(sg, {}).get(d, {}).get("pcc")
                cells.append(fmt(v))
            L.append(f"| {d} | {sg} | " + " | ".join(cells) + " |")

    L.append("\n## 配对检验（分层 bootstrap 95% CI + Wilcoxon）")
    L.append("| 对比 | ΔPCC | Δgap PCC | ΔMAE | Δgap MAE | Wilcoxon p |")
    L.append("|---|---|---|---|---|---|")
    for k, p in P.items():
        L.append(f"| {k} | {fmt_ci(p['bootstrap']['d_pcc'])} | {fmt_ci(p['bootstrap']['d_gap_pcc'])} "
                 f"| {fmt_ci(p['bootstrap']['d_mae'])} | {fmt_ci(p['bootstrap']['d_gap_mae'])} "
                 f"| {p['wilcoxon_p']:.4g} |")

    # ---------------- 规则化初步判断（排序可靠性版） ----------------
    L.append("\n## 规则化初步判断（供参考，最终以 Kimi 判断为准）")
    b2 = M.get("b2", {})
    gap_pcc = b2.get("gap", {}).get("macro", {}).get("pcc", float("nan"))
    gaps = []
    for d in DIMS:
        c = b2.get("child", {}).get(d, {}).get("pcc")
        a = b2.get("adult", {}).get(d, {}).get("pcc")
        if c is not None and a is not None and c == c and a == a:
            gaps.append((d, c, a, c - a))
    q1 = (gap_pcc == gap_pcc and gap_pcc < -0.1) or (gaps and sum(g[3] for g in gaps) / len(gaps) < -0.1)
    L.append(f"- **Q1 问题存在吗**: B2 三维 macro gap PCC = {fmt(gap_pcc, 4, True)}；逐维: "
             + ", ".join(f"{d} {fmt(g, 3, True)}" for d, _, _, g in gaps)
             + f"；同时 MAE gap = {fmt(b2.get('gap', {}).get('macro', {}).get('mae'), 4, True)}（若接近 0 即'MAE 公平假象'）")
    if "ours_rank" in M:
        o2 = P.get("ours_rank_vs_b2", {}).get("bootstrap", {})
        o2r = P.get("ours_rank_vs_b2_rank", {}).get("bootstrap", {})
        if o2:
            L.append(f"- **Q2 方法有效吗**: Ours-Rank vs B2 ΔPCC {fmt_ci(o2['d_pcc'])}（{sig(o2['d_pcc'])}）; "
                     f"Δgap PCC {fmt_ci(o2['d_gap_pcc'])}（{sig(o2['d_gap_pcc'])}，正值=gap 缩小）")
        if o2r:
            L.append(f"- **Q2b 超越'只是换 loss'吗**: Ours-Rank vs B2-Rank ΔPCC {fmt_ci(o2r['d_pcc'])}（{sig(o2r['d_pcc'])}）; "
                     f"Δgap PCC {fmt_ci(o2r['d_gap_pcc'])}（{sig(o2r['d_gap_pcc'])}）")
        q2 = bool(o2) and (o2["d_pcc"]["lo"] > 0 or o2["d_gap_pcc"]["lo"] > 0)
        q2b = bool(o2r) and (o2r["d_pcc"]["lo"] > 0 or o2r["d_gap_pcc"]["lo"] > 0)
    else:
        L.append("- Q2: 未跑 ranking 变体（去掉 --skip-rank 重跑 03）")
        q2 = q2b = False
    q3 = "b2_vs_b0" in P and P["b2_vs_b0"]["bootstrap"]["d_pcc"]["lo"] > 0
    b6p = P.get("b6_vs_b2", {}).get("bootstrap", {})
    L.append(f"- **Q3 shortcut / 廉价解**: B2 显著强于 B0 = {q3}；B6 对排序的影响 ΔPCC "
             f"{fmt_ci(b6p['d_pcc']) if b6p else '-'}（若不显著即'后校准救不了排序'）")
    if q1 and q2 and q2b:
        verdict = "倾向 GO（主会）：排序 gap 真实巨大，方法有效且超越 loss 对照"
    elif q1 and q2:
        verdict = "倾向 GO-weak：方法有效但未明显超越 B2-Rank，贡献收缩到现象刻画 + gap 缩小"
    elif q1:
        verdict = "倾向 收缩/降级：现象可发，方法未见效——考虑现象分析定位（OJSP）或调参再战"
    else:
        verdict = "倾向 NO-GO：连排序 gap 都不成立"
    L.append(f"- 初步结论: **{verdict}**")

    out = os.path.join(rd, "SUMMARY.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n✓ 已写入 {out} —— 把这个文件的内容整段贴给 Kimi")


if __name__ == "__main__":
    main()
