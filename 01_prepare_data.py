#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_prepare_data.py — Speechocean762 数据核验与 manifest 生成（W1 生死开关 + metadata 卡）

做什么：
  1. 自动探测 speechocean762 数据目录结构（兼容 Kaldi 布局 / GitHub 布局 / 常见变体）
  2. 解析出每条语音的：utt_id、wav 路径、speaker、age、gender、四维评分
  3. 生成 data/manifest.jsonl（后续脚本的唯一数据入口）
  4. 生成 data/metadata_card.md（年龄分布、子群划分、split 信息 —— 直接可贴给 Kimi）
  5. 生成 data/split.json（speaker-independent 的 train/dev/test 划分，固定下来不再变）

用法：
  python 01_prepare_data.py --data-root /path/to/speechocean762 --out-dir ./data
  python 01_prepare_data.py --data-root /path/to/speechocean762 --child-max-age 13

若自动解析失败：脚本会打印它找到的目录结构和文件清单，请把输出整段贴给 Kimi 修解析器。
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

SCORE_KEYS = ["accuracy", "completeness", "fluency", "prosodic"]
# 兼容别名
ALIASES = {"prosody": "prosodic", "prosodics": "prosodic", "total": "total"}


# ----------------------------------------------------------------------------- 文件探测
def find_files(root, names=(), exts=()):
    """在 root 下递归查找文件名命中 names 或后缀命中 exts 的文件，返回绝对路径列表。"""
    hits = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            fl = f.lower()
            if fl in names or any(fl.endswith(e) for e in exts):
                hits.append(os.path.join(dirpath, f))
    return sorted(hits)


def read_kv_file(path, key_col=0, val_col=1):
    """读取 'key value' 两列格式的 Kaldi 文件。"""
    d = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) > max(key_col, val_col):
                d[parts[key_col]] = parts[val_col]
    return d


# ----------------------------------------------------------------------------- 评分解析
def norm_score_dict(d):
    """把任意大小写/别名的评分子典规范成 SCORE_KEYS 四键（若有）。"""
    out = {}
    for k, v in d.items():
        kl = str(k).strip().lower()
        kl = ALIASES.get(kl, kl)
        if kl in SCORE_KEYS + ["total"]:
            try:
                out[kl] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def parse_scores_json(path):
    """解析 {utt: {accuracy:.., completeness:.., ...}} 形式的 json。"""
    with open(path, encoding="utf-8", errors="replace") as fh:
        obj = json.load(fh)
    res = {}
    if isinstance(obj, dict):
        for utt, val in obj.items():
            if isinstance(val, dict):
                sd = norm_score_dict(val)
                if len(sd) >= 4:
                    res[utt] = sd
    return res


def parse_scores_table(path):
    """启发式解析文本评分表：每行 'utt v1 v2 v3 v4 [v5]'，列序默认 acc/comp/flu/pros[/total]。"""
    res = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt = parts[0]
            try:
                vals = [float(x) for x in parts[1:6]]
            except ValueError:
                continue
            sd = {k: vals[i] for i, k in enumerate(SCORE_KEYS)}
            if len(vals) >= 5:
                sd["total"] = vals[4]
            res[utt] = sd
    return res


def find_scores(root, diag):
    """返回 {utt: {accuracy,completeness,fluency,prosodic[,total]}}。合并所有候选文件（train+test）。"""
    res = {}
    for p in find_files(root, exts=(".json",)):
        try:
            part = parse_scores_json(p)
        except Exception:
            continue
        if len(part) > 50:
            res.update(part)
            diag.append(f"[scores] 命中 JSON 评分文件: {p}（{len(part)} 条）")
    if res:
        return res
    for p in find_files(root, names=("scores", "scores.txt", "utt_scores", "utt2score", "scores.tsv")):
        part = parse_scores_table(p)
        if len(part) > 50:
            res.update(part)
            diag.append(f"[scores] 命中文本评分文件: {p}（{len(part)} 条）")
    if not res:
        diag.append("[scores] 未找到评分文件！请检查数据目录。")
    return res


# ----------------------------------------------------------------------------- speaker 元数据
def parse_speaker_meta(root, diag):
    """返回 {spk: {"age": float|None, "gender": str|None}}。尝试多种布局。"""
    meta = defaultdict(lambda: {"age": None, "gender": None})

    # 布局 A：Kaldi spk2age / spk2gender 两列文件
    spk2age = find_files(root, names=("spk2age", "spk2age.txt"))
    spk2gender = find_files(root, names=("spk2gender", "spk2gender.txt"))
    for p in spk2age:
        for spk, a in read_kv_file(p).items():
            try:
                meta[spk]["age"] = float(a)
            except ValueError:
                pass
        if spk2age:
            diag.append(f"[meta] 命中 spk2age: {p}")
    for p in spk2gender:
        for spk, g in read_kv_file(p).items():
            meta[spk]["gender"] = g.lower()
        diag.append(f"[meta] 命中 spk2gender: {p}")

    # 布局 B：json / csv 的 speaker 信息文件
    if not any(v["age"] is not None for v in meta.values()):
        for p in find_files(root, names=("spk2info.json", "speakers.json", "speaker_info.json", "spkinfo.json")):
            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    obj = json.load(fh)
                for spk, info in obj.items():
                    if isinstance(info, dict):
                        il = {str(k).lower(): v for k, v in info.items()}
                        if "age" in il:
                            meta[spk]["age"] = float(il["age"])
                        if "gender" in il:
                            meta[spk]["gender"] = str(il["gender"]).lower()
                diag.append(f"[meta] 命中 speaker JSON: {p}")
            except Exception:
                continue

    # 布局 C：utt2spk + utt 级 json 里带 speaker 信息（部分发布版把 age 放在 utterance 级）
    return dict(meta)


def find_utt2spk(root, diag):
    d = {}
    for p in find_files(root, names=("utt2spk", "utt2spk.txt")):
        part = read_kv_file(p)
        if len(part) > 50:
            d.update(part)
            diag.append(f"[meta] 命中 utt2spk: {p}（{len(part)} 条）")
    if not d:
        diag.append("[meta] 未找到 utt2spk！")
    return d


def find_wavs(root, diag):
    """返回 ({utt: wav_path}, {utt: 来源 split})。

    split 从 wav.scp 所在目录名推断（train/test/dev）——这是官方划分最可靠的来源；
    不能从音频路径推断，因为音频可能统一放在数据根下的 WAVE/ 目录。
    """
    d, utt_split = {}, {}
    for p in find_files(root, names=("wav.scp",)):
        part_name = os.path.basename(os.path.dirname(p)).lower()
        src = part_name if part_name in ("train", "test", "dev", "valid", "validation") else None
        with open(p, encoding="utf-8", errors="replace") as fh:
            cnt = 0
            for line in fh:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt, w = parts
                    if not os.path.isabs(w):
                        w = os.path.normpath(os.path.join(os.path.dirname(p), w))
                    d[utt] = w
                    utt_split[utt] = src
                    cnt += 1
        if cnt > 50:
            diag.append(f"[wav] 命中 wav.scp: {p}（{cnt} 条, 来源目录: {part_name}）")
    if d:
        return d, utt_split
    wavs = find_files(root, exts=(".wav", ".flac"))
    if wavs:
        d = {os.path.splitext(os.path.basename(w))[0]: w for w in wavs}
        diag.append(f"[wav] 未找到 wav.scp，按文件名索引 {len(d)} 个音频")
        return d, {}
    diag.append("[wav] 未找到音频！")
    return {}, {}


# ----------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="speechocean762 解压目录")
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--child-max-age", type=float, default=None,
                    help="child/adult 划分阈值（默认按年龄中位数自动定）")
    ap.add_argument("--dev-ratio", type=float, default=0.1, help="从 train 切 dev 的 speaker 比例")
    ap.add_argument("--rebalance", action="store_true",
                    help="官方 split 子群失衡时，按子群分层重切 train/dev/test（仍 speaker-independent）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    diag = []
    root = args.data_root
    diag.append(f"[root] {root}")
    diag.append("[tree] " + "; ".join(
        os.path.relpath(os.path.join(dp, f), root)
        for dp, _, fs in os.walk(root) for f in fs[:3]
    )[:1500])

    scores = find_scores(root, diag)
    utt2spk = find_utt2spk(root, diag)
    spk_meta = parse_speaker_meta(root, diag)
    wavs, utt_src_split = find_wavs(root, diag)

    # ---- wav 路径校验与修复 ----
    # wav.scp 里的相对路径约定不一：有的相对 wav.scp 所在目录，有的相对数据根，
    # 有的根本对不上。按顺序尝试：原样 → 相对数据根 → 按文件名全局索引，并统一转绝对路径。
    audio_index = {}
    for p in find_files(root, exts=(".wav", ".flac")):
        audio_index[os.path.splitext(os.path.basename(p))[0]] = os.path.abspath(p)
    n_ok = n_fixed = n_root = 0
    for utt, w in list(wavs.items()):
        if os.path.exists(w):
            wavs[utt] = os.path.abspath(w)
            n_ok += 1
            continue
        cand = os.path.join(root, w)
        if os.path.exists(cand):
            wavs[utt] = os.path.abspath(cand)
            n_root += 1
        elif utt in audio_index:
            wavs[utt] = audio_index[utt]
            n_fixed += 1
        else:
            wavs[utt] = None
    diag.append(f"[wav] 路径校验：原样有效 {n_ok} 条；按数据根修复 {n_root} 条；"
                f"按文件名索引修复 {n_fixed} 条；失效 {sum(1 for v in wavs.values() if v is None)} 条")

    # 组装 manifest
    rows, miss = [], Counter()
    for utt, sd in scores.items():
        spk = utt2spk.get(utt)
        if spk is None:
            # 兼容：有的版本 utt 前缀即 speaker
            m = re.match(r"^([A-Za-z]*\d+?)[-_]", utt)
            spk = m.group(1) if m else None
        if spk is None:
            miss["no_speaker"] += 1
            continue
        info = spk_meta.get(spk, {"age": None, "gender": None})
        wav = wavs.get(utt)
        if wav is None:
            miss["no_wav"] += 1
            continue
        rows.append({
            "utt": utt, "wav": wav, "speaker": spk,
            "age": info["age"], "gender": info["gender"],
            "scores": {k: sd[k] for k in SCORE_KEYS if k in sd},
        })
    diag.append(f"[join] 有评分 {len(scores)} 条；成功关联音频 {len(rows)} 条；"
                f"缺音频跳过 {miss['no_wav']} 条；缺 speaker 跳过 {miss['no_speaker']} 条")

    n_age = sum(1 for r in rows if r["age"] is not None)
    diag.append(f"[age] 带年龄信息的样本 {n_age}/{len(rows)}")

    # 年龄分布与阈值
    ages = sorted(r["age"] for r in rows if r["age"] is not None)
    child_max = args.child_max_age
    if ages:
        med = ages[len(ages) // 2]
        if child_max is None:
            # 双峰间隙检测：找儿童团与成人团之间的空隙。
            # 关键约束：间隙两侧必须各占总样本的 [min_side, 1-min_side]——
            # 否则成人尾部稀疏区（如 38→43 岁）的间隙会带偏阈值。
            min_side = 0.15
            n = len(ages)
            cands = []
            for i in range(n - 1):
                frac_left = (i + 1) / n
                if min_side <= frac_left <= 1 - min_side:
                    cands.append((ages[i + 1] - ages[i], ages[i]))
            if cands:
                max_gap, pivot = max(cands)
            else:
                max_gap, pivot = 0, None
            if max_gap >= 3 and pivot is not None:
                child_max = float(pivot)
                left_n = sum(1 for a in ages if a <= pivot)
                diag.append(f"[age] 检测到双峰间隙（{pivot} 岁 → 下一个 {pivot + max_gap} 岁，间隔 {max_gap} 岁；"
                            f"左侧 {left_n}/{n} 条），阈值取 child ≤ {child_max}")
            else:
                child_max = float(med)
                diag.append(f"[age] 中段无 ≥3 岁双峰间隙，阈值取中位数 child ≤ {child_max}（请人工核对分布直方图）")
        diag.append(f"[age] 分布 min={ages[0]} median={med} max={ages[-1]} → child/adult 阈值取 {child_max}")
    else:
        diag.append("[age] 警告：没有任何年龄信息！子群分析无法进行，请把本输出贴给 Kimi。")

    # split：优先用 utt 来源的 wav.scp 目录名（最可靠），其次音频路径推断，最后按 speaker 重切
    import random
    rng = random.Random(args.seed)
    def split_of(wav_path):
        parts = [p.lower() for p in wav_path.replace("\\", "/").split("/")]
        if "test" in parts:
            return "test"
        if "train" in parts:
            return "train"
        return None

    n_from_scp = 0
    for r in rows:
        src = utt_src_split.get(r["utt"])
        if src == "validation":
            src = "dev"
        if src is not None:
            r["split"] = src
            n_from_scp += 1
        else:
            r["split"] = split_of(r["wav"])
    if n_from_scp:
        diag.append(f"[split] {n_from_scp}/{len(rows)} 条的划分来自 wav.scp 所在目录（官方 split）")

    if any(r["split"] is None for r in rows):
        diag.append("[split] 无法推断官方 train/test，按 speaker 8:1:1 重切")
        spks = sorted({r["speaker"] for r in rows if r["speaker"] is not None})
        rng.shuffle(spks)
        n = len(spks)
        assign = {}
        for i, s in enumerate(spks):
            assign[s] = "train" if i < 0.8 * n else ("dev" if i < 0.9 * n else "test")
        for r in rows:
            r["split"] = assign[r["speaker"]]
    else:
        diag.append("[split] 沿用官方 train/test 目录划分")
        # 官方 split 的 speaker 独立性检查
        tr = {r["speaker"] for r in rows if r["split"] == "train"}
        te = {r["speaker"] for r in rows if r["split"] == "test"}
        overlap = tr & te
        diag.append(f"[split] 官方 split speaker 重叠检查: train {len(tr)} 人, test {len(te)} 人, 重叠 {len(overlap)} 人"
                    + ("（⚠ 有泄漏！将按 speaker 重切）" if overlap else "（无重叠 ✓）"))
        if overlap:
            spks = sorted(tr | te)
            rng.shuffle(spks)
            assign = {s: ("test" if i < 0.1 * len(spks) else "train") for i, s in enumerate(spks)}
            for r in rows:
                r["split"] = assign[r["speaker"]]
        # 官方 split 的 test 子群平衡检查：某子群样本过少时醒目警告
        te_child = sum(1 for r in rows if r["split"] == "test" and r["age"] is not None and r["age"] <= child_max)
        te_adult = sum(1 for r in rows if r["split"] == "test" and r["age"] is not None and r["age"] > child_max)
        if min(te_child, te_adult) < 20:
            diag.append(f"[split] ⚠⚠ 官方 test 子群失衡（child {te_child} / adult {te_adult}）："
                        f"子群 gap 估计将不可用或不可靠。建议加 --rebalance 按子群分层重切，"
                        f"或在论文中报告此局限。")
        # 从 train 里切 dev（按 speaker）
        tr_spks = sorted({r["speaker"] for r in rows if r["split"] == "train"})
        rng.shuffle(tr_spks)
        dev_set = set(tr_spks[: max(1, int(len(tr_spks) * args.dev_ratio))])
        for r in rows:
            if r["split"] == "train" and r["speaker"] in dev_set:
                r["split"] = "dev"

    # 可选：子群分层 speaker-independent 重切（官方 split 失衡时使用）
    if args.rebalance:
        diag.append("[split] --rebalance 启用：按子群分层 8:1:1 重切（speaker-independent）")
        for r in rows:
            r["split"] = "train"
        for sg_label, cond in (("child", lambda a: a <= child_max), ("adult", lambda a: a > child_max)):
            spks = sorted({r["speaker"] for r in rows if r["age"] is not None and cond(r["age"])})
            rng.shuffle(spks)
            n = len(spks)
            for i, s in enumerate(spks):
                sp = "train" if i < 0.8 * n else ("dev" if i < 0.9 * n else "test")
                for r in rows:
                    if r["speaker"] == s:
                        r["split"] = sp

    # 子群标签
    for r in rows:
        r["subgroup"] = ("child" if r["age"] is not None and r["age"] <= child_max else
                         ("adult" if r["age"] is not None else "unknown"))

    # 写 manifest
    mf = os.path.join(args.out_dir, "manifest.jsonl")
    with open(mf, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # metadata 卡
    card = build_card(rows, child_max, diag)
    with open(os.path.join(args.out_dir, "metadata_card.md"), "w", encoding="utf-8") as fh:
        fh.write(card)
    with open(os.path.join(args.out_dir, "split.json"), "w", encoding="utf-8") as fh:
        json.dump({"child_max_age": child_max, "seed": args.seed,
                   "counts": dict(Counter(r["split"] for r in rows))}, fh, ensure_ascii=False, indent=2)

    print("\n".join(diag))
    print(f"\n✓ manifest: {mf}（{len(rows)} 条）")
    print(f"✓ metadata 卡: {os.path.join(args.out_dir, 'metadata_card.md')}")
    print("下一步：python 02_extract_feats.py --data-dir ./data")


def build_card(rows, child_max, diag):
    c = Counter(r["split"] for r in rows)
    sg = Counter((r["split"], r["subgroup"]) for r in rows)
    g = Counter(r["gender"] for r in rows)
    lines = ["# Speechocean762 Metadata 卡（W1 生死开关产物）", ""]
    lines += ["## 解析诊断", "```"] + diag + ["```", ""]
    lines += ["## 样本量", f"- 总样本 {len(rows)}；train {c.get('train',0)} / dev {c.get('dev',0)} / test {c.get('test',0)}"]
    lines.append(f"- speaker 数 {len({r['speaker'] for r in rows})}；gender 分布 {dict(g)}")
    lines.append(f"- 子群阈值 child ≤ {child_max} < adult；子群×split：")
    for (sp, sgrp), n in sorted(sg.items()):
        lines.append(f"  - {sp:5s} {sgrp:7s}: {n}")
    ages = sorted(r["age"] for r in rows if r["age"] is not None)
    if ages:
        lines += ["", "## 年龄分布（每岁样本数）", "```"]
        ac = Counter(int(a) for a in ages)
        for a in sorted(ac):
            lines.append(f"{a:3d} | {'#' * min(ac[a], 80)} {ac[a]}")
        lines.append("```")
    lines += ["", "## 分数分布（均值±std）", "```"]
    import statistics
    for k in SCORE_KEYS:
        vals = [r["scores"][k] for r in rows if k in r["scores"]]
        if vals:
            lines.append(f"{k:13s} overall {statistics.mean(vals):.2f}±{statistics.pstdev(vals):.2f} | "
                         f"child {statistics.mean([r['scores'][k] for r in rows if r['subgroup']=='child' and k in r['scores']]) if any(r['subgroup']=='child' and k in r['scores'] for r in rows) else float('nan'):.2f} | "
                         f"adult {statistics.mean([r['scores'][k] for r in rows if r['subgroup']=='adult' and k in r['scores']]) if any(r['subgroup']=='adult' and k in r['scores'] for r in rows) else float('nan'):.2f}")
    lines.append("```")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
