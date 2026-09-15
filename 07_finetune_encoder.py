#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_finetune_encoder.py — B2-FT:解冻 SSL encoder 的端到端微调(表征级干预,P0 实验)

动机(导师/评审一致意见):四类被证伪的手段全部发生在冻结 encoder 的下游,
"瓶颈在表征"此前只是推断。本实验直接适配表征本身:
  - 若微调缩小 child/adult 排序 gap → 故事升级为"下游无效、表征级有效"(问题+方法,主会最认的形态)
  - 若微调不缩小 gap → 坐实问题深在表征层,审计框架不受影响
两种结果都堵住最大的科学漏洞。

与 03 的区别:03 用 02 缓存的冻结特征;本脚本读原始 wav,encoder 参与梯度。
输出兼容 04/05/06:results[_suffix]/b2ft_preds.jsonl + b2ft_metrics.json

用法:
  python 07_finetune_encoder.py --data-dir ./data                      # wav2vec2-base
  python 07_finetune_encoder.py --data-dir ./data --model microsoft/wavlm-base --out-suffix wavlm
  python 07_finetune_encoder.py --data-dir ./data --max-utts 60 --epochs 2   # 冒烟
"""
import argparse
import json
import os
import random
import time

import numpy as np


def _setup_china_network(no_auto=False):
    """国内/AutoDL 网络适配:HF_HOME 指向数据盘;不可达时自动切 hf-mirror。与 02 一致。"""
    if os.path.exists("/root/autodl-tmp"):
        os.environ.setdefault("HF_HOME", "/root/autodl-tmp/cache/huggingface")
    if no_auto or os.environ.get("HF_ENDPOINT") or os.environ.get("http_proxy"):
        return
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
    except OSError:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[net] huggingface.co 不可达,已切换 HF_ENDPOINT=https://hf-mirror.com")


DIMS = ["accuracy", "completeness", "fluency", "prosodic"]  # 与 01/03 一致(此前误序,仅影响逐维标签)


def load_manifest(data_dir):
    rows = [json.loads(l) for l in open(os.path.join(data_dir, "manifest.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r.get("split") in ("train", "test") and r.get("subgroup") in ("child", "adult")]
    return rows


def split_train_dev(train, seed=0, dev_frac=0.1):
    """dev 从 train 按 speaker 切 10%(与 03 的回退逻辑一致)。"""
    spks = sorted({r["speaker"] for r in train})
    rng = random.Random(seed)
    dev_spks = set(rng.sample(spks, max(1, int(len(spks) * dev_frac))))
    dev = [r for r in train if r["speaker"] in dev_spks]
    tr = [r for r in train if r["speaker"] not in dev_spks]
    return tr, dev


def read_audio(path, sr_target=16000, max_sec=20):
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != sr_target:
        import resampy
        y = resampy.resample(y, sr, sr_target)
    return y[: sr_target * max_sec]


def macro_metrics(Y, P):
    """逐维 PCC/MAE + macro(忽略退化维度)。"""
    from scipy import stats
    pccs, maes = [], []
    for i in range(Y.shape[1]):
        if Y[:, i].std() > 1e-9 and P[:, i].std() > 1e-9:
            pccs.append(float(stats.pearsonr(Y[:, i], P[:, i])[0]))
        else:
            pccs.append(float("nan"))
        maes.append(float(np.mean(np.abs(Y[:, i] - P[:, i]))))
    valid = [v for v in pccs if v == v]
    return {"pcc_per_dim": pccs, "mae_per_dim": maes,
            "macro_pcc": float(np.mean(valid)) if valid else float("nan"),
            "macro_mae": float(np.mean(maes))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--model", default="facebook/wav2vec2-base")
    ap.add_argument("--out-suffix", default="", help="写入 results_<suffix>/(与 03 对齐)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-utts", type=int, default=0, help="冒烟:每个 split 最多取 N 条")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-auto-net", action="store_true")
    ap.add_argument("--local-random-init", action="store_true",
                    help="离线自测:用随机初始化的微型 wav2vec2,不下载权重")
    args = ap.parse_args()

    _setup_china_network(args.no_auto_net)
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_manifest(args.data_dir)
    if args.max_utts:
        rows = [r for r in rows if r["split"] == "train"][: args.max_utts] + \
               [r for r in rows if r["split"] == "test"][: args.max_utts]
    train_all = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    train, dev = split_train_dev(train_all, args.seed)
    print(f"[data] train {len(train)} / dev {len(dev)} / test {len(test)} "
          f"(test child {sum(1 for r in test if r['subgroup'] == 'child')} / "
          f"adult {sum(1 for r in test if r['subgroup'] == 'adult')})")

    # ---------- 模型 ----------
    if args.local_random_init:
        from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2FeatureExtractor
        cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                             conv_dim=(8, 8), conv_stride=(5, 2), conv_kernel=(5, 3))
        encoder = Wav2Vec2Model(cfg)
        fe = Wav2Vec2FeatureExtractor()
        hid = 32
    else:
        from transformers import AutoFeatureExtractor, AutoModel
        fe = AutoFeatureExtractor.from_pretrained(args.model)
        encoder = AutoModel.from_pretrained(args.model)
        hid = encoder.config.hidden_size
    if device == "cuda":
        encoder.gradient_checkpointing_enable()
    encoder.to(device)

    head = nn.Sequential(nn.Linear(hid, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, len(DIMS))).to(device)
    opt = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.lr_encoder},
        {"params": head.parameters(), "lr": args.lr_head}], weight_decay=0.01)

    def forward(wavs):
        inp = fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            h = encoder(**inp).last_hidden_state          # [B, T, H]
            mask = inp.get("attention_mask")
            if mask is None:
                pooled = h.mean(dim=1)
            else:
                m = mask.unsqueeze(-1).float()
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return head(pooled.float())  # head 在 autocast 外跑 fp32,避免 Half/Float 反向冲突

    def y_of(rs):
        return torch.tensor([[r["scores"][k] for k in DIMS] for r in rs], dtype=torch.float32)

    def run_eval(rs, bs):
        encoder.eval(); head.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(rs), bs):
                batch = rs[i: i + bs]
                wavs = [read_audio(r["wav"]) for r in batch]
                preds.append(forward(wavs).float().cpu())
        encoder.train(); head.train()
        P = torch.cat(preds).numpy() if preds else np.zeros((0, len(DIMS)))
        Y = y_of(rs).numpy()
        return Y, P

    # ---------- 训练 ----------
    best_pcc, best_state, bad = -2.0, None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        random.shuffle(train)
        tot, nb = 0.0, 0
        for i in range(0, len(train), args.batch_size):
            batch = train[i: i + args.batch_size]
            wavs = [read_audio(r["wav"]) for r in batch]
            y = y_of(batch).to(device)
            pred = forward(wavs)
            loss = nn.functional.mse_loss(pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        Yd, Pd = run_eval(dev, args.batch_size)
        dev_pcc = macro_metrics(Yd, Pd)["macro_pcc"]
        print(f"[ep {ep:02d}] train MSE {tot / max(nb, 1):.4f} | dev macro PCC {dev_pcc:.4f} | {time.time() - t0:.0f}s")
        if dev_pcc > best_pcc:
            best_pcc, bad = dev_pcc, 0
            best_state = {"enc": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                          "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early stop] patience {args.patience}")
                break

    if best_state:
        encoder.load_state_dict({k: v.to(device) for k, v in best_state["enc"].items()})
        head.load_state_dict({k: v.to(device) for k, v in best_state["head"].items()})

    # ---------- 测试与输出 ----------
    Yt, Pt = run_eval(test, args.batch_size)
    rd = os.path.join(args.data_dir, "results" + (f"_{args.out_suffix}" if args.out_suffix else ""))
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, "b2ft_preds.jsonl"), "w", encoding="utf-8") as fh:
        for r, y, p in zip(test, Yt, Pt):
            fh.write(json.dumps({"utt": r["utt"], "y": [float(v) for v in y],
                                 "pred": [float(v) for v in p]}) + "\n")

    rep = {"model": "b2ft", "backbone": args.model, "n_test": len(test)}
    for sg in ("overall", "child", "adult"):
        idx = [i for i, r in enumerate(test) if sg == "overall" or r["subgroup"] == sg]
        rep[sg] = macro_metrics(Yt[idx], Pt[idx]) if idx else None
    if rep["child"] and rep["adult"]:
        rep["gap_pcc"] = rep["child"]["macro_pcc"] - rep["adult"]["macro_pcc"]
        rep["gap_mae"] = rep["child"]["macro_mae"] - rep["adult"]["macro_mae"]
    with open(os.path.join(rd, "b2ft_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)

    print("\n===== B2-FT 判决 =====")
    print(f"overall macro PCC {rep['overall']['macro_pcc']:.4f}")
    print(f"child   macro PCC {rep['child']['macro_pcc']:.4f}")
    print(f"adult   macro PCC {rep['adult']['macro_pcc']:.4f}")
    print(f"gap PCC {rep['gap_pcc']:+.4f}(对照:冻结 B2 wav2vec2 = -0.460 / WavLM = -0.553)")
    if rep["gap_pcc"] > -0.30:
        print("→ gap 明显缩小:故事升级为\"下游无效、表征级有效\"——按 V4.2 §6 的 3.4 分支写")
    else:
        print("→ gap 未明显缩小:表征级微调也救不了,坐实问题深度——审计框架不受影响,3.3 结尾句加一笔")
    print(f"✓ 输出 {rd}/b2ft_preds.jsonl + b2ft_metrics.json(04/05/06 可直接读)")


if __name__ == "__main__":
    main()
