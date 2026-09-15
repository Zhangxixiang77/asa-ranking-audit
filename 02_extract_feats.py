#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_extract_feats.py — 用 frozen wav2vec 2.0 Base 提取 utterance 级特征并缓存为 .npy

所有下游模型（B0/B2/B6/Ours）共用这一套特征：最慢的一步只做一次。
特征 = 最后一层 hidden states 在时间维的 mean pooling（768 维），可选加 std pooling（1536 维）。

用法：
  python 02_extract_feats.py --data-dir ./data                 # 全量提取（GPU 约 20-60 分钟）
  python 02_extract_feats.py --data-dir ./data --max-utts 200  # 冒烟测试，先跑通流程
  python 02_extract_feats.py --data-dir ./data --device cpu    # 无 GPU（慢，建议先用 --max-utts 验证）
  python 02_extract_feats.py --data-dir ./data --model microsoft/wavlm-base  # 换 WavLM

国内下载 HF 模型慢/失败时，先执行：export HF_ENDPOINT=https://hf-mirror.com
支持断点续跑：已存在的 .npy 自动跳过。
"""
import argparse
import json
import os
import sys

import numpy as np


def _setup_china_network():
    """国内 / AutoDL 网络适配。必须在 import transformers 之前调用。

    按优先级处理（用户显式设置 > 已有代理 > 自动检测）：
    1. AutoDL：HF 缓存默认在系统盘（仅 ~30GB），自动改到数据盘 /root/autodl-tmp；
    2. 若已设 HF_ENDPOINT 或已开 AutoDL 学术加速（存在 http_proxy），不动；
    3. 否则探测 huggingface.co 连通性，不可达则自动切 hf-mirror.com。
    """
    if "HF_HOME" not in os.environ and os.path.isdir("/root/autodl-tmp"):
        os.environ["HF_HOME"] = "/root/autodl-tmp/cache/huggingface"
        print(f"[net] 检测到 AutoDL 数据盘，HF_HOME → {os.environ['HF_HOME']}（避免撑爆系统盘）")
    if os.environ.get("HF_ENDPOINT"):
        print(f"[net] 使用已配置的 HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
        return
    if os.environ.get("http_proxy") or os.environ.get("https_proxy"):
        print("[net] 检测到代理（AutoDL 学术加速？），HF 走代理直连")
        return
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        print("[net] huggingface.co 可达，直连下载")
    except OSError:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"[net] huggingface.co 不可达，自动切国内镜像 {os.environ['HF_ENDPOINT']}")


def load_audio_16k(path):
    """读音频并重采样到 16kHz mono，返回 float32 numpy。"""
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        try:
            import resampy
            wav = resampy.resample(wav, sr, 16000)
        except ImportError:
            # 简易线性插值回退（质量略降，仅应急）
            n = int(len(wav) * 16000 / sr)
            wav = np.interp(np.linspace(0, len(wav), n, endpoint=False),
                            np.arange(len(wav)), wav).astype("float32")
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--model", default="facebook/wav2vec2-base",
                    help="HuggingFace 模型名，如 facebook/wav2vec2-base / microsoft/wavlm-base")
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "" else "cuda")
    ap.add_argument("--max-utts", type=int, default=None)
    ap.add_argument("--pooling", choices=["mean", "meanstd"], default="mean")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--hf-mirror", action="store_true",
                    help="强制使用 hf-mirror.com 下载模型（跳过连通性检测）")
    ap.add_argument("--no-auto-net", action="store_true",
                    help="关闭国内网络自动适配（完全按当前环境变量来）")
    args = ap.parse_args()

    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    if not args.no_auto_net:
        _setup_china_network()

    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    from tqdm import tqdm

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ 未检测到 CUDA，改用 CPU（很慢，建议先 --max-utts 200 冒烟）")
        device = "cpu"

    feat_dir = os.path.join(args.data_dir, "feats")
    os.makedirs(feat_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "manifest.jsonl"), encoding="utf-8")]
    todo = [r for r in rows if not os.path.exists(os.path.join(feat_dir, r["utt"] + ".npy"))]
    if args.max_utts:
        todo = todo[: args.max_utts]
    print(f"共 {len(rows)} 条，待提取 {len(todo)} 条（已缓存 {len(rows)-len(todo)} 条跳过）")
    if not todo:
        print("全部已缓存，无需提取。")
        return

    print(f"加载模型 {args.model} → {device} ...")
    processor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    bad = 0
    for i in tqdm(range(0, len(todo), args.batch_size), desc="extract"):
        batch = todo[i: i + args.batch_size]
        waves = []
        keep = []
        for r in batch:
            try:
                waves.append(load_audio_16k(r["wav"]))
                keep.append(r)
            except Exception as e:
                bad += 1
                if bad <= 5:
                    print(f"\n⚠ 音频读取失败 {r['wav']}: {e}")
        if not waves:
            continue
        inputs = processor(waves, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(inputs.input_values.to(device),
                        attention_mask=inputs.get("attention_mask", None).to(device)
                        if inputs.get("attention_mask") is not None else None)
        hs = out.last_hidden_state  # (B, T, D)
        if inputs.get("attention_mask") is not None:
            mask = model._get_feature_vector_attention_mask(hs.shape[1], inputs["attention_mask"]).to(device)
        else:
            mask = torch.ones(hs.shape[:2], device=device)
        for b, r in enumerate(keep):
            m = mask[b].bool()
            h = hs[b][m]  # (T_valid, D)
            feats = h.mean(dim=0)
            if args.pooling == "meanstd":
                feats = torch.cat([feats, h.std(dim=0)])
            np.save(os.path.join(feat_dir, r["utt"] + ".npy"),
                    feats.float().cpu().numpy())
    print(f"✓ 完成，失败 {bad} 条。特征目录: {feat_dir}")
    print("下一步：python 03_run_experiments.py --data-dir ./data")


if __name__ == "__main__":
    main()
