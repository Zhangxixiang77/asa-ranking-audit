#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_run_experiments.py — 训练并评估 B0 / B2 / B6 / Ours，输出子群 gap 与显著性检验

模型：
  B0   Profile-only：仅 age(+gender) → 分数。shortcut 下界检测。
  B2   Profile-blind：SSL 特征 → MLP → 四维分数。★最关键对照。
  B6   Per-group 后校准：B2 预测 → 训练集按子群拟合 affine（每子群每维 a,b）。★廉价上限基线。
  Ours 受限画像残差校准：ŷ = clip(G(h) + delta·tanh(R([h; e_age])))，loss = MSE + λ·||r||。

评估：
  每维 PCC / Spearman / MAE / signed bias（E[ŷ−y]），overall 与 child/adult 子群分别报告；
  macro 平均；分层配对 bootstrap 95% CI；Wilcoxon 符号秩检验。

用法：
  python 03_run_experiments.py --data-dir ./data
  python 03_run_experiments.py --data-dir ./data --epochs 60 --delta 1.0 --lambda-r 0.01
输出（data/results/）：
  {b0,b2,b6,ours}_preds.jsonl、{...}_metrics.json、pairwise.json
"""
import argparse
import json
import os
import random

import numpy as np
from scipy import stats

DIMS = ["accuracy", "completeness", "fluency", "prosodic"]


# ------------------------------------------------------------------ 数据
def load_split(data_dir):
    rows = [json.loads(l) for l in open(os.path.join(data_dir, "manifest.jsonl"), encoding="utf-8")]
    cfg = json.load(open(os.path.join(data_dir, "split.json"), encoding="utf-8"))
    child_max = cfg["child_max_age"]
    feat_dir = os.path.join(data_dir, "feats")
    data = {"train": [], "dev": [], "test": []}
    for r in rows:
        fp = os.path.join(feat_dir, r["utt"] + ".npy")
        if not os.path.exists(fp) or r["age"] is None:
            continue
        x = np.load(fp).astype("float32")
        if hasattr(load_split, "_dim") and x.shape[0] != load_split._dim:
            raise SystemExit(
                f"特征维度不一致：{fp} 是 {x.shape[0]} 维，此前样本是 {load_split._dim} 维。"
                f"可能是混用了不同 backbone / pooling 提取的特征——请清空 feats/ 后重跑 02。")
        load_split._dim = x.shape[0]
        y = np.array([r["scores"][k] for k in DIMS], dtype="float32")
        data[r["split"]].append({
            "utt": r["utt"], "x": x, "y": y, "age": float(r["age"]),
            "speaker": r.get("speaker"),
            "gender": r.get("gender") or "unknown",
            "subgroup": "child" if r["age"] <= child_max else "adult",
        })
    return data, child_max, len(rows)


def standardize(train, others):
    X = np.stack([d["x"] for d in train])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ages = np.array([d["age"] for d in train])
    amu, asd = ages.mean(), ages.std() + 1e-6
    for split in (train,) + tuple(others):
        for d in split:
            d["x"] = (d["x"] - mu) / sd
            d["age_n"] = (d["age"] - amu) / asd


# ------------------------------------------------------------------ 模型
def build_models(feat_dim, delta, device):
    import torch
    nn = torch.nn

    class Head(nn.Module):
        """B2：profile-blind MLP 评分头"""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(),
                                     nn.Dropout(0.2), nn.Linear(256, 4))
        def forward(self, h):
            return self.net(h)

    class ProfileOnly(nn.Module):
        """B0：只看 metadata（age 标准化、gender、交互）"""
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 4))
        def forward(self, age_n, gender_f):
            z = torch.stack([age_n, gender_f, age_n * gender_f], dim=-1)
            return self.net(z)

    class ResidualCalibrator(nn.Module):
        """Ours：ŷ = clip(G(h) + delta·tanh(R([h; e_age])))"""
        def __init__(self):
            super().__init__()
            self.base = Head()
            self.age_emb = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
            self.res = nn.Sequential(nn.Linear(feat_dim + 32, 128), nn.ReLU(),
                                     nn.Linear(128, 4))
            self.delta = delta
        def forward(self, h, age_n):
            y_base = self.base(h)
            e = self.age_emb(age_n.unsqueeze(-1))
            r = self.delta * torch.tanh(self.res(torch.cat([h, e], dim=-1)))
            return y_base, r, torch.clamp(y_base + r, 0.0, 10.0)

    return Head().to(device), ProfileOnly().to(device), ResidualCalibrator().to(device)


def tensors(split, device):
    import torch
    return (
        torch.from_numpy(np.stack([d["x"] for d in split])).float().to(device),
        torch.from_numpy(np.stack([d["y"] for d in split])).float().to(device),
        torch.tensor(np.array([d["age_n"] for d in split]), dtype=torch.float32, device=device),
        torch.tensor(np.array([1.0 if str(d["gender"]).startswith("f") else 0.0 for d in split]),
                     dtype=torch.float32, device=device),
    )


def macro_pcc(y_true, y_pred):
    vals = [stats.pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])]
    return float(np.nanmean(vals))


# ------------------------------------------------------------------ PCC loss（可微）
def pcc_loss_torch(pred, y, eps=1e-6):
    """batch 内可微 Pearson 相关损失：loss = 1 − mean_dim PCC。
    注意是 batch 级估计（batch≥32 时够用）；维度退化成常数时自动给出大 loss 的梯度为零。"""
    import torch
    vx = pred - pred.mean(dim=0, keepdim=True)
    vy = y - y.mean(dim=0, keepdim=True)
    cov = (vx * vy).mean(dim=0)
    sx = torch.sqrt((vx ** 2).mean(dim=0) + eps)
    sy = torch.sqrt((vy ** 2).mean(dim=0) + eps)
    pcc = cov / (sx * sy)
    return 1.0 - pcc.mean()


def group_pcc_loss_torch(pred, y, is_child, w_child=2.0, min_n=8):
    """子群加权 PCC loss：child 组的排序损失权重 w_child 倍——子群鲁棒的落点。"""
    losses, ws = [], []
    for mask, w in ((is_child, w_child), (~is_child, 1.0)):
        if int(mask.sum()) >= min_n:
            losses.append(pcc_loss_torch(pred[mask], y[mask]))
            ws.append(w)
    if not losses:
        return pcc_loss_torch(pred, y)
    return sum(l * w for l, w in zip(losses, ws)) / sum(ws)


def train_model(model, kind, tr, dv, device, epochs, lr, lambda_r, seed=0,
                lambda_pcc=0.0, w_child=2.0):
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    Xtr, Ytr, Atr, Gtr = tensors(tr, device)
    Xdv, Ydv, Adv, Gdv = tensors(dv, device)
    Ctr = torch.tensor(np.array([d["subgroup"] == "child" for d in tr]),
                       dtype=torch.bool, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_pcc, best_state, patience = -9.0, None, 0
    n, bs = len(tr), 32
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n)
        for s in range(0, n, bs):
            b = torch.from_numpy(idx[s: s + bs]).long().to(device)
            opt.zero_grad()
            if kind == "b2":
                pred = model(Xtr[b])
                loss = torch.mean((pred - Ytr[b]) ** 2)
            elif kind == "b0":
                pred = model(Atr[b], Gtr[b])
                loss = torch.mean((pred - Ytr[b]) ** 2)
            else:  # ours
                _, r, pred = model(Xtr[b], Atr[b])
                loss = torch.mean((pred - Ytr[b]) ** 2) + lambda_r * torch.mean(r ** 2)
            if lambda_pcc > 0 and kind in ("b2", "ours"):
                loss = loss + lambda_pcc * group_pcc_loss_torch(pred, Ytr[b], Ctr[b], w_child)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            if kind == "b2":
                pd = model(Xdv)
            elif kind == "b0":
                pd = model(Adv, Gdv)
            else:
                _, _, pd = model(Xdv, Adv)
            pcc = macro_pcc(Ydv.cpu().numpy(), pd.cpu().numpy())
        if pcc > best_pcc:
            best_pcc = pcc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, kind, split, device):
    import torch
    X, Y, A, G = tensors(split, device)
    model.eval()
    with torch.no_grad():
        if kind == "b2":
            p = model(X)
        elif kind == "b0":
            p = model(A, G)
        else:
            _, _, p = model(X, A)
    return p.cpu().numpy()


# ------------------------------------------------------------------ B6：per-group 最小二乘 affine
def fit_b6(train_preds, train, test_preds, test):
    out = test_preds.copy()
    for sg in ("child", "adult"):
        tr_idx = [i for i, d in enumerate(train) if d["subgroup"] == sg]
        te_idx = [i for i, d in enumerate(test) if d["subgroup"] == sg]
        if not tr_idx or not te_idx:
            continue
        Ytr = np.stack([train[i]["y"] for i in tr_idx])
        Ptr = train_preds[tr_idx]
        for j in range(4):
            A = np.stack([Ptr[:, j], np.ones(len(Ptr))], 1)
            coef, *_ = np.linalg.lstsq(A, Ytr[:, j], rcond=None)
            out[te_idx, j] = coef[0] * out[te_idx, j] + coef[1]
    return np.clip(out, 0, 10)


# ------------------------------------------------------------------ 指标
def per_dim_metrics(y, p):
    m = {}
    for i, k in enumerate(DIMS):
        m[k] = {
            "pcc": float(stats.pearsonr(y[:, i], p[:, i])[0]),
            "spearman": float(stats.spearmanr(y[:, i], p[:, i])[0]),
            "mae": float(np.mean(np.abs(y[:, i] - p[:, i]))),
            "bias": float(np.mean(p[:, i] - y[:, i])),
        }
    # macro 用 nanmean：completeness 等维度在某子群退化成常数时 PCC 为 NaN，不应传染整体
    m["macro"] = {mk: float(np.nanmean([m[k][mk] for k in DIMS]))
                  for mk in ("pcc", "spearman", "mae", "bias")}
    m["macro_valid_dims"] = [k for k in DIMS if not np.isnan(m[k]["pcc"])]
    return m


def eval_by_subgroup(y, p, split):
    res = {"overall": per_dim_metrics(y, p)}
    for sg in ("child", "adult"):
        idx = [i for i, d in enumerate(split) if d["subgroup"] == sg]
        if len(idx) >= 8:
            res[sg] = per_dim_metrics(y[idx], p[idx])
    if "child" in res and "adult" in res:
        res["gap"] = {"macro": {
            "pcc": res["child"]["macro"]["pcc"] - res["adult"]["macro"]["pcc"],
            "mae": res["child"]["macro"]["mae"] - res["adult"]["macro"]["mae"],
            "bias": res["child"]["macro"]["bias"] - res["adult"]["macro"]["bias"],
        }}
    return res


def paired_bootstrap(y, p_a, p_b, test, n_boot=1000, seed=0):
    """分层（child/adult）配对 bootstrap，比较 A 与 B：
    d_pcc / d_mae（整体），d_gap_mae / d_gap_bias（子群差距，gap = child − adult）。"""
    rng = np.random.default_rng(seed)
    sg = np.array([d["subgroup"] for d in test])
    child_idx = np.where(sg == "child")[0]
    adult_idx = np.where(sg == "adult")[0]

    def calc(ii, sgii):
        yy, pa, pb = y[ii], p_a[ii], p_b[ii]
        is_child = (sgii == "child")
        def gap(p):
            cm = float(np.mean(np.abs(yy[is_child] - p[is_child])))
            am = float(np.mean(np.abs(yy[~is_child] - p[~is_child])))
            cb = float(np.mean(p[is_child] - yy[is_child]))
            ab = float(np.mean(p[~is_child] - yy[~is_child]))
            cp = macro_pcc(yy[is_child], p[is_child])
            ap = macro_pcc(yy[~is_child], p[~is_child])
            return cm - am, cb - ab, cp - ap
        ga = gap(pa)
        gb = gap(pb)
        return {
            "d_pcc": macro_pcc(yy, pa) - macro_pcc(yy, pb),
            "d_mae": float(np.mean(np.abs(yy - pa)) - np.mean(np.abs(yy - pb))),
            "d_gap_mae": ga[0] - gb[0],
            "d_gap_bias": ga[1] - gb[1],
            "d_gap_pcc": ga[2] - gb[2],
        }

    ii0 = np.concatenate([child_idx, adult_idx])
    point = calc(ii0, sg[ii0])
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        sa = rng.choice(child_idx, size=len(child_idx), replace=True)
        sb = rng.choice(adult_idx, size=len(adult_idx), replace=True)
        ii = np.concatenate([sa, sb])
        b = calc(ii, sg[ii])
        for k in point:
            boots[k].append(b[k])
    out = {}
    for k, v in boots.items():
        vv = np.asarray([x for x in v if not np.isnan(x)])
        if len(vv) < max(30, 0.5 * n_boot):
            out[k] = (point[k], float("nan"), float("nan"))
        else:
            out[k] = (point[k], float(np.percentile(vv, 2.5)), float(np.percentile(vv, 97.5)))
    return out


def wilcoxon_abs_err(y, p_a, p_b):
    ea = np.abs(y - p_a).mean(axis=1)
    eb = np.abs(y - p_b).mean(axis=1)
    try:
        _, p = stats.wilcoxon(ea, eb)
    except ValueError:
        p = float("nan")
    return float(np.mean(ea) - np.mean(eb)), float(p)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--lambda-r", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--lambda-pcc", type=float, default=0.3,
                    help="PCC loss 权重（0 = 纯 MSE）。B2-Rank / Ours-Rank 使用")
    ap.add_argument("--w-child", type=float, default=2.0,
                    help="子群加权 PCC loss 中 child 组权重（1 = 不加权）。Ours-Rank 使用")
    ap.add_argument("--skip-rank", action="store_true",
                    help="只跑 B0/B2/B6/Ours-MSE 四个基础模型，跳过 ranking 变体")
    ap.add_argument("--out-suffix", default="",
                    help="结果输出到 results_<suffix>/（换 backbone 对比时不覆盖旧结果）")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    data, child_max, n_total = load_split(args.data_dir)
    tr, dv, te = data["train"], data["dev"], data["test"]
    if len(dv) == 0:
        # dev 为空：通常是特征未提全（如 --max-utts 冒烟时样本未覆盖 dev speaker）。
        # 自动从 train 按 speaker 切 10% 作 dev，保证流程可跑；全量特征下不会触发。
        spks = sorted({d["speaker"] for d in tr if d["speaker"] is not None})
        _rng = random.Random(args.seed)
        _rng.shuffle(spks)
        dev_spks = set(spks[: max(1, len(spks) // 10)])
        dv = [d for d in tr if d["speaker"] in dev_spks]
        tr = [d for d in tr if d["speaker"] not in dev_spks]
        print(f"⚠ dev 为空，已从 train 自动切分（全量提取特征后不会出现此提示）")
    standardize(tr, (dv, te))
    n_loaded = len(tr) + len(dv) + len(te)
    if n_loaded < 0.8 * n_total:
        print(f"⚠ 仅加载到 {n_loaded}/{n_total} 条带特征的样本——"
              f"若你只跑了 02 的 --max-utts 冒烟，请先跑全量 02 再正式看结果（当前数字仅供流程验证）。")
    print(f"train {len(tr)} / dev {len(dv)} / test {len(te)}（child ≤ {child_max}）")
    print("test 子群:", {s: sum(1 for d in te if d['subgroup'] == s) for s in ('child', 'adult')})
    assert len(te) >= 16, "测试集太小：检查特征是否已提取（02）与 manifest 是否完整（01）"

    feat_dim = tr[0]["x"].shape[0]
    b2, b0, ours = build_models(feat_dim, args.delta, device)

    print("== 训练 B2 (profile-blind) ==")
    b2 = train_model(b2, "b2", tr, dv, device, args.epochs, args.lr, 0.0, args.seed)
    print("== 训练 B0 (profile-only) ==")
    b0 = train_model(b0, "b0", tr, dv, device, args.epochs, args.lr, 0.0, args.seed)
    print("== 训练 Ours (受限残差校准, MSE) ==")
    ours = train_model(ours, "ours", tr, dv, device, args.epochs, args.lr, args.lambda_r, args.seed)

    # ---- ranking 变体：pivot 后的主实验 ----
    b2_rank = ours_rank = None
    if not args.skip_rank:
        b2_rank, _, ours_rank = build_models(feat_dim, args.delta, device)
        print(f"== 训练 B2-Rank (profile-blind + 子群加权PCC loss, w={args.w_child}) ==")
        b2_rank = train_model(b2_rank, "b2", tr, dv, device, args.epochs, args.lr, 0.0,
                              args.seed, lambda_pcc=args.lambda_pcc, w_child=args.w_child)
        print("== 训练 Ours-Rank (受限残差校准 + 子群加权PCC loss) ==")
        ours_rank = train_model(ours_rank, "ours", tr, dv, device, args.epochs, args.lr,
                                args.lambda_r, args.seed,
                                lambda_pcc=args.lambda_pcc, w_child=args.w_child)

    Yte = np.stack([d["y"] for d in te])
    p_b2_tr = predict(b2, "b2", tr, device)
    p_b2 = predict(b2, "b2", te, device)
    p_b0 = predict(b0, "b0", te, device)
    p_ours = predict(ours, "ours", te, device)
    p_b6 = fit_b6(p_b2_tr, tr, p_b2, te)
    p_b2_rank = predict(b2_rank, "b2", te, device) if b2_rank is not None else None
    p_ours_rank = predict(ours_rank, "ours", te, device) if ours_rank is not None else None

    out_dir = os.path.join(args.data_dir, "results" + (f"_{args.out_suffix}" if args.out_suffix else ""))
    os.makedirs(out_dir, exist_ok=True)
    preds = {"b0": p_b0, "b2": p_b2, "b6": p_b6, "ours": p_ours}
    if p_b2_rank is not None:
        preds["b2_rank"] = p_b2_rank
        preds["ours_rank"] = p_ours_rank
    for name, p in preds.items():
        metrics = eval_by_subgroup(Yte, p, te)
        with open(os.path.join(out_dir, f"{name}_preds.jsonl"), "w", encoding="utf-8") as fh:
            for i, d in enumerate(te):
                fh.write(json.dumps({
                    "utt": d["utt"], "subgroup": d["subgroup"], "age": d["age"],
                    "y": [round(float(v), 3) for v in Yte[i]],
                    "pred": [round(float(v), 3) for v in p[i]],
                }, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)

    pairwise = {}
    pairs = [("ours_vs_b2", p_ours, p_b2),
             ("ours_vs_b6", p_ours, p_b6),
             ("b2_vs_b0", p_b2, p_b0),
             ("b6_vs_b2", p_b6, p_b2)]
    if p_b2_rank is not None:
        pairs += [("ours_rank_vs_b2", p_ours_rank, p_b2),
                  ("ours_rank_vs_b2_rank", p_ours_rank, p_b2_rank),
                  ("b2_rank_vs_b2", p_b2_rank, p_b2)]
    for pair, pa, pb in pairs:
        boot = paired_bootstrap(Yte, pa, pb, te, n_boot=args.n_boot, seed=args.seed)
        dmae, pval = wilcoxon_abs_err(Yte, pa, pb)
        pairwise[pair] = {
            "bootstrap": {k: {"est": v[0], "lo": v[1], "hi": v[2]} for k, v in boot.items()},
            "wilcoxon_d_mae": dmae, "wilcoxon_p": pval,
        }
    with open(os.path.join(out_dir, "pairwise.json"), "w") as fh:
        json.dump(pairwise, fh, indent=2)

    print(f"\n✓ 结果写入 {out_dir}/")
    print("下一步：python 04_make_report.py --data-dir ./data")


if __name__ == "__main__":
    main()
