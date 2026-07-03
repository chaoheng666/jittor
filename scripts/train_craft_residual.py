import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.craft_residual import CraftResidual, save_craft_model
from src.data_loader import find_dataset_dirs, iter_train_edges, split_by_time
from src.samplers import MixedNegativeSampler
from src.seq_tower import build_dst_vocab, build_source_histories, jittor_available


def write_disabled(out_dir, dataset_name, reason):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset_name}_craft_residual.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name,
            "type": "craft_residual",
            "enabled": False,
            "disabled_reason": reason,
        }, f, indent=2, ensure_ascii=False)
    print(f"{dataset_name}: craft_residual disabled: {reason}")


def hist_vector(hist, seq_len):
    values = list(hist)[-seq_len:]
    out = np.zeros(seq_len, dtype=np.int32)
    if values:
        out[-len(values):] = values
    return out


def build_samples(dataset_name, history, supervision, dst_to_id, args, seed):
    from src.jt_ranker import CandidateFeatureBuilder, FEATURE_NAMES

    sampler = MixedNegativeSampler(history, seed=seed)
    histories = build_source_histories(history, dst_to_id, seq_len=args.seq_len)
    builder = CandidateFeatureBuilder(dataset_name)
    builder.fit(history)
    rows = list(supervision)
    if args.sample_edges and args.sample_edges < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, args.sample_edges)
        rows.sort(key=lambda x: x[2])

    hist_rows = []
    cand_rows = []
    feat_rows = []
    skipped = 0
    for src, dst, time in rows:
        pos_id = dst_to_id.get(dst, 0)
        hist = list(histories.get(src, ()))
        if pos_id == 0 or not hist:
            skipped += 1
            histories.setdefault(src, deque(maxlen=args.seq_len)).append(pos_id)
            continue
        negs = sampler.sample(src, dst, args.neg_per_pos)
        candidates = [dst] + negs
        hist_rows.append(hist_vector(hist, args.seq_len))
        cand_rows.append([dst_to_id.get(value, 0) for value in candidates])
        feat_rows.append([
            builder.vector(src, time, value, feature_names=FEATURE_NAMES)
            for value in candidates
        ])
        histories.setdefault(src, deque(maxlen=args.seq_len)).append(pos_id)

    return (
        np.asarray(hist_rows, dtype=np.int32),
        np.asarray(cand_rows, dtype=np.int32),
        np.asarray(feat_rows, dtype=np.float32),
        FEATURE_NAMES,
        skipped,
    )


def train_dataset(dataset_dir, out_dir, args):
    if not jittor_available():
        write_disabled(out_dir, dataset_dir.name, "jittor_not_available")
        return
    import jittor as jt
    from jittor import nn

    if args.cuda:
        jt.flags.use_cuda = 1

    edges = list(iter_train_edges(dataset_dir / "train.csv"))
    history, supervision = split_by_time(edges, args.history_ratio)
    dst_to_id = build_dst_vocab(history)
    hist_arr, cand_arr, feat_raw, feature_names, skipped = build_samples(dataset_dir.name, history, supervision, dst_to_id, args, args.seed)
    if len(hist_arr) < 10:
        write_disabled(out_dir, dataset_dir.name, f"too_few_craft_samples:{len(hist_arr)} skipped:{skipped}")
        return
    flat = feat_raw.reshape(-1, feat_raw.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    feat_arr = ((feat_raw - mean) / std).astype(np.float32)

    cut = max(1, int(len(hist_arr) * (1.0 - args.eval_ratio)))
    train_idx = np.arange(cut)
    eval_idx = np.arange(cut, len(hist_arr))
    model = CraftResidual(len(dst_to_id), feat_arr.shape[-1], emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{dataset_dir.name}_craft_residual.pkl"
    best_mrr = -1.0
    labels_cache = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_idx)
        loss_sum = 0.0
        steps = 0
        for start in range(0, len(train_idx), args.batch_size):
            idx = train_idx[start:start + args.batch_size]
            scores = model(jt.array(hist_arr[idx]), jt.array(cand_arr[idx]), jt.array(feat_arr[idx]))
            if len(idx) not in labels_cache:
                labels_cache[len(idx)] = jt.array(np.zeros(len(idx), dtype=np.int32))
            loss = nn.cross_entropy_loss(scores, labels_cache[len(idx)])
            optimizer.step(loss)
            loss_sum += float(loss.numpy())
            steps += 1

        model.eval()
        rr = []
        for start in range(0, len(eval_idx), args.batch_size):
            idx = eval_idx[start:start + args.batch_size]
            scores = model(jt.array(hist_arr[idx]), jt.array(cand_arr[idx]), jt.array(feat_arr[idx])).numpy()
            rr.extend(1.0 / (1 + (scores[:, 1:] > scores[:, :1]).sum(axis=1)))
        mrr = float(np.mean(rr)) if rr else 0.0
        print(f"{dataset_dir.name}: craft epoch={epoch} loss={loss_sum / max(steps, 1):.6f} sampled_mrr={mrr:.6f}")
        if mrr > best_mrr:
            best_mrr = mrr
            save_craft_model(best_path, model, {
                "dataset": dataset_dir.name,
                "type": "craft_residual",
                "enabled": True,
                "num_dst": len(dst_to_id),
                "dst_to_id": {str(k): int(v) for k, v in dst_to_id.items()},
                "feature_dim": feat_arr.shape[-1],
                "feature_names": feature_names,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "seq_len": args.seq_len,
                "emb_dim": args.emb_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "sampled_mrr": best_mrr,
                "skipped": skipped,
                "cold_policy": "zero_id",
            })
    print(f"{dataset_dir.name}: saved {best_path} best_sampled_mrr={best_mrr:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--model-dir", default="models_v2/craft")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--history-ratio", type=float, default=0.8)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--sample-edges", type=int, default=30000)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--neg-per-pos", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset_dirs = find_dataset_dirs(args.data_dir)
    if args.dataset != "all":
        wanted = {name.strip() for name in args.dataset.split(",") if name.strip()}
        dataset_dirs = [path for path in dataset_dirs if path.name in wanted]
    out_dir = Path(args.model_dir)
    for dataset_dir in dataset_dirs:
        train_dataset(dataset_dir, out_dir, args)


if __name__ == "__main__":
    main()
