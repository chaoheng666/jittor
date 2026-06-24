import argparse
import csv
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import jittor as jt
from jittor import nn

from src.data_loader import iter_train_edges
from src.jt_ranker import CandidateFeatureBuilder, FEATURE_NAMES, MLPRanker, normalize_features, save_model


def iter_valid_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates = [int(row[f"c{i}"]) for i in range(1, 101)]
            yield int(row["src"]), int(row["time"]), int(row["label"]), candidates


def rank_of_label(scores, label):
    positive_score = scores[label]
    rank = 1
    for i, score in enumerate(scores):
        if i != label and score > positive_score:
            rank += 1
    return rank


def build_arrays(dataset_dir, dataset_name, max_rows):
    train_edges = list(iter_train_edges(dataset_dir / "train.csv"))
    feature_builder = CandidateFeatureBuilder(dataset_name)
    feature_builder.fit(train_edges)

    xs = []
    labels = []
    used = 0
    for src, time, label, candidates in iter_valid_rows(dataset_dir / "valid.csv"):
        if max_rows and used >= max_rows:
            break
        xs.append(feature_builder.matrix(src, time, candidates))
        labels.append(label)
        used += 1

    if not xs:
        raise ValueError(f"{dataset_dir / 'valid.csv'} has no training rows")
    return np.asarray(xs, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def evaluate(model, x_raw, y, mean, std, fuse_rule):
    x = normalize_features(x_raw, mean, std).astype(np.float32)
    rr_mlp = 0.0
    rr_fused = 0.0
    rows = 0
    rule_idx = FEATURE_NAMES.index("rule_score")

    model.eval()
    for start in range(0, len(x), 512):
        end = min(start + 512, len(x))
        scores = model(jt.array(x[start:end])).numpy()
        raw_rule = x_raw[start:end, :, rule_idx]
        fused = scores + raw_rule * fuse_rule
        for i in range(end - start):
            label = int(y[start + i])
            rr_mlp += 1.0 / rank_of_label(scores[i], label)
            rr_fused += 1.0 / rank_of_label(fused[i], label)
            rows += 1
    return rr_mlp / rows, rr_fused / rows


def train_one_dataset(args, dataset_name):
    dataset_dir = Path(args.valid_dir) / dataset_name
    x_raw, y = build_arrays(dataset_dir, dataset_name, args.max_rows)
    mean = x_raw.reshape(-1, x_raw.shape[-1]).mean(axis=0)
    std = x_raw.reshape(-1, x_raw.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    mean = mean.astype(np.float32)
    x = normalize_features(x_raw, mean, std).astype(np.float32)

    model = MLPRanker(x.shape[-1], args.hidden_dim)
    optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    order = list(range(len(x)))
    rng = random.Random(args.seed)

    best_mrr = -1.0
    best_path = Path(args.model_dir) / f"{dataset_name}_jt_ranker.pkl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(order)
        loss_sum = 0.0
        steps = 0
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start:start + args.batch_size]
            bx = jt.array(x[batch_idx])
            by = jt.array(y[batch_idx])
            scores = model(bx)
            loss = nn.cross_entropy_loss(scores, by)
            optimizer.step(loss)
            loss_sum += float(loss.numpy())
            steps += 1

        mlp_mrr, fused_mrr = evaluate(model, x_raw, y, mean, std, args.fuse_rule)
        print(
            f"{dataset_name} epoch={epoch} loss={loss_sum / max(steps, 1):.6f} "
            f"mlp_mrr={mlp_mrr:.8f} fused_mrr={fused_mrr:.8f}"
        )
        if fused_mrr > best_mrr:
            best_mrr = fused_mrr
            save_model(best_path, model, {
                "dataset_name": dataset_name,
                "feature_dim": int(x.shape[-1]),
                "hidden_dim": int(args.hidden_dim),
                "feature_names": FEATURE_NAMES,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "fuse_rule": float(args.fuse_rule),
            })
    print(f"{dataset_name}: saved {best_path} best_fused_mrr={best_mrr:.8f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-dir", default="validation")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--dataset", choices=["all", "dataset1", "dataset2"], default="all")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--fuse-rule", type=float, default=1.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    if args.cuda:
        jt.flags.use_cuda = 1

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)

    names = ["dataset1", "dataset2"] if args.dataset == "all" else [args.dataset]
    for name in names:
        train_one_dataset(args, name)


if __name__ == "__main__":
    main()
