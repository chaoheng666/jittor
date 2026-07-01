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


def load_cached_arrays(cache_dir, dataset_name, max_rows):
    dataset_cache = Path(cache_dir) / dataset_name
    x = np.load(dataset_cache / "x_valid.npy", mmap_mode="r")
    y = np.load(dataset_cache / "y_valid.npy", mmap_mode="r")
    if max_rows:
        x = x[:max_rows]
        y = y[:max_rows]
    return x, y


def feature_mean_std(x_raw, chunk_size=2048):
    feature_dim = x_raw.shape[-1]
    total = np.zeros(feature_dim, dtype=np.float64)
    total_sq = np.zeros(feature_dim, dtype=np.float64)
    count = 0
    for start in range(0, len(x_raw), chunk_size):
        chunk = np.asarray(x_raw[start:start + chunk_size], dtype=np.float32).reshape(-1, feature_dim)
        total += chunk.sum(axis=0)
        total_sq += (chunk.astype(np.float64) ** 2).sum(axis=0)
        count += len(chunk)
    mean = (total / max(count, 1)).astype(np.float32)
    var = total_sq / max(count, 1) - mean.astype(np.float64) ** 2
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def hard_candidate_mask(rule_scores, labels, hard_negatives):
    if hard_negatives <= 0 or hard_negatives >= rule_scores.shape[1] - 1:
        return np.ones(rule_scores.shape, dtype=np.float32)

    mask = np.zeros(rule_scores.shape, dtype=np.float32)
    for row_idx, label in enumerate(labels):
        label = int(label)
        mask[row_idx, label] = 1.0
        row = np.asarray(rule_scores[row_idx], dtype=np.float32).copy()
        row[label] = -np.inf
        hard_idx = np.argpartition(row, -hard_negatives)[-hard_negatives:]
        mask[row_idx, hard_idx] = 1.0
    return mask


def bpr_hard_loss(logits, rule_scores, labels, hard_negatives):
    if hard_negatives <= 0:
        return None
    diffs = []
    for row_idx, label in enumerate(labels):
        label = int(label)
        row = np.asarray(rule_scores[row_idx], dtype=np.float32).copy()
        row[label] = -np.inf
        count = min(hard_negatives, row.shape[0] - 1)
        hard_idx = np.argpartition(row, -count)[-count:]
        pos = logits[row_idx, label]
        for neg in hard_idx:
            diffs.append(pos - logits[row_idx, int(neg)])
    if not diffs:
        return None
    diff = jt.stack(diffs)
    return -jt.log(jt.sigmoid(diff) + 1e-8).mean()


def evaluate(model, x_raw, y, mean, std, fuse_rule, mlp_weight):
    rr_rule = 0.0
    rr_mlp = 0.0
    rr_fused = 0.0
    rows = 0
    rule_idx = FEATURE_NAMES.index("rule_score")

    model.eval()
    for start in range(0, len(x_raw), 512):
        end = min(start + 512, len(x_raw))
        x = normalize_features(np.asarray(x_raw[start:end], dtype=np.float32), mean, std).astype(np.float32)
        scores = model(jt.array(x)).numpy()
        raw_rule = x_raw[start:end, :, rule_idx]
        fused = raw_rule * fuse_rule + np.tanh(scores) * mlp_weight
        for i in range(end - start):
            label = int(y[start + i])
            rr_rule += 1.0 / rank_of_label(raw_rule[i], label)
            rr_mlp += 1.0 / rank_of_label(scores[i], label)
            rr_fused += 1.0 / rank_of_label(fused[i], label)
            rows += 1
    return rr_rule / rows, rr_mlp / rows, rr_fused / rows


def train_one_dataset(args, dataset_name):
    dataset_dir = Path(args.valid_dir) / dataset_name
    if args.cache_dir:
        x_all, y_all = load_cached_arrays(args.cache_dir, dataset_name, args.max_rows)
    else:
        x_all, y_all = build_arrays(dataset_dir, dataset_name, args.max_rows)
    if len(x_all) >= 5:
        cut = max(1, int(len(x_all) * (1.0 - args.eval_ratio)))
        x_raw, y = x_all[:cut], y_all[:cut]
        eval_x_raw, eval_y = x_all[cut:], y_all[cut:]
    else:
        x_raw, y = x_all, y_all
        eval_x_raw, eval_y = x_all, y_all

    mean, std = feature_mean_std(x_raw)

    model = MLPRanker(x_raw.shape[-1], args.hidden_dim)
    optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    order = list(range(len(x_raw)))
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
            bx_raw = np.asarray(x_raw[batch_idx], dtype=np.float32)
            bx = jt.array(normalize_features(bx_raw, mean, std).astype(np.float32))
            by = jt.array(y[batch_idx])
            br = jt.array(bx_raw[:, :, FEATURE_NAMES.index("rule_score")])
            scores = model(bx)
            logits = br * args.fuse_rule + jt.tanh(scores) * args.mlp_weight
            if args.hard_negatives > 0:
                mask = hard_candidate_mask(
                    bx_raw[:, :, FEATURE_NAMES.index("rule_score")],
                    y[batch_idx],
                    args.hard_negatives,
                )
                logits = logits + (jt.array(mask) - 1.0) * 10000.0
            loss = nn.cross_entropy_loss(logits, by)
            if args.bpr_weight > 0:
                bpr = bpr_hard_loss(scores, bx_raw[:, :, FEATURE_NAMES.index("rule_score")], y[batch_idx], args.hard_negatives)
                if bpr is not None:
                    loss = loss + bpr * args.bpr_weight
            optimizer.step(loss)
            loss_sum += float(loss.numpy())
            steps += 1

        rule_mrr, mlp_mrr, fused_mrr = evaluate(
            model, eval_x_raw, eval_y, mean, std, args.fuse_rule, args.mlp_weight
        )
        print(
            f"{dataset_name} epoch={epoch} loss={loss_sum / max(steps, 1):.6f} "
            f"rule_mrr={rule_mrr:.8f} mlp_mrr={mlp_mrr:.8f} fused_mrr={fused_mrr:.8f}"
        )
        if fused_mrr > best_mrr:
            best_mrr = fused_mrr
            save_model(best_path, model, {
                "dataset_name": dataset_name,
                "feature_dim": int(x_raw.shape[-1]),
                "hidden_dim": int(args.hidden_dim),
                "feature_names": FEATURE_NAMES,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "fuse_rule": float(args.fuse_rule),
                "mlp_weight": float(args.mlp_weight),
                "hard_negatives": int(args.hard_negatives),
                "bpr_weight": float(args.bpr_weight),
                "use_mlp": True,
            })
    print(f"{dataset_name}: saved {best_path} best_fused_mrr={best_mrr:.8f}")


def find_dataset_names(valid_dir, dataset_arg):
    if dataset_arg != "all":
        return [name.strip() for name in dataset_arg.split(",") if name.strip()]
    valid_path = Path(valid_dir)
    return sorted(
        p.name for p in valid_path.iterdir()
        if p.is_dir() and (p / "train.csv").exists() and (p / "valid.csv").exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-dir", default="validation")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--dataset", default="all", help="all or comma-separated dataset names")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--fuse-rule", type=float, default=1.0)
    parser.add_argument("--mlp-weight", type=float, default=0.2)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--hard-negatives", type=int, default=30)
    parser.add_argument("--bpr-weight", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    if args.cuda:
        jt.flags.use_cuda = 1

    names = find_dataset_names(args.valid_dir, args.dataset)
    if not names:
        raise ValueError(f"no validation datasets found in {args.valid_dir}")
    seeds = [int(x) for x in args.seed_list.split(",") if x.strip()]
    if not seeds:
        seeds = [args.seed]

    base_model_dir = Path(args.model_dir)
    for seed in seeds:
        args.seed = seed
        args.model_dir = str(base_model_dir / f"seed_{seed}") if len(seeds) > 1 else str(base_model_dir)
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        random.seed(seed)
        np.random.seed(seed)
        jt.set_global_seed(seed)
        for name in names:
            train_one_dataset(args, name)


if __name__ == "__main__":
    main()
