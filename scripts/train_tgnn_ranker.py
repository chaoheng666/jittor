import argparse
import csv
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.data_loader import iter_train_edges


def normalize_features(x, mean, std):
    return (x - mean) / std


def iter_valid_rows(path, max_rows=0):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_rows and len(rows) >= max_rows:
                break
            candidates = [int(row[f"c{i}"]) for i in range(1, 101)]
            rows.append((int(row["src"]), int(row["time"]), int(row["label"]), candidates))
    return rows


def iter_test_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append((int(row[0]), int(row[1]), [int(x) for x in row[2:]]))
    return rows


def rank_of_label(scores, label):
    positive_score = scores[label]
    rank = 1
    for i, score in enumerate(scores):
        if i != label and score > positive_score:
            rank += 1
    return rank


def mrr(scores, labels):
    if len(labels) == 0:
        return 0.0
    total = 0.0
    for row, label in zip(scores, labels):
        total += 1.0 / rank_of_label(row, int(label))
    return total / len(labels)


def feature_mean_std(x_raw, chunk_size=1024):
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


def node_values_from_data(valid_train_edges, full_train_edges, valid_rows, test_rows):
    values = set()
    for src, dst, _ in valid_train_edges:
        values.add(src)
        values.add(dst)
    for src, dst, _ in full_train_edges:
        values.add(src)
        values.add(dst)
    for src, _, _, candidates in valid_rows:
        values.add(src)
        values.update(candidates)
    for src, _, candidates in test_rows:
        values.add(src)
        values.update(candidates)
    return values


def jt_batch(jt, batch):
    return {name: jt.array(value) for name, value in batch.items()}


def bpr_hard_loss(jt, logits, rule_scores, labels, hard_negatives):
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


def predict_scores(jt, model, builder, index, queries, x_raw, mean, std, batch_size):
    model.eval()
    out = []
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch_queries = [(src, time, candidates) for src, time, candidates in queries[start:end]]
        graph = jt_batch(jt, builder.build_batch(index, batch_queries))
        features = normalize_features(
            np.asarray(x_raw[start:end], dtype=np.float32),
            mean,
            std,
        ).astype(np.float32)
        scores = model(
            graph["src_idx"],
            graph["cand_idx"],
            graph["src_nbr"],
            graph["src_gap"],
            graph["src_dir"],
            graph["cand_nbr"],
            graph["cand_gap"],
            graph["cand_dir"],
            graph["hop_nbr"],
            graph["hop_gap"],
            graph["hop_dir"],
            jt.array(features),
        ).numpy()
        out.append(scores)
    return np.vstack(out)


def train_one_dataset(args, dataset_name):
    import jittor as jt
    from jittor import nn
    from src.temporal_gnn import (
        TemporalGNNDatasetBuilder,
        TemporalGNNRanker,
        TemporalNeighborIndex,
        save_tgnn_model,
    )

    if args.cuda:
        jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)

    valid_dataset_dir = Path(args.valid_dir) / dataset_name
    full_dataset_dir = Path(args.data_dir) / dataset_name
    cache_dir = Path(args.cache_dir) / dataset_name

    valid_rows_full = iter_valid_rows(valid_dataset_dir / "valid.csv", args.max_rows)
    valid_queries = [(src, time, candidates) for src, time, _, candidates in valid_rows_full]
    y_all = np.asarray([label for _, _, label, _ in valid_rows_full], dtype=np.int32)
    x_all = np.load(cache_dir / "x_valid.npy", mmap_mode="r")[:len(y_all)]
    test_queries = iter_test_rows(full_dataset_dir / "test.csv")
    x_test = np.load(cache_dir / "x_test.npy", mmap_mode="r")

    valid_train_edges = list(iter_train_edges(valid_dataset_dir / "train.csv"))
    full_train_edges = list(iter_train_edges(full_dataset_dir / "train.csv"))
    node_values = node_values_from_data(valid_train_edges, full_train_edges, valid_rows_full, test_queries)
    valid_index = TemporalNeighborIndex(valid_train_edges)
    full_index = TemporalNeighborIndex(full_train_edges)
    builder = TemporalGNNDatasetBuilder(
        node_values,
        src_neighbors=args.src_neighbors,
        cand_neighbors=args.cand_neighbors,
        second_hop=args.second_hop,
        max_time_bucket=args.max_time_bucket,
    )

    if len(y_all) < 5:
        raise ValueError(f"{dataset_name}: not enough rows for TGNN training")
    cut = max(1, int(len(y_all) * (1.0 - args.eval_ratio)))
    if cut >= len(y_all):
        cut = len(y_all) - 1
    train_idx = list(range(cut))
    eval_queries = valid_queries[cut:]
    eval_x = x_all[cut:]
    eval_y = y_all[cut:]

    mean, std = feature_mean_std(x_all[:cut])
    model = TemporalGNNRanker(
        len(builder.node_values),
        x_all.shape[-1],
        node_emb_dim=args.node_emb_dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_time_bucket=args.max_time_bucket,
    )
    optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    best_mrr = -1.0
    model_dir = Path(args.model_dir)
    score_dir = Path(args.score_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / f"{dataset_name}_tgnn_ranker.pkl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_idx)
        loss_sum = 0.0
        steps = 0
        for start in range(0, len(train_idx), args.batch_size):
            batch_idx = train_idx[start:start + args.batch_size]
            batch_queries = [valid_queries[i] for i in batch_idx]
            bx_raw = np.asarray(x_all[batch_idx], dtype=np.float32)
            bx = normalize_features(bx_raw, mean, std).astype(np.float32)
            by_np = y_all[batch_idx]
            graph = jt_batch(jt, builder.build_batch(valid_index, batch_queries))
            scores = model(
                graph["src_idx"],
                graph["cand_idx"],
                graph["src_nbr"],
                graph["src_gap"],
                graph["src_dir"],
                graph["cand_nbr"],
                graph["cand_gap"],
                graph["cand_dir"],
                graph["hop_nbr"],
                graph["hop_gap"],
                graph["hop_dir"],
                jt.array(bx),
            )
            by = jt.array(by_np)
            loss = nn.cross_entropy_loss(scores, by)
            if args.bpr_weight > 0:
                rule_scores = bx_raw[:, :, args.rule_feature_idx]
                bpr = bpr_hard_loss(jt, scores, rule_scores, by_np, args.hard_negatives)
                if bpr is not None:
                    loss = loss + bpr * args.bpr_weight
            optimizer.step(loss)
            loss_sum += float(loss.numpy())
            steps += 1

        eval_scores = predict_scores(
            jt, model, builder, valid_index, eval_queries, eval_x, mean, std, args.batch_size
        )
        eval_mrr = mrr(eval_scores, eval_y)
        print(f"{dataset_name} epoch={epoch} loss={loss_sum / max(steps, 1):.6f} tgnn_mrr={eval_mrr:.8f}")
        if eval_mrr > best_mrr:
            best_mrr = eval_mrr
            save_tgnn_model(best_path, model, {
                "dataset_name": dataset_name,
                "n_nodes": len(builder.node_values),
                "node_values": builder.node_values,
                "feature_dim": int(x_all.shape[-1]),
                "node_emb_dim": int(args.node_emb_dim),
                "time_emb_dim": int(args.time_emb_dim),
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "max_time_bucket": int(args.max_time_bucket),
                "src_neighbors": int(args.src_neighbors),
                "cand_neighbors": int(args.cand_neighbors),
                "second_hop": int(args.second_hop),
                "mean": mean.tolist(),
                "std": std.tolist(),
            })

    if best_path.exists():
        data = jt.load(str(best_path))
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(data["state_dict"])
        else:
            model.load_parameters(data["state_dict"])

    valid_scores = predict_scores(
        jt, model, builder, valid_index, valid_queries, x_all, mean, std, args.batch_size
    )
    np.save(score_dir / f"{dataset_name}_tgnn_valid.npy", valid_scores.astype(np.float32))
    test_scores = predict_scores(
        jt, model, builder, full_index, test_queries, x_test, mean, std, args.batch_size
    )
    np.save(score_dir / f"{dataset_name}_tgnn_test.npy", test_scores.astype(np.float32))
    print(
        f"{dataset_name}: saved TGNN scores best_mrr={best_mrr:.8f} "
        f"valid_shape={valid_scores.shape} test_shape={test_scores.shape}"
    )


def find_dataset_names(valid_dir, dataset_arg):
    if dataset_arg != "all":
        return [name.strip() for name in dataset_arg.split(",") if name.strip()]
    return sorted(
        p.name for p in Path(valid_dir).iterdir()
        if p.is_dir() and (p / "train.csv").exists() and (p / "valid.csv").exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--valid-dir", default="validation")
    parser.add_argument("--cache-dir", default="feature_cache")
    parser.add_argument("--model-dir", default="models_tgnn")
    parser.add_argument("--score-dir", default="scores_tgnn")
    parser.add_argument("--dataset", default="all", help="all or comma-separated dataset names")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--node-emb-dim", type=int, default=128)
    parser.add_argument("--time-emb-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--src-neighbors", type=int, default=50)
    parser.add_argument("--cand-neighbors", type=int, default=30)
    parser.add_argument("--second-hop", type=int, default=20)
    parser.add_argument("--max-time-bucket", type=int, default=127)
    parser.add_argument("--hard-negatives", type=int, default=30)
    parser.add_argument("--bpr-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rule-feature-idx", type=int, default=-1)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    if args.rule_feature_idx < 0:
        import json
        cache_root = Path(args.cache_dir)
        first_dataset = find_dataset_names(args.valid_dir, args.dataset)[0]
        with open(cache_root / first_dataset / "feature_names.json", encoding="utf-8") as f:
            names = json.load(f)
        args.rule_feature_idx = names.index("rule_score")

    random.seed(args.seed)
    np.random.seed(args.seed)
    names = find_dataset_names(args.valid_dir, args.dataset)
    if not names:
        raise ValueError(f"no validation datasets found in {args.valid_dir}")
    for name in names:
        train_one_dataset(args, name)


if __name__ == "__main__":
    main()
