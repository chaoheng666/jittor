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
from src.jt_ranker import CandidateFeatureBuilder, FEATURE_NAMES, normalize_features
from src.seq_ranker import SequenceFeatureBuilder, SeqResidualRanker, save_seq_model


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


def build_arrays(dataset_dir, dataset_name, seq_len, max_rows):
    train_edges = list(iter_train_edges(dataset_dir / "train.csv"))
    rows = []
    extra_dsts = []
    for src, time, label, candidates in iter_valid_rows(dataset_dir / "valid.csv"):
        if max_rows and len(rows) >= max_rows:
            break
        rows.append((src, time, label, candidates))
        extra_dsts.extend(candidates)

    if not rows:
        raise ValueError(f"{dataset_dir / 'valid.csv'} has no training rows")

    feature_builder = CandidateFeatureBuilder(dataset_name)
    feature_builder.fit(train_edges)
    seq_builder = SequenceFeatureBuilder(seq_len)
    seq_builder.fit(train_edges, extra_dsts)

    xs = []
    seq_dst = []
    seq_gap = []
    cand_idx = []
    labels = []
    for src, time, label, candidates in rows:
        sdst, sgap, cidx = seq_builder.build_query(src, time, candidates)
        xs.append(feature_builder.matrix(src, time, candidates))
        seq_dst.append(sdst)
        seq_gap.append(sgap)
        cand_idx.append(cidx)
        labels.append(label)

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(seq_dst, dtype=np.int32),
        np.asarray(seq_gap, dtype=np.int32),
        np.asarray(cand_idx, dtype=np.int32),
        np.asarray(labels, dtype=np.int32),
        seq_builder.dst_values,
    )


def evaluate(model, x_raw, seq_dst, seq_gap, cand_idx, y, mean, std, fuse_rule, gamma):
    x = normalize_features(x_raw, mean, std).astype(np.float32)
    rule_idx = FEATURE_NAMES.index("rule_score")
    rr_rule = 0.0
    rr_seq = 0.0
    rr_fused = 0.0
    rows = 0

    model.eval()
    for start in range(0, len(x), 256):
        end = min(start + 256, len(x))
        scores = model(
            jt.array(seq_dst[start:end]),
            jt.array(seq_gap[start:end]),
            jt.array(cand_idx[start:end]),
            jt.array(x[start:end]),
        ).numpy()
        raw_rule = x_raw[start:end, :, rule_idx]
        fused = raw_rule * fuse_rule + np.tanh(scores) * gamma
        for i in range(end - start):
            label = int(y[start + i])
            rr_rule += 1.0 / rank_of_label(raw_rule[i], label)
            rr_seq += 1.0 / rank_of_label(scores[i], label)
            rr_fused += 1.0 / rank_of_label(fused[i], label)
            rows += 1
    return rr_rule / rows, rr_seq / rows, rr_fused / rows


def train_one_dataset(args, dataset_name):
    dataset_dir = Path(args.valid_dir) / dataset_name
    x_all, sd_all, sg_all, ci_all, y_all, dst_values = build_arrays(
        dataset_dir, dataset_name, args.seq_len, args.max_rows
    )
    if len(x_all) >= 5:
        cut = max(1, int(len(x_all) * (1.0 - args.eval_ratio)))
        train_slice = slice(0, cut)
        eval_slice = slice(cut, None)
    else:
        train_slice = slice(None)
        eval_slice = slice(None)

    x_raw = x_all[train_slice]
    seq_dst = sd_all[train_slice]
    seq_gap = sg_all[train_slice]
    cand_idx = ci_all[train_slice]
    y = y_all[train_slice]

    eval_x_raw = x_all[eval_slice]
    eval_seq_dst = sd_all[eval_slice]
    eval_seq_gap = sg_all[eval_slice]
    eval_cand_idx = ci_all[eval_slice]
    eval_y = y_all[eval_slice]

    mean = x_raw.reshape(-1, x_raw.shape[-1]).mean(axis=0).astype(np.float32)
    std = x_raw.reshape(-1, x_raw.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    x = normalize_features(x_raw, mean, std).astype(np.float32)

    model = SeqResidualRanker(
        len(dst_values),
        x.shape[-1],
        seq_len=args.seq_len,
        dst_emb_dim=args.dst_emb_dim,
        time_emb_dim=args.time_emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    order = list(range(len(x)))
    rng = random.Random(args.seed)
    rule_idx = FEATURE_NAMES.index("rule_score")

    best_mrr = -1.0
    best_path = Path(args.model_dir) / f"{dataset_name}_seq_ranker.pkl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(order)
        loss_sum = 0.0
        steps = 0
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start:start + args.batch_size]
            scores = model(
                jt.array(seq_dst[batch_idx]),
                jt.array(seq_gap[batch_idx]),
                jt.array(cand_idx[batch_idx]),
                jt.array(x[batch_idx]),
            )
            br = jt.array(x_raw[batch_idx, :, rule_idx])
            by = jt.array(y[batch_idx])
            loss = nn.cross_entropy_loss(br * args.fuse_rule + jt.tanh(scores) * args.gamma, by)
            optimizer.step(loss)
            loss_sum += float(loss.numpy())
            steps += 1

        rule_mrr, seq_mrr, fused_mrr = evaluate(
            model, eval_x_raw, eval_seq_dst, eval_seq_gap, eval_cand_idx,
            eval_y, mean, std, args.fuse_rule, args.gamma
        )
        print(
            f"{dataset_name} epoch={epoch} loss={loss_sum / max(steps, 1):.6f} "
            f"rule_mrr={rule_mrr:.8f} seq_mrr={seq_mrr:.8f} fused_mrr={fused_mrr:.8f}"
        )
        if fused_mrr > best_mrr:
            best_mrr = fused_mrr
            save_seq_model(best_path, model, {
                "dataset_name": dataset_name,
                "n_dst": len(dst_values),
                "dst_values": dst_values,
                "feature_dim": int(x.shape[-1]),
                "feature_names": FEATURE_NAMES,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "seq_len": int(args.seq_len),
                "dst_emb_dim": int(args.dst_emb_dim),
                "time_emb_dim": int(args.time_emb_dim),
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "fuse_rule": float(args.fuse_rule),
                "gamma": float(args.gamma),
                "use_seq": bool(fused_mrr > rule_mrr),
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
    parser.add_argument("--model-dir", default="models_seq")
    parser.add_argument("--dataset", default="all", help="all or comma-separated dataset names")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--dst-emb-dim", type=int, default=128)
    parser.add_argument("--time-emb-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--fuse-rule", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-list", default="")
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
