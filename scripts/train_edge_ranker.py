import argparse
import csv
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import jittor as jt
from jittor import nn

from src.data_loader import find_dataset_dirs
from src.jt_ranker import CandidateFeatureBuilder, FEATURE_NAMES, MLPRanker, normalize_features, save_model


def read_edges(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield int(row["src"]), int(row["dst"]), int(row["time"])


def split_by_time(edges, history_ratio):
    rows = sorted(edges, key=lambda x: x[2])
    if len(rows) < 2:
        raise ValueError("need at least two edges for history/supervision split")
    cut = int(len(rows) * history_ratio)
    cut = min(max(cut, 1), len(rows) - 1)
    return rows[:cut], rows[cut:]


class EdgeNegativeSampler:
    def __init__(
        self,
        history_edges,
        seed,
        recent_limit=100,
        transition_limit=200,
        popular_limit=3000,
        cooc_window=8,
    ):
        self.rng = random.Random(seed)
        self.transition_limit = transition_limit
        self.recent_by_src = defaultdict(lambda: deque(maxlen=recent_limit))
        self.transition = defaultdict(Counter)
        self.cooc = defaultdict(Counter)

        dst_counter = Counter()
        by_src = defaultdict(list)
        for src, dst, time in sorted(history_edges, key=lambda x: x[2]):
            dst_counter[dst] += 1
            self.recent_by_src[src].append(dst)
            by_src[src].append((time, dst))

        self.dst_unique = list(dst_counter.keys())
        self.popular = [dst for dst, _ in dst_counter.most_common(popular_limit)]
        if not self.dst_unique:
            raise ValueError("history split has no destination nodes")

        for src, rows in by_src.items():
            rows.sort()
            dsts = [dst for _, dst in rows]
            for i in range(1, len(dsts)):
                self.transition[dsts[i - 1]][dsts[i]] += 1
            recent = dsts[-300:]
            for i, dst in enumerate(recent):
                start = max(0, i - cooc_window)
                for j in range(start, i):
                    prev = recent[j]
                    if prev != dst:
                        self.cooc[prev][dst] += 1.0 / (i - j)

    def _allowed(self, src, positive, dst, seen):
        return dst != positive and dst not in seen

    def sample(self, src, positive, count):
        seen = set()
        negatives = []
        recent = list(self.recent_by_src.get(src, ()))
        hard = list(reversed(recent))
        if recent:
            last_dst = recent[-1]
            hard.extend(dst for dst, _ in self.transition[last_dst].most_common(self.transition_limit))
            hard.extend(dst for dst, _ in self.cooc[last_dst].most_common(self.transition_limit))
        hard.extend(self.popular)

        for dst in hard:
            if len(negatives) >= count:
                return negatives
            if self._allowed(src, positive, dst, seen):
                seen.add(dst)
                negatives.append(dst)

        tries = 0
        max_tries = max(1000, count * 500)
        while len(negatives) < count and tries < max_tries:
            tries += 1
            dst = self.rng.choice(self.dst_unique)
            if self._allowed(src, positive, dst, seen):
                seen.add(dst)
                negatives.append(dst)
        return negatives


def choose_positive_edges(edges, sample_edges, rng):
    if sample_edges and sample_edges < len(edges):
        rows = rng.sample(edges, sample_edges)
        rows.sort(key=lambda x: x[2])
        return rows
    return list(edges)


def build_pair_array(builder, sampler, positives, negatives):
    rows = []
    skipped = 0
    for src, dst, time in positives:
        negs = sampler.sample(src, dst, negatives)
        if not negs:
            skipped += 1
            continue
        pos_vec = builder.vector(src, time, dst)
        for neg_dst in negs:
            rows.append([pos_vec, builder.vector(src, time, neg_dst)])
    if not rows:
        raise ValueError("could not build any edge training pairs")
    return np.asarray(rows, dtype=np.float32), skipped


def pair_metrics(model, x_raw, mean, std, fuse_rule, gamma, batch_size):
    x = normalize_features(x_raw, mean, std).astype(np.float32)
    rule_idx = FEATURE_NAMES.index("rule_score")
    rule_diff_all = x_raw[:, 0, rule_idx] - x_raw[:, 1, rule_idx]
    rule_acc = float(np.mean(rule_diff_all > 0))

    fused_correct = 0
    residual_correct = 0
    loss_sum = 0.0
    rows = 0
    model.eval()
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        scores = model(jt.array(x[start:end])).numpy()
        residual_diff = scores[:, 0] - scores[:, 1]
        rule_diff = rule_diff_all[start:end]
        fused_diff = rule_diff * fuse_rule + residual_diff * gamma
        loss_sum += float(np.log1p(np.exp(-np.clip(fused_diff, -50, 50))).sum())
        fused_correct += int((fused_diff > 0).sum())
        residual_correct += int((residual_diff > 0).sum())
        rows += end - start

    return {
        "rule_acc": rule_acc,
        "residual_acc": residual_correct / max(rows, 1),
        "fused_acc": fused_correct / max(rows, 1),
        "loss": loss_sum / max(rows, 1),
    }


def train_one_dataset(args, dataset_dir, dataset_name):
    all_edges = list(read_edges(dataset_dir / "train.csv"))
    history_edges, supervision_edges = split_by_time(all_edges, args.history_ratio)
    rng = random.Random(args.seed)
    positives = choose_positive_edges(supervision_edges, args.sample_edges, rng)

    if len(positives) >= 5:
        cut = max(1, int(len(positives) * (1.0 - args.eval_ratio)))
        train_pos = positives[:cut]
        eval_pos = positives[cut:]
    else:
        train_pos = positives
        eval_pos = positives

    builder = CandidateFeatureBuilder(dataset_name)
    builder.fit(history_edges)
    sampler = EdgeNegativeSampler(
        history_edges,
        seed=args.seed,
        recent_limit=args.recent_limit,
        transition_limit=args.transition_limit,
        popular_limit=args.popular_limit,
    )

    train_raw, train_skipped = build_pair_array(builder, sampler, train_pos, args.negatives)
    eval_raw, eval_skipped = build_pair_array(builder, sampler, eval_pos, args.negatives)
    mean = train_raw.reshape(-1, train_raw.shape[-1]).mean(axis=0).astype(np.float32)
    std = train_raw.reshape(-1, train_raw.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    x_train = normalize_features(train_raw, mean, std).astype(np.float32)
    rule_idx = FEATURE_NAMES.index("rule_score")

    model = MLPRanker(x_train.shape[-1], args.hidden_dim)
    optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    order = list(range(len(x_train)))
    best_score = -1.0
    best_path = Path(args.model_dir) / f"{dataset_name}_edge_ranker.pkl"
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"{dataset_name}: history={len(history_edges)} supervision={len(supervision_edges)} "
        f"train_pos={len(train_pos)} eval_pos={len(eval_pos)} "
        f"train_pairs={len(train_raw)} eval_pairs={len(eval_raw)} "
        f"skipped={train_skipped + eval_skipped}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(order)
        loss_sum = 0.0
        steps = 0
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start:start + args.batch_size]
            bx = jt.array(x_train[batch_idx])
            br = jt.array(train_raw[batch_idx, :, rule_idx])
            scores = model(bx)
            diff = (br[:, 0] - br[:, 1]) * args.fuse_rule + (scores[:, 0] - scores[:, 1]) * args.gamma
            logits = jt.stack([diff * 0.0, diff], dim=1)
            targets = jt.array(np.ones(len(batch_idx), dtype=np.int32))
            loss = nn.cross_entropy_loss(logits, targets)
            optimizer.step(loss)
            loss_sum += float(loss.numpy())
            steps += 1

        metrics = pair_metrics(model, eval_raw, mean, std, args.fuse_rule, args.gamma, args.batch_size)
        print(
            f"{dataset_name} epoch={epoch} loss={loss_sum / max(steps, 1):.6f} "
            f"rule_acc={metrics['rule_acc']:.6f} residual_acc={metrics['residual_acc']:.6f} "
            f"fused_acc={metrics['fused_acc']:.6f} eval_loss={metrics['loss']:.6f}"
        )
        if metrics["fused_acc"] > best_score + 1e-12:
            best_score = metrics["fused_acc"]
            save_model(best_path, model, {
                "dataset_name": dataset_name,
                "feature_dim": int(x_train.shape[-1]),
                "hidden_dim": int(args.hidden_dim),
                "feature_names": FEATURE_NAMES,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "fuse_rule": float(args.fuse_rule),
                "gamma": float(args.gamma),
                "training_mode": "edge_residual",
                "zero_row_context": True,
                "history_ratio": float(args.history_ratio),
                "negatives": int(args.negatives),
                "sample_edges": int(args.sample_edges),
                "use_edge_mlp": bool(metrics["fused_acc"] > metrics["rule_acc"]),
            })
    print(f"{dataset_name}: saved {best_path} best_fused_acc={best_score:.6f}")


def find_dataset_names(data_dir, dataset_arg):
    if dataset_arg != "all":
        return [name.strip() for name in dataset_arg.split(",") if name.strip()]
    return [p.name for p in find_dataset_dirs(data_dir)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--model-dir", default="models_edge")
    parser.add_argument("--dataset", default="all", help="all or comma-separated dataset names")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--fuse-rule", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--negatives", type=int, default=8)
    parser.add_argument("--history-ratio", type=float, default=0.8)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--sample-edges", type=int, default=0)
    parser.add_argument("--recent-limit", type=int, default=100)
    parser.add_argument("--transition-limit", type=int, default=200)
    parser.add_argument("--popular-limit", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    if args.cuda:
        jt.flags.use_cuda = 1

    data_dir = Path(args.data_dir)
    names = find_dataset_names(data_dir, args.dataset)
    if not names:
        raise ValueError(f"no datasets found in {data_dir}")
    seeds = [int(x) for x in args.seed_list.split(",") if x.strip()]
    if not seeds:
        seeds = [args.seed]

    base_model_dir = Path(args.model_dir)
    dataset_dirs = {p.name: p for p in find_dataset_dirs(data_dir)}
    for seed in seeds:
        args.seed = seed
        args.model_dir = str(base_model_dir / f"seed_{seed}") if len(seeds) > 1 else str(base_model_dir)
        random.seed(seed)
        np.random.seed(seed)
        jt.set_global_seed(seed)
        for name in names:
            train_one_dataset(args, dataset_dirs[name], name)


if __name__ == "__main__":
    main()
