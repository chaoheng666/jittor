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
from src.jt_ranker import (
    CandidateFeatureBuilder,
    CraftRerankModel,
    FEATURE_NAMES,
    QUERY_FEATURE_NAMES,
    normalize_features,
    normalize_query_features,
    save_model,
)


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
        recent_limit=120,
        transition_limit=300,
        popular_limit=5000,
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

        for _, rows in by_src.items():
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

    @staticmethod
    def _allowed(positive, dst, seen):
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
            if self._allowed(positive, dst, seen):
                seen.add(dst)
                negatives.append(dst)

        tries = 0
        max_tries = max(1000, count * 500)
        while len(negatives) < count and tries < max_tries:
            tries += 1
            dst = self.rng.choice(self.dst_unique)
            if self._allowed(positive, dst, seen):
                seen.add(dst)
                negatives.append(dst)
        return negatives


def choose_positive_edges(edges, sample_edges, rng):
    if sample_edges and sample_edges < len(edges):
        rows = rng.sample(edges, sample_edges)
        rows.sort(key=lambda x: x[2])
        return rows
    return list(edges)


def build_listwise_queries(builder, sampler, positives, negatives):
    queries = []
    seen_flags = []
    skipped = 0
    for src, dst, time in positives:
        negs = sampler.sample(src, dst, negatives)
        if len(negs) < negatives:
            skipped += 1
            continue
        queries.append((src, time, [dst] + negs))
        seen_flags.append(1 if builder.features.pair_count[(src, dst)] else 0)
    if not queries:
        raise ValueError("could not build any listwise training queries")
    targets = np.zeros(len(queries), dtype=np.int32)
    return queries, targets, np.asarray(seen_flags, dtype=np.int32), skipped


def normalize_batch_arrays(arrays, mean, std, query_mean, query_std):
    return {
        "x": normalize_features(arrays["x"], mean, std).astype(np.float32),
        "candidate_idx": arrays["candidate_idx"].astype(np.int32),
        "history_idx": arrays["history_idx"].astype(np.int32),
        "history_delta": arrays["history_delta"].astype(np.float32),
        "history_mask": arrays["history_mask"].astype(np.float32),
        "query": normalize_query_features(arrays["query"], query_mean, query_std).astype(np.float32),
    }


def model_scores(model, arrays, batch_size):
    model.eval()
    out = []
    for start in range(0, len(arrays["x"]), batch_size):
        end = min(start + batch_size, len(arrays["x"]))
        scores = model(
            jt.array(arrays["x"][start:end]),
            jt.array(arrays["candidate_idx"][start:end]),
            jt.array(arrays["history_idx"][start:end]),
            jt.array(arrays["history_delta"][start:end]),
            jt.array(arrays["history_mask"][start:end]),
            jt.array(arrays["query"][start:end]),
        ).numpy()
        out.append(scores)
    return np.vstack(out)


def ranking_metrics(scores, targets, seen_flags):
    ranks = []
    for row, target in zip(scores, targets):
        target_score = row[target]
        rank = 1 + int(np.sum(row > target_score))
        ranks.append(rank)
    ranks = np.asarray(ranks, dtype=np.float64)
    rr = 1.0 / ranks
    seen_flags = np.asarray(seen_flags, dtype=np.int32)
    seen_mask = seen_flags == 1
    unseen_mask = ~seen_mask
    return {
        "validation_mrr": float(rr.mean()) if len(rr) else 0.0,
        "validation_hit1": float(np.mean(ranks == 1)) if len(ranks) else 0.0,
        "validation_seen_mrr": float(rr[seen_mask].mean()) if np.any(seen_mask) else 0.0,
        "validation_unseen_mrr": float(rr[unseen_mask].mean()) if np.any(unseen_mask) else 0.0,
    }


def evaluate_model(model, eval_arrays, eval_raw_arrays, targets, seen_flags, batch_size):
    scores = model_scores(model, eval_arrays, batch_size)
    metrics = ranking_metrics(scores, targets, seen_flags)
    rule_idx = FEATURE_NAMES.index("rule_score")
    rule_scores = eval_raw_arrays["x"][:, :, rule_idx]
    rule_metrics = ranking_metrics(rule_scores, targets, seen_flags)
    metrics.update({
        "rule_mrr": rule_metrics["validation_mrr"],
        "rule_hit1": rule_metrics["validation_hit1"],
        "rule_seen_mrr": rule_metrics["validation_seen_mrr"],
        "rule_unseen_mrr": rule_metrics["validation_unseen_mrr"],
    })
    return metrics, scores, rule_scores


def save_eval_cache(path, scores, rule_scores, targets, seen_flags, queries):
    candidates = np.asarray([row[2] for row in queries], dtype=np.int64)
    keys = np.asarray([[src, time, row[0]] for src, time, row in queries], dtype=np.int64)
    np.savez_compressed(
        path,
        scores=scores.astype(np.float32),
        rule_scores=rule_scores.astype(np.float32),
        targets=targets.astype(np.int32),
        seen_flags=seen_flags.astype(np.int32),
        keys=keys,
        candidates=candidates,
    )


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

    builder = CandidateFeatureBuilder(dataset_name, history_len=args.history_len)
    builder.fit(history_edges)
    sampler = EdgeNegativeSampler(
        history_edges,
        seed=args.seed,
        recent_limit=max(args.history_len, args.recent_limit),
        transition_limit=args.transition_limit,
        popular_limit=args.popular_limit,
    )

    train_queries, train_targets, train_seen, train_skipped = build_listwise_queries(
        builder, sampler, train_pos, args.negatives
    )
    eval_queries, eval_targets, eval_seen, eval_skipped = build_listwise_queries(
        builder, sampler, eval_pos, args.negatives
    )
    train_raw = builder.arrays_for_queries(train_queries)
    eval_raw = builder.arrays_for_queries(eval_queries)

    flat_train = train_raw["x"].reshape(-1, train_raw["x"].shape[-1])
    mean = flat_train.mean(axis=0).astype(np.float32)
    std = flat_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    query_mean = train_raw["query"].mean(axis=0).astype(np.float32)
    query_std = train_raw["query"].std(axis=0)
    query_std = np.where(query_std < 1e-6, 1.0, query_std).astype(np.float32)

    train_arrays = normalize_batch_arrays(train_raw, mean, std, query_mean, query_std)
    eval_arrays = normalize_batch_arrays(eval_raw, mean, std, query_mean, query_std)

    model = CraftRerankModel(
        feature_dim=len(FEATURE_NAMES),
        query_dim=len(QUERY_FEATURE_NAMES),
        node_count=builder.node_count,
        history_len=args.history_len,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
    )
    optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    order = list(range(len(train_arrays["x"])))
    best_mrr = -1.0
    best_scores = None
    best_rule_scores = None
    best_metrics = None
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / f"{dataset_name}_edge_ranker.pkl"
    eval_cache_path = model_dir / f"{dataset_name}_edge_ranker_eval.npz"

    print(
        f"{dataset_name}: history={len(history_edges)} supervision={len(supervision_edges)} "
        f"train_queries={len(train_queries)} eval_queries={len(eval_queries)} "
        f"candidates={args.negatives + 1} nodes={builder.node_count} "
        f"repeat_fraction={builder.features.repeat_edge_fraction:.6f} "
        f"bipartite={int(builder.features.is_bipartite_like)} "
        f"skipped={train_skipped + eval_skipped}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(order)
        loss_sum = 0.0
        steps = 0
        candidate_size = train_arrays["x"].shape[1]
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start:start + args.batch_size]
            scores = model(
                jt.array(train_arrays["x"][batch_idx]),
                jt.array(train_arrays["candidate_idx"][batch_idx]),
                jt.array(train_arrays["history_idx"][batch_idx]),
                jt.array(train_arrays["history_delta"][batch_idx]),
                jt.array(train_arrays["history_mask"][batch_idx]),
                jt.array(train_arrays["query"][batch_idx]),
            )
            targets = jt.array(train_targets[batch_idx])
            list_loss = nn.cross_entropy_loss(scores, targets)
            diff = scores[:, 0:1] - scores[:, 1:]
            flat_diff = diff.reshape(-1)
            pair_logits = jt.stack([flat_diff * 0.0, flat_diff], dim=1)
            pair_targets = jt.array(np.ones(len(batch_idx) * (candidate_size - 1), dtype=np.int32))
            pair_loss = nn.cross_entropy_loss(pair_logits, pair_targets)
            loss = list_loss + pair_loss * args.pairwise_weight
            optimizer.step(loss)
            loss_sum += float(loss.numpy())
            steps += 1

        metrics, scores, rule_scores = evaluate_model(
            model, eval_arrays, eval_raw, eval_targets, eval_seen, args.batch_size
        )
        print(
            f"{dataset_name} epoch={epoch} loss={loss_sum / max(steps, 1):.6f} "
            f"mrr={metrics['validation_mrr']:.6f} hit1={metrics['validation_hit1']:.6f} "
            f"seen_mrr={metrics['validation_seen_mrr']:.6f} "
            f"unseen_mrr={metrics['validation_unseen_mrr']:.6f} "
            f"rule_mrr={metrics['rule_mrr']:.6f}"
        )
        if metrics["validation_mrr"] > best_mrr + 1e-12:
            best_mrr = metrics["validation_mrr"]
            best_scores = scores
            best_rule_scores = rule_scores
            best_metrics = metrics
            save_model(best_path, model, {
                "dataset_name": dataset_name,
                "feature_dim": len(FEATURE_NAMES),
                "query_dim": len(QUERY_FEATURE_NAMES),
                "node_count": int(builder.node_count),
                "node_values": builder.node_values,
                "hidden_dim": int(args.hidden_dim),
                "embed_dim": int(args.embed_dim),
                "history_len": int(args.history_len),
                "feature_names": FEATURE_NAMES,
                "query_feature_names": QUERY_FEATURE_NAMES,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "query_mean": query_mean.tolist(),
                "query_std": query_std.tolist(),
                "training_mode": "craft_rerank_listwise",
                "history_ratio": float(args.history_ratio),
                "negatives": int(args.negatives),
                "sample_edges": int(args.sample_edges),
                "train_queries": int(len(train_queries)),
                "eval_queries": int(len(eval_queries)),
                "eval_cache_path": str(eval_cache_path),
                "use_craft_model": bool(metrics["validation_mrr"] > metrics["rule_mrr"]),
                **metrics,
            })
            save_eval_cache(eval_cache_path, best_scores, best_rule_scores, eval_targets, eval_seen, eval_queries)

    print(f"{dataset_name}: saved {best_path} best_validation_mrr={best_mrr:.6f}")


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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--history-len", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pairwise-weight", type=float, default=0.15)
    parser.add_argument("--negatives", type=int, default=32)
    parser.add_argument("--history-ratio", type=float, default=0.8)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--sample-edges", type=int, default=0)
    parser.add_argument("--recent-limit", type=int, default=120)
    parser.add_argument("--transition-limit", type=int, default=300)
    parser.add_argument("--popular-limit", type=int, default=5000)
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
