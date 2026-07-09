import argparse
import json
import math
import os
import os.path as osp
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JT_SYNC", "0")

root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, root)

import jittor as jt
from jittor import nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from jittor_geometric.data import TemporalData
from jittor_geometric.dataloader.temporal_dataloader import get_neighbor_sampler
from jittor_geometric.nn.models.craft import CRAFT


jt.flags.use_cuda = 1


POP_BUCKETS = [
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101-500", 101, 500),
    (">500", 501, 10**12),
]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def bucket_name(count):
    count = int(count)
    if count <= 0:
        return "0"
    for name, lo, hi in POP_BUCKETS:
        if lo <= count <= hi:
            return name
    return ">500"


def normalize_counter(counter, labels):
    vals = np.asarray([float(counter.get(k, 0.0)) for k in labels], dtype=np.float64)
    if vals.sum() <= 0:
        vals[:] = 1.0 / max(len(vals), 1)
    else:
        vals /= vals.sum()
    return {k: float(v) for k, v in zip(labels, vals)}


def row_zscore(x):
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x - mean) / std


def softmax_rows(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    denom = probs.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return probs / denom


def tie_aware_mrr(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    rr = []
    for row, label in zip(scores, labels):
        pos = row[int(label)]
        greater = int((row > pos).sum())
        equal = int((row == pos).sum())
        rank = greater + (equal + 1.0) / 2.0
        rr.append(1.0 / rank)
    return float(np.mean(rr)) if rr else 0.0


def load_dataset(data_dir, dataset):
    ds_dir = Path(data_dir) / dataset
    train_df = pd.read_csv(ds_dir / "train.csv")
    test_df = pd.read_csv(ds_dir / "test.csv")
    train_df = train_df.sort_values("time").reset_index(drop=True)
    return train_df, test_df


def split_train_valid(train_df):
    if "split" in train_df.columns:
        split_values = train_df["split"].astype(str)
        train_part = train_df[split_values == "0"].sort_values("time").reset_index(drop=True)
        valid_part = train_df[split_values == "1"].sort_values("time").reset_index(drop=True)
        if len(train_part) and len(valid_part):
            return train_part, valid_part, "split"
    num_val = max(1, int(len(train_df) * 0.15))
    return (
        train_df.iloc[:-num_val].reset_index(drop=True),
        train_df.iloc[-num_val:].reset_index(drop=True),
        "tail15",
    )


def make_temporal_data(df):
    src = df["src"].to_numpy(np.int32)
    dst = df["dst"].to_numpy(np.int32)
    t = df["time"].to_numpy(np.int32)
    edge_ids = np.arange(len(df), dtype=np.int32) + 1
    return TemporalData(src=jt.Var(src), dst=jt.Var(dst), t=jt.Var(t), edge_ids=jt.Var(edge_ids))


def rows_from_df(df):
    return [
        (int(t), int(s), int(d))
        for s, d, t in df[["src", "dst", "time"]].itertuples(index=False, name=None)
    ]


class HistoryState:
    def __init__(self, recent_window=200000):
        self.recent_window = int(recent_window)
        self.dst_counts = Counter()
        self.recent_counts = Counter()
        self.recent_queue = deque()
        self.src_history = defaultdict(set)
        self.src_last = {}
        self.transition = defaultdict(Counter)
        self.known_dsts = set()

    def copy_from_rows(self, rows):
        self.update(rows)
        return self

    def update(self, rows):
        for _t, src, dst in rows:
            prev = self.src_last.get(src)
            if prev is not None:
                self.transition[prev][dst] += 1
            self.src_history[src].add(dst)
            self.src_last[src] = dst
            self.dst_counts[dst] += 1
            self.known_dsts.add(dst)
            self.recent_queue.append(dst)
            self.recent_counts[dst] += 1
            while len(self.recent_queue) > self.recent_window:
                old = self.recent_queue.popleft()
                self.recent_counts[old] -= 1
                if self.recent_counts[old] <= 0:
                    del self.recent_counts[old]

    def build_pools(self):
        labels = [name for name, _lo, _hi in POP_BUCKETS]
        pop = {name: [] for name in labels}
        recent = {name: [] for name in labels}
        for dst, count in self.dst_counts.items():
            pop[bucket_name(count)].append(int(dst))
        for dst, count in self.recent_counts.items():
            if dst in self.known_dsts:
                recent[bucket_name(count)].append(int(dst))
        return {
            "known": list(map(int, self.known_dsts)),
            "pop": pop,
            "recent": recent,
        }

    def sample_from_pool(self, pool, exclude, rng, tries=60):
        if not pool:
            return None
        n = len(pool)
        for _ in range(tries):
            cand = int(pool[int(rng.integers(0, n))])
            if cand not in exclude:
                return cand
        return None

    def sample_bucketed(self, pools, probs, exclude, rng):
        labels = list(probs.keys())
        weights = np.asarray([probs[k] for k in labels], dtype=np.float64)
        if weights.sum() <= 0:
            weights[:] = 1.0 / max(len(weights), 1)
        else:
            weights = weights / weights.sum()
        for _ in range(20):
            label = labels[int(rng.choice(len(labels), p=weights))]
            cand = self.sample_from_pool(pools.get(label, []), exclude, rng)
            if cand is not None:
                return cand
        return None

    def add_unique(self, cands, cand, exclude, target_size):
        if cand is None or cand in exclude or cand in cands:
            return False
        cands.append(int(cand))
        exclude.add(int(cand))
        return len(cands) >= target_size

    def build_candidates(
        self,
        src,
        pos_dst,
        pools,
        profile,
        rng,
        num_candidates=100,
        src_history_neg_quota=0,
        exclude_src_history=True,
    ):
        if len(self.known_dsts) < num_candidates:
            return None, None
        target = int(num_candidates)
        pos_dst = int(pos_dst)
        src = int(src)
        src_seen = set(self.src_history.get(src, set()))
        exclude = {pos_dst}
        cands = [pos_dst]

        if src_history_neg_quota > 0 and src_seen:
            src_pool = [d for d in src_seen if d != pos_dst]
            rng.shuffle(src_pool)
            for cand in src_pool[: int(src_history_neg_quota)]:
                if self.add_unique(cands, cand, exclude, target):
                    break

        if exclude_src_history:
            exclude.update(d for d in src_seen if d not in cands)

        transition_quota = int(profile.get("transition_neg_quota", 0))
        last_dst = self.src_last.get(src)
        if transition_quota > 0 and last_dst is not None:
            trans_pool = [d for d, _c in self.transition.get(last_dst, Counter()).most_common(200)]
            rng.shuffle(trans_pool)
            for cand in trans_pool[:transition_quota]:
                if self.add_unique(cands, cand, exclude, target):
                    break

        pop_probs = profile.get("pop_bucket_probs", {})
        recent_probs = profile.get("recent_bucket_probs", {})
        remaining = target - len(cands)
        recent_quota = int(round(remaining * float(profile.get("recent_negative_fraction", 0.5))))
        pop_quota = remaining - recent_quota

        for _ in range(max(pop_quota, 0)):
            cand = self.sample_bucketed(pools["pop"], pop_probs, exclude, rng)
            if self.add_unique(cands, cand, exclude, target):
                break
        while len(cands) < target and recent_quota > 0:
            recent_quota -= 1
            cand = self.sample_bucketed(pools["recent"], recent_probs, exclude, rng)
            if self.add_unique(cands, cand, exclude, target):
                break
        while len(cands) < target:
            cand = self.sample_from_pool(pools["known"], exclude, rng)
            if cand is None:
                return None, None
            self.add_unique(cands, cand, exclude, target)

        rng.shuffle(cands)
        label_idx = int(cands.index(pos_dst))
        return np.asarray(cands, dtype=np.int32), label_idx

    def rule_scores(self, src_np, cand_np):
        scores_pop = np.zeros(cand_np.shape, dtype=np.float32)
        scores_recent = np.zeros(cand_np.shape, dtype=np.float32)
        scores_trans = np.zeros(cand_np.shape, dtype=np.float32)
        for i, src in enumerate(src_np):
            last = self.src_last.get(int(src))
            trans = self.transition.get(last, {}) if last is not None else {}
            for j, dst in enumerate(cand_np[i]):
                d = int(dst)
                scores_pop[i, j] = math.log1p(self.dst_counts.get(d, 0))
                scores_recent[i, j] = math.log1p(self.recent_counts.get(d, 0))
                scores_trans[i, j] = math.log1p(trans.get(d, 0))
        scores_rule = row_zscore(scores_pop) + row_zscore(scores_recent) + row_zscore(scores_trans)
        return {
            "pop": scores_pop,
            "recent": scores_recent,
            "transition": scores_trans,
            "rule": scores_rule,
        }


def compute_profile(train_rows, valid_rows, recent_window=200000):
    state = HistoryState(recent_window=recent_window).copy_from_rows(train_rows)
    labels = [name for name, _lo, _hi in POP_BUCKETS]
    pop_counter = Counter()
    recent_counter = Counter()
    seen = 0
    repeated_pair = 0
    transition_hit = 0
    pop_counts = []
    recent_counts = []
    valid_count = 0
    for _t, src, dst in valid_rows:
        valid_count += 1
        if dst in state.known_dsts:
            seen += 1
        if dst in state.src_history.get(src, set()):
            repeated_pair += 1
        last = state.src_last.get(src)
        if last is not None and dst in state.transition.get(last, {}):
            transition_hit += 1
        pc = int(state.dst_counts.get(dst, 0))
        rc = int(state.recent_counts.get(dst, 0))
        pop_counts.append(pc)
        recent_counts.append(rc)
        pop_counter[bucket_name(pc)] += 1
        recent_counter[bucket_name(rc)] += 1

    pop_counts_np = np.asarray(pop_counts, dtype=np.int32) if pop_counts else np.zeros(0, dtype=np.int32)
    recent_counts_np = np.asarray(recent_counts, dtype=np.int32) if recent_counts else np.zeros(0, dtype=np.int32)
    transition_ratio = float(transition_hit / max(valid_count, 1))
    repeated_ratio = float(repeated_pair / max(valid_count, 1))
    recent_positive_ratio = float((recent_counts_np > 0).mean()) if len(recent_counts_np) else 0.0
    transition_neg_quota = min(12, max(0, int(round(99 * transition_ratio * 0.5))))
    return {
        "train_rows": int(len(train_rows)),
        "valid_rows": int(len(valid_rows)),
        "dst_seen_ratio": float(seen / max(valid_count, 1)),
        "cold_dst_ratio": float(1.0 - seen / max(valid_count, 1)),
        "src_history_hit_ratio": repeated_ratio,
        "new_pair_ratio": float(1.0 - repeated_ratio),
        "transition_hit_ratio": transition_ratio,
        "transition_neg_quota": int(transition_neg_quota),
        "recent_negative_fraction": 0.5,
        "pop_bucket_probs": normalize_counter(pop_counter, labels),
        "recent_bucket_probs": normalize_counter(recent_counter, labels),
        "pop_count_percentiles": percentiles(pop_counts_np),
        "recent_count_percentiles": percentiles(recent_counts_np),
    }


def percentiles(arr):
    if len(arr) == 0:
        return {}
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(arr, qs)
    return {str(q): float(v) for q, v in zip(qs, vals)}


@dataclass
class CandidateSet:
    src: np.ndarray
    time: np.ndarray
    candidates: np.ndarray
    labels: np.ndarray

    def save(self, path):
        ensure_dir(Path(path).parent)
        np.savez(
            path,
            src=self.src.astype(np.int32),
            time=self.time.astype(np.int32),
            candidates=self.candidates.astype(np.int32),
            labels=self.labels.astype(np.int32),
        )

    @staticmethod
    def load(path):
        data = np.load(path)
        return CandidateSet(
            data["src"].astype(np.int32),
            data["time"].astype(np.int32),
            data["candidates"].astype(np.int32),
            data["labels"].astype(np.int32),
        )


def select_rows_for_limit(rows, max_events):
    if max_events <= 0 or max_events >= len(rows):
        return rows
    idx = np.linspace(0, len(rows) - 1, int(max_events), dtype=np.int64)
    return [rows[int(i)] for i in idx]


def generate_rolling_candidates(
    rows,
    profile,
    args,
    cache_path,
    require_src_history=True,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not args.rebuild_cache:
        return CandidateSet.load(cache_path), {"cache": str(cache_path), "loaded": True}

    rng = np.random.default_rng(int(args.candidate_seed))
    state = HistoryState(recent_window=args.recent_window)
    src_list, time_list, cand_list, label_list = [], [], [], []
    skipped_no_pool = 0
    skipped_no_history = 0
    blocks = 0
    rows = sorted(rows)
    max_events = int(args.max_train_events)
    for start in tqdm(range(0, len(rows), args.block_size), desc="build train candidates", ncols=120):
        block = rows[start : start + args.block_size]
        blocks += 1
        pools = state.build_pools()
        for t, src, dst in block:
            if max_events > 0 and len(label_list) >= max_events:
                break
            if require_src_history and not state.src_history.get(src):
                skipped_no_history += 1
                continue
            cands, label = state.build_candidates(
                src,
                dst,
                pools,
                profile,
                rng,
                num_candidates=args.num_candidates,
                src_history_neg_quota=args.src_history_neg_quota,
                exclude_src_history=True,
            )
            if cands is None:
                skipped_no_pool += 1
                continue
            src_list.append(src)
            time_list.append(t)
            cand_list.append(cands)
            label_list.append(label)
        state.update(block)
        if max_events > 0 and len(label_list) >= max_events:
            break

    result = CandidateSet(
        np.asarray(src_list, dtype=np.int32),
        np.asarray(time_list, dtype=np.int32),
        np.asarray(cand_list, dtype=np.int32),
        np.asarray(label_list, dtype=np.int32),
    )
    result.save(cache_path)
    meta = {
        "cache": str(cache_path),
        "loaded": False,
        "rows_input": int(len(rows)),
        "events": int(len(result.labels)),
        "blocks": int(blocks),
        "skipped_no_history": int(skipped_no_history),
        "skipped_no_pool": int(skipped_no_pool),
    }
    return result, meta


def generate_static_candidates(rows, history_rows, profile, args, cache_path, max_events):
    cache_path = Path(cache_path)
    if cache_path.exists() and not args.rebuild_cache:
        return CandidateSet.load(cache_path), {"cache": str(cache_path), "loaded": True}

    rng = np.random.default_rng(int(args.candidate_seed) + 17)
    state = HistoryState(recent_window=args.recent_window).copy_from_rows(history_rows)
    pools = state.build_pools()
    selected = select_rows_for_limit(sorted(rows), int(max_events))
    src_list, time_list, cand_list, label_list = [], [], [], []
    skipped = 0
    for t, src, dst in tqdm(selected, desc="build valid candidates", ncols=120):
        cands, label = state.build_candidates(
            src,
            dst,
            pools,
            profile,
            rng,
            num_candidates=args.num_candidates,
            src_history_neg_quota=args.src_history_neg_quota,
            exclude_src_history=True,
        )
        if cands is None:
            skipped += 1
            continue
        src_list.append(src)
        time_list.append(t)
        cand_list.append(cands)
        label_list.append(label)
    result = CandidateSet(
        np.asarray(src_list, dtype=np.int32),
        np.asarray(time_list, dtype=np.int32),
        np.asarray(cand_list, dtype=np.int32),
        np.asarray(label_list, dtype=np.int32),
    )
    result.save(cache_path)
    meta = {
        "cache": str(cache_path),
        "loaded": False,
        "rows_input": int(len(selected)),
        "events": int(len(result.labels)),
        "skipped": int(skipped),
    }
    return result, meta


def iter_batches(candidate_set, batch_size, shuffle, seed):
    n = len(candidate_set.labels)
    idx = np.arange(n, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        part = idx[start : start + batch_size]
        yield (
            candidate_set.src[part],
            candidate_set.time[part],
            candidate_set.candidates[part],
            candidate_set.labels[part],
        )


def score_candidates(model, src_np, time_np, cand_np, sampler, num_neighbors):
    src_np = np.asarray(src_np, dtype=np.int32)
    time_np = np.asarray(time_np, dtype=np.int32)
    cand_np = np.asarray(cand_np, dtype=np.int32)
    src_neighb_seq, _, src_neighb_interact_times = sampler.get_historical_neighbors_left(
        node_ids=src_np, node_interact_times=time_np, num_neighbors=int(num_neighbors)
    )
    neighbor_num = (src_neighb_seq != 0).sum(axis=1).astype(np.int32)
    flat_cand = cand_np.reshape(-1)
    flat_time = np.broadcast_to(time_np[:, None], cand_np.shape).reshape(-1).astype(np.int32)
    dst_last_neighbor, _, dst_last_update_time = sampler.get_historical_neighbors_left(
        node_ids=flat_cand, node_interact_times=flat_time, num_neighbors=1
    )
    dst_last_update_time = np.asarray(dst_last_update_time).reshape(len(cand_np), -1)
    dst_last_update_time[np.asarray(dst_last_neighbor).reshape(len(cand_np), -1) == 0] = -100000

    src_adj = jt.Var(src_neighb_seq.astype(np.int32)) - model.dst_min_idx + 1
    dst_adj = jt.Var(cand_np.astype(np.int32)) - model.dst_min_idx + 1
    src_adj = jt.where(src_adj < 0, jt.zeros_like(src_adj), src_adj)
    dst_adj = jt.where(dst_adj < 0, jt.zeros_like(dst_adj), dst_adj)
    logits = model.forward(
        src_adj,
        jt.Var(neighbor_num),
        jt.Var(np.asarray(src_neighb_interact_times, dtype=np.int32)),
        jt.Var(time_np),
        test_dst=dst_adj,
        dst_last_update_times=jt.Var(dst_last_update_time.astype(np.int32)),
    )
    return logits.squeeze(-1), neighbor_num


def apply_unseen_dst_fallback(logits, cand_np, known_dsts, margin=2.0):
    logits = np.asarray(logits, dtype=np.float32).copy()
    known_mask = np.vectorize(lambda x: int(x) in known_dsts)(cand_np)
    for i in range(logits.shape[0]):
        if known_mask[i].any():
            floor = float(logits[i][known_mask[i]].min()) - float(margin)
            logits[i][~known_mask[i]] = floor
        else:
            logits[i, :] = 0.0
    return logits


def evaluate_baselines(candidate_set, state):
    scores = state.rule_scores(candidate_set.src, candidate_set.candidates)
    return {name + "_mrr": tie_aware_mrr(val, candidate_set.labels) for name, val in scores.items()}


def evaluate_model(model, candidate_set, sampler, state, args):
    model.eval()
    all_logits = []
    for src, t, cand, _label in tqdm(
        iter_batches(candidate_set, args.eval_batch_size, False, args.seed),
        total=(len(candidate_set.labels) + args.eval_batch_size - 1) // args.eval_batch_size,
        desc="eval model",
        ncols=120,
    ):
        logits, _neighbor_num = score_candidates(model, src, t, cand, sampler, args.num_neighbors)
        all_logits.append(logits.numpy())
    logits_np = np.vstack(all_logits)
    rule = state.rule_scores(candidate_set.src, candidate_set.candidates)["rule"]
    fusion = row_zscore(logits_np) + row_zscore(rule)
    return {
        "craft_mrr": tie_aware_mrr(logits_np, candidate_set.labels),
        "fusion_mrr": tie_aware_mrr(fusion, candidate_set.labels),
    }


def train_model(model, optimizer, train_set, val_set, train_sampler, val_sampler, val_state, args, save_prefix):
    best_metric = -1.0
    best_epoch = 0
    best_score_mode = "craft"
    patience = 0
    history = []
    baseline_metrics = evaluate_baselines(val_set, val_state) if val_set is not None else {}
    if baseline_metrics:
        print("Validation baselines:", baseline_metrics, flush=True)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        total_batches = (len(train_set.labels) + args.batch_size - 1) // args.batch_size
        pbar = tqdm(
            iter_batches(train_set, args.batch_size, True, args.seed + epoch),
            total=total_batches,
            ncols=120,
            desc=f"epoch {epoch}",
        )
        for src, t, cand, labels in pbar:
            logits, neighbor_num = score_candidates(model, src, t, cand, train_sampler, args.num_neighbors)
            loss = nn.cross_entropy_loss(logits, jt.Var(labels.astype(np.int32)))
            optimizer.zero_grad()
            optimizer.step(loss)
            if args.sync_each_batch:
                jt.sync_all()
            val = float(loss.item())
            losses.append(val)
            pbar.set_description(f"epoch {epoch} loss={val:.5f}")

        mean_loss = float(np.mean(losses)) if losses else 0.0
        row = {"epoch": epoch, "loss": mean_loss}
        print(f"epoch={epoch} loss={mean_loss:.6f}", flush=True)
        stop_now = False
        if val_set is not None:
            metrics = evaluate_model(model, val_set, val_sampler, val_state, args)
            row.update(metrics)
            if float(metrics["fusion_mrr"]) > float(metrics["craft_mrr"]):
                select_metric = float(metrics["fusion_mrr"])
                select_score_mode = "fusion"
            else:
                select_metric = float(metrics["craft_mrr"])
                select_score_mode = "craft"
            row["select_mrr"] = select_metric
            row["select_score_mode"] = select_score_mode
            print(
                f"epoch={epoch} metrics={metrics} select_mrr={select_metric:.6f} "
                f"select_score_mode={select_score_mode}",
                flush=True,
            )
            if select_metric > best_metric:
                best_metric = select_metric
                best_epoch = epoch
                best_score_mode = select_score_mode
                patience = 0
                jt.save(model.state_dict(), str(save_prefix) + "_best.pkl")
                print(f"new best select_mrr={best_metric:.6f}", flush=True)
            else:
                patience += 1
                print(f"no improvement patience={patience}/{args.early_stop}", flush=True)
            if patience >= args.early_stop:
                stop_now = True
        else:
            jt.save(model.state_dict(), str(save_prefix) + "_latest.pkl")
        history.append(row)
        if stop_now:
            break

    if val_set is not None and Path(str(save_prefix) + "_best.pkl").exists():
        model.load_state_dict(jt.load(str(save_prefix) + "_best.pkl"))
    return {
        "history": history,
        "baseline_metrics": baseline_metrics,
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "best_score_mode": best_score_mode,
    }


def build_model(node_size, dst_min, src_min, args):
    model = CRAFT(
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        hidden_dropout_prob=0.1,
        attn_dropout_prob=0.1,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        n_nodes=int(node_size),
        max_seq_length=int(args.num_neighbors),
        loss_type="CE",
        use_pos=True,
        input_cat_time_intervals=False,
        output_cat_time_intervals=True,
        output_cat_repeat_times=True,
        num_output_layer=1,
        emb_dropout_prob=0.1,
        skip_connection=True,
    )
    model.set_min_idx(int(src_min), int(dst_min))
    return model


def predict_test(model, test_df, sampler, final_state, args):
    model.eval()
    cand_cols = [c for c in test_df.columns if c.startswith("c")]
    test_src = test_df["src"].to_numpy(np.int32)
    test_time = test_df["time"].to_numpy(np.int32)
    test_candidates = test_df[cand_cols].to_numpy(np.int32)
    all_scores = []
    total = (len(test_src) + args.eval_batch_size - 1) // args.eval_batch_size
    for start in tqdm(range(0, len(test_src), args.eval_batch_size), total=total, desc="predict test", ncols=120):
        end = min(start + args.eval_batch_size, len(test_src))
        src = test_src[start:end]
        t = test_time[start:end]
        cand = test_candidates[start:end]
        logits, _neighbor_num = score_candidates(model, src, t, cand, sampler, args.num_neighbors)
        logits_np = apply_unseen_dst_fallback(logits.numpy(), cand, final_state.known_dsts, args.unseen_margin)
        if args.score_mode == "fusion":
            rule = final_state.rule_scores(src, cand)["rule"]
            logits_np = row_zscore(logits_np) + row_zscore(rule)
        all_scores.append(softmax_rows(logits_np))
    return np.vstack(all_scores)


def write_scores(path, scores):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in scores:
            f.write(",".join(f"{float(p):.8f}" for p in row) + "\n")


def validate_score_file(path, expected_rows, expected_cols):
    rows = 0
    sum_min, sum_max = 10.0, -10.0
    min_val, max_val = 10.0, -10.0
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows += 1
            vals = [float(x) for x in line.strip().split(",") if x]
            if len(vals) != expected_cols:
                bad += 1
                continue
            s = sum(vals)
            sum_min = min(sum_min, s)
            sum_max = max(sum_max, s)
            min_val = min(min_val, min(vals))
            max_val = max(max_val, max(vals))
    return {
        "rows": rows,
        "expected_rows": int(expected_rows),
        "bad": int(bad),
        "sum_min": float(sum_min),
        "sum_max": float(sum_max),
        "min_val": float(min_val),
        "max_val": float(max_val),
    }


def save_json(path, payload):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset2")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--save_dir", default="./saved_models")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--artifact_dir", default="./artifacts")
    parser.add_argument("--report_dir", default="./reports")
    parser.add_argument("--run_name", default="listwise")
    parser.add_argument("--mode", choices=["profile", "validate", "refit", "predict"], default="validate")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--early_stop", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_neighbors", type=int, default=64)
    parser.add_argument("--num_candidates", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=50000)
    parser.add_argument("--recent_window", type=int, default=200000)
    parser.add_argument("--max_train_events", type=int, default=0)
    parser.add_argument("--max_val_events", type=int, default=50000)
    parser.add_argument("--src_history_neg_quota", type=int, default=0)
    parser.add_argument("--candidate_seed", type=int, default=20260709)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--sync_each_batch", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--score_mode", choices=["craft", "fusion"], default="craft")
    parser.add_argument("--unseen_margin", type=float, default=2.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    ensure_dir(args.save_dir)
    ensure_dir(args.output_dir)
    ensure_dir(args.artifact_dir)
    ensure_dir(args.report_dir)

    train_df, test_df = load_dataset(args.data_dir, args.dataset)
    split0_df, split1_df, split_method = split_train_valid(train_df)
    split0_rows = rows_from_df(split0_df)
    split1_rows = rows_from_df(split1_df)
    all_rows = rows_from_df(train_df)
    profile = compute_profile(split0_rows, split1_rows, recent_window=args.recent_window)
    profile["split_method"] = split_method
    profile_path = Path(args.report_dir) / f"{args.run_name}_profile.json"
    save_json(profile_path, profile)
    print("profile:", json.dumps(profile, sort_keys=True), flush=True)

    if args.mode == "profile":
        return

    cand_cols = [c for c in test_df.columns if c.startswith("c")]
    max_node = max(int(train_df["src"].max()), int(train_df["dst"].max()), int(test_df[cand_cols].max().max()))
    dst_min = min(int(train_df["dst"].min()), int(test_df[cand_cols].min().min()))
    src_min = int(train_df["src"].min())
    node_size = max_node + 1
    print(f"node_size={node_size} src_min={src_min} dst_min={dst_min}", flush=True)

    if args.mode == "validate":
        train_rows_for_mode = split0_rows
        history_rows = split0_rows
        train_df_for_sampler = split0_df
        val_rows = split1_rows
        train_cache = Path(args.artifact_dir) / f"{args.run_name}_train_candidates.npz"
        val_cache = Path(args.artifact_dir) / f"{args.run_name}_valid_candidates.npz"
        train_set, train_meta = generate_rolling_candidates(train_rows_for_mode, profile, args, train_cache)
        val_set, val_meta = generate_static_candidates(val_rows, history_rows, profile, args, val_cache, args.max_val_events)
        train_sampler = get_neighbor_sampler(make_temporal_data(train_df_for_sampler), "recent", seed=1)
        val_sampler = train_sampler
        val_state = HistoryState(recent_window=args.recent_window).copy_from_rows(history_rows)
    elif args.mode == "refit":
        train_rows_for_mode = all_rows
        train_df_for_sampler = train_df
        train_cache = Path(args.artifact_dir) / f"{args.run_name}_train_candidates.npz"
        train_set, train_meta = generate_rolling_candidates(train_rows_for_mode, profile, args, train_cache)
        val_set, val_meta, val_sampler, val_state = None, {}, None, None
        train_sampler = get_neighbor_sampler(make_temporal_data(train_df_for_sampler), "recent", seed=1)
    elif args.mode == "predict":
        train_set, train_meta = None, {}
        train_sampler = get_neighbor_sampler(make_temporal_data(train_df), "recent", seed=1)
        model = build_model(node_size, dst_min, src_min, args)
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode predict")
        model.load_state_dict(jt.load(args.checkpoint))
        final_state = HistoryState(recent_window=args.recent_window).copy_from_rows(all_rows)
        scores = predict_test(model, test_df, train_sampler, final_state, args)
        output_file = Path(args.output_dir) / args.dataset / f"{args.dataset}_result.csv"
        write_scores(output_file, scores)
        report = validate_score_file(output_file, len(test_df), len(cand_cols))
        report["output_file"] = str(output_file)
        save_json(Path(args.report_dir) / f"{args.run_name}_predict_report.json", report)
        print("predict_report:", report, flush=True)
        return
    else:
        raise ValueError(args.mode)

    if train_set is None or len(train_set.labels) == 0:
        raise RuntimeError("no training candidates generated")
    if args.mode == "validate" and (val_set is None or len(val_set.labels) == 0):
        raise RuntimeError("no validation candidates generated")

    model = build_model(node_size, dst_min, src_min, args)
    optimizer = nn.Adam(list(model.parameters()), lr=float(args.lr))
    save_prefix = Path(args.save_dir) / f"{args.run_name}_{args.dataset}_CRAFT"
    train_report = train_model(model, optimizer, train_set, val_set, train_sampler, val_sampler, val_state, args, save_prefix)

    report = {
        "args": vars(args),
        "profile": profile,
        "train_candidates": train_meta,
        "valid_candidates": val_meta,
        "train": train_report,
    }
    if args.mode == "refit":
        latest_path = str(save_prefix) + "_latest.pkl"
        if Path(latest_path).exists():
            model.load_state_dict(jt.load(latest_path))
        if args.predict:
            final_state = HistoryState(recent_window=args.recent_window).copy_from_rows(all_rows)
            scores = predict_test(model, test_df, train_sampler, final_state, args)
            output_file = Path(args.output_dir) / args.dataset / f"{args.dataset}_result.csv"
            write_scores(output_file, scores)
            report["prediction"] = validate_score_file(output_file, len(test_df), len(cand_cols))
            report["prediction"]["output_file"] = str(output_file)
    save_json(Path(args.report_dir) / f"{args.run_name}_report.json", report)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
