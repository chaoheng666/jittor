import json
import math
import pickle
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from src.data_loader import iter_test_rows, iter_train_edges
from src.metrics import row_zscore, softmax


try:
    import jittor as jt
    from jittor import nn
except Exception:  # pragma: no cover - exercised only when Jittor is missing.
    jt = None
    nn = None


def jittor_available():
    return jt is not None and nn is not None


def read_edges_with_split(dataset_dir):
    rows = []
    with open(Path(dataset_dir) / "train.csv", newline="", encoding="utf-8") as f:
        import csv

        reader = csv.DictReader(f)
        has_split = "split" in (reader.fieldnames or [])
        for row in reader:
            split = row.get("split", "")
            if not has_split:
                split = ""
            rows.append((int(row["src"]), int(row["dst"]), int(row["time"]), split))
    rows.sort(key=lambda x: x[2])
    return rows


def split_dataset2_edges(dataset_dir, final_train=False):
    rows = read_edges_with_split(dataset_dir)
    if not rows:
        raise ValueError(f"{dataset_dir}: no training rows")
    if final_train:
        train = [(s, d, t) for s, d, t, _ in rows]
        return train, []
    has_split = any(split != "" for *_, split in rows)
    if has_split:
        train = [(s, d, t) for s, d, t, split in rows if str(split) == "0"]
        valid = [(s, d, t) for s, d, t, split in rows if str(split) != "0"]
        if train and valid:
            return train, valid
    cut = int(len(rows) * 0.8)
    cut = min(max(cut, 1), len(rows) - 1)
    train = [(s, d, t) for s, d, t, _ in rows[:cut]]
    valid = [(s, d, t) for s, d, t, _ in rows[cut:]]
    return train, valid


def iter_test_candidate_dsts(dataset_dir):
    for _src, _time, candidates in iter_test_rows(Path(dataset_dir) / "test.csv"):
        for dst in candidates:
            yield int(dst)


def build_id_maps(edges, extra_dsts=None):
    src_values = sorted({src for src, _, _ in edges})
    dst_set = {dst for _, dst, _ in edges}
    if extra_dsts is not None:
        dst_set.update(int(dst) for dst in extra_dsts)
    dst_values = sorted(dst_set)
    src_to_id = {value: idx + 1 for idx, value in enumerate(src_values)}
    dst_to_id = {value: idx + 1 for idx, value in enumerate(dst_values)}
    return src_to_id, dst_to_id


def _time_feature(time_value, last_time, hist_len, time_min, time_scale, gap_scale, hist_scale):
    global_pos = (float(time_value) - float(time_min)) / max(float(time_scale), 1.0)
    if last_time is None:
        gap = 0.0
    else:
        gap = math.log1p(max(float(time_value) - float(last_time), 0.0)) / max(float(gap_scale), 1.0)
    hist_value = math.log1p(float(hist_len)) / max(float(hist_scale), 1.0)
    return [global_pos, gap, hist_value]


def build_samples(
    history_edges,
    supervision_edges,
    src_to_id,
    dst_to_id,
    seq_len,
    update_with_supervision=True,
    max_events=0,
):
    history_by_src = defaultdict(lambda: deque(maxlen=seq_len))
    last_time_by_src = {}
    count_by_src = Counter()
    # Validation features are normalized from the observable history only.  When
    # history is empty during self-supervised training, fall back to the training
    # supervision window.
    time_source = history_edges if history_edges else supervision_edges
    all_times = [time for _, _, time in time_source]
    time_min = min(all_times) if all_times else 0
    time_max = max(all_times) if all_times else 1
    gaps = []
    for src, dst, time in sorted(history_edges, key=lambda x: x[2]):
        dst_id = dst_to_id.get(dst, 0)
        if dst_id:
            history_by_src[src].append((dst_id, time))
        if src in last_time_by_src:
            gaps.append(max(time - last_time_by_src[src], 0))
        last_time_by_src[src] = time
        count_by_src[src] += 1
    gap_scale = math.log1p(float(np.percentile(gaps, 90))) if gaps else 1.0
    hist_scale = math.log1p(max(count_by_src.values(), default=1))

    src_rows = []
    hist_rows = []
    hist_gap_rows = []
    time_rows = []
    labels = []
    pair_seen = set((src, dst) for src, dst, _ in history_edges)
    is_new_pair = []
    skipped = 0
    skipped_cold_dst = 0
    skipped_cold_src = 0
    skipped_no_history_src = 0
    skipped_other = 0

    supervision_rows = sorted(supervision_edges, key=lambda x: x[2])
    selected = _selected_training_indices(len(supervision_rows), max_events)

    for idx, (src, dst, time) in enumerate(supervision_rows):
        src_id = src_to_id.get(src, 0)
        dst_id = dst_to_id.get(dst, 0)
        hist = list(history_by_src.get(src, ()))
        if idx not in selected:
            if update_with_supervision and dst_id:
                history_by_src[src].append((dst_id, time))
                last_time_by_src[src] = time
                count_by_src[src] += 1
                pair_seen.add((src, dst))
            continue
        if src_id == 0 or dst_id == 0 or not hist:
            skipped += 1
            reason_known = False
            if src_id == 0:
                skipped_cold_src += 1
                reason_known = True
            elif not hist:
                skipped_no_history_src += 1
                reason_known = True
            if dst_id == 0:
                skipped_cold_dst += 1
                reason_known = True
            if not reason_known:
                skipped_other += 1
            if update_with_supervision and dst_id:
                history_by_src[src].append((dst_id, time))
                last_time_by_src[src] = time
                count_by_src[src] += 1
                pair_seen.add((src, dst))
            continue
        hist_vec = np.zeros(seq_len, dtype=np.int32)
        hist_gap_vec = np.zeros(seq_len, dtype=np.float32)
        hist_slice = hist[-seq_len:]
        offset = seq_len - len(hist_slice)
        for pos, (hist_dst_id, hist_time) in enumerate(hist_slice, start=offset):
            hist_vec[pos] = hist_dst_id
            hist_gap_vec[pos] = math.log1p(max(float(time) - float(hist_time), 0.0)) / max(gap_scale, 1.0)
        src_rows.append(src_id)
        hist_rows.append(hist_vec)
        hist_gap_rows.append(hist_gap_vec)
        time_rows.append(_time_feature(
            time,
            last_time_by_src.get(src),
            len(hist),
            time_min,
            max(time_max - time_min, 1),
            gap_scale,
            hist_scale,
        ))
        labels.append(dst_id - 1)
        is_new_pair.append(0 if (src, dst) in pair_seen else 1)
        if update_with_supervision:
            history_by_src[src].append((dst_id, time))
            last_time_by_src[src] = time
            count_by_src[src] += 1
            pair_seen.add((src, dst))

    return {
        "src": np.asarray(src_rows, dtype=np.int32),
        "hist": np.asarray(hist_rows, dtype=np.int32),
        "hist_gap": np.asarray(hist_gap_rows, dtype=np.float32),
        "time": np.asarray(time_rows, dtype=np.float32),
        "label": np.asarray(labels, dtype=np.int32),
        "is_new_pair": np.asarray(is_new_pair, dtype=np.int8),
        "skipped": int(skipped),
        "skipped_cold_dst": int(skipped_cold_dst),
        "skipped_cold_src": int(skipped_cold_src),
        "skipped_no_history_src": int(skipped_no_history_src),
        "skipped_other": int(skipped_other),
    }


def source_histories_for_prediction(edges, dst_to_id, seq_len):
    histories = defaultdict(lambda: deque(maxlen=seq_len))
    last_time = {}
    count_by_src = Counter()
    times = []
    gaps = []
    for src, dst, time in sorted(edges, key=lambda x: x[2]):
        times.append(time)
        dst_id = dst_to_id.get(dst, 0)
        if dst_id:
            histories[src].append((dst_id, time))
        if src in last_time:
            gaps.append(max(time - last_time[src], 0))
        last_time[src] = time
        count_by_src[src] += 1
    return {
        "histories": histories,
        "last_time": last_time,
        "count_by_src": count_by_src,
        "time_min": min(times) if times else 0,
        "time_scale": max((max(times) - min(times)) if times else 1, 1),
        "gap_scale": math.log1p(float(np.percentile(gaps, 90))) if gaps else 1.0,
        "hist_scale": math.log1p(max(count_by_src.values(), default=1)),
    }


def _history_arrays_for_src(state, src, time, seq_len):
    hist = list(state["histories"].get(src, ()))
    hist_vec = np.zeros(seq_len, dtype=np.int32)
    gap_vec = np.zeros(seq_len, dtype=np.float32)
    hist_slice = hist[-seq_len:]
    offset = seq_len - len(hist_slice)
    for pos, (dst_id, hist_time) in enumerate(hist_slice, start=offset):
        hist_vec[pos] = int(dst_id)
        gap_vec[pos] = math.log1p(max(float(time) - float(hist_time), 0.0)) / max(float(state["gap_scale"]), 1.0)
    return hist_vec, gap_vec, len(hist)


if nn is not None:
    class TemporalRecommender(nn.Module):
        def __init__(
            self,
            num_src,
            num_dst,
            emb_dim=96,
            hidden_dim=192,
            dropout=0.1,
            seq_len=80,
            dst_features=None,
        ):
            super().__init__()
            self.seq_len = int(seq_len)
            self.num_dst = int(num_dst)
            if dst_features is None:
                dst_features = np.zeros((int(num_dst) + 1, 5), dtype=np.float32)
            dst_features = np.asarray(dst_features, dtype=np.float32)
            if dst_features.ndim != 2 or dst_features.shape[0] == 0:
                dst_features = np.zeros((int(num_dst) + 1, 5), dtype=np.float32)
            if dst_features.shape[0] != int(num_dst) + 1:
                fixed = np.zeros((int(num_dst) + 1, dst_features.shape[1]), dtype=np.float32)
                rows = min(fixed.shape[0], dst_features.shape[0])
                fixed[:rows] = dst_features[:rows]
                dst_features = fixed
            self.dst_feature_dim = int(dst_features.shape[1])
            self.pos_weights = jt.array(np.linspace(0.25, 1.0, self.seq_len, dtype=np.float32)).reshape((1, -1))
            self.src_emb = nn.Embedding(num_src + 1, emb_dim)
            self.dst_emb = nn.Embedding(num_dst + 1, emb_dim)
            self.dst_bias = nn.Embedding(num_dst + 1, 1)
            self.dst_feat_emb = nn.Embedding(num_dst + 1, self.dst_feature_dim)
            try:
                self.dst_feat_emb.weight.assign(jt.array(dst_features))
            except Exception:
                self.dst_feat_emb.weight = jt.array(dst_features)
            self.dst_feat_proj = nn.Sequential(
                nn.Linear(self.dst_feature_dim, emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, emb_dim),
            )
            self.gap_proj = nn.Sequential(
                nn.Linear(1, emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, emb_dim),
            )
            self.time_proj = nn.Sequential(
                nn.Linear(3, emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, emb_dim),
            )
            self.state_proj = nn.Sequential(
                nn.Linear(emb_dim * 4, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, emb_dim),
            )
            self.pair_mlp = nn.Sequential(
                nn.Linear(emb_dim * 4, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def dst_repr(self, dst_ids):
            feature_rows = self.dst_feat_emb(dst_ids)
            return self.dst_emb(dst_ids) + 0.15 * self.dst_feat_proj(feature_rows)

        def all_dst_repr(self):
            ids = jt.arange(1, self.num_dst + 1).int32()
            return self.dst_repr(ids)

        def encode_state(self, src_ids, hist_ids, hist_gaps, time_feats):
            src_vec = self.src_emb(src_ids)
            hist_mask = (hist_ids > 0).float32()
            gap_vec = self.gap_proj(hist_gaps.unsqueeze(-1))
            hist_emb = (self.dst_repr(hist_ids) + gap_vec) * hist_mask.unsqueeze(-1)
            pos_weights = self.pos_weights
            weighted_mask = hist_mask * pos_weights
            denom = jt.maximum(weighted_mask.sum(dim=1, keepdims=True), jt.ones((hist_ids.shape[0], 1)))
            hist_vec = (hist_emb * pos_weights.unsqueeze(-1)).sum(dim=1) / denom
            recent_vec = hist_vec
            time_vec = self.time_proj(time_feats)
            return self.state_proj(jt.concat([src_vec, hist_vec, recent_vec, time_vec], dim=1))

        def execute(self, src_ids, hist_ids, hist_gaps, time_feats, cand_ids=None):
            state = self.encode_state(src_ids, hist_ids, hist_gaps, time_feats)
            if cand_ids is None:
                dst_weight = self.all_dst_repr()
                all_ids = jt.arange(1, self.num_dst + 1).int32()
                bias = self.dst_bias(all_ids).reshape((1, -1))
                return jt.matmul(state, dst_weight.transpose(1, 0)) + bias
            if len(cand_ids.shape) == 1:
                cand_emb = self.dst_repr(cand_ids)
                bias = self.dst_bias(cand_ids).reshape((1, -1))
                return jt.matmul(state, cand_emb.transpose(1, 0)) + bias
            cand_emb = self.dst_repr(cand_ids)
            bias = self.dst_bias(cand_ids).squeeze(-1)
            cand_mask = (cand_ids > 0).float32()
            state_expanded = state.unsqueeze(1)
            dot_scores = (state_expanded * cand_emb).sum(dim=2) + bias
            pair_features = jt.concat(
                [
                    state_expanded + cand_emb * 0.0,
                    cand_emb,
                    state_expanded * cand_emb,
                    jt.abs(state_expanded - cand_emb),
                ],
                dim=2,
            )
            scores = dot_scores + 0.2 * self.pair_mlp(pair_features).squeeze(-1)
            return scores * cand_mask - (1.0 - cand_mask) * 1e6
else:
    class TemporalRecommender:
        def __init__(self, *args, **kwargs):
            raise ImportError("jittor is required for TemporalRecommender")


def _batch_indices(size, batch_size, rng, shuffle=True):
    idx = np.arange(size)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, size, batch_size):
        yield idx[start:start + batch_size]


def _build_hard_negative_tables(train_edges, src_to_id, dst_to_id, per_src_limit=200, popular_limit=4096):
    recent_by_src = defaultdict(lambda: deque(maxlen=int(per_src_limit)))
    dst_counts = Counter()
    for src, dst, _time in sorted(train_edges, key=lambda x: x[2]):
        src_id = src_to_id.get(src, 0)
        dst_id = dst_to_id.get(dst, 0)
        if src_id and dst_id:
            recent_by_src[src_id].append(dst_id)
            dst_counts[dst_id] += 1
    hard_by_src = {}
    for src_id, values in recent_by_src.items():
        unique_recent = []
        seen = set()
        for dst_id in reversed(values):
            if dst_id not in seen:
                unique_recent.append(dst_id)
                seen.add(dst_id)
        hard_by_src[src_id] = np.asarray(unique_recent, dtype=np.int32)
    popular = np.asarray([dst_id for dst_id, _ in dst_counts.most_common(int(popular_limit))], dtype=np.int32)
    return hard_by_src, popular


def build_dst_feature_table(edges, dst_to_id):
    feature_dim = 8
    features = np.zeros((len(dst_to_id) + 1, feature_dim), dtype=np.float32)
    max_dst_value = max(dst_to_id.keys(), default=1)
    for dst, dst_id in dst_to_id.items():
        raw_norm = math.log1p(float(dst)) / max(math.log1p(float(max_dst_value)), 1.0)
        bucket = float(dst % 997) / 996.0
        features[int(dst_id), 6] = raw_norm
        features[int(dst_id), 7] = bucket
    if not edges:
        return features

    times = np.asarray([time for _, _, time in edges], dtype=np.float64)
    time_min = float(times.min())
    time_max = float(times.max())
    time_scale = max(time_max - time_min, 1.0)
    recent_cut = float(np.percentile(times, 80))
    older_cut = float(np.percentile(times, 60))
    total_counts = Counter()
    recent_counts = Counter()
    older_counts = Counter()
    last_time = {}
    recent_sources = defaultdict(set)
    for src, dst, time in edges:
        if dst not in dst_to_id:
            continue
        total_counts[dst] += 1
        last_time[dst] = max(last_time.get(dst, time), time)
        if time >= recent_cut:
            recent_counts[dst] += 1
            recent_sources[dst].add(src)
        elif time >= older_cut:
            older_counts[dst] += 1

    max_log_total = math.log1p(max(total_counts.values(), default=1))
    max_log_recent = math.log1p(max(recent_counts.values(), default=1))
    max_log_sources = math.log1p(max((len(v) for v in recent_sources.values()), default=1))
    for dst, dst_id in dst_to_id.items():
        total = float(total_counts.get(dst, 0))
        recent = float(recent_counts.get(dst, 0))
        older = float(older_counts.get(dst, 0))
        trend = math.log1p(recent) - math.log1p(older)
        trend = max(min(trend, 5.0), -5.0) / 5.0
        recency = 1.0 / (1.0 + max(time_max - float(last_time.get(dst, time_min)), 0.0) / time_scale)
        features[int(dst_id)] = np.asarray(
            [
                math.log1p(total) / max(max_log_total, 1.0),
                math.log1p(recent) / max(max_log_recent, 1.0),
                trend,
                recency,
                math.log1p(float(len(recent_sources.get(dst, ())))) / max(max_log_sources, 1.0),
                1.0,
                features[int(dst_id), 6],
                features[int(dst_id), 7],
            ],
            dtype=np.float32,
        )
    return features


class FastDataset2RuleScorer:
    def __init__(self, recent_limit=200):
        self.recent_limit = int(recent_limit)
        self.src_recent = defaultdict(lambda: deque(maxlen=self.recent_limit))
        self.seen_pairs = set()
        self.dst_count = Counter()
        self.dst_recent_count = Counter()
        self.dst_older_count = Counter()
        self.dst_last_time = {}
        self.time_scale = 1.0
        self.max_log_count = 1.0
        self.max_log_recent = 1.0

    def fit(self, edges):
        rows = sorted(edges, key=lambda x: x[2])
        if not rows:
            return self
        times = np.asarray([time for _, _, time in rows], dtype=np.float64)
        time_min = float(times.min())
        time_max = float(times.max())
        self.time_scale = max(time_max - time_min, 1.0)
        recent_cut = float(np.percentile(times, 80))
        older_cut = float(np.percentile(times, 60))
        for src, dst, time in rows:
            self.src_recent[src].append(dst)
            self.seen_pairs.add((src, dst))
            self.dst_count[dst] += 1
            self.dst_last_time[dst] = max(self.dst_last_time.get(dst, time), time)
            if time >= recent_cut:
                self.dst_recent_count[dst] += 1
            elif time >= older_cut:
                self.dst_older_count[dst] += 1
        self.max_log_count = max(math.log1p(max(self.dst_count.values(), default=1)), 1.0)
        self.max_log_recent = max(math.log1p(max(self.dst_recent_count.values(), default=1)), 1.0)
        return self

    def score_many(self, src, time, candidates):
        recent_rank = {}
        for rank, dst in enumerate(reversed(self.src_recent.get(src, ())), start=1):
            recent_rank.setdefault(dst, rank)
        out = []
        for dst in candidates:
            count = self.dst_count.get(dst, 0)
            if count <= 0:
                out.append(-8.0)
                continue
            recent = self.dst_recent_count.get(dst, 0)
            older = self.dst_older_count.get(dst, 0)
            last_time = self.dst_last_time.get(dst, time)
            pop = math.log1p(float(count)) / self.max_log_count
            recent_pop = math.log1p(float(recent)) / self.max_log_recent
            trend = max(min(math.log1p(float(recent)) - math.log1p(float(older)), 5.0), -5.0) / 5.0
            recency = 1.0 / (1.0 + max(float(time) - float(last_time), 0.0) / self.time_scale)
            rank = recent_rank.get(dst)
            src_recent_score = 1.0 / float(rank) if rank else 0.0
            repeated_penalty = -0.35 if (src, dst) in self.seen_pairs else 0.0
            score = (
                1.10 * pop
                + 1.35 * recent_pop
                + 0.55 * trend
                + 0.65 * recency
                + 0.12 * src_recent_score
                + repeated_penalty
            )
            out.append(float(score))
        return out


FEATURE_NAMES = [
    "rule_score",
    "pop",
    "recent_pop",
    "trend",
    "recency",
    "src_recent_score",
    "pair_seen",
    "pair_count",
    "src_degree",
    "dst_degree",
    "src_hist_len",
    "dst_seen",
    "src_seen",
    "dst_time_gap",
    "rank_rule",
    "rank_pop",
    "rank_recent",
    "dst_unknown",
]

LISTWISE_EXTRA_FEATURE_NAMES = [
    "rule_z",
    "pop_z",
    "recent_pop_z",
    "trend_z",
    "recency_z",
    "src_recent_z",
    "dst_gap_z",
    "rule_minmax",
    "pop_minmax",
    "recent_minmax",
    "candidate_log_id",
    "candidate_id_bucket",
    "row_unknown_frac",
    "row_known_frac",
    "row_rule_max",
    "row_rule_spread",
    "time_pos",
]

LISTWISE_FEATURE_NAMES = FEATURE_NAMES + LISTWISE_EXTRA_FEATURE_NAMES


class Dataset2FeatureState:
    def __init__(self, scale_edges, recent_limit=200):
        self.recent_limit = int(recent_limit)
        self.src_recent = defaultdict(lambda: deque(maxlen=self.recent_limit))
        self.src_count = Counter()
        self.dst_count = Counter()
        self.dst_recent_count = Counter()
        self.dst_older_count = Counter()
        self.pair_count = Counter()
        self.dst_last_time = {}
        rows = sorted(scale_edges, key=lambda x: x[2])
        if rows:
            times = np.asarray([time for _, _, time in rows], dtype=np.float64)
            self.time_min = float(times.min())
            self.time_max = float(times.max())
            self.time_scale = max(self.time_max - self.time_min, 1.0)
            self.recent_cut = float(np.percentile(times, 80))
            self.older_cut = float(np.percentile(times, 60))
            src_counts = Counter(src for src, _, _ in rows)
            dst_counts = Counter(dst for _, dst, _ in rows)
            self.max_log_src = max(math.log1p(max(src_counts.values(), default=1)), 1.0)
            self.max_log_dst = max(math.log1p(max(dst_counts.values(), default=1)), 1.0)
            self.max_log_pair = max(
                math.log1p(max(Counter((src, dst) for src, dst, _ in rows).values(), default=1)),
                1.0,
            )
        else:
            self.time_min = 0.0
            self.time_max = 1.0
            self.time_scale = 1.0
            self.recent_cut = 1.0
            self.older_cut = 0.0
            self.max_log_src = 1.0
            self.max_log_dst = 1.0
            self.max_log_pair = 1.0

    def fit(self, edges):
        for src, dst, time in sorted(edges, key=lambda x: x[2]):
            self.update(src, dst, time)
        return self

    def update(self, src, dst, time):
        self.src_recent[src].append(dst)
        self.src_count[src] += 1
        self.dst_count[dst] += 1
        self.pair_count[(src, dst)] += 1
        self.dst_last_time[dst] = max(self.dst_last_time.get(dst, time), time)
        if time >= self.recent_cut:
            self.dst_recent_count[dst] += 1
        elif time >= self.older_cut:
            self.dst_older_count[dst] += 1

    def popular_dsts(self, limit=4096):
        return [dst for dst, _ in self.dst_count.most_common(int(limit))]

    def score_features(self, src, time, candidates):
        recent_rank = {}
        for rank, dst in enumerate(reversed(self.src_recent.get(src, ())), start=1):
            recent_rank.setdefault(dst, rank)
        src_degree = math.log1p(float(self.src_count.get(src, 0))) / self.max_log_src
        src_hist_len = math.log1p(float(len(self.src_recent.get(src, ())))) / self.max_log_src
        src_seen = 1.0 if self.src_count.get(src, 0) > 0 else 0.0
        rows = []
        rule_scores = []
        pop_scores = []
        recent_scores = []
        for dst in candidates:
            count = float(self.dst_count.get(dst, 0))
            recent = float(self.dst_recent_count.get(dst, 0))
            older = float(self.dst_older_count.get(dst, 0))
            pop = math.log1p(count) / self.max_log_dst
            recent_pop = math.log1p(recent) / self.max_log_dst
            trend = max(min(math.log1p(recent) - math.log1p(older), 5.0), -5.0) / 5.0
            last_time = self.dst_last_time.get(dst)
            if last_time is None:
                recency = 0.0
                dst_time_gap = 1.0
            else:
                gap = max(float(time) - float(last_time), 0.0)
                recency = 1.0 / (1.0 + gap / self.time_scale)
                dst_time_gap = math.log1p(gap) / math.log1p(self.time_scale)
            rank = recent_rank.get(dst)
            src_recent_score = 1.0 / float(rank) if rank else 0.0
            pair = float(self.pair_count.get((src, dst), 0))
            pair_seen = 1.0 if pair > 0.0 else 0.0
            pair_count = math.log1p(pair) / self.max_log_pair
            dst_degree = math.log1p(count) / self.max_log_dst
            dst_seen = 1.0 if count > 0.0 else 0.0
            dst_unknown = 1.0 - dst_seen
            repeated_penalty = -0.35 if pair_seen else 0.0
            rule_score = (
                1.10 * pop
                + 1.35 * recent_pop
                + 0.55 * trend
                + 0.65 * recency
                + 0.12 * src_recent_score
                + repeated_penalty
                - 8.0 * dst_unknown
            )
            rule_scores.append(rule_score)
            pop_scores.append(pop)
            recent_scores.append(src_recent_score + recency)
            rows.append([
                rule_score,
                pop,
                recent_pop,
                trend,
                recency,
                src_recent_score,
                pair_seen,
                pair_count,
                src_degree,
                dst_degree,
                src_hist_len,
                dst_seen,
                src_seen,
                dst_time_gap,
                0.0,
                0.0,
                0.0,
                dst_unknown,
            ])
        features = np.asarray(rows, dtype=np.float32)
        for col, values in ((14, rule_scores), (15, pop_scores), (16, recent_scores)):
            values = np.asarray(values, dtype=np.float64)
            order = np.argsort(-values)
            ranks = np.empty(len(values), dtype=np.float32)
            ranks[order] = np.arange(1, len(values) + 1, dtype=np.float32)
            features[:, col] = 1.0 / ranks
        return features

    def pair_features(self, src, time, dst):
        recent_rank = {}
        for rank, recent_dst in enumerate(reversed(self.src_recent.get(src, ())), start=1):
            recent_rank.setdefault(recent_dst, rank)
        src_degree = math.log1p(float(self.src_count.get(src, 0))) / self.max_log_src
        src_hist_len = math.log1p(float(len(self.src_recent.get(src, ())))) / self.max_log_src
        src_seen = 1.0 if self.src_count.get(src, 0) > 0 else 0.0
        count = float(self.dst_count.get(dst, 0))
        recent = float(self.dst_recent_count.get(dst, 0))
        older = float(self.dst_older_count.get(dst, 0))
        pop = math.log1p(count) / self.max_log_dst
        recent_pop = math.log1p(recent) / self.max_log_dst
        trend = max(min(math.log1p(recent) - math.log1p(older), 5.0), -5.0) / 5.0
        last_time = self.dst_last_time.get(dst)
        if last_time is None:
            recency = 0.0
            dst_time_gap = 1.0
        else:
            gap = max(float(time) - float(last_time), 0.0)
            recency = 1.0 / (1.0 + gap / self.time_scale)
            dst_time_gap = math.log1p(gap) / math.log1p(self.time_scale)
        rank = recent_rank.get(dst)
        src_recent_score = 1.0 / float(rank) if rank else 0.0
        pair = float(self.pair_count.get((src, dst), 0))
        pair_seen = 1.0 if pair > 0.0 else 0.0
        pair_count = math.log1p(pair) / self.max_log_pair
        dst_degree = math.log1p(count) / self.max_log_dst
        dst_seen = 1.0 if count > 0.0 else 0.0
        dst_unknown = 1.0 - dst_seen
        repeated_penalty = -0.35 if pair_seen else 0.0
        rule_score = (
            1.10 * pop
            + 1.35 * recent_pop
            + 0.55 * trend
            + 0.65 * recency
            + 0.12 * src_recent_score
            + repeated_penalty
            - 8.0 * dst_unknown
        )
        return np.asarray(
            [
                rule_score,
                pop,
                recent_pop,
                trend,
                recency,
                src_recent_score,
                pair_seen,
                pair_count,
                src_degree,
                dst_degree,
                src_hist_len,
                dst_seen,
                src_seen,
                dst_time_gap,
                1.0,
                1.0,
                1.0,
                dst_unknown,
            ],
            dtype=np.float32,
        )


def _row_zscore_1d(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.mean())) / std).astype(np.float32)


def _row_minmax_1d(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    spread = high - low
    if spread < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / spread).astype(np.float32)


def _listwise_score_features(state, src, time, candidates):
    base = state.score_features(src, time, candidates)
    candidates_arr = np.asarray(candidates, dtype=np.float32)
    if candidates_arr.size == 0:
        return np.zeros((0, len(LISTWISE_FEATURE_NAMES)), dtype=np.float32)

    rule = base[:, FEATURE_NAMES.index("rule_score")]
    pop = base[:, FEATURE_NAMES.index("pop")]
    recent_pop = base[:, FEATURE_NAMES.index("recent_pop")]
    trend = base[:, FEATURE_NAMES.index("trend")]
    recency = base[:, FEATURE_NAMES.index("recency")]
    src_recent = base[:, FEATURE_NAMES.index("src_recent_score")]
    dst_gap = base[:, FEATURE_NAMES.index("dst_time_gap")]
    unknown = base[:, FEATURE_NAMES.index("dst_unknown")]

    max_candidate = max(float(candidates_arr.max()), 1.0)
    log_scale = max(math.log1p(max_candidate), 1.0)
    candidate_log_id = np.log1p(np.maximum(candidates_arr, 0.0)) / log_scale
    candidate_bucket = (np.mod(candidates_arr, 997.0) / 996.0).astype(np.float32)
    unknown_frac = float(np.mean(unknown > 0.5))
    known_frac = 1.0 - unknown_frac
    rule_max = float(rule.max()) if rule.size else 0.0
    rule_spread = float(rule.max() - rule.min()) if rule.size else 0.0
    time_pos = (float(time) - float(state.time_min)) / max(float(state.time_scale), 1.0)
    time_pos = max(0.0, min(time_pos, 1.5))

    extras = np.stack(
        [
            _row_zscore_1d(rule),
            _row_zscore_1d(pop),
            _row_zscore_1d(recent_pop),
            _row_zscore_1d(trend),
            _row_zscore_1d(recency),
            _row_zscore_1d(src_recent),
            _row_zscore_1d(dst_gap),
            _row_minmax_1d(rule),
            _row_minmax_1d(pop),
            _row_minmax_1d(recent_pop),
            candidate_log_id.astype(np.float32),
            candidate_bucket.astype(np.float32),
            np.full(len(candidates_arr), unknown_frac, dtype=np.float32),
            np.full(len(candidates_arr), known_frac, dtype=np.float32),
            np.full(len(candidates_arr), rule_max, dtype=np.float32),
            np.full(len(candidates_arr), rule_spread, dtype=np.float32),
            np.full(len(candidates_arr), time_pos, dtype=np.float32),
        ],
        axis=1,
    )
    return np.concatenate([base, extras], axis=1).astype(np.float32)


def _feature_candidates(src, positive_dst, state, known_dsts, popular_dsts, neg_count, rng):
    selected = [positive_dst]
    seen = {positive_dst}
    for dst in reversed(state.src_recent.get(src, ())):
        if dst not in seen:
            selected.append(dst)
            seen.add(dst)
        if len(selected) >= neg_count + 1:
            break
    for dst in popular_dsts:
        if dst not in seen:
            selected.append(dst)
            seen.add(dst)
        if len(selected) >= neg_count + 1:
            break
    known_dsts = list(known_dsts)
    while len(selected) < neg_count + 1 and known_dsts:
        dst = int(known_dsts[int(rng.integers(0, len(known_dsts)))])
        if dst not in seen:
            selected.append(dst)
            seen.add(dst)
    return selected[:neg_count + 1]


def _selected_training_indices(size, max_events):
    size = int(size)
    max_events = int(max_events or 0)
    if max_events <= 0 or max_events >= size:
        return set(range(size))
    return set(int(x) for x in np.linspace(0, size - 1, max_events, dtype=np.int64))


def _build_feature_training_data(history_edges, supervision_edges, neg_count, max_events, seed):
    rng = np.random.default_rng(int(seed))
    scale_edges = history_edges if history_edges else supervision_edges
    state = Dataset2FeatureState(scale_edges, recent_limit=200).fit(history_edges)
    full_state = Dataset2FeatureState(scale_edges, recent_limit=200).fit(scale_edges)
    known_dsts = sorted({dst for _, dst, _ in scale_edges})
    popular_dsts = full_state.popular_dsts(limit=4096)
    selected = _selected_training_indices(len(supervision_edges), max_events)
    xs = []
    ys = []
    events_used = 0
    for idx, (src, dst, time) in enumerate(sorted(supervision_edges, key=lambda x: x[2])):
        if idx in selected:
            candidates = _feature_candidates(src, dst, state, known_dsts, popular_dsts, int(neg_count), rng)
            features = state.score_features(src, time, candidates)
            label = np.zeros(len(candidates), dtype=np.int8)
            label[0] = 1
            xs.append(features)
            ys.append(label)
            events_used += 1
        state.update(src, dst, time)
    if not xs:
        raise RuntimeError("dataset2 feature reranker: no training rows generated")
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(np.int8), events_used


def _fit_feature_classifier(x, y, model_kind, seed):
    if model_kind == "jittor_mlp":
        model = JittorFeatureMLP(input_dim=x.shape[1], seed=int(seed))
        model.fit(x, y)
        return model

    if model_kind == "torch_mlp":
        model = TorchFeatureMLP(input_dim=x.shape[1], seed=int(seed))
        model.fit(x, y)
        return model

    positive_weight = max((len(y) - int(y.sum())) / max(int(y.sum()), 1), 1.0)
    sample_weight = np.where(y > 0, positive_weight, 1.0).astype(np.float32)
    if model_kind == "sgd":
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-5,
                max_iter=8,
                tol=1e-4,
                random_state=int(seed),
                n_jobs=-1,
            ),
        )
        model.fit(x, y, sgdclassifier__sample_weight=sample_weight)
        return model

    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_iter=120,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        min_samples_leaf=80,
        random_state=int(seed),
        verbose=0,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model


class JittorFeatureMLP:
    def __init__(self, input_dim, hidden_dim=96, epochs=8, batch_size=65536, lr=0.001, seed=2026):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.seed = int(seed)
        self.mean = None
        self.std = None
        self.state_dict = None
        self.loss_history = []
        self._runtime_model = None

    def _make_model(self):
        if not jittor_available():
            raise ImportError("Jittor is required for jittor_mlp feature reranker")

        class DenseMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
                    nn.ReLU(),
                    nn.Linear(max(hidden_dim // 2, 8), 1),
                )

            def execute(self, x):
                return self.net(x).reshape((-1,))

        return DenseMLP(self.input_dim, self.hidden_dim)

    def _normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            return (x - self.mean.reshape((-1,))) / self.std.reshape((-1,))
        return (x - self.mean) / self.std

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_runtime_model"] = None
        return state

    def _get_runtime_model(self):
        runtime_model = getattr(self, "_runtime_model", None)
        if runtime_model is not None:
            return runtime_model
        model = self._make_model()
        if hasattr(model, "load_state_dict"):
            model.load_state_dict({key: jt.array(value) for key, value in self.state_dict.items()})
        else:
            model.load_parameters({key: jt.array(value) for key, value in self.state_dict.items()})
        model.eval()
        self._runtime_model = model
        return model

    def fit(self, x, y):
        if not jittor_available():
            raise ImportError("Jittor is required for jittor_mlp feature reranker")
        jt.flags.use_cuda = 1
        random.seed(self.seed)
        np.random.seed(self.seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.mean = x.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = x.std(axis=0, keepdims=True).astype(np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)
        x = self._normalize(x)
        model = self._make_model()
        optimizer = nn.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        positive = max(float(y.sum()), 1.0)
        negative = max(float(len(y) - y.sum()), 1.0)
        pos_weight = negative / positive
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(y))
        for _epoch in range(self.epochs):
            rng.shuffle(indices)
            losses = []
            model.train()
            for start in range(0, len(indices), self.batch_size):
                idx = indices[start:start + self.batch_size]
                xb = jt.array(x[idx])
                yb = jt.array(y[idx])
                logits = model(xb)
                bce = jt.maximum(logits, jt.zeros_like(logits)) - logits * yb + jt.log(1.0 + jt.exp(-jt.abs(logits)))
                weights = yb * (pos_weight - 1.0) + 1.0
                loss = (bce * weights).mean()
                optimizer.step(loss)
                losses.append(float(loss.numpy()))
            self.loss_history.append(float(np.mean(losses)) if losses else 0.0)
        self.state_dict = {
            key: np.asarray(value.numpy(), dtype=np.float32)
            for key, value in model.state_dict().items()
        }
        model.eval()
        self._runtime_model = model
        return self

    def predict_proba(self, x):
        if not jittor_available():
            raise ImportError("Jittor is required for jittor_mlp feature reranker")
        jt.flags.use_cuda = 1
        x = self._normalize(x)
        model = self._get_runtime_model()
        out = []
        for start in range(0, len(x), 262144):
            logits = model(jt.array(x[start:start + 262144]))
            scores = (1.0 / (1.0 + np.exp(-logits.numpy()))).astype(np.float32)
            out.append(scores)
        pos = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
        return np.stack([1.0 - pos, pos], axis=1)


class JittorListwiseFeatureMLP:
    def __init__(
        self,
        input_dim,
        hidden_dim=160,
        epochs=10,
        batch_size=4096,
        lr=0.001,
        margin_weight=0.05,
        seed=2026,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.margin_weight = float(margin_weight)
        self.seed = int(seed)
        self.mean = None
        self.std = None
        self.state_dict = None
        self.loss_history = []
        self._runtime_model = None

    def _make_model(self):
        if not jittor_available():
            raise ImportError("Jittor is required for listwise feature ranker")

        class DenseListwiseMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.08),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
                    nn.ReLU(),
                    nn.Linear(max(hidden_dim // 2, 16), 1),
                )

            def execute(self, x):
                shape = x.shape
                flat = x.reshape((-1, shape[-1]))
                scores = self.net(flat).reshape((-1,))
                if len(shape) == 3:
                    return scores.reshape((shape[0], shape[1]))
                return scores

        return DenseListwiseMLP(self.input_dim, self.hidden_dim)

    def _normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            return (x - self.mean.reshape((-1,))) / self.std.reshape((-1,))
        return (x - self.mean) / self.std

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_runtime_model"] = None
        return state

    def _get_runtime_model(self):
        runtime_model = getattr(self, "_runtime_model", None)
        if runtime_model is not None:
            return runtime_model
        model = self._make_model()
        if hasattr(model, "load_state_dict"):
            model.load_state_dict({key: jt.array(value) for key, value in self.state_dict.items()})
        else:
            model.load_parameters({key: jt.array(value) for key, value in self.state_dict.items()})
        model.eval()
        self._runtime_model = model
        return model

    def fit(self, x, label_positions):
        if not jittor_available():
            raise ImportError("Jittor is required for listwise feature ranker")
        jt.flags.use_cuda = 1
        random.seed(self.seed)
        np.random.seed(self.seed)
        x = np.asarray(x, dtype=np.float32)
        label_positions = np.asarray(label_positions, dtype=np.int32)
        if x.ndim != 3:
            raise ValueError(f"listwise training expects 3D features, got shape={x.shape}")
        self.mean = x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        self.std = x.std(axis=(0, 1), keepdims=True).astype(np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)
        x = self._normalize(x)
        model = self._make_model()
        optimizer = nn.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        rng = np.random.default_rng(self.seed)
        indices = np.arange(x.shape[0])
        for epoch in range(1, self.epochs + 1):
            rng.shuffle(indices)
            losses = []
            model.train()
            for start in range(0, len(indices), self.batch_size):
                idx = indices[start:start + self.batch_size]
                xb = jt.array(x[idx])
                yb_np = label_positions[idx].astype(np.int32)
                yb = jt.array(yb_np)
                logits = model(xb)
                loss = nn.cross_entropy_loss(logits, yb)
                if self.margin_weight > 0.0:
                    one_hot = np.zeros((len(idx), x.shape[1]), dtype=np.float32)
                    one_hot[np.arange(len(idx)), yb_np] = 1.0
                    one_hot_jt = jt.array(one_hot)
                    pos = (logits * one_hot_jt).sum(dim=1)
                    neg_logits = logits - one_hot_jt * 1e6
                    max_neg = neg_logits.max(dim=1)
                    if isinstance(max_neg, tuple):
                        max_neg = max_neg[0]
                    margin = jt.log(1.0 + jt.exp(max_neg - pos)).mean()
                    loss = loss + float(self.margin_weight) * margin
                optimizer.step(loss)
                losses.append(float(loss.numpy()))
            mean_loss = float(np.mean(losses)) if losses else 0.0
            self.loss_history.append(mean_loss)
            print(f"dataset2 listwise: epoch={epoch} loss={mean_loss:.6f}", flush=True)
        self.state_dict = {
            key: np.asarray(value.numpy(), dtype=np.float32)
            for key, value in model.state_dict().items()
        }
        model.eval()
        self._runtime_model = model
        return self

    def predict_scores(self, x):
        if not jittor_available():
            raise ImportError("Jittor is required for listwise feature ranker")
        jt.flags.use_cuda = 1
        x = np.asarray(x, dtype=np.float32)
        x = self._normalize(x)
        model = self._get_runtime_model()
        out = []
        flat = x.reshape((-1, x.shape[-1]))
        for start in range(0, len(flat), 262144):
            logits = model(jt.array(flat[start:start + 262144])).numpy().astype(np.float32)
            out.append(logits)
        scores = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
        if x.ndim == 3:
            return scores.reshape((x.shape[0], x.shape[1]))
        return scores


class TorchFeatureMLP:
    def __init__(self, input_dim, hidden_dim=96, epochs=8, batch_size=65536, lr=0.001, seed=2026):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.seed = int(seed)
        self.state_dict = None
        self.loss_history = []

    def _device(self):
        import torch

        try:
            import torch_npu  # noqa: F401
        except Exception:
            pass
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu:0")
        return torch.device("cpu")

    def _make_model(self):
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim // 2, 1),
        )

    def fit(self, x, y):
        import torch

        torch.manual_seed(self.seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        device = self._device()
        model = self._make_model().to(device)
        positive = max(float(y.sum()), 1.0)
        negative = max(float(len(y) - y.sum()), 1.0)
        pos_weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(y))
        for _epoch in range(self.epochs):
            rng.shuffle(indices)
            losses = []
            model.train()
            for start in range(0, len(indices), self.batch_size):
                idx = indices[start:start + self.batch_size]
                xb = torch.from_numpy(x[idx]).to(device)
                yb = torch.from_numpy(y[idx]).to(device)
                logits = model(xb).squeeze(1)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.loss_history.append(float(np.mean(losses)) if losses else 0.0)
        self.state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        return self

    def predict_proba(self, x):
        import torch

        x = np.asarray(x, dtype=np.float32)
        device = self._device()
        model = self._make_model().to(device)
        model.load_state_dict(self.state_dict)
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), 262144):
                xb = torch.from_numpy(x[start:start + 262144]).to(device)
                score = torch.sigmoid(model(xb).squeeze(1)).detach().cpu().numpy()
                out.append(score.astype(np.float32))
        pos = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
        return np.stack([1.0 - pos, pos], axis=1)


def _feature_model_scores(model, features):
    if hasattr(model, "predict_scores"):
        return model.predict_scores(features).reshape(-1).astype(np.float32)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(np.float32)
        return proba.reshape(-1).astype(np.float32)
    return model.decision_function(features).astype(np.float32)


def _reciprocal_rank_at_zero(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return 1.0 / (1 + int(np.sum(scores > scores[0])))


def _feature_eval_candidate_ids(src, dst, state, known_dsts, popular_dsts, neg_count, rng, hard):
    if hard:
        return _feature_candidates(src, dst, state, known_dsts, popular_dsts, int(neg_count), rng)
    selected = [dst]
    seen = {dst}
    known_dsts = list(known_dsts)
    while len(selected) < neg_count + 1 and known_dsts:
        cand = int(known_dsts[int(rng.integers(0, len(known_dsts)))])
        if cand not in seen:
            selected.append(cand)
            seen.add(cand)
    return selected


def evaluate_feature_reranker(model, history_edges, valid_edges, neg_count, max_events, seed, fusion_weights):
    state = Dataset2FeatureState(history_edges, recent_limit=200).fit(history_edges)
    known_dsts = sorted({dst for _, dst, _ in history_edges})
    popular_dsts = state.popular_dsts(limit=4096)
    selected = _evaluation_indices(len(valid_edges), max_events=max_events)
    out = {
        "pseudo100": {"rule_only": [], "model_only": [], "fusion": {str(w): [] for w in fusion_weights}},
        "hard-pseudo100": {"rule_only": [], "model_only": [], "fusion": {str(w): [] for w in fusion_weights}},
    }
    for eval_name, hard in (("pseudo100", False), ("hard-pseudo100", True)):
        rng = np.random.default_rng(int(seed) + (17 if hard else 0))
        for idx in selected:
            src, dst, time = valid_edges[int(idx)]
            candidates = _feature_eval_candidate_ids(
                src,
                dst,
                state,
                known_dsts,
                popular_dsts,
                int(neg_count),
                rng,
                hard=hard,
            )
            features = state.score_features(src, time, candidates)
            rule_scores = features[:, 0]
            model_scores = _feature_model_scores(model, features)
            out[eval_name]["rule_only"].append(_reciprocal_rank_at_zero(rule_scores))
            out[eval_name]["model_only"].append(_reciprocal_rank_at_zero(model_scores))
            for weight in fusion_weights:
                fused = row_zscore(model_scores.reshape(1, -1))[0] * float(weight) + row_zscore(rule_scores.reshape(1, -1))[0]
                out[eval_name]["fusion"][str(weight)].append(_reciprocal_rank_at_zero(fused))
    metrics = {}
    for eval_name, payload in out.items():
        metrics[eval_name] = {
            "rule_only": _bucket_metrics(payload["rule_only"]),
            "model_only": _bucket_metrics(payload["model_only"]),
            "fusion": {weight: _bucket_metrics(values) for weight, values in payload["fusion"].items()},
        }
    return metrics


def _select_feature_fusion_weight(metrics, weights):
    hard = metrics.get("hard-pseudo100", {})
    pseudo = metrics.get("pseudo100", {})
    rule_mrr = hard.get("rule_only", {}).get("mrr", 0.0)
    pseudo_rule_mrr = pseudo.get("rule_only", {}).get("mrr", 0.0)
    model_mrr = hard.get("model_only", {}).get("mrr", 0.0)
    if model_mrr + 1e-12 < rule_mrr:
        return 0.0
    best_weight = 0.0
    best_score = 0.0
    for weight in weights:
        hard_value = hard.get("fusion", {}).get(str(weight), {}).get("mrr", 0.0)
        pseudo_value = pseudo.get("fusion", {}).get(str(weight), {}).get("mrr", 0.0)
        hard_gain = hard_value - rule_mrr
        pseudo_gain = pseudo_value - pseudo_rule_mrr
        if pseudo_gain < -0.002:
            continue
        score = hard_gain + 0.5 * pseudo_gain
        if score > best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def _reciprocal_rank_at_index(scores, positive_index):
    scores = np.asarray(scores, dtype=np.float64)
    positive_index = int(positive_index)
    return 1.0 / (1 + int(np.sum(scores > scores[positive_index])))


def _extend_unique(selected, seen, values, limit):
    for value in values:
        value = int(value)
        if value in seen:
            continue
        selected.append(value)
        seen.add(value)
        if len(selected) >= limit:
            return True
    return False


def _listwise_candidate_dsts(
    src,
    positive_dst,
    state,
    known_dsts,
    popular_dsts,
    test_known_dsts,
    cold_dsts,
    neg_count,
    rng,
    mode="mixed",
):
    target_size = int(neg_count) + 1
    selected = [int(positive_dst)]
    seen = {int(positive_dst)}
    if target_size <= 1:
        return selected

    recent_values = [int(value) for value in reversed(state.src_recent.get(src, ()))]
    known_dsts = list(known_dsts)
    popular_dsts = list(popular_dsts)
    test_known_dsts = list(test_known_dsts)
    cold_dsts = list(cold_dsts)

    if mode == "random":
        quotas = {"recent": 0, "popular": 0, "test": 0, "cold": 0}
    elif mode == "cold":
        quotas = {
            "recent": max(2, neg_count // 8),
            "popular": max(4, neg_count // 5),
            "test": max(4, neg_count // 6),
            "cold": max(8, neg_count // 2),
        }
    elif mode == "hard":
        quotas = {
            "recent": max(4, neg_count // 3),
            "popular": max(8, neg_count // 2),
            "test": max(2, neg_count // 10),
            "cold": 0,
        }
    else:
        quotas = {
            "recent": max(4, neg_count // 4),
            "popular": max(8, neg_count // 3),
            "test": max(4, neg_count // 5),
            "cold": max(2, neg_count // 8),
        }

    if quotas["recent"]:
        _extend_unique(selected, seen, recent_values, min(target_size, 1 + quotas["recent"]))
    if quotas["popular"] and len(selected) < target_size:
        window = popular_dsts[: max(quotas["popular"] * 8, quotas["popular"])]
        rng.shuffle(window)
        _extend_unique(selected, seen, window, min(target_size, len(selected) + quotas["popular"]))
    if quotas["test"] and len(selected) < target_size and test_known_dsts:
        draws = [test_known_dsts[int(rng.integers(0, len(test_known_dsts)))] for _ in range(quotas["test"] * 4)]
        _extend_unique(selected, seen, draws, min(target_size, len(selected) + quotas["test"]))
    if quotas["cold"] and len(selected) < target_size and cold_dsts:
        draws = [cold_dsts[int(rng.integers(0, len(cold_dsts)))] for _ in range(quotas["cold"] * 4)]
        _extend_unique(selected, seen, draws, min(target_size, len(selected) + quotas["cold"]))

    while len(selected) < target_size and known_dsts:
        value = int(known_dsts[int(rng.integers(0, len(known_dsts)))])
        if value not in seen:
            selected.append(value)
            seen.add(value)
    return selected[:target_size]


def _shuffle_candidates_with_label(candidates, positive_dst, rng):
    candidates = list(candidates)
    order = np.arange(len(candidates))
    rng.shuffle(order)
    shuffled = [candidates[int(idx)] for idx in order]
    positive_index = int(shuffled.index(int(positive_dst)))
    return shuffled, positive_index


def _build_listwise_training_data(
    dataset_dir,
    history_edges,
    supervision_edges,
    neg_count,
    max_events,
    seed,
    new_pair_only=True,
):
    rng = np.random.default_rng(int(seed))
    scale_edges = history_edges if history_edges else supervision_edges
    state = Dataset2FeatureState(scale_edges, recent_limit=200).fit(history_edges)
    full_state = Dataset2FeatureState(scale_edges, recent_limit=200).fit(scale_edges)
    known_dsts = sorted({dst for _, dst, _ in scale_edges})
    known_set = set(known_dsts)
    test_pool = _read_test_candidate_pool(dataset_dir)
    test_known_dsts = [dst for dst in test_pool if dst in known_set]
    cold_dsts = [dst for dst in test_pool if dst not in known_set]
    popular_dsts = full_state.popular_dsts(limit=8192)
    selected = _selected_training_indices(len(supervision_edges), max_events)

    xs = []
    label_positions = []
    used_events = 0
    skipped_repeated = 0
    for idx, (src, dst, time) in enumerate(sorted(supervision_edges, key=lambda row: row[2])):
        repeated_pair = state.pair_count.get((src, dst), 0) > 0
        if repeated_pair and new_pair_only:
            skipped_repeated += 1
            state.update(src, dst, time)
            continue
        if idx in selected:
            candidates = _listwise_candidate_dsts(
                src,
                dst,
                state,
                known_dsts,
                popular_dsts,
                test_known_dsts,
                cold_dsts,
                int(neg_count),
                rng,
                mode="mixed",
            )
            if len(candidates) >= 2:
                candidates, positive_index = _shuffle_candidates_with_label(candidates, dst, rng)
                xs.append(_listwise_score_features(state, src, time, candidates))
                label_positions.append(positive_index)
                used_events += 1
                if used_events % 50000 == 0:
                    print(f"dataset2 listwise: built events={used_events}", flush=True)
        state.update(src, dst, time)
    if not xs:
        raise RuntimeError("dataset2 listwise: no training rows generated")
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(label_positions, dtype=np.int32),
        used_events,
        len(cold_dsts),
        len(test_known_dsts),
        skipped_repeated,
    )


def evaluate_listwise_feature_ranker(model, dataset_dir, history_edges, valid_edges, neg_count, max_events, seed, fusion_weights):
    state = Dataset2FeatureState(history_edges, recent_limit=200).fit(history_edges)
    known_dsts = sorted({dst for _, dst, _ in history_edges})
    known_set = set(known_dsts)
    test_pool = _read_test_candidate_pool(dataset_dir)
    test_known_dsts = [dst for dst in test_pool if dst in known_set]
    cold_dsts = [dst for dst in test_pool if dst not in known_set]
    popular_dsts = state.popular_dsts(limit=8192)
    selected = _evaluation_indices(len(valid_edges), max_events=max_events)
    eval_modes = {
        "pseudo100": "random",
        "hard-pseudo100": "hard",
        "test-mixed-pseudo100": "mixed",
        "cold-interference-pseudo100": "cold",
    }
    raw = {
        name: {
            "rule_only": [],
            "model_only": [],
            "fusion": {str(weight): [] for weight in fusion_weights},
            "top1_changed": {str(weight): 0 for weight in fusion_weights},
            "events": 0,
            "cold_candidate_frac": [],
        }
        for name in eval_modes
    }
    def flush_eval(eval_name, feature_rows, positive_indices, cold_fracs):
        if not feature_rows:
            return
        feature_arr = np.stack(feature_rows).astype(np.float32)
        model_matrix = model.predict_scores(feature_arr)
        rule_matrix = feature_arr[:, :, FEATURE_NAMES.index("rule_score")]
        payload = raw[eval_name]
        for row_idx, positive_index in enumerate(positive_indices):
            rule_scores = rule_matrix[row_idx]
            model_scores = model_matrix[row_idx]
            payload["rule_only"].append(_reciprocal_rank_at_index(rule_scores, positive_index))
            payload["model_only"].append(_reciprocal_rank_at_index(model_scores, positive_index))
            rule_top = int(np.argmax(rule_scores))
            model_z = row_zscore(model_scores.reshape(1, -1))[0]
            rule_z = row_zscore(rule_scores.reshape(1, -1))[0]
            for weight in fusion_weights:
                fused = rule_z + float(weight) * model_z
                payload["fusion"][str(weight)].append(_reciprocal_rank_at_index(fused, positive_index))
                if int(np.argmax(fused)) != rule_top:
                    payload["top1_changed"][str(weight)] += 1
        payload["events"] += len(positive_indices)
        payload["cold_candidate_frac"].extend(cold_fracs)

    eval_batch_size = 512
    for eval_name, mode in eval_modes.items():
        rng = np.random.default_rng(int(seed) + sum(ord(ch) for ch in eval_name))
        feature_rows = []
        positive_indices = []
        cold_fracs = []
        for idx in selected:
            src, dst, time = valid_edges[int(idx)]
            if dst not in known_set:
                continue
            candidates = _listwise_candidate_dsts(
                src,
                dst,
                state,
                known_dsts,
                popular_dsts,
                test_known_dsts,
                cold_dsts,
                int(neg_count),
                rng,
                mode=mode,
            )
            if len(candidates) < 2:
                continue
            candidates, positive_index = _shuffle_candidates_with_label(candidates, dst, rng)
            features = _listwise_score_features(state, src, time, candidates)
            feature_rows.append(features)
            positive_indices.append(positive_index)
            cold_fracs.append(float(np.mean([cand not in known_set for cand in candidates])))
            if len(feature_rows) >= eval_batch_size:
                flush_eval(eval_name, feature_rows, positive_indices, cold_fracs)
                feature_rows.clear()
                positive_indices.clear()
                cold_fracs.clear()
        flush_eval(eval_name, feature_rows, positive_indices, cold_fracs)
    metrics = {}
    for eval_name, payload in raw.items():
        events = int(payload["events"])
        metrics[eval_name] = {
            "rule_only": _bucket_metrics(payload["rule_only"]),
            "model_only": _bucket_metrics(payload["model_only"]),
            "fusion": {weight: _bucket_metrics(values) for weight, values in payload["fusion"].items()},
            "top1_changed_rate": {
                weight: (float(count) / max(events, 1))
                for weight, count in payload["top1_changed"].items()
            },
            "cold_candidate_frac": float(np.mean(payload["cold_candidate_frac"])) if payload["cold_candidate_frac"] else 0.0,
        }
    return metrics


def _select_listwise_fusion_weight(metrics, weights):
    best_weight = 0.0
    best_score = 0.0
    hard_rule = metrics.get("hard-pseudo100", {}).get("rule_only", {}).get("mrr", 0.0)
    mixed_rule = metrics.get("test-mixed-pseudo100", {}).get("rule_only", {}).get("mrr", 0.0)
    random_rule = metrics.get("pseudo100", {}).get("rule_only", {}).get("mrr", 0.0)
    for weight in weights:
        key = str(weight)
        hard = metrics.get("hard-pseudo100", {}).get("fusion", {}).get(key, {}).get("mrr", 0.0)
        mixed = metrics.get("test-mixed-pseudo100", {}).get("fusion", {}).get(key, {}).get("mrr", 0.0)
        random_value = metrics.get("pseudo100", {}).get("fusion", {}).get(key, {}).get("mrr", 0.0)
        cold_value = metrics.get("cold-interference-pseudo100", {}).get("fusion", {}).get(key, {}).get("mrr", 0.0)
        hard_gain = hard - hard_rule
        mixed_gain = mixed - mixed_rule
        random_gain = random_value - random_rule
        if random_gain < -0.01:
            continue
        score = 0.45 * hard_gain + 0.40 * mixed_gain + 0.10 * random_gain + 0.05 * cold_value
        if score > best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def train_dataset2_listwise_feature_ranker(
    dataset_dir,
    artifact_dir,
    final_train=False,
    neg_count=99,
    max_train_events=160000,
    valid_max_events=30000,
    hidden_dim=160,
    epochs=10,
    batch_size=4096,
    lr=0.001,
    margin_weight=0.05,
    fusion_model_weight=None,
    fusion_rule_weight=1.0,
    unknown_policy="neutral",
    unknown_score=0.0,
    unknown_margin=0.0,
    cold_prior_weight=0.0,
    new_pair_only=True,
    seed=2026,
):
    train_edges, valid_edges = split_dataset2_edges(dataset_dir, final_train=final_train)
    x, labels, events_used, cold_negatives, test_known_negatives, skipped_repeated = _build_listwise_training_data(
        dataset_dir,
        [],
        train_edges,
        neg_count=int(neg_count),
        max_events=int(max_train_events),
        seed=int(seed),
        new_pair_only=bool(new_pair_only),
    )
    print(
        f"dataset2 listwise: train_events={events_used} candidates={x.shape[1]} "
        f"feature_dim={x.shape[2]} cold_pool={cold_negatives} test_known_pool={test_known_negatives} "
        f"skipped_repeated={skipped_repeated}",
        flush=True,
    )
    model = JittorListwiseFeatureMLP(
        input_dim=x.shape[2],
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        margin_weight=float(margin_weight),
        seed=int(seed),
    )
    model.fit(x, labels)
    validation_metrics = {}
    selected_weight = 0.0
    weights = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0]
    if valid_edges:
        validation_metrics = evaluate_listwise_feature_ranker(
            model,
            dataset_dir,
            train_edges,
            valid_edges,
            neg_count=int(neg_count),
            max_events=int(valid_max_events),
            seed=int(seed) + 303,
            fusion_weights=weights,
        )
        selected_weight = _select_listwise_fusion_weight(validation_metrics, weights)
    elif fusion_model_weight is None:
        selected_weight = 0.15
    if fusion_model_weight is not None:
        selected_weight = float(fusion_model_weight)

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "feature_model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "dataset": "dataset2",
        "type": "dataset2_listwise_feature_ranker",
        "feature_names": LISTWISE_FEATURE_NAMES,
        "base_feature_names": FEATURE_NAMES,
        "neg_count": int(neg_count),
        "max_train_events": int(max_train_events),
        "events_used": int(events_used),
        "candidate_rows": int(events_used * x.shape[1]),
        "feature_dim": int(x.shape[2]),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "margin_weight": float(margin_weight),
        "final_train": bool(final_train),
        "loss_history": getattr(model, "loss_history", []),
        "validation_metrics": validation_metrics,
        "fusion_model_weight": float(selected_weight),
        "fusion_rule_weight": float(fusion_rule_weight),
        "unknown_policy": str(unknown_policy),
        "unknown_score": float(unknown_score),
        "unknown_margin": float(unknown_margin),
        "cold_prior_weight": float(cold_prior_weight),
        "cold_negative_pool": int(cold_negatives),
        "test_known_negative_pool": int(test_known_negatives),
        "skipped_repeated": int(skipped_repeated),
        "new_pair_only": bool(new_pair_only),
    }
    with open(artifact_dir / "model.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def train_dataset2_feature_reranker(
    dataset_dir,
    artifact_dir,
    final_train=False,
    neg_count=99,
    max_train_events=120000,
    valid_max_events=20000,
    model_kind="jittor_mlp",
    fusion_model_weight=None,
    fusion_rule_weight=1.0,
    unknown_policy="neutral",
    unknown_score=0.0,
    unknown_margin=0.0,
    cold_prior_weight=0.0,
    seed=2026,
):
    train_edges, valid_edges = split_dataset2_edges(dataset_dir, final_train=final_train)
    history_edges = [] if final_train else train_edges
    supervision_edges = train_edges
    x, y, events_used = _build_feature_training_data(
        history_edges,
        supervision_edges,
        neg_count=int(neg_count),
        max_events=int(max_train_events),
        seed=int(seed),
    )
    model = _fit_feature_classifier(x, y, str(model_kind), seed)
    weights = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
    validation_metrics = {}
    selected_weight = 0.0
    if valid_edges:
        validation_metrics = evaluate_feature_reranker(
            model,
            train_edges,
            valid_edges,
            neg_count=int(neg_count),
            max_events=int(valid_max_events),
            seed=int(seed),
            fusion_weights=weights,
        )
        selected_weight = _select_feature_fusion_weight(validation_metrics, weights)
    elif fusion_model_weight is None:
        selected_weight = 0.05
    if fusion_model_weight is not None:
        selected_weight = float(fusion_model_weight)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "feature_model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "dataset": "dataset2",
        "type": "dataset2_feature_reranker",
        "feature_names": FEATURE_NAMES,
        "model_kind": str(model_kind),
        "loss_history": getattr(model, "loss_history", []),
        "neg_count": int(neg_count),
        "max_train_events": int(max_train_events),
        "events_used": int(events_used),
        "training_rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "final_train": bool(final_train),
        "validation_metrics": validation_metrics,
        "fusion_model_weight": float(selected_weight),
        "fusion_rule_weight": float(fusion_rule_weight),
        "unknown_policy": str(unknown_policy),
        "unknown_score": float(unknown_score),
        "unknown_margin": float(unknown_margin),
        "cold_prior_weight": float(cold_prior_weight),
    }
    with open(artifact_dir / "model.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def _read_test_candidate_pool(dataset_dir):
    values = set()
    test_path = Path(dataset_dir) / "test.csv"
    if not test_path.exists():
        return []
    for _src, _time, candidates in iter_test_rows(test_path):
        values.update(candidates)
    return sorted(values)


def _pairwise_negative_dsts(src, dst, state, known_dsts, popular_dsts, cold_dsts, neg_count, rng):
    selected = []
    seen = {dst}
    recent_quota = max(1, neg_count // 4) if state.src_recent.get(src) else 0
    popular_quota = max(1, neg_count // 4) if popular_dsts else 0
    cold_quota = max(1, neg_count // 5) if cold_dsts else 0
    random_quota = max(0, neg_count - recent_quota - popular_quota - cold_quota)

    groups = []
    recent_values = [int(value) for value in reversed(state.src_recent.get(src, ()))]
    groups.extend(recent_values[:recent_quota])
    if popular_dsts:
        offset = int(rng.integers(0, min(len(popular_dsts), max(popular_quota * 20, popular_quota))))
        groups.extend(int(value) for value in popular_dsts[offset:offset + popular_quota * 2])
    if cold_dsts:
        for _ in range(cold_quota * 2):
            groups.append(int(cold_dsts[int(rng.integers(0, len(cold_dsts)))]))
    for _ in range(max(random_quota * 3, neg_count)):
        if known_dsts:
            groups.append(int(known_dsts[int(rng.integers(0, len(known_dsts)))]))

    for cand in groups:
        cand = int(cand)
        if cand in seen:
            continue
        selected.append(cand)
        seen.add(cand)
        if len(selected) >= neg_count:
            break
    while len(selected) < neg_count and known_dsts:
        cand = int(known_dsts[int(rng.integers(0, len(known_dsts)))])
        if cand not in seen:
            selected.append(cand)
            seen.add(cand)
    return selected


def _build_pairwise_training_data(
    dataset_dir,
    history_edges,
    supervision_edges,
    neg_count,
    max_events,
    seed,
    new_pair_only=True,
):
    rng = np.random.default_rng(int(seed))
    scale_edges = history_edges if history_edges else supervision_edges
    state = Dataset2FeatureState(scale_edges, recent_limit=200).fit(history_edges)
    full_state = Dataset2FeatureState(scale_edges, recent_limit=200).fit(scale_edges)
    known_dsts = sorted({dst for _, dst, _ in scale_edges})
    test_pool = _read_test_candidate_pool(dataset_dir)
    known_set = set(known_dsts)
    cold_dsts = [dst for dst in test_pool if dst not in known_set]
    popular_dsts = full_state.popular_dsts(limit=8192)
    selected = _selected_training_indices(len(supervision_edges), max_events)
    event_count = len(selected)
    rows_per_event = int(neg_count) + 1
    x = np.zeros((event_count * rows_per_event, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(event_count * rows_per_event, dtype=np.int8)
    out = 0
    used_events = 0
    skipped_repeated = 0
    for idx, (src, dst, time) in enumerate(sorted(supervision_edges, key=lambda row: row[2])):
        repeated_pair = state.pair_count.get((src, dst), 0) > 0
        if repeated_pair and new_pair_only:
            skipped_repeated += 1
            state.update(src, dst, time)
            continue
        if idx in selected:
            negatives = _pairwise_negative_dsts(
                src,
                dst,
                state,
                known_dsts,
                popular_dsts,
                cold_dsts,
                int(neg_count),
                rng,
            )
            candidates = [dst] + negatives
            features = state.score_features(src, time, candidates)
            take = min(len(candidates), rows_per_event)
            x[out:out + take] = features[:take]
            y[out] = 1
            out += take
            used_events += 1
            if used_events % 100000 == 0:
                print(f"dataset2 pairwise: built events={used_events} rows={out}", flush=True)
        state.update(src, dst, time)
    return x[:out], y[:out], used_events, len(cold_dsts), skipped_repeated


def evaluate_pairwise_classifier(model, history_edges, valid_edges, neg_count, max_events, seed, fusion_weights=None):
    from sklearn.metrics import roc_auc_score

    state = Dataset2FeatureState(history_edges, recent_limit=200).fit(history_edges)
    known_dsts = sorted({dst for _, dst, _ in history_edges})
    popular_dsts = state.popular_dsts(limit=8192)
    rng = np.random.default_rng(int(seed))
    selected = _evaluation_indices(len(valid_edges), max_events=max_events)
    xs = []
    ys = []
    rr_random = []
    rr_hard = []
    rule_rr_random = []
    rule_rr_hard = []
    if fusion_weights is None:
        fusion_weights = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    fusion_rr_random = {str(weight): [] for weight in fusion_weights}
    fusion_rr_hard = {str(weight): [] for weight in fusion_weights}
    for idx in selected:
        src, dst, time = valid_edges[int(idx)]
        random_negs = _pairwise_negative_dsts(src, dst, state, known_dsts, [], [], int(neg_count), rng)
        hard_negs = _pairwise_negative_dsts(src, dst, state, known_dsts, popular_dsts, [], int(neg_count), rng)
        for negs, target, rule_target, fusion_target in (
            (random_negs, rr_random, rule_rr_random, fusion_rr_random),
            (hard_negs, rr_hard, rule_rr_hard, fusion_rr_hard),
        ):
            candidates = [dst] + negs
            features = state.score_features(src, time, candidates)
            scores = _feature_model_scores(model, features)
            rule_scores = features[:, 0]
            target.append(_reciprocal_rank_at_zero(scores))
            rule_target.append(_reciprocal_rank_at_zero(rule_scores))
            rule_z = row_zscore(rule_scores.reshape(1, -1))[0]
            model_z = row_zscore(scores.reshape(1, -1))[0]
            for weight in fusion_weights:
                fused = rule_z + float(weight) * model_z
                fusion_target[str(weight)].append(_reciprocal_rank_at_zero(fused))
        candidates = [dst] + random_negs
        features = state.score_features(src, time, candidates)
        xs.append(features)
        label = np.zeros(len(candidates), dtype=np.int8)
        label[0] = 1
        ys.append(label)
    if xs:
        x = np.vstack(xs)
        y = np.concatenate(ys)
        scores = _feature_model_scores(model, x)
        try:
            auc = float(roc_auc_score(y, scores))
        except ValueError:
            auc = 0.0
    else:
        auc = 0.0
        y = np.zeros(0, dtype=np.int8)
    return {
        "pairwise_auc": auc,
        "pairwise_rows": int(len(y)),
        "events": int(len(selected)),
        "random_mrr": _bucket_metrics(rr_random),
        "hard_mrr": _bucket_metrics(rr_hard),
        "rule_random_mrr": _bucket_metrics(rule_rr_random),
        "rule_hard_mrr": _bucket_metrics(rule_rr_hard),
        "fusion_random_mrr": {weight: _bucket_metrics(values) for weight, values in fusion_rr_random.items()},
        "fusion_hard_mrr": {weight: _bucket_metrics(values) for weight, values in fusion_rr_hard.items()},
    }


def train_dataset2_pairwise_classifier(
    dataset_dir,
    artifact_dir,
    final_train=False,
    neg_count=8,
    max_train_events=400000,
    valid_max_events=20000,
    model_kind="jittor_mlp",
    fusion_model_weight=0.05,
    fusion_rule_weight=1.0,
    unknown_policy="neutral",
    unknown_score=0.0,
    unknown_margin=0.0,
    cold_prior_weight=0.0,
    seed=2026,
):
    train_edges, valid_edges = split_dataset2_edges(dataset_dir, final_train=final_train)
    x, y, events_used, cold_negatives, skipped_repeated = _build_pairwise_training_data(
        dataset_dir,
        [],
        train_edges,
        neg_count=int(neg_count),
        max_events=int(max_train_events),
        seed=int(seed),
        new_pair_only=True,
    )
    print(
        f"dataset2 pairwise: training rows={len(y)} positives={int(y.sum())} "
        f"events={events_used} neg_count={neg_count} cold_negative_pool={cold_negatives} "
        f"skipped_repeated={skipped_repeated}",
        flush=True,
    )
    model = _fit_feature_classifier(x, y, str(model_kind), seed)
    validation_metrics = {}
    if valid_edges:
        validation_metrics = evaluate_pairwise_classifier(
            model,
            train_edges,
            valid_edges,
            neg_count=max(99, int(neg_count) * 8),
            max_events=int(valid_max_events),
            seed=int(seed) + 101,
        )
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "feature_model.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "dataset": "dataset2",
        "type": "dataset2_pairwise_classifier",
        "feature_names": FEATURE_NAMES,
        "model_kind": str(model_kind),
        "neg_count": int(neg_count),
        "max_train_events": int(max_train_events),
        "events_used": int(events_used),
        "training_rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "cold_negative_pool": int(cold_negatives),
        "skipped_repeated": int(skipped_repeated),
        "final_train": bool(final_train),
        "loss_history": getattr(model, "loss_history", []),
        "validation_metrics": validation_metrics,
        "fusion_model_weight": float(fusion_model_weight),
        "fusion_rule_weight": float(fusion_rule_weight),
        "unknown_policy": str(unknown_policy),
        "unknown_score": float(unknown_score),
        "unknown_margin": float(unknown_margin),
        "cold_prior_weight": float(cold_prior_weight),
    }
    with open(artifact_dir / "model.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def iter_dataset2_pairwise_proba_chunks(dataset_dir, artifact_dir, batch_size=512, max_rows=0):
    artifact_dir = Path(artifact_dir)
    with open(artifact_dir / "model.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(artifact_dir / "feature_model.pkl", "rb") as f:
        model = pickle.load(f)
    train_edges = list(iter_train_edges(Path(dataset_dir) / "train.csv"))
    state = Dataset2FeatureState(train_edges, recent_limit=200).fit(train_edges)
    batch = []
    emitted = 0
    limit = max(int(max_rows or 0), 0)
    for row in iter_test_rows(Path(dataset_dir) / "test.csv"):
        if limit and emitted + len(batch) >= limit:
            break
        batch.append(row)
        if len(batch) >= batch_size:
            yield _score_pairwise_query_batch(model, state, batch, meta)
            emitted += len(batch)
            batch = []
    if batch:
        yield _score_pairwise_query_batch(model, state, batch, meta)


def _score_pairwise_query_batch(model, state, chunk, meta):
    rows = []
    shapes = []
    rule_rows = []
    known_rows = []
    for src, time, candidates in chunk:
        features = state.score_features(src, time, candidates)
        rows.append(features)
        shapes.append(len(candidates))
        rule_rows.append(features[:, 0])
        known_rows.append(features[:, FEATURE_NAMES.index("dst_seen")] > 0.5)
    flat = np.vstack(rows).astype(np.float32)
    probs = _feature_model_scores(model, flat)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(probs / (1.0 - probs)).astype(np.float32)
    output = []
    offset = 0
    for size, rule_scores in zip(shapes, rule_rows):
        model_scores = logits[offset:offset + size]
        offset += size
        fused = (
            row_zscore(model_scores.reshape(1, -1))[0] * float(meta.get("fusion_model_weight", 0.05))
            + row_zscore(np.asarray(rule_scores, dtype=np.float32).reshape(1, -1))[0]
            * float(meta.get("fusion_rule_weight", 1.0))
        )
        output.append(fused)
    scores = np.asarray(output, dtype=np.float32)
    known_mask = np.asarray(known_rows, dtype=bool)
    scores = _apply_unknown_policy(
        scores,
        known_mask,
        chunk,
        policy=meta.get("unknown_policy", "neutral"),
        fallback=float(meta.get("unknown_score", 0.0)),
        margin=float(meta.get("unknown_margin", 0.0)),
        cold_prior_weight=float(meta.get("cold_prior_weight", 0.0)),
    )
    return softmax(scores, temperature=1.0)


def iter_dataset2_feature_proba_chunks(dataset_dir, artifact_dir, batch_size=512, max_rows=0):
    artifact_dir = Path(artifact_dir)
    with open(artifact_dir / "model.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(artifact_dir / "feature_model.pkl", "rb") as f:
        model = pickle.load(f)
    train_edges = list(iter_train_edges(Path(dataset_dir) / "train.csv"))
    state = Dataset2FeatureState(train_edges, recent_limit=200).fit(train_edges)
    batch = []
    emitted = 0
    limit = max(int(max_rows or 0), 0)
    for row in iter_test_rows(Path(dataset_dir) / "test.csv"):
        if limit and emitted + len(batch) >= limit:
            break
        batch.append(row)
        if len(batch) >= batch_size:
            yield _score_feature_query_batch(model, state, batch, meta)
            emitted += len(batch)
            batch = []
    if batch:
        yield _score_feature_query_batch(model, state, batch, meta)


def _features_for_meta(state, src, time, candidates, meta):
    if str(meta.get("type", "")) == "dataset2_listwise_feature_ranker":
        return _listwise_score_features(state, src, time, candidates)
    return state.score_features(src, time, candidates)


def _score_feature_query_batch(model, state, chunk, meta):
    rows = []
    shapes = []
    rule_rows = []
    known_rows = []
    for src, time, candidates in chunk:
        features = _features_for_meta(state, src, time, candidates, meta)
        rows.append(features)
        shapes.append(len(candidates))
        rule_rows.append(features[:, 0])
        known_rows.append(features[:, FEATURE_NAMES.index("dst_seen")] > 0.5)
    flat = np.vstack(rows).astype(np.float32)
    model_scores = _feature_model_scores(model, flat)
    output = []
    offset = 0
    for size, rule_scores in zip(shapes, rule_rows):
        scores = model_scores[offset:offset + size]
        offset += size
        fused = (
            row_zscore(scores.reshape(1, -1))[0] * float(meta.get("fusion_model_weight", 0.0))
            + row_zscore(np.asarray(rule_scores, dtype=np.float32).reshape(1, -1))[0]
            * float(meta.get("fusion_rule_weight", 1.0))
        )
        output.append(fused)
    scores = np.asarray(output, dtype=np.float32)
    known_mask = np.asarray(known_rows, dtype=bool)
    scores = _apply_unknown_policy(
        scores,
        known_mask,
        chunk,
        policy=meta.get("unknown_policy", "neutral"),
        fallback=float(meta.get("unknown_score", 0.0)),
        margin=float(meta.get("unknown_margin", 0.0)),
        cold_prior_weight=float(meta.get("cold_prior_weight", 0.0)),
    )
    return softmax(scores, temperature=1.0)


def _hard_ids_for_batch(src_ids, hard_by_src, popular_ids, hard_count):
    hard_count = int(hard_count)
    if hard_count <= 0:
        return np.zeros(0, dtype=np.int32)
    selected = []
    seen = set()
    per_src_take = max(1, hard_count // max(len(np.unique(src_ids)), 1))
    for src_id in np.unique(src_ids):
        values = hard_by_src.get(int(src_id))
        if values is None:
            continue
        for dst_id in values[:per_src_take]:
            value = int(dst_id)
            if value not in seen:
                selected.append(value)
                seen.add(value)
            if len(selected) >= hard_count:
                return np.asarray(selected, dtype=np.int32)
    for dst_id in popular_ids:
        value = int(dst_id)
        if value not in seen:
            selected.append(value)
            seen.add(value)
        if len(selected) >= hard_count:
            break
    return np.asarray(selected, dtype=np.int32)


def _shared_candidate_set(labels, num_dst, neg_count, rng, extra_ids=None, corrected=True):
    pos_ids = labels.astype(np.int32) + 1
    negs = rng.integers(1, num_dst + 1, size=int(neg_count), dtype=np.int32)
    pieces = [pos_ids, negs]
    if extra_ids is not None and len(extra_ids):
        pieces.append(np.asarray(extra_ids, dtype=np.int32))
    candidates = np.unique(np.concatenate(pieces)).astype(np.int32)
    label_positions = np.searchsorted(candidates, pos_ids).astype(np.int32)
    logq = np.zeros(len(candidates), dtype=np.float32)
    if corrected:
        q = np.full(len(candidates), min(float(neg_count) / max(float(num_dst), 1.0), 1.0), dtype=np.float32)
        if extra_ids is not None and len(extra_ids):
            extra_positions = np.searchsorted(candidates, np.asarray(extra_ids, dtype=np.int32))
            extra_positions = extra_positions[(extra_positions >= 0) & (extra_positions < len(candidates))]
            q[extra_positions] = np.minimum(q[extra_positions] + 1.0 / max(float(len(extra_ids)), 1.0), 1.0)
        q[label_positions] = 1.0
        logq = np.log(np.clip(q, 1e-8, 1.0)).astype(np.float32)
    return candidates, label_positions, logq


def _per_row_candidate_matrix(labels, src_ids, num_dst, rng, hard_by_src, popular_ids, neg_count):
    neg_count = int(neg_count)
    if neg_count < 1:
        return None
    cand = np.zeros((len(labels), neg_count + 1), dtype=np.int32)
    cand[:, 0] = labels.astype(np.int32) + 1
    for row_idx, (label, src_id) in enumerate(zip(labels, src_ids)):
        positive_id = int(label) + 1
        selected = [positive_id]
        seen = {positive_id}
        hard_values = hard_by_src.get(int(src_id))
        if hard_values is not None:
            for dst_id in hard_values:
                value = int(dst_id)
                if value not in seen:
                    selected.append(value)
                    seen.add(value)
                if len(selected) >= neg_count + 1:
                    break
        if len(selected) < neg_count + 1:
            for dst_id in popular_ids:
                value = int(dst_id)
                if value not in seen:
                    selected.append(value)
                    seen.add(value)
                if len(selected) >= neg_count + 1:
                    break
        while len(selected) < neg_count + 1:
            value = int(rng.integers(1, num_dst + 1))
            if value not in seen:
                selected.append(value)
                seen.add(value)
        cand[row_idx] = np.asarray(selected[:neg_count + 1], dtype=np.int32)
    return cand


def _save_model(path, model, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jt.save({"state_dict": model.state_dict(), "meta": meta}, str(path))


def _jsonable_meta(meta):
    if not meta:
        return {}
    out = {}
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            out[f"{key}_shape"] = list(value.shape)
            out[key] = "<stored in model.pkl>"
        else:
            out[key] = value
    return out


def load_model(path):
    if not jittor_available():
        raise ImportError("jittor is required to load dataset2 temporal model")
    payload = jt.load(str(path))
    meta = payload["meta"]
    model = TemporalRecommender(
        int(meta["num_src"]),
        int(meta["num_dst"]),
        emb_dim=int(meta.get("emb_dim", 96)),
        hidden_dim=int(meta.get("hidden_dim", 192)),
        dropout=0.0,
        seq_len=int(meta.get("seq_len", 80)),
        dst_features=np.asarray(meta.get("dst_features", []), dtype=np.float32)
        if meta.get("dst_features") is not None
        else None,
    )
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(payload["state_dict"])
    else:
        model.load_parameters(payload["state_dict"])
    model.eval()
    return model, meta


def _bucket_metrics(rr_values):
    if not rr_values:
        return {"mrr": 0.0, "events": 0}
    return {"mrr": float(np.mean(rr_values)), "events": int(len(rr_values))}


def _bucket_metrics_with_zeros(rr_values, zero_events=0):
    events = len(rr_values) + int(zero_events)
    if events <= 0:
        return {"mrr": 0.0, "events": 0}
    return {"mrr": float(np.sum(rr_values) / events), "events": int(events)}


def _evaluation_indices(size, max_events=0):
    size = int(size)
    if size <= 0:
        return np.zeros(0, dtype=np.int64)
    count = min(size, int(max_events) if max_events else size)
    if count >= size:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, count, dtype=np.int64)


def evaluate_full_mrr(model, samples, batch_size, max_events=20000):
    skipped = int(samples.get("skipped", 0))
    cold_dst_events = int(samples.get("skipped_cold_dst", 0))
    cold_src_events = int(samples.get("skipped_cold_src", 0))
    no_history_events = int(samples.get("skipped_no_history_src", 0))
    if len(samples["label"]) == 0:
        return {
            "overall": {"mrr": 0.0, "events": skipped},
            "repeated": {"mrr": 0.0, "events": 0},
            "new_pair": {"mrr": 0.0, "events": 0},
            "cold_dst": {"mrr": 0.0, "events": cold_dst_events},
            "cold_src": {"mrr": 0.0, "events": cold_src_events},
            "no_history_src": {"mrr": 0.0, "events": no_history_events},
            "skipped_other": int(samples.get("skipped_other", 0)),
        }
    idx = _evaluation_indices(len(samples["label"]), max_events=max_events)
    rr = []
    repeated_rr = []
    new_rr = []
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        logits = model(
            jt.array(samples["src"][batch_idx]),
            jt.array(samples["hist"][batch_idx]),
            jt.array(samples["hist_gap"][batch_idx]),
            jt.array(samples["time"][batch_idx]),
        ).numpy()
        labels = samples["label"][batch_idx]
        pos = logits[np.arange(len(labels)), labels]
        ranks = 1 + (logits > pos[:, None]).sum(axis=1)
        batch_rr = 1.0 / ranks
        rr.extend(batch_rr.tolist())
        is_new = samples["is_new_pair"][batch_idx].astype(bool)
        if is_new.any():
            new_rr.extend(batch_rr[is_new].tolist())
        if (~is_new).any():
            repeated_rr.extend(batch_rr[~is_new].tolist())
    return {
        "overall": _bucket_metrics_with_zeros(rr, skipped),
        "repeated": _bucket_metrics(repeated_rr),
        "new_pair": _bucket_metrics(new_rr),
        "cold_dst": {"mrr": 0.0, "events": cold_dst_events},
        "cold_src": {"mrr": 0.0, "events": cold_src_events},
        "no_history_src": {"mrr": 0.0, "events": no_history_events},
        "skipped_other": int(samples.get("skipped_other", 0)),
    }


def evaluate_candidate_mrr(
    model,
    samples,
    batch_size,
    hard_by_src,
    popular_ids,
    num_dst,
    seed,
    neg_count=99,
    max_events=20000,
):
    skipped = int(samples.get("skipped", 0))
    if len(samples["label"]) == 0:
        return {
            "overall": {"mrr": 0.0, "events": skipped},
            "repeated": {"mrr": 0.0, "events": 0},
            "new_pair": {"mrr": 0.0, "events": 0},
        }
    rng = np.random.default_rng(int(seed))
    idx = _evaluation_indices(len(samples["label"]), max_events=max_events)
    rr = []
    repeated_rr = []
    new_rr = []
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        labels = samples["label"][batch_idx]
        cand = _per_row_candidate_matrix(
            labels,
            samples["src"][batch_idx],
            num_dst,
            rng,
            hard_by_src,
            popular_ids,
            int(neg_count),
        )
        logits = model(
            jt.array(samples["src"][batch_idx]),
            jt.array(samples["hist"][batch_idx]),
            jt.array(samples["hist_gap"][batch_idx]),
            jt.array(samples["time"][batch_idx]),
            jt.array(cand),
        ).numpy()
        pos = logits[:, 0]
        ranks = 1 + (logits > pos[:, None]).sum(axis=1)
        batch_rr = 1.0 / ranks
        rr.extend(batch_rr.tolist())
        is_new = samples["is_new_pair"][batch_idx].astype(bool)
        if is_new.any():
            new_rr.extend(batch_rr[is_new].tolist())
        if (~is_new).any():
            repeated_rr.extend(batch_rr[~is_new].tolist())
    return {
        "overall": _bucket_metrics_with_zeros(rr, skipped),
        "repeated": _bucket_metrics(repeated_rr),
        "new_pair": _bucket_metrics(new_rr),
    }


def train_dataset2(
    dataset_dir,
    artifact_dir,
    final_train=False,
    cuda=True,
    softmax_mode="sampled",
    max_train_events=0,
    neg_count=4096,
    seq_len=80,
    emb_dim=96,
    hidden_dim=192,
    dropout=0.1,
    epochs=6,
    batch_size=512,
    lr=0.001,
    weight_decay=1e-6,
    bpr_weight=0.05,
    all_dst_weight=0.20,
    hard_negative_count=512,
    sampled_correction=True,
    rerank_neg_count=99,
    rerank_weight=1.00,
    fusion_model_weight=1.0,
    fusion_rule_weight=0.10,
    include_test_vocab=True,
    unknown_policy="neutral",
    unknown_score=0.0,
    unknown_margin=0.0,
    cold_prior_weight=0.0,
    valid_max_events=20000,
    seed=2026,
):
    if not jittor_available():
        raise ImportError("Jittor is required for dataset2 temporal recommender training")
    neg_count = int(neg_count)
    seq_len = int(seq_len)
    epochs = int(epochs)
    batch_size = int(batch_size)
    if neg_count < 1:
        raise ValueError(f"neg_count must be >= 1, got {neg_count}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    bpr_weight = float(bpr_weight)
    if bpr_weight < 0.0:
        raise ValueError(f"bpr_weight must be >= 0, got {bpr_weight}")
    all_dst_weight = float(all_dst_weight)
    if all_dst_weight < 0.0:
        raise ValueError(f"all_dst_weight must be >= 0, got {all_dst_weight}")
    rerank_neg_count = int(rerank_neg_count)
    if rerank_neg_count < 0:
        raise ValueError(f"rerank_neg_count must be >= 0, got {rerank_neg_count}")
    rerank_weight = float(rerank_weight)
    if rerank_weight < 0.0:
        raise ValueError(f"rerank_weight must be >= 0, got {rerank_weight}")
    hard_negative_count = int(hard_negative_count)
    if hard_negative_count < 0:
        raise ValueError(f"hard_negative_count must be >= 0, got {hard_negative_count}")
    sampled_correction = bool(sampled_correction)
    fusion_model_weight = float(fusion_model_weight)
    fusion_rule_weight = float(fusion_rule_weight)
    if cuda:
        jt.flags.use_cuda = 1
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_edges, valid_edges = split_dataset2_edges(dataset_dir, final_train=final_train)
    print(
        f"dataset2 temporal: loaded train_edges={len(train_edges)} valid_edges={len(valid_edges)} "
        f"max_train_events={int(max_train_events or 0)} include_test_vocab={bool(include_test_vocab)}",
        flush=True,
    )
    extra_dsts = None
    if include_test_vocab:
        print("dataset2 temporal: reading test candidate dst vocab", flush=True)
        extra_dsts = iter_test_candidate_dsts(dataset_dir)
    src_to_id, dst_to_id = build_id_maps(train_edges, extra_dsts=extra_dsts)
    print(f"dataset2 temporal: mapped src={len(src_to_id)} dst={len(dst_to_id)}", flush=True)
    max_train_events = int(max_train_events or 0)
    print("dataset2 temporal: building train samples", flush=True)
    train_samples = build_samples(
        [],
        train_edges,
        src_to_id,
        dst_to_id,
        seq_len,
        update_with_supervision=True,
        max_events=max_train_events,
    )
    print(f"dataset2 temporal: train samples={len(train_samples['label'])}", flush=True)
    valid_samples = None
    if valid_edges:
        print("dataset2 temporal: building valid samples", flush=True)
        valid_samples = build_samples(
            train_edges,
            valid_edges,
            src_to_id,
            dst_to_id,
            seq_len,
            update_with_supervision=False,
            max_events=valid_max_events,
        )
        print(f"dataset2 temporal: valid samples={len(valid_samples['label'])}", flush=True)
    if len(train_samples["label"]) < 100:
        raise RuntimeError(f"dataset2: too few training samples after history filtering: {len(train_samples['label'])}")

    num_src = len(src_to_id)
    num_dst = len(dst_to_id)
    hard_by_src, popular_ids = _build_hard_negative_tables(
        train_edges,
        src_to_id,
        dst_to_id,
        per_src_limit=max(200, min(seq_len * 4, 1000)),
        popular_limit=max(4096, hard_negative_count * 4),
    )
    dst_features = build_dst_feature_table(train_edges, dst_to_id)
    model = TemporalRecommender(
        num_src,
        num_dst,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        seq_len=seq_len,
        dst_features=dst_features,
    )
    optimizer = nn.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_mrr = -1.0
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pkl"
    selected_meta = None

    use_full = str(softmax_mode).lower() == "full"
    effective_neg_count = min(neg_count, max(num_dst - 1, 1))
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        steps = 0
        for idx in _batch_indices(len(train_samples["label"]), batch_size, rng, shuffle=True):
            src = jt.array(train_samples["src"][idx])
            hist = jt.array(train_samples["hist"][idx])
            hist_gap = jt.array(train_samples["hist_gap"][idx])
            time_feats = jt.array(train_samples["time"][idx])
            labels = train_samples["label"][idx]
            if use_full:
                logits = model(src, hist, hist_gap, time_feats)
                loss = float(all_dst_weight) * nn.cross_entropy_loss(logits, jt.array(labels.astype(np.int32)))
                if bpr_weight:
                    neg_ids = rng.integers(1, num_dst + 1, size=(len(idx), 1), dtype=np.int32)
                    cand = np.concatenate([(labels.astype(np.int32) + 1)[:, None], neg_ids], axis=1)
                    pair_scores = model(src, hist, hist_gap, time_feats, jt.array(cand))
                    diff = pair_scores[:, 0] - pair_scores[:, 1]
                    loss = loss + float(bpr_weight) * jt.log(1.0 + jt.exp(-diff)).mean()
                if rerank_weight and rerank_neg_count:
                    rerank_cand = _per_row_candidate_matrix(
                        labels,
                        train_samples["src"][idx],
                        num_dst,
                        rng,
                        hard_by_src,
                        popular_ids,
                        rerank_neg_count,
                    )
                    rerank_logits = model(src, hist, hist_gap, time_feats, jt.array(rerank_cand))
                    rerank_labels = np.zeros(len(idx), dtype=np.int32)
                    loss = loss + float(rerank_weight) * nn.cross_entropy_loss(rerank_logits, jt.array(rerank_labels))
            else:
                hard_ids = _hard_ids_for_batch(
                    train_samples["src"][idx],
                    hard_by_src,
                    popular_ids,
                    hard_negative_count,
                )
                cand, label_positions, logq = _shared_candidate_set(
                    labels,
                    num_dst,
                    effective_neg_count,
                    rng,
                    extra_ids=hard_ids,
                    corrected=sampled_correction,
                )
                raw_logits = model(src, hist, hist_gap, time_feats, jt.array(cand))
                logits = raw_logits - jt.array(logq).reshape((1, -1))
                loss = float(all_dst_weight) * nn.cross_entropy_loss(logits, jt.array(label_positions))
                if bpr_weight and cand.shape[0] > 1:
                    positive_mask = np.zeros((len(idx), len(cand)), dtype=np.float32)
                    positive_mask[np.arange(len(idx)), label_positions] = 1.0
                    positive_mask_jt = jt.array(positive_mask)
                    neg_logits = raw_logits - positive_mask_jt * 1e6
                    max_neg = neg_logits.max(dim=1)
                    if isinstance(max_neg, tuple):
                        max_neg = max_neg[0]
                    pos_score = (raw_logits * positive_mask_jt).sum(dim=1)
                    diff = pos_score - max_neg
                    loss = loss + float(bpr_weight) * jt.log(1.0 + jt.exp(-diff)).mean()
                if rerank_weight and rerank_neg_count:
                    rerank_cand = _per_row_candidate_matrix(
                        labels,
                        train_samples["src"][idx],
                        num_dst,
                        rng,
                        hard_by_src,
                        popular_ids,
                        rerank_neg_count,
                    )
                    rerank_logits = model(src, hist, hist_gap, time_feats, jt.array(rerank_cand))
                    rerank_labels = np.zeros(len(idx), dtype=np.int32)
                    loss = loss + float(rerank_weight) * nn.cross_entropy_loss(rerank_logits, jt.array(rerank_labels))
            optimizer.step(loss)
            loss_sum += float(loss.numpy())
            steps += 1

        metrics = {
            "overall": {"mrr": 0.0, "events": 0},
            "repeated": {"mrr": 0.0, "events": 0},
            "new_pair": {"mrr": 0.0, "events": 0},
            "cold_dst": {"mrr": 0.0, "events": 0},
            "cold_src": {"mrr": 0.0, "events": 0},
            "no_history_src": {"mrr": 0.0, "events": 0},
            "skipped_other": 0,
        }
        if valid_samples is not None:
            model.eval()
            metrics = evaluate_full_mrr(model, valid_samples, batch_size=max(32, min(256, batch_size)), max_events=valid_max_events)
            candidate_metrics = evaluate_candidate_mrr(
                model,
                valid_samples,
                batch_size=max(32, min(256, batch_size)),
                hard_by_src=hard_by_src,
                popular_ids=popular_ids,
                num_dst=num_dst,
                seed=seed + epoch,
                neg_count=max(rerank_neg_count, 99),
                max_events=valid_max_events,
            )
            score_for_selection = candidate_metrics["overall"]["mrr"]
        else:
            candidate_metrics = {
                "overall": {"mrr": 0.0, "events": 0},
                "repeated": {"mrr": 0.0, "events": 0},
                "new_pair": {"mrr": 0.0, "events": 0},
            }
            score_for_selection = float(epoch)
        print(
            f"dataset2: epoch={epoch} loss={loss_sum / max(steps, 1):.6f} "
            f"overall_mrr={metrics['overall']['mrr']:.6f} "
            f"repeated_mrr={metrics['repeated']['mrr']:.6f} "
            f"new_pair_mrr={metrics['new_pair']['mrr']:.6f} "
            f"candidate_hard_mrr={candidate_metrics['overall']['mrr']:.6f} "
            f"candidate_hard_new_pair_mrr={candidate_metrics['new_pair']['mrr']:.6f} "
            f"cold_dst_mrr={metrics['cold_dst']['mrr']:.6f} "
            f"cold_src_mrr={metrics['cold_src']['mrr']:.6f} "
            f"no_history_src_mrr={metrics['no_history_src']['mrr']:.6f} "
            f"events=(all:{metrics['overall']['events']} rep:{metrics['repeated']['events']} "
            f"new:{metrics['new_pair']['events']} cold_dst:{metrics['cold_dst']['events']} "
            f"cold_src:{metrics['cold_src']['events']} no_hist:{metrics['no_history_src']['events']})"
        )
        if score_for_selection > best_mrr:
            best_mrr = score_for_selection
            selected_meta = {
                "dataset": "dataset2",
                "type": "dataset2_temporal_recommender",
                "num_src": num_src,
                "num_dst": num_dst,
                "src_to_id": {str(k): int(v) for k, v in src_to_id.items()},
                "dst_to_id": {str(k): int(v) for k, v in dst_to_id.items()},
                "seq_len": int(seq_len),
                "emb_dim": int(emb_dim),
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "dst_features": dst_features.astype(np.float32),
                "dst_feature_dim": int(dst_features.shape[1]),
                "softmax_mode": "full" if use_full else "sampled",
                "neg_count": int(effective_neg_count),
                "hard_negative_count": int(hard_negative_count),
                "all_dst_weight": float(all_dst_weight),
                "sampled_correction": bool(sampled_correction),
                "rerank_neg_count": int(rerank_neg_count),
                "rerank_weight": float(rerank_weight),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "best_valid_mrr": float(metrics["overall"]["mrr"]),
                "best_repeated_mrr": float(metrics["repeated"]["mrr"]),
                "best_new_pair_mrr": float(metrics["new_pair"]["mrr"]),
                "best_cold_dst_mrr": float(metrics["cold_dst"]["mrr"]),
                "best_cold_src_mrr": float(metrics["cold_src"]["mrr"]),
                "best_no_history_src_mrr": float(metrics["no_history_src"]["mrr"]),
                "validation_metrics": metrics,
                "validation_candidate_hard_metrics": candidate_metrics,
                "train_samples": int(len(train_samples["label"])),
                "train_skipped": int(train_samples["skipped"]),
                "train_skipped_cold_dst": int(train_samples["skipped_cold_dst"]),
                "train_skipped_cold_src": int(train_samples["skipped_cold_src"]),
                "train_skipped_no_history_src": int(train_samples["skipped_no_history_src"]),
                "train_skipped_other": int(train_samples["skipped_other"]),
                "valid_samples": int(len(valid_samples["label"])) if valid_samples is not None else 0,
                "valid_skipped": int(valid_samples["skipped"]) if valid_samples is not None else 0,
                "valid_skipped_cold_dst": int(valid_samples["skipped_cold_dst"]) if valid_samples is not None else 0,
                "valid_skipped_cold_src": int(valid_samples["skipped_cold_src"]) if valid_samples is not None else 0,
                "valid_skipped_no_history_src": int(valid_samples["skipped_no_history_src"]) if valid_samples is not None else 0,
                "valid_skipped_other": int(valid_samples["skipped_other"]) if valid_samples is not None else 0,
                "final_train": bool(final_train),
                "include_test_vocab": bool(include_test_vocab),
                "unknown_policy": str(unknown_policy),
                "unknown_score": float(unknown_score),
                "unknown_margin": float(unknown_margin),
                "cold_prior_weight": float(cold_prior_weight),
                "fusion_model_weight": fusion_model_weight,
                "fusion_rule_weight": fusion_rule_weight,
            }
            _save_model(model_path, model, selected_meta)

    with open(artifact_dir / "model.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable_meta(selected_meta), f, indent=2, ensure_ascii=False)
    return _jsonable_meta(selected_meta or {})


def iter_dataset2_proba_chunks(dataset_dir, artifact_dir, batch_size=512, max_rows=0):
    model_json = Path(artifact_dir) / "model.json"
    if model_json.exists():
        with open(model_json, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("type") == "dataset2_pairwise_classifier":
            yield from iter_dataset2_pairwise_proba_chunks(dataset_dir, artifact_dir, batch_size=batch_size, max_rows=max_rows)
            return
        if meta.get("type") == "dataset2_feature_reranker":
            yield from iter_dataset2_feature_proba_chunks(dataset_dir, artifact_dir, batch_size=batch_size, max_rows=max_rows)
            return
        if meta.get("type") == "dataset2_listwise_feature_ranker":
            yield from iter_dataset2_feature_proba_chunks(dataset_dir, artifact_dir, batch_size=batch_size, max_rows=max_rows)
            return
    if not jittor_available():
        raise ImportError("Jittor is required for dataset2 temporal recommender prediction")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    model, meta = load_model(Path(artifact_dir) / "model.pkl")
    seq_len = int(meta.get("seq_len", 80))
    src_to_id = {int(k): int(v) for k, v in meta.get("src_to_id", {}).items()}
    dst_to_id = {int(k): int(v) for k, v in meta.get("dst_to_id", {}).items()}
    train_edges = list(iter_train_edges(Path(dataset_dir) / "train.csv"))
    pred_state = source_histories_for_prediction(train_edges, dst_to_id, seq_len)
    rule_model = FastDataset2RuleScorer(recent_limit=max(seq_len * 3, 200))
    rule_model.fit(train_edges)

    unknown_score = float(meta.get("unknown_score", -8.0))
    unknown_policy = str(meta.get("unknown_policy", "demote"))
    unknown_margin = float(meta.get("unknown_margin", 5.0))
    cold_prior_weight = float(meta.get("cold_prior_weight", 0.0))
    fusion_model_weight = float(meta.get("fusion_model_weight", 1.0))
    fusion_rule_weight = float(meta.get("fusion_rule_weight", 0.10))
    batch = []
    emitted = 0
    limit = max(int(max_rows or 0), 0)
    for row in iter_test_rows(Path(dataset_dir) / "test.csv"):
        if limit and emitted + len(batch) >= limit:
            break
        batch.append(row)
        if len(batch) >= batch_size:
            yield _score_query_batch(
                model,
                rule_model,
                pred_state,
                src_to_id,
                dst_to_id,
                seq_len,
                batch,
                unknown_score,
                unknown_policy,
                unknown_margin,
                cold_prior_weight,
                fusion_model_weight,
                fusion_rule_weight,
            )
            emitted += len(batch)
            batch = []
    if batch:
        yield _score_query_batch(
            model,
            rule_model,
            pred_state,
            src_to_id,
            dst_to_id,
            seq_len,
            batch,
            unknown_score,
            unknown_policy,
            unknown_margin,
            cold_prior_weight,
            fusion_model_weight,
            fusion_rule_weight,
        )


def _score_query_batch(
    model,
    rule_model,
    pred_state,
    src_to_id,
    dst_to_id,
    seq_len,
    chunk,
    unknown_score,
    unknown_policy,
    unknown_margin,
    cold_prior_weight,
    fusion_model_weight,
    fusion_rule_weight,
):
    src_values = [src for src, _, _ in chunk]
    src_ids = np.asarray([src_to_id.get(src, 0) for src in src_values], dtype=np.int32)
    hist_arr = np.zeros((len(chunk), seq_len), dtype=np.int32)
    gap_arr = np.zeros((len(chunk), seq_len), dtype=np.float32)
    time_arr = np.zeros((len(chunk), 3), dtype=np.float32)
    cand_arr = np.zeros((len(chunk), 100), dtype=np.int32)
    known_mask = np.zeros((len(chunk), 100), dtype=bool)
    valid_model_rows = np.zeros(len(chunk), dtype=bool)
    for row_idx, (src, time, candidates) in enumerate(chunk):
        hist_vec, gap_vec, hist_len = _history_arrays_for_src(pred_state, src, time, seq_len)
        hist_arr[row_idx] = hist_vec
        gap_arr[row_idx] = gap_vec
        valid_model_rows[row_idx] = src_ids[row_idx] > 0 and hist_len > 0
        time_arr[row_idx] = _time_feature(
            time,
            pred_state["last_time"].get(src),
            hist_len,
            pred_state["time_min"],
            pred_state["time_scale"],
            pred_state["gap_scale"],
            pred_state["hist_scale"],
        )
        for cand_idx, dst in enumerate(candidates):
            dst_id = dst_to_id.get(dst, 0)
            cand_arr[row_idx, cand_idx] = dst_id
            known_mask[row_idx, cand_idx] = dst_id > 0
    scores = model(
        jt.array(src_ids),
        jt.array(hist_arr),
        jt.array(gap_arr),
        jt.array(time_arr),
        jt.array(cand_arr),
    ).numpy().astype(np.float32)
    scores[~valid_model_rows, :] = 0.0
    scores = _apply_unknown_policy(
        scores,
        known_mask,
        chunk,
        policy=unknown_policy,
        fallback=float(unknown_score),
        margin=float(unknown_margin),
        cold_prior_weight=float(cold_prior_weight),
    )
    if fusion_rule_weight or fusion_model_weight != 1.0:
        fused = row_zscore(scores) * float(fusion_model_weight)
        if fusion_rule_weight:
            rule_scores = np.asarray(
                [rule_model.score_many(src, time, candidates) for src, time, candidates in chunk],
                dtype=np.float32,
            )
            fused = fused + row_zscore(rule_scores) * float(fusion_rule_weight)
        scores = _apply_unknown_policy(
            fused,
            known_mask,
            chunk,
            policy=unknown_policy,
            fallback=float(unknown_score),
            margin=float(unknown_margin),
            cold_prior_weight=float(cold_prior_weight),
        )
    return softmax(scores.astype(np.float32), temperature=1.0)


def _apply_unknown_policy(scores, known_mask, chunk, policy="demote", fallback=-8.0, margin=5.0, cold_prior_weight=0.0):
    policy = str(policy or "neutral").lower()
    scores = np.asarray(scores, dtype=np.float32).copy()
    known_mask = np.asarray(known_mask, dtype=bool)
    if known_mask.all() or policy in {"neutral", "none"}:
        if cold_prior_weight:
            return scores + _cold_id_prior(chunk, known_mask) * float(cold_prior_weight)
        return scores
    if policy in {"mild_penalty", "mild"}:
        scores[~known_mask] -= abs(float(margin))
    elif policy in {"boost_by_id_prior", "boost"}:
        scores += _cold_id_prior(chunk, known_mask) * max(float(cold_prior_weight), 0.25)
    elif policy in {"constant", "fallback"}:
        scores[~known_mask] = float(fallback)
    else:
        scores = _demote_unknown_scores(scores, known_mask, fallback=float(fallback), margin=float(margin))
    if cold_prior_weight and policy not in {"boost_by_id_prior", "boost"}:
        scores += _cold_id_prior(chunk, known_mask) * float(cold_prior_weight)
    return scores


def _cold_id_prior(chunk, known_mask):
    prior = np.zeros_like(np.asarray(known_mask, dtype=np.float32), dtype=np.float32)
    for row_idx, (_src, _time, candidates) in enumerate(chunk):
        row = np.asarray(candidates, dtype=np.float32)
        if row.size == 0:
            continue
        log_row = np.log1p(np.maximum(row, 0.0))
        scale = max(float(log_row.max() - log_row.min()), 1e-6)
        norm = (log_row - float(log_row.mean())) / scale
        norm = np.clip(norm, -1.0, 1.0)
        unknown = ~known_mask[row_idx]
        prior[row_idx, :len(candidates)] = norm
        prior[row_idx, ~unknown] *= 0.25
    return prior


def _demote_unknown_scores(scores, known_mask, fallback=-8.0, margin=5.0):
    scores = np.asarray(scores, dtype=np.float32).copy()
    known_mask = np.asarray(known_mask, dtype=bool)
    if known_mask.all():
        return scores
    for row_idx in range(scores.shape[0]):
        unknown = ~known_mask[row_idx]
        if not unknown.any():
            continue
        known = known_mask[row_idx]
        if known.any():
            floor = float(np.min(scores[row_idx, known])) - abs(float(margin))
            scores[row_idx, unknown] = min(float(fallback), floor)
        else:
            scores[row_idx, unknown] = 0.0
    return scores
