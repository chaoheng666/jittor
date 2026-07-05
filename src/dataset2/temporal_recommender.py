import json
import math
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


def build_id_maps(edges):
    src_values = sorted({src for src, _, _ in edges})
    dst_values = sorted({dst for _, dst, _ in edges})
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


def build_samples(history_edges, supervision_edges, src_to_id, dst_to_id, seq_len, update_with_supervision=True):
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

    for src, dst, time in sorted(supervision_edges, key=lambda x: x[2]):
        src_id = src_to_id.get(src, 0)
        dst_id = dst_to_id.get(dst, 0)
        hist = list(history_by_src.get(src, ()))
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
            self.dst_features = jt.array(dst_features)
            self.pos_weights = jt.array(np.linspace(0.25, 1.0, self.seq_len, dtype=np.float32)).reshape((1, -1))
            self.src_emb = nn.Embedding(num_src + 1, emb_dim)
            self.dst_emb = nn.Embedding(num_dst + 1, emb_dim)
            self.dst_bias = nn.Embedding(num_dst + 1, 1)
            self.dst_feat_proj = nn.Sequential(
                nn.Linear(dst_features.shape[1], emb_dim),
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
            return self.dst_emb(dst_ids) + self.dst_feat_proj(self.dst_features[dst_ids])

        def all_dst_repr(self):
            return self.dst_emb.weight[1:] + self.dst_feat_proj(self.dst_features[1:])

        def encode_state(self, src_ids, hist_ids, hist_gaps, time_feats):
            src_vec = self.src_emb(src_ids)
            hist_mask = (hist_ids > 0).float32()
            gap_vec = self.gap_proj(hist_gaps.unsqueeze(-1))
            hist_emb = (self.dst_repr(hist_ids) + gap_vec) * hist_mask.unsqueeze(-1)
            pos_weights = self.pos_weights
            weighted_mask = hist_mask * pos_weights
            denom = jt.maximum(weighted_mask.sum(dim=1, keepdims=True), jt.ones((hist_ids.shape[0], 1)))
            hist_vec = (hist_emb * pos_weights.unsqueeze(-1)).sum(dim=1) / denom
            recent_vec = hist_emb[:, -1, :]
            time_vec = self.time_proj(time_feats)
            return self.state_proj(jt.concat([src_vec, hist_vec, recent_vec, time_vec], dim=1))

        def execute(self, src_ids, hist_ids, hist_gaps, time_feats, cand_ids=None):
            state = self.encode_state(src_ids, hist_ids, hist_gaps, time_feats)
            if cand_ids is None:
                dst_weight = self.all_dst_repr()
                bias = self.dst_bias.weight[1:].reshape((1, -1))
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
    feature_dim = 5
    features = np.zeros((len(dst_to_id) + 1, feature_dim), dtype=np.float32)
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


def train_dataset2(
    dataset_dir,
    artifact_dir,
    final_train=False,
    cuda=True,
    softmax_mode="sampled",
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
    hard_negative_count=512,
    sampled_correction=True,
    rerank_neg_count=64,
    rerank_weight=0.10,
    fusion_model_weight=1.0,
    fusion_rule_weight=0.10,
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
    src_to_id, dst_to_id = build_id_maps(train_edges)
    train_samples = build_samples([], train_edges, src_to_id, dst_to_id, seq_len, update_with_supervision=True)
    valid_samples = None
    if valid_edges:
        valid_samples = build_samples(train_edges, valid_edges, src_to_id, dst_to_id, seq_len, update_with_supervision=False)
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
                loss = nn.cross_entropy_loss(logits, jt.array(labels.astype(np.int32)))
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
                loss = nn.cross_entropy_loss(logits, jt.array(label_positions))
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
            score_for_selection = metrics["overall"]["mrr"]
        else:
            score_for_selection = float(epoch)
        print(
            f"dataset2: epoch={epoch} loss={loss_sum / max(steps, 1):.6f} "
            f"overall_mrr={metrics['overall']['mrr']:.6f} "
            f"repeated_mrr={metrics['repeated']['mrr']:.6f} "
            f"new_pair_mrr={metrics['new_pair']['mrr']:.6f} "
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
                "unknown_score": -8.0,
                "fusion_model_weight": fusion_model_weight,
                "fusion_rule_weight": fusion_rule_weight,
            }
            _save_model(model_path, model, selected_meta)

    with open(artifact_dir / "model.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable_meta(selected_meta), f, indent=2, ensure_ascii=False)
    return selected_meta or {}


def iter_dataset2_proba_chunks(dataset_dir, artifact_dir, batch_size=512, max_rows=0):
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
    scores = _demote_unknown_scores(scores, known_mask, fallback=float(unknown_score), margin=5.0)
    if fusion_rule_weight or fusion_model_weight != 1.0:
        fused = row_zscore(scores) * float(fusion_model_weight)
        if fusion_rule_weight:
            rule_scores = np.asarray(
                [rule_model.score_many(src, time, candidates) for src, time, candidates in chunk],
                dtype=np.float32,
            )
            fused = fused + row_zscore(rule_scores) * float(fusion_rule_weight)
        scores = _demote_unknown_scores(fused, known_mask, fallback=float(unknown_score), margin=5.0)
    return softmax(scores.astype(np.float32), temperature=1.0)


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
