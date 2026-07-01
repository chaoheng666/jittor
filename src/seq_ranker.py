import bisect
import math
from collections import defaultdict

import jittor as jt
from jittor import nn


class SequenceFeatureBuilder:
    def __init__(self, seq_len=50, max_time_bucket=63):
        self.seq_len = seq_len
        self.max_time_bucket = max_time_bucket
        self.by_src = defaultdict(list)
        self.by_src_times = {}
        self.dst_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.dst_values = []

    def fit(self, train_edges, extra_dsts=None):
        self.fit_history(train_edges)
        for _, dst, _ in train_edges:
            self._add_dst(dst)
        if extra_dsts:
            for dst in extra_dsts:
                self._add_dst(dst)
        self.dst_values = [None, None]
        values = sorted(dst for dst in self.dst_to_idx if isinstance(dst, int))
        for dst in values:
            idx = self.dst_to_idx[dst]
            if idx >= len(self.dst_values):
                self.dst_values.extend([None] * (idx - len(self.dst_values) + 1))
            self.dst_values[idx] = dst

    def load_dst_values(self, dst_values):
        self.dst_values = list(dst_values)
        self.dst_to_idx = {"<PAD>": 0, "<UNK>": 1}
        for idx, dst in enumerate(self.dst_values):
            if idx >= 2 and dst is not None:
                self.dst_to_idx[int(dst)] = idx

    def fit_history(self, train_edges):
        self.by_src.clear()
        self.by_src_times.clear()
        for src, dst, time in sorted(train_edges, key=lambda x: x[2]):
            self.by_src[src].append((time, dst))
        self.by_src_times = {
            src: [row[0] for row in rows]
            for src, rows in self.by_src.items()
        }

    def _add_dst(self, dst):
        if dst not in self.dst_to_idx:
            self.dst_to_idx[dst] = len(self.dst_to_idx)

    def dst_index(self, dst):
        return self.dst_to_idx.get(dst, 1)

    def build_query(self, src, time, candidates):
        rows = self.by_src.get(src, [])
        times = self.by_src_times.get(src, [])
        end = bisect.bisect_left(times, time)
        hist = rows[max(0, end - self.seq_len):end]

        seq_dst = [0] * self.seq_len
        seq_gap = [0] * self.seq_len
        start = self.seq_len - len(hist)
        for i, (hist_time, dst) in enumerate(hist, start=start):
            seq_dst[i] = self.dst_index(dst)
            gap = max(time - hist_time, 0)
            seq_gap[i] = min(int(math.log1p(gap)), self.max_time_bucket)

        cand_idx = [self.dst_index(dst) for dst in candidates]
        return seq_dst, seq_gap, cand_idx


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def execute(self, x, mask):
        h = jt.tanh(self.proj(x))
        score = self.attn(h).squeeze(-1)
        score = score + (mask - 1.0) * 10000.0
        weight = nn.softmax(score, dim=1).unsqueeze(-1)
        return (self.drop(x) * weight).sum(dim=1)


class SeqResidualRanker(nn.Module):
    def __init__(
        self, n_dst, feature_dim, seq_len=50, dst_emb_dim=128,
        time_emb_dim=32, hidden_dim=128, dropout=0.1, max_time_bucket=63
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dst_emb = nn.Embedding(n_dst, dst_emb_dim)
        self.time_emb = nn.Embedding(max_time_bucket + 1, time_emb_dim)
        seq_dim = dst_emb_dim + time_emb_dim
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.pool1 = AttentionPool(hidden_dim, dropout)
        self.ff1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ff2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cand_proj = nn.Linear(dst_emb_dim + feature_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def execute(self, seq_dst, seq_gap, cand_idx, cand_features):
        mask = (seq_dst > 0).float32()
        seq = jt.cat([self.dst_emb(seq_dst), self.time_emb(seq_gap)], dim=-1)
        seq = self.seq_proj(seq)
        pooled = self.pool1(seq, mask)
        pooled = pooled + self.ff1(pooled)
        pooled = pooled + self.ff2(pooled)

        cand_emb = self.dst_emb(cand_idx)
        cand = self.cand_proj(jt.cat([cand_emb, cand_features], dim=-1))
        pooled = pooled.unsqueeze(1) + jt.zeros_like(cand)

        q = self.query_proj(cand)
        k = self.key_proj(seq)
        v = self.value_proj(seq)
        attn_score = (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=-1) / math.sqrt(float(self.hidden_dim))
        attn_score = attn_score + (mask.unsqueeze(1) - 1.0) * 10000.0
        attn = nn.softmax(attn_score, dim=2)
        context = (attn.unsqueeze(-1) * v.unsqueeze(1)).sum(dim=2)
        dot = (cand * pooled).sum(dim=-1).unsqueeze(-1) / math.sqrt(float(self.hidden_dim))

        return self.out(jt.cat([pooled, cand, context, cand * context, dot], dim=-1)).squeeze(-1)


def save_seq_model(path, model, meta):
    jt.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_seq_model(path):
    data = jt.load(path)
    meta = data["meta"]
    model = SeqResidualRanker(
        meta["n_dst"],
        meta["feature_dim"],
        seq_len=meta.get("seq_len", 50),
        dst_emb_dim=meta.get("dst_emb_dim", 128),
        time_emb_dim=meta.get("time_emb_dim", 32),
        hidden_dim=meta.get("hidden_dim", 128),
        dropout=meta.get("dropout", 0.1),
        max_time_bucket=meta.get("max_time_bucket", 63),
    )
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(data["state_dict"])
    else:
        model.load_parameters(data["state_dict"])
    model.eval()
    return model, meta
