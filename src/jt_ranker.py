import math

import numpy as np
import jittor as jt
from jittor import nn

from .feature_builder import FeatureBuilder
from .rule_ranker_v2 import DEFAULT_WEIGHTS, NEW_LINK_WEIGHTS, REPEAT_EDGE_WEIGHTS


FEATURE_NAMES = [
    "bias",
    "has_pair",
    "is_new_pair",
    "pair_count",
    "pair_recent_count",
    "pair_recency",
    "pair_time_gap",
    "in_recent_5",
    "in_recent_10",
    "in_recent_20",
    "in_recent_50",
    "recent_count",
    "recent_decay_count",
    "recent_rank_score",
    "is_last_dst",
    "dst_seen",
    "is_cold_dst",
    "dst_popularity",
    "dst_recent_popularity",
    "dst_recent_popularity_10",
    "dst_recent_popularity_05",
    "dst_trend",
    "dst_trend_10",
    "dst_recency",
    "dst_time_gap",
    "dst_unique_src",
    "src_activity",
    "src_unique_dst",
    "src_recent_unique_20",
    "src_repeat_rate",
    "src_avg_gap",
    "last_transition",
    "last_reverse_transition",
    "last_cooc",
    "recent_transition_score",
    "reverse_recent_transition_score",
    "recent_transition_hits",
    "recent_cooc_score",
    "reverse_recent_cooc_score",
    "recent_cooc_hits",
    "temporal_cn",
    "recent_cn",
    "temporal_aa",
    "temporal_ra",
    "preferential_attachment",
    "two_hop_overlap",
    "shared_recent_neighbor",
    "static_common_neighbors",
    "item_transition",
    "rule_score",
]


QUERY_FEATURE_NAMES = [
    "query_bias",
    "query_src_activity",
    "query_src_unique_dst",
    "query_src_repeat_rate",
    "query_recent_unique_20",
    "query_candidate_pair_hit_ratio",
    "query_candidate_cold_ratio",
    "query_candidate_avg_popularity",
    "query_candidate_avg_recent_popularity",
    "query_repeat_edge_fraction",
    "query_is_bipartite_like",
    "query_time_after_train",
]


class CandidateFeatureBuilder:
    def __init__(self, dataset_name, history_len=60, node_values=None):
        self.dataset_name = dataset_name
        self.history_len = history_len
        self.features = FeatureBuilder(recent_limit=max(120, history_len))
        self.rule_weights = dict(DEFAULT_WEIGHTS.get(dataset_name, {}))
        self.node_values = list(node_values) if node_values is not None else None
        self.node_to_idx = None
        if self.node_values is not None:
            self._build_node_index(self.node_values)

    def _build_node_index(self, node_values):
        self.node_values = list(node_values)
        self.node_to_idx = {"__PAD__": 0, "__UNK__": 1}
        for value in self.node_values:
            if value not in self.node_to_idx:
                self.node_to_idx[value] = len(self.node_to_idx)

    def fit(self, train_edges):
        edges = list(train_edges)
        self.features.fit(edges)
        if not self.rule_weights:
            if self.features.is_bipartite_like or self.features.repeat_edge_fraction < 0.08:
                self.rule_weights = dict(NEW_LINK_WEIGHTS)
            else:
                self.rule_weights = dict(REPEAT_EDGE_WEIGHTS)
        if self.node_to_idx is None:
            nodes = set()
            for src, dst, _ in edges:
                nodes.add(src)
                nodes.add(dst)
            self._build_node_index(sorted(nodes))

    @property
    def node_count(self):
        return len(self.node_to_idx or {"__PAD__": 0, "__UNK__": 1})

    def vector(self, src, time, dst):
        feats = self.features.features(src, time, dst)
        feats["rule_score"] = sum(
            weight * feats.get(name, 0.0)
            for name, weight in self.rule_weights.items()
        )
        return [float(feats.get(name, 0.0)) for name in FEATURE_NAMES]

    def query_vector(self, src, time, candidates):
        feats = self.features.query_features(src, time, candidates)
        return [float(feats.get(name, 0.0)) for name in QUERY_FEATURE_NAMES]

    def candidate_indices(self, candidates):
        unk = self.node_to_idx.get("__UNK__", 1)
        return [self.node_to_idx.get(dst, unk) for dst in candidates]

    def history(self, src, time):
        return self.features.history_arrays(src, time, self.node_to_idx, self.history_len)

    def arrays_for_queries(self, queries):
        x_rows = []
        cand_rows = []
        hist_rows = []
        delta_rows = []
        mask_rows = []
        query_rows = []
        for src, time, candidates in queries:
            x_rows.append([self.vector(src, time, dst) for dst in candidates])
            cand_rows.append(self.candidate_indices(candidates))
            hist_ids, hist_delta, hist_mask = self.history(src, time)
            hist_rows.append(hist_ids)
            delta_rows.append(hist_delta)
            mask_rows.append(hist_mask)
            query_rows.append(self.query_vector(src, time, candidates))
        return {
            "x": np.asarray(x_rows, dtype=np.float32),
            "candidate_idx": np.asarray(cand_rows, dtype=np.int32),
            "history_idx": np.asarray(hist_rows, dtype=np.int32),
            "history_delta": np.asarray(delta_rows, dtype=np.float32),
            "history_mask": np.asarray(mask_rows, dtype=np.float32),
            "query": np.asarray(query_rows, dtype=np.float32),
        }


def normalize_features(x, mean, std):
    return (x - mean) / std


def normalize_query_features(x, mean, std):
    return (x - mean) / std


class CraftRerankModel(nn.Module):
    """Candidate-aware temporal reranker with gated expert branches."""

    def __init__(
        self,
        feature_dim,
        query_dim,
        node_count,
        history_len=60,
        hidden_dim=128,
        embed_dim=64,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.query_dim = query_dim
        self.node_count = node_count
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.node_emb = nn.Embedding(node_count, embed_dim)
        self.pos_emb = nn.Embedding(history_len, embed_dim)
        self.hist_time_proj = nn.Linear(1, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.craft_head = nn.Sequential(
            nn.Linear(embed_dim * 3 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.repeat_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.structure_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.rule_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def execute(self, x, candidate_idx, history_idx, history_delta, history_mask, query):
        batch_size = x.shape[0]
        candidate_size = x.shape[1]
        history_len = history_idx.shape[1]

        flat_x = x.reshape(batch_size * candidate_size, x.shape[2])
        feat_h = self.feature_net(flat_x).reshape(batch_size, candidate_size, self.hidden_dim)

        cand = self.node_emb(candidate_idx.reshape(-1)).reshape(
            batch_size, candidate_size, self.embed_dim
        )
        hist = self.node_emb(history_idx.reshape(-1)).reshape(
            batch_size, history_len, self.embed_dim
        )
        pos_ids = jt.array(np.arange(history_len, dtype=np.int32)).reshape(1, history_len)
        pos = self.pos_emb(pos_ids)
        delta = self.hist_time_proj(history_delta.reshape(batch_size * history_len, 1)).reshape(
            batch_size, history_len, self.embed_dim
        )
        hist = hist + pos + delta

        q = self.query_proj(cand)
        k = self.key_proj(hist)
        v = self.value_proj(hist)
        attn = (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=3) / math.sqrt(float(self.embed_dim))
        attn = attn + (history_mask.unsqueeze(1) - 1.0) * 1e9
        attn = nn.softmax(attn, dim=2)
        context = (attn.unsqueeze(3) * v.unsqueeze(1)).sum(dim=2)

        craft_in = jt.concat([cand, context, cand * context, feat_h], dim=2)
        craft = self.craft_head(craft_in.reshape(batch_size * candidate_size, -1)).reshape(
            batch_size, candidate_size
        )
        repeat = self.repeat_head(feat_h.reshape(batch_size * candidate_size, -1)).reshape(
            batch_size, candidate_size
        )
        structure = self.structure_head(feat_h.reshape(batch_size * candidate_size, -1)).reshape(
            batch_size, candidate_size
        )
        rule = self.rule_head(feat_h.reshape(batch_size * candidate_size, -1)).reshape(
            batch_size, candidate_size
        )

        branches = jt.stack([craft, repeat, structure, rule], dim=2)
        gate = nn.softmax(self.gate_net(query), dim=1)
        scores = (branches * gate.unsqueeze(1)).sum(dim=2)
        return scores


def save_model(path, model, meta):
    jt.save({"state_dict": model.state_dict(), "meta": meta}, str(path))


def load_model(path):
    data = jt.load(str(path))
    meta = data["meta"]
    model = CraftRerankModel(
        feature_dim=meta["feature_dim"],
        query_dim=meta["query_dim"],
        node_count=meta["node_count"],
        history_len=meta.get("history_len", 60),
        hidden_dim=meta.get("hidden_dim", 128),
        embed_dim=meta.get("embed_dim", 64),
    )
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(data["state_dict"])
    else:
        model.load_parameters(data["state_dict"])
    model.eval()
    return model, meta
