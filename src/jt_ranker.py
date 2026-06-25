import math
from collections import Counter, defaultdict

import jittor as jt
from jittor import nn

from .feature_builder import FeatureBuilder
from .rule_ranker_v2 import RuleRankerV2


FEATURE_NAMES = [
    "bias",
    "has_pair",
    "pair_count",
    "pair_recency",
    "in_recent_5",
    "in_recent_10",
    "in_recent_20",
    "recent_count",
    "recent_rank_score",
    "is_last_dst",
    "dst_popularity",
    "dst_recent_popularity",
    "dst_trend",
    "src_activity",
    "is_cold_dst",
    "dst_recency",
    "item_transition",
    "rule_score",
    "pair_recent_count",
    "pair_time_gap",
    "dst_time_gap",
    "src_unique_dst",
    "dst_unique_src",
    "src_repeat_rate",
]


class CandidateFeatureBuilder:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.features = FeatureBuilder()
        self.rule_ranker = RuleRankerV2(dataset_name)
        self.transition = defaultdict(Counter)

    def fit(self, train_edges):
        edges = list(train_edges)
        self.features.fit(edges)
        self.rule_ranker.fit(edges)

        by_src = defaultdict(list)
        for src, dst, time in edges:
            by_src[src].append((time, dst))
        for rows in by_src.values():
            rows.sort()
            for i in range(1, len(rows)):
                self.transition[rows[i - 1][1]][rows[i][1]] += 1

    def vector(self, src, time, dst):
        feats = self.features.features(src, time, dst)
        last_dst = self.features.src_last_dst.get(src)
        if last_dst is not None:
            feats["item_transition"] = math.log1p(self.transition[last_dst][dst])
        feats["rule_score"] = math.log1p(self.rule_ranker.score(src, time, dst))
        return [float(feats.get(name, 0.0)) for name in FEATURE_NAMES]

    def matrix(self, src, time, candidates):
        return [self.vector(src, time, dst) for dst in candidates]


class MLPRanker(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def execute(self, x):
        batch_size = x.shape[0]
        candidate_size = x.shape[1]
        x = x.reshape(batch_size * candidate_size, x.shape[2])
        scores = self.net(x)
        return scores.reshape(batch_size, candidate_size)


def normalize_features(x, mean, std):
    return (x - mean) / std


def save_model(path, model, meta):
    jt.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_model(path):
    data = jt.load(path)
    meta = data["meta"]
    model = MLPRanker(meta["feature_dim"], meta.get("hidden_dim", 64))
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(data["state_dict"])
    else:
        model.load_parameters(data["state_dict"])
    model.eval()
    return model, meta
