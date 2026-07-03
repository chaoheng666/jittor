import jittor as jt
from jittor import nn

from .feature_builder import FeatureBuilder
from .rule_ranker_v2 import RuleRankerV2


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
    "item_transition",
    "rule_score",
]


class CandidateFeatureBuilder:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.features = FeatureBuilder()
        self.rule_ranker = RuleRankerV2(dataset_name)

    def fit(self, train_edges):
        edges = list(train_edges)
        self.features.fit(edges)
        self.rule_ranker.fit(edges)

    def vector(self, src, time, dst):
        feats = self.features.features(src, time, dst)
        feats["rule_score"] = self.rule_ranker.score(src, time, dst)
        return [float(feats.get(name, 0.0)) for name in FEATURE_NAMES]


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
    jt.save({"state_dict": model.state_dict(), "meta": meta}, str(path))


def load_model(path):
    data = jt.load(str(path))
    meta = data["meta"]
    model = MLPRanker(meta["feature_dim"], meta.get("hidden_dim", 64))
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(data["state_dict"])
    else:
        model.load_parameters(data["state_dict"])
    model.eval()
    return model, meta
