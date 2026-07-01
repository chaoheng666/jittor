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
    "row_pair_rank",
    "row_dst_pop_rank",
    "row_dst_recent_rank",
    "row_dst_recency_rank",
    "row_transition_rank",
    "row_cooc_rank",
    "row_rule_rank",
    "row_is_rule_top1",
]


class CandidateFeatureBuilder:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.features = FeatureBuilder()
        self.rule_ranker = RuleRankerV2(dataset_name)

    def fit(self, train_edges):
        edges = list(train_edges)
        self.features.fit(edges)
        self.rule_ranker.use_feature_builder(self.features)

    def vector(self, src, time, dst):
        feats = self.features.features(src, time, dst)
        feats["rule_score"] = self.rule_ranker.score_from_features(feats)
        return [float(feats.get(name, 0.0)) for name in FEATURE_NAMES]

    def matrix(self, src, time, candidates):
        rows = [self.vector(src, time, dst) for dst in candidates]
        self._add_row_context(rows)
        return rows

    @staticmethod
    def _rank_percentiles(values):
        n = len(values)
        if n <= 1:
            return [1.0] * n
        order = sorted(range(n), key=lambda i: (values[i], i))
        ranks = [0.0] * n
        for rank, idx in enumerate(order):
            ranks[idx] = rank / (n - 1)
        return ranks

    def _add_row_context(self, rows):
        rank_pairs = [
            ("pair_count", "row_pair_rank"),
            ("dst_popularity", "row_dst_pop_rank"),
            ("dst_recent_popularity", "row_dst_recent_rank"),
            ("dst_recency", "row_dst_recency_rank"),
            ("recent_transition_score", "row_transition_rank"),
            ("recent_cooc_score", "row_cooc_rank"),
            ("rule_score", "row_rule_rank"),
        ]
        for source_name, target_name in rank_pairs:
            source_idx = FEATURE_NAMES.index(source_name)
            target_idx = FEATURE_NAMES.index(target_name)
            values = [row[source_idx] for row in rows]
            ranks = self._rank_percentiles(values)
            for row, rank in zip(rows, ranks):
                row[target_idx] = rank

        rule_idx = FEATURE_NAMES.index("rule_score")
        top_idx = FEATURE_NAMES.index("row_is_rule_top1")
        if rows:
            best = max(range(len(rows)), key=lambda i: rows[i][rule_idx])
            rows[best][top_idx] = 1.0


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
