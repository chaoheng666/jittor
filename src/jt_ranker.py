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
    "dst_recent_popularity_w01",
    "dst_recent_popularity_w02",
    "dst_recent_popularity_w05",
    "dst_recent_popularity_w10",
    "dst_recent_popularity_w20",
    "dst_trend",
    "dst_trend_10",
    "dst_trend_w01",
    "dst_trend_w02",
    "dst_trend_w05",
    "dst_trend_w10",
    "dst_trend_w20",
    "dst_recency",
    "dst_time_gap",
    "dst_unique_src",
    "src_activity",
    "src_unique_dst",
    "src_recent_unique_20",
    "src_repeat_rate",
    "src_avg_gap",
    "last_transition",
    "last_transition_prob",
    "last_reverse_transition",
    "last_cooc",
    "last_cooc_prob",
    "recent_transition_score",
    "reverse_recent_transition_score",
    "recent_transition_hits",
    "recent_cooc_score",
    "reverse_recent_cooc_score",
    "recent_cooc_hits",
    "recent_transition_prob_score",
    "recent_transition_pmi_score",
    "recent_cooc_prob_score",
    "recent_cooc_pmi_score",
    "recent_dst_source_overlap",
    "item_transition",
    "pair_recent_count_w01",
    "pair_recent_count_w02",
    "pair_recent_count_w05",
    "pair_recent_count_w10",
    "pair_recent_count_w20",
    "pair_trend_w01",
    "pair_trend_w02",
    "pair_trend_w05",
    "pair_trend_w10",
    "pair_trend_w20",
    "test_candidate_count",
    "test_candidate_rank",
    "test_candidate_seen",
    "rule_score",
    "row_pair_rank",
    "row_dst_pop_rank",
    "row_dst_recent_rank",
    "row_dst_recency_rank",
    "row_transition_rank",
    "row_cooc_rank",
    "row_rule_rank",
    "row_rule_zscore",
    "row_rule_top1_gap",
    "row_rule_top3_gap",
    "row_pair_zscore",
    "row_dst_pop_zscore",
    "row_dst_recent_zscore",
    "row_transition_zscore",
    "row_cooc_zscore",
    "row_test_prior_rank",
    "row_test_prior_zscore",
    "row_cold_fraction",
    "row_is_rule_top1",
]


class CandidateFeatureBuilder:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.features = FeatureBuilder()
        self.rule_ranker = RuleRankerV2(dataset_name)
        self.test_candidate_count = {}
        self.test_candidate_rank = {}
        self.test_candidate_total = 0

    def fit(self, train_edges):
        edges = list(train_edges)
        self.features.fit(edges)
        self.rule_ranker.use_feature_builder(self.features)

    def fit_candidate_priors(self, candidate_rows):
        counts = {}
        total = 0
        for row in candidate_rows:
            candidates = row[-1]
            for dst in candidates:
                counts[dst] = counts.get(dst, 0) + 1
                total += 1
        self.test_candidate_count = counts
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        denom = max(len(ranked) - 1, 1)
        self.test_candidate_rank = {
            dst: 1.0 - rank / denom
            for rank, (dst, _) in enumerate(ranked)
        }
        self.test_candidate_total = total

    def vector(self, src, time, dst):
        feats = self.features.features(src, time, dst)
        candidate_count = self.test_candidate_count.get(dst, 0)
        feats["test_candidate_count"] = float(candidate_count)
        feats["test_candidate_rank"] = float(self.test_candidate_rank.get(dst, 0.0))
        feats["test_candidate_seen"] = 1.0 if candidate_count else 0.0
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

    @staticmethod
    def _zscores(values):
        n = len(values)
        if n == 0:
            return []
        mean = sum(values) / n
        var = sum((value - mean) ** 2 for value in values) / n
        std = var ** 0.5
        if std < 1e-6:
            return [0.0] * n
        return [(value - mean) / std for value in values]

    @staticmethod
    def _top_gap(values, top_k):
        if not values:
            return []
        ordered = sorted(values, reverse=True)
        top_idx = min(top_k - 1, len(ordered) - 1)
        threshold = ordered[top_idx]
        return [threshold - value for value in values]

    def _add_row_context(self, rows):
        rank_pairs = [
            ("pair_count", "row_pair_rank"),
            ("dst_popularity", "row_dst_pop_rank"),
            ("dst_recent_popularity", "row_dst_recent_rank"),
            ("dst_recency", "row_dst_recency_rank"),
            ("recent_transition_score", "row_transition_rank"),
            ("recent_cooc_score", "row_cooc_rank"),
            ("test_candidate_count", "row_test_prior_rank"),
            ("rule_score", "row_rule_rank"),
        ]
        for source_name, target_name in rank_pairs:
            source_idx = FEATURE_NAMES.index(source_name)
            target_idx = FEATURE_NAMES.index(target_name)
            values = [row[source_idx] for row in rows]
            ranks = self._rank_percentiles(values)
            for row, rank in zip(rows, ranks):
                row[target_idx] = rank

        zscore_pairs = [
            ("rule_score", "row_rule_zscore"),
            ("pair_count", "row_pair_zscore"),
            ("dst_popularity", "row_dst_pop_zscore"),
            ("dst_recent_popularity", "row_dst_recent_zscore"),
            ("recent_transition_score", "row_transition_zscore"),
            ("recent_cooc_score", "row_cooc_zscore"),
            ("test_candidate_count", "row_test_prior_zscore"),
        ]
        for source_name, target_name in zscore_pairs:
            source_idx = FEATURE_NAMES.index(source_name)
            target_idx = FEATURE_NAMES.index(target_name)
            values = [row[source_idx] for row in rows]
            zscores = self._zscores(values)
            for row, zscore in zip(rows, zscores):
                row[target_idx] = zscore

        if rows:
            rule_idx = FEATURE_NAMES.index("rule_score")
            gap1_idx = FEATURE_NAMES.index("row_rule_top1_gap")
            gap3_idx = FEATURE_NAMES.index("row_rule_top3_gap")
            values = [row[rule_idx] for row in rows]
            gap1 = self._top_gap(values, 1)
            gap3 = self._top_gap(values, 3)
            for row, value1, value3 in zip(rows, gap1, gap3):
                row[gap1_idx] = value1
                row[gap3_idx] = value3

            cold_idx = FEATURE_NAMES.index("is_cold_dst")
            row_cold_idx = FEATURE_NAMES.index("row_cold_fraction")
            cold_fraction = sum(row[cold_idx] for row in rows) / len(rows)
            for row in rows:
                row[row_cold_idx] = cold_fraction

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
