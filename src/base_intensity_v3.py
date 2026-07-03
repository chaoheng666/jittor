from .rule_ranker_v2 import RuleRankerV2


DEFAULT_BASE_WEIGHTS = {
    "dataset1": {
        "edge": 0.42,
        "pop": 0.15,
        "seq": 0.16,
        "struct": 0.12,
        "rule": 0.15,
        "cold": -1.75,
    },
    "dataset2": {
        "edge": 0.05,
        "pop": 0.35,
        "seq": 0.28,
        "struct": 0.10,
        "rule": 0.22,
        "cold": -2.35,
    },
}


class BaseIntensityV3:
    """Rule-compatible future-edge intensity scorer.

    This keeps the repository's strong statistical features as the main model
    and adds explicit decay/temporal-structure terms from FeatureBuilder.
    """

    def __init__(self, dataset_name="dataset1", weights=None):
        self.dataset_name = dataset_name
        self.rule = RuleRankerV2(dataset_name)
        self.weights = dict(DEFAULT_BASE_WEIGHTS.get(dataset_name, DEFAULT_BASE_WEIGHTS["dataset1"]))
        if weights:
            self.weights.update(weights)

    def fit(self, train_edges):
        self.rule.fit(train_edges)

    def _rule_score_from_features(self, feats):
        rule_score = 0.0
        for name, weight in self.rule.weights.items():
            rule_score += weight * feats.get(name, 0.0)
        return rule_score

    def _score_from_parts(self, parts):
        return float(
            self.weights.get("edge", 0.0) * parts["edge"]
            + self.weights.get("pop", 0.0) * parts["pop"]
            + self.weights.get("seq", 0.0) * parts["seq"]
            + self.weights.get("struct", 0.0) * parts["struct"]
            + self.weights.get("rule", 0.0) * parts["rule"]
            + self.weights.get("cold", 0.0) * parts["cold"]
        )

    def _score_parts_from_features(self, feats):
        rule_score = self._rule_score_from_features(feats)
        edge = (
            2.0 * feats.get("edge_decay", 0.0)
            + 1.4 * feats.get("pair_count", 0.0)
            + 1.8 * feats.get("pair_recent_count", 0.0)
            + 2.0 * feats.get("pair_recency", 0.0)
            + 1.0 * feats.get("recent_decay_count", 0.0)
            + 1.0 * feats.get("in_recent_5", 0.0)
            + 0.7 * feats.get("in_recent_20", 0.0)
        )
        pop = (
            0.8 * feats.get("dst_popularity", 0.0)
            + 1.2 * feats.get("dst_recent_popularity", 0.0)
            + 1.0 * feats.get("dst_recent_popularity_10", 0.0)
            + 0.7 * feats.get("dst_recent_popularity_05", 0.0)
            + 0.9 * feats.get("dst_pop_decay", 0.0)
            + 0.8 * feats.get("dst_trend", 0.0)
            + 0.5 * feats.get("dst_trend_10", 0.0)
            + 0.6 * feats.get("dst_recency", 0.0)
        )
        seq = (
            1.4 * feats.get("last_transition", 0.0)
            + 1.0 * feats.get("last_cooc", 0.0)
            + 1.4 * feats.get("recent_transition_score", 0.0)
            + 1.0 * feats.get("recent_cooc_score", 0.0)
            + 0.6 * feats.get("recent_transition_hits", 0.0)
            + 0.5 * feats.get("recent_cooc_hits", 0.0)
            + 0.4 * feats.get("reverse_recent_transition_score", 0.0)
        )
        struct = (
            1.0 * feats.get("temporal_cn", 0.0)
            + 0.6 * feats.get("temporal_aa", 0.0)
            + 0.4 * feats.get("temporal_ra", 0.0)
        )
        cold = feats.get("is_cold_dst", 0.0)
        return {
            "edge": edge,
            "pop": pop,
            "seq": seq,
            "struct": struct,
            "rule": rule_score,
            "cold": cold,
        }

    def score_parts(self, src, time, dst):
        feats = self.rule.features.features(src, time, dst)
        return self._score_parts_from_features(feats)

    def score(self, src, time, dst):
        parts = self.score_parts(src, time, dst)
        return self._score_from_parts(parts)

    def score_with_rule(self, src, time, dst):
        feats = self.rule.features.features(src, time, dst)
        parts = self._score_parts_from_features(feats)
        return self._score_from_parts(parts), float(parts["rule"])

    def score_many(self, src, time, candidates):
        return [self.score(src, time, dst) for dst in candidates]

    def score_many_with_rule(self, src, time, candidates):
        base_scores = []
        rule_scores = []
        for dst in candidates:
            base_score, rule_score = self.score_with_rule(src, time, dst)
            base_scores.append(base_score)
            rule_scores.append(rule_score)
        return base_scores, rule_scores
