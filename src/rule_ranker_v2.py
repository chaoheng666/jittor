import math

import numpy as np

from .feature_builder import FeatureBuilder


REPEAT_EDGE_WEIGHTS = {
    "bias": 0.0,
    "has_pair": 4.0,
    "pair_count": 6.5,
    "pair_recent_count": 2.0,
    "pair_recency": 5.5,
    "in_recent_5": 3.0,
    "in_recent_10": 2.0,
    "in_recent_20": 1.2,
    "in_recent_50": 0.7,
    "recent_count": 1.2,
    "recent_decay_count": 1.5,
    "recent_rank_score": 2.0,
    "is_last_dst": 2.5,
    "dst_seen": 0.2,
    "dst_popularity": 0.35,
    "dst_recent_popularity": 0.55,
    "dst_recent_popularity_10": 0.25,
    "dst_trend": 0.25,
    "dst_recency": 0.45,
    "dst_unique_src": 0.15,
    "src_repeat_rate": 0.8,
    "is_cold_dst": -2.0,
    "last_transition": 1.8,
    "last_cooc": 0.8,
    "recent_transition_score": 1.1,
    "recent_transition_hits": 0.4,
    "recent_cooc_score": 0.9,
    "recent_cooc_hits": 0.35,
    "reverse_recent_transition_score": 0.3,
    "reverse_recent_cooc_score": 0.25,
    "temporal_cn": 0.8,
    "recent_cn": 0.6,
    "temporal_aa": 0.4,
    "temporal_ra": 0.2,
    "preferential_attachment": 0.12,
    "two_hop_overlap": 0.35,
    "shared_recent_neighbor": 0.5,
    "static_common_neighbors": 0.25,
}


NEW_LINK_WEIGHTS = {
    "bias": 0.0,
    "has_pair": -2.8,
    "pair_count": -1.8,
    "pair_recent_count": -1.2,
    "pair_recency": -0.5,
    "in_recent_5": -1.2,
    "in_recent_10": -0.9,
    "in_recent_20": -0.6,
    "in_recent_50": -0.35,
    "recent_count": -0.4,
    "recent_decay_count": -0.4,
    "is_last_dst": -1.4,
    "is_new_pair": 0.8,
    "dst_seen": 2.4,
    "is_cold_dst": -5.5,
    "dst_popularity": 0.9,
    "dst_unique_src": 0.65,
    "dst_recent_popularity": 1.4,
    "dst_recent_popularity_10": 1.2,
    "dst_recent_popularity_05": 0.9,
    "dst_trend": 0.7,
    "dst_trend_10": 0.45,
    "dst_recency": 1.2,
    "src_activity": 0.05,
    "src_unique_dst": 0.05,
    "last_transition": 2.2,
    "last_reverse_transition": 0.15,
    "last_cooc": 1.6,
    "recent_transition_score": 2.8,
    "recent_transition_hits": 0.7,
    "recent_cooc_score": 2.2,
    "recent_cooc_hits": 0.65,
    "reverse_recent_transition_score": 0.25,
    "reverse_recent_cooc_score": 0.25,
    "temporal_cn": 1.5,
    "recent_cn": 1.0,
    "temporal_aa": 0.75,
    "temporal_ra": 0.35,
    "preferential_attachment": 0.2,
    "two_hop_overlap": 0.65,
    "shared_recent_neighbor": 0.7,
    "static_common_neighbors": 0.15,
}


DEFAULT_WEIGHTS = {
    "dataset1": REPEAT_EDGE_WEIGHTS,
    "dataset2": NEW_LINK_WEIGHTS,
}


class RuleRankerV2:
    """Rule-based future-edge rerank scorer.

    Scores how likely a candidate destination is to receive the next edge from
    a source at a given time, using only history before that time.
    """

    def __init__(self, dataset_name="dataset1", weights=None):
        self.dataset_name = dataset_name
        self._explicit_weights = weights is not None
        self.weights = dict(DEFAULT_WEIGHTS.get(dataset_name, {}))
        if weights:
            self.weights.update(weights)
        self.features = FeatureBuilder()

    def fit(self, train_edges):
        edges = list(train_edges)
        self.features.fit(edges)
        if not self._explicit_weights and not self.weights:
            self.weights = dict(self._auto_weights())

    def _auto_weights(self):
        fb = self.features
        if fb.is_bipartite_like or fb.repeat_edge_fraction < 0.08:
            return NEW_LINK_WEIGHTS
        return REPEAT_EDGE_WEIGHTS

    def score(self, src, time, dst):
        feats = self.features.features(src, time, dst)
        score = 0.0
        for name, weight in self.weights.items():
            score += weight * feats.get(name, 0.0)
        return float(score)

    def score_many(self, src, time, candidates):
        return [self.score(src, time, dst) for dst in candidates]

    def predict_proba(self, src, time, candidates):
        scores = np.asarray(self.score_many(src, time, candidates), dtype=np.float64)
        scores = scores - scores.max()
        exp_scores = np.exp(scores)
        total = exp_scores.sum()
        if not math.isfinite(total) or total <= 0:
            return [1.0 / len(candidates)] * len(candidates)
        return (exp_scores / total).tolist()
