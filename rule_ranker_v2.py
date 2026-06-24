import math
from collections import Counter, defaultdict

from feature_builder import FeatureBuilder


DEFAULT_WEIGHTS = {
    "dataset1": {
        "has_pair": 4.0,
        "pair_count": 8.0,
        "pair_recency": 6.0,
        "in_recent_5": 3.0,
        "in_recent_10": 2.0,
        "in_recent_20": 1.0,
        "is_last_dst": 3.0,
        "dst_popularity": 0.4,
        "dst_recent_popularity": 0.6,
        "dst_recency": 0.5,
        "item_transition": 2.0,
    },
    "dataset2": {
        "has_pair": 1.0,
        "pair_count": 2.0,
        "pair_recency": 1.5,
        "dst_popularity": 1.8,
        "dst_recent_popularity": 2.2,
        "dst_trend": 1.0,
        "src_activity": 0.2,
        "is_cold_dst": -1.0,
        "dst_recency": 1.0,
        "item_transition": 2.5,
    },
}


class RuleRankerV2:
    def __init__(self, dataset_name="dataset1", weights=None):
        self.dataset_name = dataset_name
        self.weights = dict(DEFAULT_WEIGHTS.get(dataset_name, {}))
        if weights:
            self.weights.update(weights)
        self.features = FeatureBuilder()
        self.transition = defaultdict(Counter)

    def fit(self, train_edges):
        edges = list(train_edges)
        self.features.fit(edges)
        by_src = defaultdict(list)
        for src, dst, time in edges:
            by_src[src].append((time, dst))
        for rows in by_src.values():
            rows.sort()
            for i in range(1, len(rows)):
                self.transition[rows[i - 1][1]][rows[i][1]] += 1

    def score(self, src, time, dst):
        fb = self.features
        pair = (src, dst)
        score = self.weights.get("bias", 0.0)

        pair_count = fb.pair_count[pair]
        if pair_count:
            score += self.weights.get("has_pair", 0.0)
            score += self.weights.get("pair_count", 0.0) * math.log1p(pair_count)
            score += self.weights.get("pair_recency", 0.0) * fb.recency(fb.pair_last_time[pair])

        if dst in fb.src_recent_5.get(src, ()):
            score += self.weights.get("in_recent_5", 0.0)
        if dst in fb.src_recent_10.get(src, ()):
            score += self.weights.get("in_recent_10", 0.0)
        if dst in fb.src_recent_20.get(src, ()):
            score += self.weights.get("in_recent_20", 0.0)
        if fb.src_last_dst.get(src) == dst:
            score += self.weights.get("is_last_dst", 0.0)

        dst_count = fb.dst_count[dst]
        dst_recent = fb.dst_recent_count[dst]
        dst_old = fb.dst_old_count[dst]
        score += self.weights.get("dst_popularity", 0.0) * math.log1p(dst_count)
        score += self.weights.get("dst_recent_popularity", 0.0) * math.log1p(dst_recent)
        trend = math.log1p(dst_recent) - math.log1p(dst_old)
        score += self.weights.get("dst_trend", 0.0) * trend
        score += self.weights.get("src_activity", 0.0) * math.log1p(fb.src_count[src])
        if dst_count == 0:
            score += self.weights.get("is_cold_dst", 0.0)
        dst_last_time = fb.dst_last_time.get(dst)
        if dst_last_time is not None:
            score += self.weights.get("dst_recency", 0.0) * fb.recency(dst_last_time)

        last_dst = self.features.src_last_dst.get(src)
        if last_dst is not None:
            score += self.weights.get("item_transition", 0.0) * self.transition[last_dst][dst]
        return max(score, 0.0)

    def predict_proba(self, src, time, candidates):
        scores = [self.score(src, time, dst) for dst in candidates]
        total = sum(scores)
        if total <= 0:
            return [1.0 / len(candidates)] * len(candidates)
        return [s / total for s in scores]
