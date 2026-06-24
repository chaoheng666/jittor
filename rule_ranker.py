import math
from collections import Counter, defaultdict


class RuleRanker:
    def __init__(self):
        self.pair_count = Counter()
        self.pair_recent_count = Counter()
        self.pair_last_time = {}
        self.dst_popularity = Counter()
        self.recent_dst_popularity = Counter()
        self.dst_last_time = {}
        self.src_history = defaultdict(set)
        self.min_time = 0
        self.max_time = 0
        self.recent_start = 0

    def fit(self, train_edges):
        edges = []
        for src, dst, time in train_edges:
            edges.append((src, dst, time))
            if time > self.max_time:
                self.max_time = time

        if edges:
            self.min_time = min(time for _, _, time in edges)
            self.recent_start = self.max_time - (self.max_time - self.min_time) // 5

        for src, dst, time in edges:
            pair = (src, dst)
            self.pair_count[pair] += 1
            if time > self.pair_last_time.get(pair, -1):
                self.pair_last_time[pair] = time
            if time > self.dst_last_time.get(dst, -1):
                self.dst_last_time[dst] = time
            self.dst_popularity[dst] += 1
            self.src_history[src].add(dst)
            if time >= self.recent_start:
                self.pair_recent_count[pair] += 1
                self.recent_dst_popularity[dst] += 1

    def recency(self, last_time):
        span = max(self.max_time - self.min_time, 1)
        age = max(self.max_time - last_time, 0) / span
        return 1.0 / (1.0 + age * 20.0)

    def score(self, src, time, dst):
        pair = (src, dst)
        score = 0.0
        pair_count = self.pair_count[pair]
        if pair_count:
            score += math.log1p(pair_count) * 8.0
            score += math.log1p(self.pair_recent_count[pair]) * 4.0

        last_time = self.pair_last_time.get(pair)
        if last_time is not None:
            score += self.recency(last_time) * 6.0

        score += math.log1p(self.dst_popularity[dst]) * 0.8
        score += math.log1p(self.recent_dst_popularity[dst]) * 1.2

        dst_last_time = self.dst_last_time.get(dst)
        if dst_last_time is not None:
            score += self.recency(dst_last_time) * 0.8

        if dst in self.src_history.get(src, ()):
            score += 1.0
        return score

    def predict_proba(self, src, time, candidates):
        scores = [self.score(src, time, dst) for dst in candidates]
        total = sum(scores)
        if total <= 0:
            return [1.0 / len(candidates)] * len(candidates)
        return [s / total for s in scores]
