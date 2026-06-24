import math
from collections import Counter, defaultdict, deque


class FeatureBuilder:
    def __init__(self, recent_limit=20):
        self.recent_limit = recent_limit
        self.pair_count = Counter()
        self.pair_last_time = {}
        self.dst_count = Counter()
        self.dst_recent_count = Counter()
        self.dst_old_count = Counter()
        self.dst_last_time = {}
        self.src_count = Counter()
        self.src_last_dst = {}
        self.src_recent = defaultdict(lambda: deque(maxlen=recent_limit))
        self.src_recent_5 = {}
        self.src_recent_10 = {}
        self.src_recent_20 = {}
        self.min_time = 0
        self.max_time = 0
        self.recent_start = 0

    def fit(self, train_edges):
        edges = sorted(train_edges, key=lambda x: x[2])
        if not edges:
            return

        self.min_time = edges[0][2]
        self.max_time = edges[-1][2]
        self.recent_start = self.max_time - (self.max_time - self.min_time) // 5

        for src, dst, time in edges:
            pair = (src, dst)
            self.pair_count[pair] += 1
            self.pair_last_time[pair] = time
            self.dst_count[dst] += 1
            self.dst_last_time[dst] = time
            self.src_count[src] += 1
            self.src_last_dst[src] = dst
            self.src_recent[src].append(dst)
            if time >= self.recent_start:
                self.dst_recent_count[dst] += 1
            else:
                self.dst_old_count[dst] += 1

        for src, recent in self.src_recent.items():
            values = list(recent)
            self.src_recent_5[src] = set(values[-5:])
            self.src_recent_10[src] = set(values[-10:])
            self.src_recent_20[src] = set(values[-20:])

    def recency(self, last_time):
        span = max(self.max_time - self.min_time, 1)
        age = max(self.max_time - last_time, 0) / span
        return 1.0 / (1.0 + age * 20.0)

    def features(self, src, time, dst):
        pair = (src, dst)
        recent = list(self.src_recent.get(src, ()))
        dst_count = self.dst_count[dst]
        dst_old = self.dst_old_count[dst]
        dst_recent = self.dst_recent_count[dst]
        last_time = self.pair_last_time.get(pair)
        dst_last_time = self.dst_last_time.get(dst)

        feats = {
            "bias": 1.0,
            "has_pair": 1.0 if self.pair_count[pair] else 0.0,
            "pair_count": math.log1p(self.pair_count[pair]),
            "pair_recency": self.recency(last_time) if last_time is not None else 0.0,
            "in_recent_5": 1.0 if dst in recent[-5:] else 0.0,
            "in_recent_10": 1.0 if dst in recent[-10:] else 0.0,
            "in_recent_20": 1.0 if dst in recent[-20:] else 0.0,
            "is_last_dst": 1.0 if self.src_last_dst.get(src) == dst else 0.0,
            "dst_popularity": math.log1p(dst_count),
            "dst_recent_popularity": math.log1p(dst_recent),
            "dst_trend": math.log1p(dst_recent) - math.log1p(dst_old),
            "src_activity": math.log1p(self.src_count[src]),
            "is_cold_dst": 1.0 if dst_count == 0 else 0.0,
            "dst_recency": self.recency(dst_last_time) if dst_last_time is not None else 0.0,
            "item_transition": 0.0,
        }
        return feats
