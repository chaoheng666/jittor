import math
from collections import Counter, defaultdict, deque


class FeatureBuilder:
    """Streaming temporal graph features for candidate reranking.

    The builder intentionally keeps all features graph-internal.  It supports
    the three signals used by the final scorer:
    - repeat/memory hazard for seen-heavy scenes;
    - source sequence transition and destination prior for new-link scenes;
    - TNCN-lite structural hints derived from recent temporal neighborhoods.
    """

    def __init__(
        self,
        recent_limit=120,
        cooc_window=8,
        cooc_recent_limit=300,
        cooc_topk=400,
    ):
        self.recent_limit = recent_limit
        self.cooc_window = cooc_window
        self.cooc_recent_limit = cooc_recent_limit
        self.cooc_topk = cooc_topk
        self._reset()

    def _reset(self):
        self.pair_count = Counter()
        self.pair_recent_count = Counter()
        self.pair_last_time = {}
        self.dst_count = Counter()
        self.dst_recent_count = Counter()
        self.dst_recent_count_10 = Counter()
        self.dst_recent_count_05 = Counter()
        self.dst_old_count = Counter()
        self.dst_last_time = {}
        self.dst_unique_src_count = Counter()
        self.src_count = Counter()
        self.src_unique_dst_count = Counter()
        self.src_repeat_rate = defaultdict(float)
        self.src_last_dst = {}
        self.src_recent = defaultdict(lambda: deque(maxlen=self.recent_limit))
        self.src_recent_times = defaultdict(lambda: deque(maxlen=self.recent_limit))
        self.src_recent_5 = {}
        self.src_recent_10 = {}
        self.src_recent_20 = {}
        self.src_recent_50 = {}
        self.src_recent_unique_20 = Counter()
        self.src_avg_gap = defaultdict(float)
        self.transition = defaultdict(Counter)
        self.reverse_transition = defaultdict(Counter)
        self.cooc_transition = defaultdict(Counter)
        self.reverse_cooc_transition = defaultdict(Counter)
        self.src_dst_sets = defaultdict(set)
        self.dst_src_sets = defaultdict(set)
        self.node_neighbors = defaultdict(set)
        self.min_time = 0
        self.max_time = 0
        self.recent_start = 0
        self.recent_start_10 = 0
        self.recent_start_05 = 0
        self.src_dst_overlap = 0
        self.repeat_edge_fraction = 0.0
        self.is_bipartite_like = False

    def fit(self, train_edges):
        self._reset()
        edges = sorted(train_edges, key=lambda x: x[2])
        if not edges:
            return

        self.min_time = edges[0][2]
        self.max_time = edges[-1][2]
        span = max(self.max_time - self.min_time, 1)
        self.recent_start = self.max_time - span // 5
        self.recent_start_10 = self.max_time - span // 10
        self.recent_start_05 = self.max_time - span // 20

        by_src = defaultdict(list)
        src_values = set()
        dst_values = set()

        for src, dst, time in edges:
            pair = (src, dst)
            self.pair_count[pair] += 1
            self.pair_last_time[pair] = time
            self.dst_count[dst] += 1
            self.dst_last_time[dst] = time
            self.src_count[src] += 1
            self.src_dst_sets[src].add(dst)
            self.dst_src_sets[dst].add(src)
            self.node_neighbors[src].add(dst)
            self.node_neighbors[dst].add(src)
            by_src[src].append((time, dst))
            src_values.add(src)
            dst_values.add(dst)
            if time >= self.recent_start:
                self.pair_recent_count[pair] += 1
                self.dst_recent_count[dst] += 1
            else:
                self.dst_old_count[dst] += 1
            if time >= self.recent_start_10:
                self.dst_recent_count_10[dst] += 1
            if time >= self.recent_start_05:
                self.dst_recent_count_05[dst] += 1

        repeated_extra = sum(count - 1 for count in self.pair_count.values())
        self.repeat_edge_fraction = repeated_extra / max(len(edges), 1)
        self.src_dst_overlap = len(src_values & dst_values)
        overlap_ratio = self.src_dst_overlap / max(min(len(src_values), len(dst_values)), 1)
        self.is_bipartite_like = overlap_ratio < 0.01

        for src, rows in by_src.items():
            rows.sort()
            gaps = []
            for idx, (time, dst) in enumerate(rows):
                self.src_last_dst[src] = dst
                self.src_recent[src].append(dst)
                self.src_recent_times[src].append(time)
                if idx > 0:
                    prev_time, prev_dst = rows[idx - 1]
                    gaps.append(max(time - prev_time, 0))
                    self.transition[prev_dst][dst] += 1
                    self.reverse_transition[dst][prev_dst] += 1
            if gaps:
                self.src_avg_gap[src] = sum(gaps) / len(gaps)
            self._add_window_cooc(rows)

        self._prune_nested_counter(self.transition, self.cooc_topk)
        self._prune_nested_counter(self.reverse_transition, self.cooc_topk)
        self._prune_nested_counter(self.cooc_transition, self.cooc_topk)
        self._prune_nested_counter(self.reverse_cooc_transition, self.cooc_topk)

        for src, dsts in self.src_dst_sets.items():
            unique_count = len(dsts)
            self.src_unique_dst_count[src] = unique_count
            total_count = self.src_count[src]
            if total_count:
                self.src_repeat_rate[src] = 1.0 - unique_count / total_count
        for dst, srcs in self.dst_src_sets.items():
            self.dst_unique_src_count[dst] = len(srcs)

        for src, recent in self.src_recent.items():
            values = list(recent)
            self.src_recent_5[src] = set(values[-5:])
            self.src_recent_10[src] = set(values[-10:])
            self.src_recent_20[src] = set(values[-20:])
            self.src_recent_50[src] = set(values[-50:])
            self.src_recent_unique_20[src] = len(set(values[-20:]))

    def _add_window_cooc(self, rows):
        if self.cooc_window <= 0:
            return
        if self.cooc_recent_limit:
            rows = rows[-self.cooc_recent_limit:]
        dsts = [dst for _, dst in rows]
        for i, dst in enumerate(dsts):
            start = max(0, i - self.cooc_window)
            for j in range(start, i):
                prev = dsts[j]
                if prev == dst:
                    continue
                weight = 1.0 / (i - j)
                self.cooc_transition[prev][dst] += weight
                self.reverse_cooc_transition[dst][prev] += weight

    @staticmethod
    def _prune_nested_counter(table, topk):
        if topk <= 0:
            return
        for key in list(table.keys()):
            counter = table[key]
            if len(counter) > topk:
                table[key] = Counter(dict(counter.most_common(topk)))

    def recency(self, last_time, current_time=None, scale=20.0):
        if last_time is None:
            return 0.0
        if current_time is None:
            current_time = self.max_time
        span = max(self.max_time - self.min_time, 1)
        age = max(current_time - last_time, 0) / span
        return 1.0 / (1.0 + age * scale)

    @staticmethod
    def _log_gap(current_time, last_time):
        if last_time is None:
            return 0.0
        return math.log1p(max(current_time - last_time, 0))

    def _recent_sequence_scores(self, recent, dst):
        same_decay = 0.0
        recent_rank_score = 0.0
        transition_score = 0.0
        reverse_transition_score = 0.0
        cooc_score = 0.0
        reverse_cooc_score = 0.0
        transition_hits = 0
        cooc_hits = 0
        temporal_cn = 0.0
        temporal_aa = 0.0
        temporal_ra = 0.0
        recent_cn = 0.0

        dst_degree = max(self.dst_count[dst], 1)
        for rank, hist_dst in enumerate(reversed(recent), start=1):
            discount = 1.0 / (rank ** 0.7)
            if hist_dst == dst:
                same_decay += discount
                if recent_rank_score == 0.0:
                    recent_rank_score = 1.0 / rank

            trans_value = self.transition[hist_dst][dst]
            rev_trans_value = self.reverse_transition[hist_dst][dst]
            cooc_value = self.cooc_transition[hist_dst][dst]
            reverse_cooc_value = self.reverse_cooc_transition[hist_dst][dst]
            combined = trans_value + rev_trans_value + cooc_value + reverse_cooc_value

            if trans_value:
                transition_hits += 1
                transition_score += math.log1p(trans_value) * discount
            if rev_trans_value:
                reverse_transition_score += math.log1p(rev_trans_value) * discount
            if cooc_value:
                cooc_hits += 1
                cooc_score += math.log1p(cooc_value) * discount
            if reverse_cooc_value:
                reverse_cooc_score += math.log1p(reverse_cooc_value) * discount
            if combined:
                recent_cn += 1.0 / rank
                temporal_cn += math.log1p(combined) * discount
                hist_degree = max(self.dst_count[hist_dst], 1)
                temporal_aa += combined / max(math.log1p(hist_degree + dst_degree), 1.0) * discount
                temporal_ra += combined / max(hist_degree + dst_degree, 1) * discount

        return {
            "recent_decay_count": same_decay,
            "recent_rank_score": recent_rank_score,
            "recent_transition_score": transition_score,
            "reverse_recent_transition_score": reverse_transition_score,
            "recent_cooc_score": cooc_score,
            "reverse_recent_cooc_score": reverse_cooc_score,
            "recent_transition_hits": math.log1p(transition_hits),
            "recent_cooc_hits": math.log1p(cooc_hits),
            "temporal_cn": temporal_cn,
            "temporal_aa": temporal_aa,
            "temporal_ra": temporal_ra,
            "recent_cn": recent_cn,
            "two_hop_overlap": math.log1p(transition_hits + cooc_hits),
            "shared_recent_neighbor": 1.0 if transition_hits or cooc_hits else 0.0,
        }

    def _small_common_neighbor_features(self, src, dst, max_scan=512):
        if self.is_bipartite_like:
            return 0.0, 0.0, 0.0
        left = self.node_neighbors.get(src, ())
        right = self.node_neighbors.get(dst, ())
        if not left or not right:
            return 0.0, 0.0, 0.0
        small, large = (left, right) if len(left) <= len(right) else (right, left)
        if len(small) > max_scan:
            return 0.0, 0.0, 0.0
        cn = 0
        aa = 0.0
        ra = 0.0
        for node in small:
            if node in large:
                cn += 1
                degree = max(len(self.node_neighbors.get(node, ())), 1)
                aa += 1.0 / max(math.log1p(degree), 1.0)
                ra += 1.0 / degree
        return math.log1p(cn), aa, ra

    def features(self, src, time, dst):
        pair = (src, dst)
        recent = list(self.src_recent.get(src, ()))
        dst_count = self.dst_count[dst]
        dst_old = self.dst_old_count[dst]
        dst_recent = self.dst_recent_count[dst]
        dst_recent_10 = self.dst_recent_count_10[dst]
        dst_recent_05 = self.dst_recent_count_05[dst]
        last_time = self.pair_last_time.get(pair)
        dst_last_time = self.dst_last_time.get(dst)
        recent_count = recent.count(dst)
        seq_scores = self._recent_sequence_scores(recent, dst)
        last_dst = self.src_last_dst.get(src)
        last_transition = self.transition[last_dst][dst] if last_dst is not None else 0
        last_reverse_transition = self.reverse_transition[last_dst][dst] if last_dst is not None else 0
        last_cooc = self.cooc_transition[last_dst][dst] if last_dst is not None else 0
        static_cn, static_aa, static_ra = self._small_common_neighbor_features(src, dst)
        src_degree = max(len(self.node_neighbors.get(src, ())), 0)
        dst_degree = max(len(self.node_neighbors.get(dst, ())), 0)

        feats = {
            "bias": 1.0,
            "has_pair": 1.0 if self.pair_count[pair] else 0.0,
            "is_new_pair": 0.0 if self.pair_count[pair] else 1.0,
            "pair_count": math.log1p(self.pair_count[pair]),
            "pair_recent_count": math.log1p(self.pair_recent_count[pair]),
            "pair_recency": self.recency(last_time, time) if last_time is not None else 0.0,
            "pair_time_gap": self._log_gap(time, last_time),
            "in_recent_5": 1.0 if dst in self.src_recent_5.get(src, ()) else 0.0,
            "in_recent_10": 1.0 if dst in self.src_recent_10.get(src, ()) else 0.0,
            "in_recent_20": 1.0 if dst in self.src_recent_20.get(src, ()) else 0.0,
            "in_recent_50": 1.0 if dst in self.src_recent_50.get(src, ()) else 0.0,
            "recent_count": math.log1p(recent_count),
            "recent_decay_count": seq_scores["recent_decay_count"],
            "recent_rank_score": seq_scores["recent_rank_score"],
            "is_last_dst": 1.0 if last_dst == dst else 0.0,
            "dst_seen": 1.0 if dst_count else 0.0,
            "dst_popularity": math.log1p(dst_count),
            "dst_recent_popularity": math.log1p(dst_recent),
            "dst_recent_popularity_10": math.log1p(dst_recent_10),
            "dst_recent_popularity_05": math.log1p(dst_recent_05),
            "dst_trend": math.log1p(dst_recent) - math.log1p(dst_old),
            "dst_trend_10": math.log1p(dst_recent_10) - math.log1p(max(dst_count - dst_recent_10, 0)),
            "src_activity": math.log1p(self.src_count[src]),
            "src_unique_dst": math.log1p(self.src_unique_dst_count[src]),
            "src_recent_unique_20": math.log1p(self.src_recent_unique_20[src]),
            "src_repeat_rate": self.src_repeat_rate[src],
            "src_avg_gap": math.log1p(self.src_avg_gap[src]),
            "is_cold_dst": 1.0 if dst_count == 0 else 0.0,
            "dst_recency": self.recency(dst_last_time, time) if dst_last_time is not None else 0.0,
            "dst_time_gap": self._log_gap(time, dst_last_time),
            "dst_unique_src": math.log1p(self.dst_unique_src_count[dst]),
            "last_transition": math.log1p(last_transition),
            "last_reverse_transition": math.log1p(last_reverse_transition),
            "last_cooc": math.log1p(last_cooc),
            "recent_transition_score": seq_scores["recent_transition_score"],
            "reverse_recent_transition_score": seq_scores["reverse_recent_transition_score"],
            "recent_transition_hits": seq_scores["recent_transition_hits"],
            "recent_cooc_score": seq_scores["recent_cooc_score"],
            "reverse_recent_cooc_score": seq_scores["reverse_recent_cooc_score"],
            "recent_cooc_hits": seq_scores["recent_cooc_hits"],
            "temporal_cn": seq_scores["temporal_cn"],
            "recent_cn": seq_scores["recent_cn"],
            "temporal_aa": seq_scores["temporal_aa"] + static_aa,
            "temporal_ra": seq_scores["temporal_ra"] + static_ra,
            "preferential_attachment": math.log1p(src_degree) * math.log1p(dst_degree),
            "two_hop_overlap": seq_scores["two_hop_overlap"],
            "shared_recent_neighbor": seq_scores["shared_recent_neighbor"],
            "static_common_neighbors": static_cn,
        }
        feats["item_transition"] = feats["last_transition"]
        return feats

    def query_features(self, src, time, candidates):
        candidates = list(candidates)
        if not candidates:
            candidates = [0]
        history = list(self.src_recent.get(src, ()))
        hits = sum(1 for dst in candidates if self.pair_count[(src, dst)] > 0)
        cold = sum(1 for dst in candidates if self.dst_count[dst] == 0)
        popularities = [math.log1p(self.dst_count[dst]) for dst in candidates]
        recent_pops = [math.log1p(self.dst_recent_count[dst]) for dst in candidates]
        unique_recent = len(set(history[-20:]))
        return {
            "query_bias": 1.0,
            "query_src_activity": math.log1p(self.src_count[src]),
            "query_src_unique_dst": math.log1p(self.src_unique_dst_count[src]),
            "query_src_repeat_rate": self.src_repeat_rate[src],
            "query_recent_unique_20": math.log1p(unique_recent),
            "query_candidate_pair_hit_ratio": hits / len(candidates),
            "query_candidate_cold_ratio": cold / len(candidates),
            "query_candidate_avg_popularity": sum(popularities) / len(popularities),
            "query_candidate_avg_recent_popularity": sum(recent_pops) / len(recent_pops),
            "query_repeat_edge_fraction": self.repeat_edge_fraction,
            "query_is_bipartite_like": 1.0 if self.is_bipartite_like else 0.0,
            "query_time_after_train": max(time - self.max_time, 0) / max(self.max_time - self.min_time, 1),
        }

    def history_arrays(self, src, time, node_to_idx, history_len):
        dsts = list(self.src_recent.get(src, ()))
        times = list(self.src_recent_times.get(src, ()))
        rows = [(dst, ts) for dst, ts in zip(dsts, times) if ts < time]
        rows = rows[-history_len:]
        rows.reverse()

        pad_idx = node_to_idx.get("__PAD__", 0)
        unk_idx = node_to_idx.get("__UNK__", 1)
        ids = [pad_idx] * history_len
        deltas = [0.0] * history_len
        mask = [0.0] * history_len
        span = max(self.max_time - self.min_time, 1)
        for i, (dst, ts) in enumerate(rows):
            ids[i] = node_to_idx.get(dst, unk_idx)
            deltas[i] = math.log1p(max(time - ts, 0) / span)
            mask[i] = 1.0
        return ids, deltas, mask
