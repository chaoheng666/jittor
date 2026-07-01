import math
from collections import Counter, defaultdict, deque


class FeatureBuilder:
    """Time-aware candidate features for candidate reranking.

    The two public A-list scenes behave very differently:
    - dataset1 has many repeated edges, so pair memory is very strong.
    - dataset2 is bipartite and the official split contains new source-dst
      links, so item-to-item dynamics and destination freshness matter more.

    This builder keeps both families of signals and lets the rule/model layer
    decide how to weight them per dataset.
    """

    def __init__(
        self,
        recent_limit=100,
        cooc_window=8,
        cooc_recent_limit=300,
        cooc_topk=300,
        window_fractions=(0.01, 0.02, 0.05, 0.10, 0.20),
    ):
        self.recent_limit = recent_limit
        self.cooc_window = cooc_window
        self.cooc_recent_limit = cooc_recent_limit
        self.cooc_topk = cooc_topk
        self.window_fractions = tuple(window_fractions)

        self.pair_count = Counter()
        self.pair_recent_count = Counter()
        self.pair_window_count = {fraction: Counter() for fraction in self.window_fractions}
        self.pair_last_time = {}
        self.dst_count = Counter()
        self.dst_recent_count = Counter()
        self.dst_recent_count_10 = Counter()
        self.dst_recent_count_05 = Counter()
        self.dst_window_count = {fraction: Counter() for fraction in self.window_fractions}
        self.dst_old_count = Counter()
        self.dst_last_time = {}
        self.dst_unique_src_count = Counter()
        self.dst_src_sets = {}
        self.src_count = Counter()
        self.src_unique_dst_count = Counter()
        self.src_repeat_rate = defaultdict(float)
        self.src_last_dst = {}
        self.src_recent = defaultdict(lambda: deque(maxlen=recent_limit))
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
        self.transition_total = Counter()
        self.transition_event_total = 0
        self.cooc_total = Counter()
        self.cooc_event_total = 0.0
        self.min_time = 0
        self.max_time = 0
        self.recent_start = 0
        self.recent_start_10 = 0
        self.recent_start_05 = 0
        self.window_starts = {}
        self.src_dst_overlap = 0
        self.repeat_edge_fraction = 0.0
        self.is_bipartite_like = False

    def fit(self, train_edges):
        edges = sorted(train_edges, key=lambda x: x[2])
        if not edges:
            return

        self.min_time = edges[0][2]
        self.max_time = edges[-1][2]
        span = max(self.max_time - self.min_time, 1)
        self.recent_start = self.max_time - span // 5
        self.recent_start_10 = self.max_time - span // 10
        self.recent_start_05 = self.max_time - span // 20
        self.window_starts = {
            fraction: self.max_time - max(1, int(span * fraction))
            for fraction in self.window_fractions
        }
        src_dst_sets = defaultdict(set)
        dst_src_sets = defaultdict(set)
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
            src_dst_sets[src].add(dst)
            dst_src_sets[dst].add(src)
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
            for fraction, start_time in self.window_starts.items():
                if time >= start_time:
                    self.pair_window_count[fraction][pair] += 1
                    self.dst_window_count[fraction][dst] += 1

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
                if idx > 0:
                    prev_time, prev_dst = rows[idx - 1]
                    gaps.append(max(time - prev_time, 0))
                    self.transition[prev_dst][dst] += 1
                    self.reverse_transition[dst][prev_dst] += 1
                    self.transition_total[prev_dst] += 1
                    self.transition_event_total += 1
            if gaps:
                self.src_avg_gap[src] = sum(gaps) / len(gaps)
            self._add_window_cooc(rows)

        self._prune_nested_counter(self.transition, self.cooc_topk)
        self._prune_nested_counter(self.reverse_transition, self.cooc_topk)
        self._prune_nested_counter(self.cooc_transition, self.cooc_topk)
        self._prune_nested_counter(self.reverse_cooc_transition, self.cooc_topk)

        for src, dsts in src_dst_sets.items():
            unique_count = len(dsts)
            self.src_unique_dst_count[src] = unique_count
            total_count = self.src_count[src]
            if total_count:
                self.src_repeat_rate[src] = 1.0 - unique_count / total_count
        for dst, srcs in dst_src_sets.items():
            self.dst_unique_src_count[dst] = len(srcs)
        self.dst_src_sets = {dst: set(srcs) for dst, srcs in dst_src_sets.items()}

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
                self.cooc_total[prev] += weight
                self.cooc_event_total += weight

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
        transition_prob_score = 0.0
        transition_pmi_score = 0.0
        cooc_prob_score = 0.0
        cooc_pmi_score = 0.0
        overlap_score = 0.0
        transition_hits = 0
        cooc_hits = 0

        for rank, hist_dst in enumerate(reversed(recent), start=1):
            discount = 1.0 / (rank ** 0.7)
            if hist_dst == dst:
                same_decay += discount
                if recent_rank_score == 0.0:
                    recent_rank_score = 1.0 / rank

            trans_value = self.transition[hist_dst][dst]
            if trans_value:
                transition_hits += 1
                transition_score += math.log1p(trans_value) * discount
                transition_prob = trans_value / max(self.transition_total[hist_dst], 1)
            else:
                transition_prob = 0.0

            rev_trans_value = self.reverse_transition[hist_dst][dst]
            if rev_trans_value:
                reverse_transition_score += math.log1p(rev_trans_value) * discount

            cooc_value = self.cooc_transition[hist_dst][dst]
            if cooc_value:
                cooc_hits += 1
                cooc_score += math.log1p(cooc_value) * discount
                cooc_prob = cooc_value / max(self.cooc_total[hist_dst], 1e-12)
            else:
                cooc_prob = 0.0

            reverse_cooc_value = self.reverse_cooc_transition[hist_dst][dst]
            if reverse_cooc_value:
                reverse_cooc_score += math.log1p(reverse_cooc_value) * discount

            if trans_value:
                transition_prob_score += transition_prob * discount
                transition_pmi_score += self._pmi(trans_value, hist_dst, dst, self.transition_event_total) * discount
            if cooc_value:
                cooc_prob_score += cooc_prob * discount
                cooc_pmi_score += self._pmi(cooc_value, hist_dst, dst, self.cooc_event_total) * discount
            if rank <= 20:
                overlap_score += self._dst_source_overlap(hist_dst, dst) * discount

        return {
            "recent_decay_count": same_decay,
            "recent_rank_score": recent_rank_score,
            "recent_transition_score": transition_score,
            "reverse_recent_transition_score": reverse_transition_score,
            "recent_cooc_score": cooc_score,
            "reverse_recent_cooc_score": reverse_cooc_score,
            "recent_transition_hits": math.log1p(transition_hits),
            "recent_cooc_hits": math.log1p(cooc_hits),
            "recent_transition_prob_score": transition_prob_score,
            "recent_transition_pmi_score": transition_pmi_score,
            "recent_cooc_prob_score": cooc_prob_score,
            "recent_cooc_pmi_score": cooc_pmi_score,
            "recent_dst_source_overlap": overlap_score,
        }

    def _pmi(self, value, src_dst, dst, total):
        if value <= 0 or total <= 0:
            return 0.0
        left = self.dst_count[src_dst]
        right = self.dst_count[dst]
        if left <= 0 or right <= 0:
            return 0.0
        return max(0.0, math.log((value * total) / max(left * right, 1e-12)))

    def _dst_source_overlap(self, left_dst, right_dst):
        if left_dst == right_dst:
            return 1.0
        left = self.dst_src_sets.get(left_dst)
        right = self.dst_src_sets.get(right_dst)
        if not left or not right:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        overlap = sum(1 for src in left if src in right)
        if overlap <= 0:
            return 0.0
        return overlap / math.sqrt(len(left) * len(right))

    def features(self, src, time, dst):
        pair = (src, dst)
        recent = list(self.src_recent.get(src, ()))
        recent_20 = recent[-20:]
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
        last_transition_prob = (
            last_transition / max(self.transition_total[last_dst], 1)
            if last_dst is not None else 0.0
        )
        last_cooc_prob = (
            last_cooc / max(self.cooc_total[last_dst], 1e-12)
            if last_dst is not None else 0.0
        )

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
            "last_transition_prob": last_transition_prob,
            "last_reverse_transition": math.log1p(last_reverse_transition),
            "last_cooc": math.log1p(last_cooc),
            "last_cooc_prob": last_cooc_prob,
            "recent_transition_score": seq_scores["recent_transition_score"],
            "reverse_recent_transition_score": seq_scores["reverse_recent_transition_score"],
            "recent_transition_hits": seq_scores["recent_transition_hits"],
            "recent_cooc_score": seq_scores["recent_cooc_score"],
            "reverse_recent_cooc_score": seq_scores["reverse_recent_cooc_score"],
            "recent_cooc_hits": seq_scores["recent_cooc_hits"],
            "recent_transition_prob_score": seq_scores["recent_transition_prob_score"],
            "recent_transition_pmi_score": seq_scores["recent_transition_pmi_score"],
            "recent_cooc_prob_score": seq_scores["recent_cooc_prob_score"],
            "recent_cooc_pmi_score": seq_scores["recent_cooc_pmi_score"],
            "recent_dst_source_overlap": seq_scores["recent_dst_source_overlap"],
        }
        for fraction in self.window_fractions:
            suffix = f"w{int(round(fraction * 100)):02d}"
            pair_window = self.pair_window_count[fraction][pair]
            dst_window = self.dst_window_count[fraction][dst]
            feats[f"pair_recent_count_{suffix}"] = math.log1p(pair_window)
            feats[f"pair_trend_{suffix}"] = math.log1p(pair_window) - math.log1p(max(self.pair_count[pair] - pair_window, 0))
            feats[f"dst_recent_popularity_{suffix}"] = math.log1p(dst_window)
            feats[f"dst_trend_{suffix}"] = math.log1p(dst_window) - math.log1p(max(dst_count - dst_window, 0))
        # Backward-compatible alias used by older model metadata.
        feats["item_transition"] = feats["last_transition"]
        return feats
