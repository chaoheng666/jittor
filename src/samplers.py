import math
import random
from collections import Counter, defaultdict, deque

import numpy as np


DEFAULT_MIX = {
    "history_hard": 0.35,
    "global_pop": 0.25,
    "random_seen": 0.20,
    "random_cold": 0.20,
}


class MixedNegativeSampler:
    """Proposal sampler for future-edge negatives.

    The sampler is intentionally not tied to official 100-candidate rows. It
    mixes hard source-local negatives, popular destinations, random seen
    destinations, and random all-space/cold nodes, then exposes an approximate
    proposal probability for logQ correction.
    """

    def __init__(
        self,
        history_edges,
        seed=2026,
        recent_limit=100,
        transition_limit=200,
        popular_limit=3000,
        cooc_window=8,
        mix=None,
        node_min=None,
        node_max=None,
    ):
        self.rng = random.Random(seed)
        self.mix = dict(DEFAULT_MIX)
        if mix:
            self.mix.update(mix)
        total = sum(max(v, 0.0) for v in self.mix.values())
        if total <= 0:
            self.mix = dict(DEFAULT_MIX)
            total = sum(self.mix.values())
        self.mix = {k: max(v, 0.0) / total for k, v in self.mix.items()}

        self.transition_limit = int(transition_limit)
        self.recent_by_src = defaultdict(lambda: deque(maxlen=recent_limit))
        self.transition = defaultdict(Counter)
        self.cooc = defaultdict(Counter)
        self.dst_counter = Counter()
        self.node_values = set()
        by_src = defaultdict(list)
        all_src = []
        all_dst = []

        for src, dst, time in sorted(history_edges, key=lambda x: x[2]):
            all_src.append(src)
            all_dst.append(dst)
            self.dst_counter[dst] += 1
            self.node_values.add(src)
            self.node_values.add(dst)
            self.recent_by_src[src].append(dst)
            by_src[src].append((time, dst))

        self.edge_src = np.asarray(all_src, dtype=np.int64)
        self.edge_dst = np.asarray(all_dst, dtype=np.int64)
        self.dst_unique = list(self.dst_counter.keys())
        if not self.dst_unique:
            raise ValueError("history split has no destination nodes")
        self.popular = [dst for dst, _ in self.dst_counter.most_common(popular_limit)]

        for rows in by_src.values():
            rows.sort()
            dsts = [dst for _, dst in rows]
            for i in range(1, len(dsts)):
                self.transition[dsts[i - 1]][dsts[i]] += 1
            recent = dsts[-300:]
            for i, dst in enumerate(recent):
                start = max(0, i - cooc_window)
                for j in range(start, i):
                    prev = recent[j]
                    if prev != dst:
                        self.cooc[prev][dst] += 1.0 / (i - j)

        if node_min is None:
            node_min = min(self.node_values) if self.node_values else min(self.dst_unique)
        if node_max is None:
            node_max = max(self.node_values) if self.node_values else max(self.dst_unique)
        self.node_min = int(node_min)
        self.node_max = int(node_max)
        if self.node_max < self.node_min:
            self.node_min, self.node_max = self.node_max, self.node_min

    def _allowed(self, positive, dst, seen):
        return dst != positive and dst not in seen

    def hard_candidates(self, src):
        recent = list(self.recent_by_src.get(src, ()))
        hard = list(reversed(recent))
        if recent:
            last_dst = recent[-1]
            hard.extend(dst for dst, _ in self.transition[last_dst].most_common(self.transition_limit))
            hard.extend(dst for dst, _ in self.cooc[last_dst].most_common(self.transition_limit))
        return hard

    def _take_from_list(self, values, positive, need, seen):
        out = []
        for dst in values:
            if len(out) >= need:
                break
            if self._allowed(positive, dst, seen):
                seen.add(dst)
                out.append(dst)
        return out

    def _random_seen(self, positive, need, seen):
        out = []
        tries = 0
        max_tries = max(1000, need * 500)
        while len(out) < need and tries < max_tries:
            tries += 1
            dst = self.rng.choice(self.dst_unique)
            if self._allowed(positive, dst, seen):
                seen.add(dst)
                out.append(dst)
        return out

    def _random_cold(self, positive, need, seen):
        out = []
        tries = 0
        width = max(self.node_max - self.node_min + 1, 1)
        max_tries = max(1000, need * 1000)
        while len(out) < need and tries < max_tries:
            tries += 1
            dst = self.rng.randint(self.node_min, self.node_max)
            if dst in self.dst_counter:
                continue
            if self._allowed(positive, dst, seen):
                seen.add(dst)
                out.append(dst)
        if len(out) < need:
            out.extend(self._random_seen(positive, need - len(out), seen))
        return out

    def sample(self, src, positive, count):
        count = int(count)
        if count <= 0:
            return []

        seen = set()
        out = []
        quotas = {
            name: int(round(count * weight))
            for name, weight in self.mix.items()
        }
        while sum(quotas.values()) < count:
            name = max(self.mix, key=self.mix.get)
            quotas[name] += 1
        while sum(quotas.values()) > count:
            name = max(quotas, key=quotas.get)
            quotas[name] -= 1

        out.extend(self._take_from_list(self.hard_candidates(src), positive, quotas.get("history_hard", 0), seen))
        out.extend(self._take_from_list(self.popular, positive, quotas.get("global_pop", 0), seen))
        out.extend(self._random_seen(positive, quotas.get("random_seen", 0), seen))
        out.extend(self._random_cold(positive, quotas.get("random_cold", 0), seen))

        if len(out) < count:
            hard_backfill = self.hard_candidates(src) + self.popular
            out.extend(self._take_from_list(hard_backfill, positive, count - len(out), seen))
        if len(out) < count:
            out.extend(self._random_seen(positive, count - len(out), seen))
        if len(out) < count:
            out.extend(self._random_cold(positive, count - len(out), seen))
        if len(out) < count:
            for dst in range(self.node_min, self.node_max + 1):
                if len(out) >= count:
                    break
                if dst in self.dst_counter:
                    continue
                if self._allowed(positive, dst, seen):
                    seen.add(dst)
                    out.append(dst)
        return out[:count]

    def large_pool(self, src, positive, count):
        return self.sample(src, positive, count)

    def lage_pool(self, src, positive, count):
        return self.large_pool(src, positive, count)

    def proposal_probability(self, src, dst):
        q = 0.0
        if dst in self.hard_candidates(src):
            q += self.mix.get("history_hard", 0.0) / max(len(set(self.hard_candidates(src))), 1)
        if dst in self.popular:
            q += self.mix.get("global_pop", 0.0) / max(len(self.popular), 1)
        if dst in self.dst_counter:
            q += self.mix.get("random_seen", 0.0) / max(len(self.dst_unique), 1)
        else:
            width = max(self.node_max - self.node_min + 1 - len(self.dst_unique), 1)
            q += self.mix.get("random_cold", 0.0) / width
        return max(q, 1e-12)

    def logq(self, src, dst):
        return math.log(self.proposal_probability(src, dst))
