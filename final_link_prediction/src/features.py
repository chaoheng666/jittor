import math
import pickle
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from .data import Edge, TestRow, ensure_dir, row_rank_score


FEATURE_NAMES = [
    "rule",
    "pop",
    "recent_pop",
    "trend",
    "recency",
    "src_recent_exact",
    "pair_log",
    "dst_known",
    "degree_cap",
    "candidate_seen_in_test",
    "svd",
    "profile",
    "transition",
    "rank_rule",
    "rank_pop",
    "rank_recent_pop",
    "rank_pair",
    "rank_recency",
    "rank_svd",
    "rank_profile",
    "rank_transition",
]


SEARCH_COMPONENTS = [
    "rule",
    "pop",
    "recent_pop",
    "trend",
    "recency",
    "src_recent_exact",
    "pair_log",
    "dst_known",
    "degree_cap",
    "candidate_seen_in_test",
    "svd",
    "profile",
    "transition",
    "rank_rule",
    "rank_pop",
    "rank_recent_pop",
    "rank_pair",
    "rank_recency",
    "rank_svd",
    "rank_profile",
    "rank_transition",
]


class GraphFeatureModel:
    def __init__(
        self,
        dataset: str,
        svd_dim: int = 128,
        recent_limit: int = 160,
        transition_window: int = 16,
        transition_topk: int = 256,
        seed: int = 2026,
    ):
        self.dataset = dataset
        self.svd_dim = int(svd_dim)
        self.recent_limit = int(recent_limit)
        self.transition_window = int(transition_window)
        self.transition_topk = int(transition_topk)
        self.seed = int(seed)

        self.src_recent = defaultdict(lambda: deque(maxlen=self.recent_limit))
        self.src_count = Counter()
        self.dst_count = Counter()
        self.dst_recent_count = Counter()
        self.dst_mid_count = Counter()
        self.dst_older_count = Counter()
        self.dst_last_time = {}
        self.pair_count = Counter()
        self.test_candidate_count = Counter()

        self.max_log_src = 1.0
        self.max_log_dst = 1.0
        self.max_log_recent = 1.0
        self.max_log_mid = 1.0
        self.max_log_pair = 1.0
        self.max_log_test_candidate = 1.0
        self.time_min = 0.0
        self.time_max = 1.0
        self.time_scale = 1.0
        self.recent_cut = 0.0
        self.mid_cut = 0.0

        self.src_to_id = {}
        self.dst_to_id = {}
        self.src_emb = None
        self.dst_emb = None
        self.transition = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["src_recent"] = {k: list(v) for k, v in self.src_recent.items()}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        recent = defaultdict(lambda: deque(maxlen=self.recent_limit))
        for key, values in state.get("src_recent", {}).items():
            recent[key] = deque(values, maxlen=self.recent_limit)
        self.src_recent = recent

    def fit(self, edges: Sequence[Edge], test_rows: Sequence[TestRow] = ()) -> "GraphFeatureModel":
        rows = sorted(edges, key=lambda x: (x[2], x[0], x[1]))
        if not rows:
            raise ValueError("cannot fit feature model with no edges")

        times = np.asarray([t for _, _, t in rows], dtype=np.float64)
        self.time_min = float(times.min())
        self.time_max = float(times.max())
        self.time_scale = max(self.time_max - self.time_min, 1.0)
        self.mid_cut = float(np.percentile(times, 60))
        self.recent_cut = float(np.percentile(times, 82))

        self.src_to_id = {v: i for i, v in enumerate(sorted({s for s, _, _ in rows}))}
        self.dst_to_id = {v: i for i, v in enumerate(sorted({d for _, d, _ in rows}))}

        for src, dst, time in rows:
            self.src_recent[src].append(dst)
            self.src_count[src] += 1
            self.dst_count[dst] += 1
            self.pair_count[(src, dst)] += 1
            self.dst_last_time[dst] = max(self.dst_last_time.get(dst, time), time)
            if time >= self.recent_cut:
                self.dst_recent_count[dst] += 1
            elif time >= self.mid_cut:
                self.dst_mid_count[dst] += 1
            else:
                self.dst_older_count[dst] += 1

        for row in test_rows:
            self.test_candidate_count.update(row.candidates)

        self.max_log_src = max(math.log1p(max(self.src_count.values(), default=1)), 1.0)
        self.max_log_dst = max(math.log1p(max(self.dst_count.values(), default=1)), 1.0)
        self.max_log_recent = max(math.log1p(max(self.dst_recent_count.values(), default=1)), 1.0)
        self.max_log_mid = max(math.log1p(max(self.dst_mid_count.values(), default=1)), 1.0)
        self.max_log_pair = max(math.log1p(max(self.pair_count.values(), default=1)), 1.0)
        self.max_log_test_candidate = max(math.log1p(max(self.test_candidate_count.values(), default=1)), 1.0)

        self._fit_svd(rows)
        self._fit_transition(rows)
        return self

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "GraphFeatureModel":
        with path.open("rb") as f:
            return pickle.load(f)

    def _fit_svd(self, rows: Sequence[Edge]) -> None:
        n_src = len(self.src_to_id)
        n_dst = len(self.dst_to_id)
        if min(n_src, n_dst) <= 2:
            self.src_emb = np.zeros((n_src, 2), dtype=np.float32)
            self.dst_emb = np.zeros((n_dst, 2), dtype=np.float32)
            return
        src_ids = np.fromiter((self.src_to_id[s] for s, _, _ in rows), dtype=np.int32)
        dst_ids = np.fromiter((self.dst_to_id[d] for _, d, _ in rows), dtype=np.int32)
        time_values = np.fromiter((t for _, _, t in rows), dtype=np.float32)
        time_norm = (time_values - float(self.time_min)) / max(float(self.time_scale), 1.0)
        values = (1.0 + 2.8 * time_norm).astype(np.float32)
        matrix = sparse.coo_matrix((values, (src_ids, dst_ids)), shape=(n_src, n_dst), dtype=np.float32).tocsr()
        matrix.data = np.log1p(matrix.data).astype(np.float32)
        dim = min(self.svd_dim, max(2, min(matrix.shape) - 1))
        svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=self.seed)
        src_emb = svd.fit_transform(matrix).astype(np.float32)
        dst_emb = svd.components_.T.astype(np.float32)
        self.src_emb = src_emb / np.maximum(np.linalg.norm(src_emb, axis=1, keepdims=True), 1e-6)
        self.dst_emb = dst_emb / np.maximum(np.linalg.norm(dst_emb, axis=1, keepdims=True), 1e-6)

    def _fit_transition(self, rows: Sequence[Edge]) -> None:
        by_src = defaultdict(list)
        for src, dst, time in rows:
            by_src[src].append((time, dst))
        raw = defaultdict(Counter)
        for seq in by_src.values():
            seq.sort()
            recent = deque(maxlen=self.transition_window)
            for _time, dst in seq:
                for rank, prev_dst in enumerate(reversed(recent), start=1):
                    if prev_dst != dst:
                        raw[prev_dst][dst] += 1.0 / math.sqrt(rank)
                recent.append(dst)
        transition = {}
        for prev_dst, counter in raw.items():
            prev_norm = math.sqrt(math.log1p(float(self.dst_count.get(prev_dst, 1))))
            keep = {}
            for dst, value in counter.most_common(self.transition_topk):
                dst_norm = math.sqrt(math.log1p(float(self.dst_count.get(dst, 1))))
                keep[dst] = math.log1p(float(value)) / max(prev_norm * dst_norm, 1e-6)
            if keep:
                transition[prev_dst] = keep
        self.transition = transition

    def _svd_scores(self, src: int, candidates: Sequence[int]) -> np.ndarray:
        src_id = self.src_to_id.get(src)
        if src_id is None or self.src_emb is None or self.dst_emb is None:
            return np.zeros(len(candidates), dtype=np.float32)
        src_vec = self.src_emb[src_id]
        out = np.zeros(len(candidates), dtype=np.float32)
        for i, dst in enumerate(candidates):
            dst_id = self.dst_to_id.get(dst)
            if dst_id is not None:
                out[i] = float(np.dot(src_vec, self.dst_emb[dst_id]))
        return out

    def _profile_scores(self, src: int, candidates: Sequence[int]) -> np.ndarray:
        if self.dst_emb is None:
            return np.zeros(len(candidates), dtype=np.float32)
        hist = list(self.src_recent.get(src, ()))
        if not hist:
            return np.zeros(len(candidates), dtype=np.float32)
        profile = np.zeros(self.dst_emb.shape[1], dtype=np.float32)
        total = 0.0
        for rank, hist_dst in enumerate(reversed(hist[-self.recent_limit:]), start=1):
            dst_id = self.dst_to_id.get(hist_dst)
            if dst_id is None:
                continue
            weight = 1.0 / math.sqrt(rank)
            profile += self.dst_emb[dst_id] * weight
            total += weight
        if total <= 0:
            return np.zeros(len(candidates), dtype=np.float32)
        profile /= total
        profile /= max(float(np.linalg.norm(profile)), 1e-6)
        out = np.zeros(len(candidates), dtype=np.float32)
        for i, dst in enumerate(candidates):
            dst_id = self.dst_to_id.get(dst)
            if dst_id is not None:
                out[i] = float(np.dot(profile, self.dst_emb[dst_id]))
        return out

    def _transition_scores(self, src: int, candidates: Sequence[int]) -> np.ndarray:
        hist = list(self.src_recent.get(src, ()))
        if not hist:
            return np.zeros(len(candidates), dtype=np.float32)
        cand_pos = {dst: i for i, dst in enumerate(candidates)}
        out = np.zeros(len(candidates), dtype=np.float32)
        for rank, hist_dst in enumerate(reversed(hist[-self.transition_window:]), start=1):
            table = self.transition.get(hist_dst)
            if not table:
                continue
            weight = 1.0 / math.sqrt(rank)
            for dst, idx in cand_pos.items():
                value = table.get(dst)
                if value is not None:
                    out[idx] += weight * float(value)
        return out

    def feature_row(self, src: int, time: int, candidates: Sequence[int]) -> np.ndarray:
        candidates = tuple(candidates)
        recent_rank: Dict[int, int] = {}
        for rank, dst in enumerate(reversed(self.src_recent.get(src, ())), start=1):
            recent_rank.setdefault(dst, rank)

        pop = np.zeros(len(candidates), dtype=np.float32)
        recent_pop = np.zeros(len(candidates), dtype=np.float32)
        trend = np.zeros(len(candidates), dtype=np.float32)
        recency = np.zeros(len(candidates), dtype=np.float32)
        exact = np.zeros(len(candidates), dtype=np.float32)
        pair_log = np.zeros(len(candidates), dtype=np.float32)
        known = np.zeros(len(candidates), dtype=np.float32)
        degree_cap = np.zeros(len(candidates), dtype=np.float32)
        test_seen = np.zeros(len(candidates), dtype=np.float32)

        for i, dst in enumerate(candidates):
            dst_count = float(self.dst_count.get(dst, 0))
            recent = float(self.dst_recent_count.get(dst, 0))
            mid = float(self.dst_mid_count.get(dst, 0))
            older = float(self.dst_older_count.get(dst, 0))
            pair = float(self.pair_count.get((src, dst), 0))
            last = self.dst_last_time.get(dst)
            rank = recent_rank.get(dst)

            pop[i] = math.log1p(dst_count) / self.max_log_dst
            recent_pop[i] = math.log1p(recent) / self.max_log_recent
            trend[i] = max(min(math.log1p(recent) - math.log1p(mid + older * 0.25), 5.0), -5.0) / 5.0
            recency[i] = 0.0 if last is None else 1.0 / (1.0 + max(float(time - last), 0.0) / self.time_scale)
            exact[i] = 1.0 / math.sqrt(float(rank)) if rank else 0.0
            pair_log[i] = math.log1p(pair) / self.max_log_pair
            known[i] = 1.0 if dst_count > 0 else 0.0
            degree_cap[i] = min(pop[i], 0.72)
            test_seen[i] = math.log1p(float(self.test_candidate_count.get(dst, 0))) / self.max_log_test_candidate

        svd = self._svd_scores(src, candidates)
        profile = self._profile_scores(src, candidates)
        transition = self._transition_scores(src, candidates)

        if self.dataset == "dataset1":
            rule = (
                1.45 * pair_log
                + 1.15 * exact
                + 0.85 * pop
                + 1.00 * recent_pop
                + 0.45 * trend
                + 0.45 * recency
                + 0.25 * profile
                + 0.25 * transition
                - 0.20 * (1.0 - known)
            )
        else:
            rule = (
                1.00 * degree_cap
                + 1.15 * recent_pop
                + 0.50 * trend
                + 0.65 * recency
                + 0.15 * profile
                + 0.10 * transition
                - 0.28 * pair_log
                - 0.08 * exact
                - 0.80 * (1.0 - known)
            )

        base = [
            rule,
            pop,
            recent_pop,
            trend,
            recency,
            exact,
            pair_log,
            known,
            degree_cap,
            test_seen,
            svd,
            profile,
            transition,
            row_rank_score(rule)[0],
            row_rank_score(pop)[0],
            row_rank_score(recent_pop)[0],
            row_rank_score(pair_log)[0],
            row_rank_score(recency)[0],
            row_rank_score(svd)[0],
            row_rank_score(profile)[0],
            row_rank_score(transition)[0],
        ]
        return np.stack(base, axis=1).astype(np.float32)

    def feature_tensor(self, rows: Sequence[TestRow], progress_every: int = 20000) -> np.ndarray:
        feats = np.zeros((len(rows), 100, len(FEATURE_NAMES)), dtype=np.float32)
        for i, row in enumerate(rows):
            feats[i] = self.feature_row(row.src, row.time, row.candidates)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"features dataset={self.dataset} rows={i + 1}", flush=True)
        return feats

    def score_rows(self, rows: Sequence[TestRow], weights: Dict[str, float], batch_size: int = 4096) -> np.ndarray:
        from .validation import score_feature_tensor

        scores = np.zeros((len(rows), 100), dtype=np.float32)
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            feats = self.feature_tensor(chunk, progress_every=0)
            scores[start:start + len(chunk)] = score_feature_tensor(feats, weights)
            print(f"scored dataset={self.dataset} rows={start + len(chunk)}", flush=True)
        return scores
