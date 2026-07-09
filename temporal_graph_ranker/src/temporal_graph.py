import math
import os
import pickle
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from threadpoolctl import threadpool_limits

from .io_data import Edge, TestRow, ensure_dir, row_rank_score


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


def _rank_score_1d(values: np.ndarray) -> np.ndarray:
    """Return reciprocal ranks for one candidate row without a Python row loop."""
    order = np.argsort(-values)
    out = np.empty(len(values), dtype=np.float32)
    out[order] = 1.0 / np.arange(1, len(values) + 1, dtype=np.float32)
    return out


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
        self.src_last_time = {}
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
        self.src_profile = None
        self.transition = {}
        self._dst_keys = np.asarray([], dtype=np.int64)
        self._dst_count_arr = np.asarray([], dtype=np.float32)
        self._dst_recent_arr = np.asarray([], dtype=np.float32)
        self._dst_mid_arr = np.asarray([], dtype=np.float32)
        self._dst_older_arr = np.asarray([], dtype=np.float32)
        self._dst_last_arr = np.asarray([], dtype=np.float64)
        self._test_candidate_arr = np.asarray([], dtype=np.float32)

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
            self.src_last_time[src] = max(self.src_last_time.get(src, time), time)
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

        self._build_numeric_tables()
        self._fit_svd(rows)
        self._build_source_profiles()
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

    def _build_numeric_tables(self) -> None:
        self._dst_keys = np.asarray(sorted(self.dst_to_id), dtype=np.int64)
        self._dst_count_arr = np.asarray([self.dst_count.get(int(dst), 0) for dst in self._dst_keys], dtype=np.float32)
        self._dst_recent_arr = np.asarray([self.dst_recent_count.get(int(dst), 0) for dst in self._dst_keys], dtype=np.float32)
        self._dst_mid_arr = np.asarray([self.dst_mid_count.get(int(dst), 0) for dst in self._dst_keys], dtype=np.float32)
        self._dst_older_arr = np.asarray([self.dst_older_count.get(int(dst), 0) for dst in self._dst_keys], dtype=np.float32)
        self._dst_last_arr = np.asarray([self.dst_last_time.get(int(dst), np.nan) for dst in self._dst_keys], dtype=np.float64)
        self._test_candidate_arr = np.asarray(
            [self.test_candidate_count.get(int(dst), 0) for dst in self._dst_keys], dtype=np.float32
        )

    def _build_source_profiles(self) -> None:
        if self.src_emb is None or self.dst_emb is None:
            return
        self.src_profile = np.zeros((len(self.src_to_id), self.dst_emb.shape[1]), dtype=np.float32)
        for src, hist in self.src_recent.items():
            src_id = self.src_to_id.get(int(src))
            if src_id is None:
                continue
            dst_ids = [self.dst_to_id.get(int(dst)) for dst in reversed(hist)]
            dst_ids = [idx for idx in dst_ids if idx is not None]
            if not dst_ids:
                continue
            weights = 1.0 / np.sqrt(np.arange(1, len(dst_ids) + 1, dtype=np.float32))
            profile = (self.dst_emb[np.asarray(dst_ids)] * weights[:, None]).sum(axis=0) / weights.sum()
            self.src_profile[src_id] = profile / max(float(np.linalg.norm(profile)), 1e-6)

    def _candidate_indices(self, candidates: Sequence[int]) -> tuple:
        values = np.asarray(candidates, dtype=np.int64)
        positions = np.searchsorted(self._dst_keys, values)
        known = positions < len(self._dst_keys)
        check = np.flatnonzero(known)
        if len(check):
            known[check] = self._dst_keys[positions[check]] == values[check]
        return values, positions, known

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
        # Sparse SVD benefits from a bounded BLAS pool; feature workers later
        # run one thread each, so they do not oversubscribe all 192 cores.
        with threadpool_limits(limits=max(1, int(os.environ.get("SVD_THREADS", "32")))):
            src_emb = svd.fit_transform(matrix).astype(np.float32)
        dst_emb = svd.components_.T.astype(np.float32)
        self.src_emb = src_emb / np.maximum(np.linalg.norm(src_emb, axis=1, keepdims=True), 1e-6)
        self.dst_emb = dst_emb / np.maximum(np.linalg.norm(dst_emb, axis=1, keepdims=True), 1e-6)

    def _fit_transition(self, rows: Sequence[Edge]) -> None:
        edge_array = np.asarray(rows, dtype=np.int64)
        if len(edge_array) < 2:
            self.transition = {}
            return
        # Group each source's interactions by time once, then form every
        # within-source lag with vector operations.  The old implementation
        # executed this exact 16-lag logic in nested Python loops.
        order = np.lexsort((edge_array[:, 1], edge_array[:, 2], edge_array[:, 0]))
        src = edge_array[order, 0]
        dst = edge_array[order, 1]
        key_base = int(max(int(dst.max()) + 1, 1))
        key_parts = []
        value_parts = []
        for lag in range(1, self.transition_window + 1):
            prev = dst[:-lag]
            curr = dst[lag:]
            same_src = src[:-lag] == src[lag:]
            valid = same_src & (prev != curr)
            if not np.any(valid):
                continue
            key_parts.append(prev[valid] * key_base + curr[valid])
            value_parts.append(np.full(int(valid.sum()), 1.0 / math.sqrt(lag), dtype=np.float32))
        if not key_parts:
            self.transition = {}
            return
        keys = np.concatenate(key_parts)
        values = np.concatenate(value_parts)
        order = np.argsort(keys)
        keys = keys[order]
        values = values[order]
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        unique_keys = keys[starts]
        unique_values = np.add.reduceat(values, starts)
        prev_values = unique_keys // key_base
        dst_values = unique_keys % key_base
        prev_starts = np.r_[0, np.flatnonzero(prev_values[1:] != prev_values[:-1]) + 1]
        prev_ends = np.r_[prev_starts[1:], len(prev_values)]
        transition = {}
        for start, end in zip(prev_starts, prev_ends):
            prev_dst = int(prev_values[start])
            local_values = unique_values[start:end]
            local_dst = dst_values[start:end]
            take = min(self.transition_topk, len(local_values))
            if take <= 0:
                continue
            if take < len(local_values):
                selected = np.argpartition(local_values, -take)[-take:]
                selected = selected[np.argsort(-local_values[selected])]
            else:
                selected = np.argsort(-local_values)
            prev_norm = math.sqrt(math.log1p(float(self.dst_count.get(prev_dst, 1))))
            keep = {}
            for idx in selected:
                candidate = int(local_dst[int(idx)])
                dst_norm = math.sqrt(math.log1p(float(self.dst_count.get(candidate, 1))))
                keep[candidate] = math.log1p(float(local_values[int(idx)])) / max(prev_norm * dst_norm, 1e-6)
            transition[prev_dst] = keep
        self.transition = transition

    def _svd_scores(self, src: int, candidates: Sequence[int]) -> np.ndarray:
        src_id = self.src_to_id.get(src)
        if src_id is None or self.src_emb is None or self.dst_emb is None:
            return np.zeros(len(candidates), dtype=np.float32)
        _values, positions, known = self._candidate_indices(candidates)
        out = np.zeros(len(candidates), dtype=np.float32)
        if np.any(known):
            out[known] = self.dst_emb[positions[known]] @ self.src_emb[src_id]
        return out

    def _profile_scores(self, src: int, candidates: Sequence[int]) -> np.ndarray:
        if self.dst_emb is None:
            return np.zeros(len(candidates), dtype=np.float32)
        src_id = self.src_to_id.get(src)
        if src_id is None or self.src_profile is None:
            return np.zeros(len(candidates), dtype=np.float32)
        profile = self.src_profile[src_id]
        out = np.zeros(len(candidates), dtype=np.float32)
        _values, positions, known = self._candidate_indices(candidates)
        if np.any(known):
            out[known] = self.dst_emb[positions[known]] @ profile
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
        candidate_values, candidate_pos, candidate_known = self._candidate_indices(candidates)
        recent_rank: Dict[int, int] = {}
        for rank, dst in enumerate(reversed(self.src_recent.get(src, ())), start=1):
            recent_rank.setdefault(dst, rank)

        dst_count = np.zeros(len(candidates), dtype=np.float32)
        recent = np.zeros(len(candidates), dtype=np.float32)
        mid = np.zeros(len(candidates), dtype=np.float32)
        older = np.zeros(len(candidates), dtype=np.float32)
        last = np.full(len(candidates), np.nan, dtype=np.float64)
        test_count = np.fromiter(
            (self.test_candidate_count.get(int(dst), 0) for dst in candidate_values), dtype=np.float32
        )
        if np.any(candidate_known):
            dst_count[candidate_known] = self._dst_count_arr[candidate_pos[candidate_known]]
            recent[candidate_known] = self._dst_recent_arr[candidate_pos[candidate_known]]
            mid[candidate_known] = self._dst_mid_arr[candidate_pos[candidate_known]]
            older[candidate_known] = self._dst_older_arr[candidate_pos[candidate_known]]
            last[candidate_known] = self._dst_last_arr[candidate_pos[candidate_known]]

        pop = np.log1p(dst_count) / self.max_log_dst
        recent_pop = np.log1p(recent) / self.max_log_recent
        trend = np.clip(np.log1p(recent) - np.log1p(mid + older * 0.25), -5.0, 5.0) / 5.0
        recency = np.zeros(len(candidates), dtype=np.float32)
        has_last = np.isfinite(last)
        recency[has_last] = 1.0 / (1.0 + np.maximum(float(time) - last[has_last], 0.0) / self.time_scale)
        ranks = np.asarray([recent_rank.get(int(dst), 0) for dst in candidate_values], dtype=np.float32)
        exact = np.zeros(len(candidates), dtype=np.float32)
        has_rank = ranks > 0
        exact[has_rank] = 1.0 / np.sqrt(ranks[has_rank])
        pair_values = np.fromiter((self.pair_count.get((src, int(dst)), 0) for dst in candidate_values), dtype=np.float32)
        pair_log = np.log1p(pair_values) / self.max_log_pair
        known = (dst_count > 0).astype(np.float32)
        degree_cap = np.minimum(pop, 0.72)
        test_seen = np.log1p(test_count) / self.max_log_test_candidate

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
            _rank_score_1d(rule),
            _rank_score_1d(pop),
            _rank_score_1d(recent_pop),
            _rank_score_1d(pair_log),
            _rank_score_1d(recency),
            _rank_score_1d(svd),
            _rank_score_1d(profile),
            _rank_score_1d(transition),
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
        from .evaluation import score_feature_tensor

        scores = np.zeros((len(rows), 100), dtype=np.float32)
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            feats = self.feature_tensor(chunk, progress_every=0)
            scores[start:start + len(chunk)] = score_feature_tensor(feats, weights)
            print(f"scored dataset={self.dataset} rows={start + len(chunk)}", flush=True)
        return scores
