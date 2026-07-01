import bisect
import math
from collections import defaultdict

import numpy as np
import jittor as jt
from jittor import nn


PAD_NODE = 0
UNK_NODE = 1
DIR_OUT = 1
DIR_IN = 2


class TemporalNeighborIndex:
    def __init__(self, edges):
        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)
        self.out_times = {}
        self.in_times = {}
        for src, dst, time in sorted(edges, key=lambda row: row[2]):
            self.out_edges[src].append((time, dst))
            self.in_edges[dst].append((time, src))
        self.out_times = {node: [time for time, _ in rows] for node, rows in self.out_edges.items()}
        self.in_times = {node: [time for time, _ in rows] for node, rows in self.in_edges.items()}

    @staticmethod
    def _recent(rows, times, query_time, limit):
        if not rows or limit <= 0:
            return []
        end = bisect.bisect_left(times, query_time)
        start = max(0, end - limit)
        return rows[start:end]

    def recent_directed(self, node, query_time, limit, direction):
        if direction == DIR_OUT:
            return [(time, nbr, DIR_OUT) for time, nbr in self._recent(
                self.out_edges.get(node, []), self.out_times.get(node, []), query_time, limit
            )]
        return [(time, nbr, DIR_IN) for time, nbr in self._recent(
            self.in_edges.get(node, []), self.in_times.get(node, []), query_time, limit
        )]

    def recent_neighbors(self, node, query_time, limit):
        out_limit = max(1, limit // 2)
        in_limit = max(1, limit - out_limit)
        rows = self.recent_directed(node, query_time, out_limit, DIR_OUT)
        rows.extend(self.recent_directed(node, query_time, in_limit, DIR_IN))
        rows.sort(key=lambda row: row[0], reverse=True)
        return rows[:limit]

    def second_hop_neighbors(self, node, query_time, first_hop_limit, per_mid_limit, total_limit):
        output = []
        first_hop = self.recent_neighbors(node, query_time, first_hop_limit)
        for _, mid, _ in first_hop:
            if len(output) >= total_limit:
                break
            rows = self.recent_neighbors(mid, query_time, per_mid_limit)
            for time, nbr, direction in rows:
                if nbr == node:
                    continue
                output.append((time, nbr, direction))
                if len(output) >= total_limit:
                    break
        output.sort(key=lambda row: row[0], reverse=True)
        return output[:total_limit]


class TemporalGNNDatasetBuilder:
    def __init__(
        self,
        node_values,
        src_neighbors=50,
        cand_neighbors=30,
        second_hop=20,
        max_time_bucket=127,
    ):
        self.node_to_idx = {int(node): idx + 2 for idx, node in enumerate(sorted(node_values))}
        self.node_values = [None, None] + sorted(int(node) for node in node_values)
        self.src_neighbors = src_neighbors
        self.cand_neighbors = cand_neighbors
        self.second_hop = second_hop
        self.max_time_bucket = max_time_bucket

    def node_index(self, node):
        return self.node_to_idx.get(int(node), UNK_NODE)

    def _gap_bucket(self, query_time, edge_time):
        gap = max(int(query_time) - int(edge_time), 0)
        return min(int(math.log1p(gap)), self.max_time_bucket)

    def _fill_neighbor_arrays(self, rows, query_time, limit):
        node_idx = np.zeros(limit, dtype=np.int32)
        gap = np.zeros(limit, dtype=np.int32)
        direction = np.zeros(limit, dtype=np.int32)
        for i, (edge_time, node, edge_direction) in enumerate(rows[:limit]):
            node_idx[i] = self.node_index(node)
            gap[i] = self._gap_bucket(query_time, edge_time)
            direction[i] = edge_direction
        return node_idx, gap, direction

    def build_batch(self, index, queries):
        batch_size = len(queries)
        src_idx = np.zeros(batch_size, dtype=np.int32)
        cand_idx = np.zeros((batch_size, 100), dtype=np.int32)
        src_nbr = np.zeros((batch_size, self.src_neighbors), dtype=np.int32)
        src_gap = np.zeros((batch_size, self.src_neighbors), dtype=np.int32)
        src_dir = np.zeros((batch_size, self.src_neighbors), dtype=np.int32)
        cand_nbr = np.zeros((batch_size, 100, self.cand_neighbors), dtype=np.int32)
        cand_gap = np.zeros((batch_size, 100, self.cand_neighbors), dtype=np.int32)
        cand_dir = np.zeros((batch_size, 100, self.cand_neighbors), dtype=np.int32)
        hop_nbr = np.zeros((batch_size, self.second_hop), dtype=np.int32)
        hop_gap = np.zeros((batch_size, self.second_hop), dtype=np.int32)
        hop_dir = np.zeros((batch_size, self.second_hop), dtype=np.int32)

        for row_idx, (src, query_time, candidates) in enumerate(queries):
            src_idx[row_idx] = self.node_index(src)
            cand_idx[row_idx] = [self.node_index(dst) for dst in candidates]
            rows = index.recent_neighbors(src, query_time, self.src_neighbors)
            src_nbr[row_idx], src_gap[row_idx], src_dir[row_idx] = self._fill_neighbor_arrays(
                rows, query_time, self.src_neighbors
            )
            hop_rows = index.second_hop_neighbors(
                src, query_time, self.src_neighbors, max(1, self.second_hop), self.second_hop
            )
            hop_nbr[row_idx], hop_gap[row_idx], hop_dir[row_idx] = self._fill_neighbor_arrays(
                hop_rows, query_time, self.second_hop
            )
            for cand_pos, dst in enumerate(candidates):
                cand_rows = index.recent_neighbors(dst, query_time, self.cand_neighbors)
                nodes, gaps, dirs = self._fill_neighbor_arrays(cand_rows, query_time, self.cand_neighbors)
                cand_nbr[row_idx, cand_pos] = nodes
                cand_gap[row_idx, cand_pos] = gaps
                cand_dir[row_idx, cand_pos] = dirs

        return {
            "src_idx": src_idx,
            "cand_idx": cand_idx,
            "src_nbr": src_nbr,
            "src_gap": src_gap,
            "src_dir": src_dir,
            "cand_nbr": cand_nbr,
            "cand_gap": cand_gap,
            "cand_dir": cand_dir,
            "hop_nbr": hop_nbr,
            "hop_gap": hop_gap,
            "hop_dir": hop_dir,
        }


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def execute(self, messages, node_ids):
        mask = (node_ids > 0).float32()
        scores = self.score(messages).squeeze(-1)
        scores = scores + (mask - 1.0) * 10000.0
        weight = nn.softmax(scores, dim=-1).unsqueeze(-1)
        return (self.drop(messages) * weight * mask.unsqueeze(-1)).sum(dim=-2)


class TemporalGNNRanker(nn.Module):
    def __init__(
        self,
        n_nodes,
        feature_dim,
        node_emb_dim=128,
        time_emb_dim=32,
        hidden_dim=192,
        dropout=0.1,
        max_time_bucket=127,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_emb = nn.Embedding(n_nodes, node_emb_dim)
        self.time_emb = nn.Embedding(max_time_bucket + 1, time_emb_dim)
        self.dir_emb = nn.Embedding(3, time_emb_dim)
        msg_dim = node_emb_dim + time_emb_dim * 2
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.src_pool = TemporalAttentionPool(hidden_dim, dropout)
        self.cand_pool = TemporalAttentionPool(hidden_dim, dropout)
        self.hop_pool = TemporalAttentionPool(hidden_dim, dropout)
        self.src_proj = nn.Linear(node_emb_dim + hidden_dim * 2, hidden_dim)
        self.cand_proj = nn.Linear(node_emb_dim + hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 6 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _messages(self, nbr_idx, gap, direction):
        emb = self.node_emb(nbr_idx)
        time = self.time_emb(gap)
        direction_emb = self.dir_emb(direction)
        return self.msg_mlp(jt.cat([emb, time, direction_emb], dim=-1))

    def execute(
        self,
        src_idx,
        cand_idx,
        src_nbr,
        src_gap,
        src_dir,
        cand_nbr,
        cand_gap,
        cand_dir,
        hop_nbr,
        hop_gap,
        hop_dir,
        cand_features,
    ):
        batch_size = src_idx.shape[0]
        candidate_size = cand_idx.shape[1]

        src_msgs = self._messages(src_nbr, src_gap, src_dir)
        src_context = self.src_pool(src_msgs, src_nbr)

        hop_msgs = self._messages(hop_nbr, hop_gap, hop_dir)
        hop_context = self.hop_pool(hop_msgs, hop_nbr)

        flat_cand_nbr = cand_nbr.reshape(batch_size * candidate_size, cand_nbr.shape[-1])
        flat_cand_gap = cand_gap.reshape(batch_size * candidate_size, cand_gap.shape[-1])
        flat_cand_dir = cand_dir.reshape(batch_size * candidate_size, cand_dir.shape[-1])
        cand_msgs = self._messages(flat_cand_nbr, flat_cand_gap, flat_cand_dir)
        cand_context = self.cand_pool(cand_msgs, flat_cand_nbr).reshape(batch_size, candidate_size, -1)

        src_base = self.node_emb(src_idx)
        src_hidden = self.src_proj(jt.cat([src_base, src_context, hop_context], dim=-1))
        cand_base = self.node_emb(cand_idx)
        cand_hidden = self.cand_proj(jt.cat([cand_base, cand_context], dim=-1))

        q = self.query_proj(cand_hidden)
        k = self.key_proj(src_msgs)
        v = self.value_proj(src_msgs)
        src_mask = (src_nbr > 0).float32()
        attn_score = (q.unsqueeze(2) * k.unsqueeze(1)).sum(dim=-1) / math.sqrt(float(self.hidden_dim))
        attn_score = attn_score + (src_mask.unsqueeze(1) - 1.0) * 10000.0
        attn = nn.softmax(attn_score, dim=2)
        cross_context = (attn.unsqueeze(-1) * v.unsqueeze(1) * src_mask.unsqueeze(1).unsqueeze(-1)).sum(dim=2)

        src_hidden_expanded = src_hidden.unsqueeze(1) + jt.zeros_like(cand_hidden)
        hop_expanded = hop_context.unsqueeze(1) + jt.zeros_like(cand_hidden)
        feature_hidden = self.feature_proj(cand_features)
        dot = (src_hidden_expanded * cand_hidden).sum(dim=-1).unsqueeze(-1) / math.sqrt(float(self.hidden_dim))

        return self.out(jt.cat([
            src_hidden_expanded,
            cand_hidden,
            cross_context,
            hop_expanded,
            feature_hidden,
            src_hidden_expanded * cand_hidden,
            dot,
        ], dim=-1)).squeeze(-1)


def save_tgnn_model(path, model, meta):
    jt.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_tgnn_model(path):
    data = jt.load(path)
    meta = data["meta"]
    model = TemporalGNNRanker(
        meta["n_nodes"],
        meta["feature_dim"],
        node_emb_dim=meta.get("node_emb_dim", 128),
        time_emb_dim=meta.get("time_emb_dim", 32),
        hidden_dim=meta.get("hidden_dim", 192),
        dropout=meta.get("dropout", 0.1),
        max_time_bucket=meta.get("max_time_bucket", 127),
    )
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(data["state_dict"])
    else:
        model.load_parameters(data["state_dict"])
    model.eval()
    return model, meta
