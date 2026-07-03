from collections import deque

import numpy as np


try:
    import jittor as jt
    from jittor import nn
except Exception:  # pragma: no cover - exercised only when Jittor is missing.
    jt = None
    nn = None


def jittor_available():
    return jt is not None and nn is not None


if nn is not None:
    class NextDstSeqTower(nn.Module):
        def __init__(self, num_dst, emb_dim=64, hidden_dim=64, dropout=0.1):
            super().__init__()
            self.emb = nn.Embedding(num_dst + 1, emb_dim)
            self.bias = nn.Embedding(num_dst + 1, 1)
            self.proj = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, emb_dim),
            )

        def execute(self, src_hist_ids, cand_ids):
            hist_mask = (src_hist_ids > 0).float32()
            cand_mask = (cand_ids > 0).float32()
            hist_emb = self.emb(src_hist_ids) * hist_mask.unsqueeze(-1)
            denom = jt.maximum(hist_mask.sum(dim=1, keepdims=True), jt.ones((src_hist_ids.shape[0], 1)))
            hidden = self.proj(hist_emb.sum(dim=1) / denom)
            cand_emb = self.emb(cand_ids) * cand_mask.unsqueeze(-1)
            bias = self.bias(cand_ids).squeeze(-1) * cand_mask
            return (hidden.unsqueeze(1) * cand_emb).sum(dim=2) + bias
else:
    class NextDstSeqTower:
        def __init__(self, *args, **kwargs):
            raise ImportError("jittor is required for NextDstSeqTower")


def build_dst_vocab(edges, min_count=1):
    counts = {}
    for _, dst, _ in edges:
        counts[dst] = counts.get(dst, 0) + 1
    dsts = [dst for dst, count in counts.items() if count >= min_count]
    dsts.sort()
    return {dst: idx + 1 for idx, dst in enumerate(dsts)}


def build_source_histories(edges, dst_to_id, seq_len=50):
    histories = {}
    for src, dst, _ in sorted(edges, key=lambda x: x[2]):
        if src not in histories:
            histories[src] = deque(maxlen=seq_len)
        histories[src].append(dst_to_id.get(dst, 0))
    return histories


def history_array(histories, src_values, seq_len):
    rows = np.zeros((len(src_values), seq_len), dtype=np.int32)
    for i, src in enumerate(src_values):
        values = list(histories.get(src, ()))
        if values:
            rows[i, -len(values):] = values[-seq_len:]
    return rows


def save_seq_model(path, model, meta):
    if jt is None:
        raise ImportError("jittor is required to save seq model")
    jt.save({"state_dict": model.state_dict(), "meta": meta}, str(path))


def load_seq_model(path):
    if jt is None:
        raise ImportError("jittor is required to load seq model")
    data = jt.load(str(path))
    meta = data["meta"]
    model = NextDstSeqTower(
        int(meta["num_dst"]),
        emb_dim=int(meta.get("emb_dim", 64)),
        hidden_dim=int(meta.get("hidden_dim", 64)),
        dropout=float(meta.get("dropout", 0.0)),
    )
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(data["state_dict"])
    else:
        model.load_parameters(data["state_dict"])
    model.eval()
    return model, meta


def score_seq_model(model_path, train_edges, queries, batch_size=512):
    if jt is None:
        raise ImportError("jittor is required to score seq_nextdst models")
    model, meta = load_seq_model(model_path)
    seq_len = int(meta.get("seq_len", 50))
    dst_to_id = {int(k): int(v) for k, v in meta.get("dst_to_id", {}).items()}
    histories = build_source_histories(train_edges, dst_to_id, seq_len=seq_len)
    out = []
    for start in range(0, len(queries), batch_size):
        chunk = queries[start:start + batch_size]
        hist = history_array(histories, [src for src, _, _ in chunk], seq_len)
        cand = np.asarray(
            [[dst_to_id.get(dst, 0) for dst in candidates] for _, _, candidates in chunk],
            dtype=np.int32,
        )
        scores = model(jt.array(hist), jt.array(cand)).numpy()
        out.append(scores)
    return np.vstack(out).astype(np.float32)
