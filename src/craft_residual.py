import numpy as np

from .seq_tower import build_dst_vocab, build_source_histories, history_array


try:
    import jittor as jt
    from jittor import nn
except Exception:  # pragma: no cover - exercised only when Jittor is missing.
    jt = None
    nn = None


if nn is not None:
    class CraftResidual(nn.Module):
        def __init__(self, num_dst, feature_dim, emb_dim=64, hidden_dim=64, dropout=0.1):
            super().__init__()
            self.emb = nn.Embedding(num_dst + 1, emb_dim)
            self.feat = nn.Sequential(
                nn.Linear(feature_dim + 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def execute(self, src_hist_ids, cand_ids, base_feats):
            hist_mask = (src_hist_ids > 0).float32()
            cand_mask = (cand_ids > 0).float32()
            hist_emb = self.emb(src_hist_ids) * hist_mask.unsqueeze(-1)
            cand_emb = self.emb(cand_ids) * cand_mask.unsqueeze(-1)
            match = (hist_emb.unsqueeze(1) * cand_emb.unsqueeze(2)).sum(dim=3)
            match = match.max(dim=2)
            if isinstance(match, tuple):
                match = match[0]
            hist_count = hist_mask.sum(dim=1, keepdims=True)
            hist_count = hist_count.broadcast(cand_ids.shape)
            x = jt.concat([base_feats, match.unsqueeze(-1), hist_count.unsqueeze(-1)], dim=2)
            flat = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            return self.feat(flat).reshape(x.shape[0], x.shape[1])
else:
    class CraftResidual:
        def __init__(self, *args, **kwargs):
            raise ImportError("jittor is required for CraftResidual")


def save_craft_model(path, model, meta):
    if jt is None:
        raise ImportError("jittor is required to save craft model")
    jt.save({"state_dict": model.state_dict(), "meta": meta}, str(path))


def load_craft_model(path):
    if jt is None:
        raise ImportError("jittor is required to load craft model")
    data = jt.load(str(path))
    meta = data["meta"]
    model = CraftResidual(
        int(meta["num_dst"]),
        int(meta["feature_dim"]),
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


def score_craft_residual(model_path, dataset_name, train_edges, queries, batch_size=256):
    if jt is None:
        raise ImportError("jittor is required to score craft residual models")
    from .jt_ranker import CandidateFeatureBuilder, FEATURE_NAMES, normalize_features

    model, meta = load_craft_model(model_path)
    seq_len = int(meta.get("seq_len", 50))
    feature_names = meta.get("feature_names", FEATURE_NAMES)
    dst_to_id = {int(k): int(v) for k, v in meta.get("dst_to_id", {}).items()}
    histories = build_source_histories(train_edges, dst_to_id, seq_len=seq_len)
    builder = CandidateFeatureBuilder(dataset_name)
    builder.fit(train_edges)
    mean = np.asarray(meta.get("mean", [0.0] * len(feature_names)), dtype=np.float32)
    std = np.asarray(meta.get("std", [1.0] * len(feature_names)), dtype=np.float32)
    out = []
    for start in range(0, len(queries), batch_size):
        chunk = queries[start:start + batch_size]
        hist = history_array(histories, [src for src, _, _ in chunk], seq_len)
        cand = np.asarray(
            [[dst_to_id.get(dst, 0) for dst in candidates] for _, _, candidates in chunk],
            dtype=np.int32,
        )
        raw = np.asarray(
            [
                [builder.vector(src, time, dst, feature_names=feature_names) for dst in candidates]
                for src, time, candidates in chunk
            ],
            dtype=np.float32,
        )
        feats = normalize_features(raw, mean, std).astype(np.float32)
        scores = model(jt.array(hist), jt.array(cand), jt.array(feats)).numpy()
        out.append(np.tanh(scores))
    return np.vstack(out).astype(np.float32)
