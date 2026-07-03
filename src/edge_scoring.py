from pathlib import Path

import numpy as np

from .base_intensity_v3 import BaseIntensityV3
from .data_loader import iter_test_rows, iter_train_edges
from .metrics import row_zscore, softmax
from .rule_ranker_v2 import RuleRankerV2


def load_jittor():
    import jittor as jt

    return jt


def load_jt_ranker_parts():
    from .jt_ranker import CandidateFeatureBuilder, FEATURE_NAMES, load_model, normalize_features

    return CandidateFeatureBuilder, FEATURE_NAMES, load_model, normalize_features


def load_train_edges(dataset_dir):
    return list(iter_train_edges(Path(dataset_dir) / "train.csv"))


def load_test_queries(dataset_dir):
    return [(src, time, candidates) for src, time, candidates in iter_test_rows(Path(dataset_dir) / "test.csv")]


def score_rule(dataset_name, train_edges, queries):
    ranker = RuleRankerV2(dataset_name)
    ranker.fit(train_edges)
    rows = [ranker.score_many(src, time, candidates) for src, time, candidates in queries]
    return np.asarray(rows, dtype=np.float32)


def score_base_intensity_model(dataset_name, train_edges, queries, weights=None):
    ranker = BaseIntensityV3(dataset_name, weights=weights)
    ranker.fit(train_edges)
    rows = [ranker.score_many(src, time, candidates) for src, time, candidates in queries]
    return np.asarray(rows, dtype=np.float32)


def score_edge_mlp_model(model_path, dataset_name, train_edges, queries, batch_size=512):
    jt = load_jittor()
    CandidateFeatureBuilder, FEATURE_NAMES, load_model, normalize_features = load_jt_ranker_parts()
    model, meta = load_model(model_path)
    builder = CandidateFeatureBuilder(dataset_name)
    builder.fit(train_edges)

    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    fuse_rule = float(meta.get("fuse_rule", 1.0))
    gamma = float(meta.get("gamma", 0.15))
    use_edge_mlp = bool(meta.get("use_edge_mlp", True))
    feature_names = meta.get("feature_names", FEATURE_NAMES)
    rule_idx = feature_names.index("rule_score")

    model.eval()
    out = []
    for start in range(0, len(queries), batch_size):
        chunk = queries[start:start + batch_size]
        x_raw = np.asarray(
            [
                [builder.vector(src, time, dst, feature_names=feature_names) for dst in candidates]
                for src, time, candidates in chunk
            ],
            dtype=np.float32,
        )
        rule = x_raw[:, :, rule_idx]
        if not use_edge_mlp:
            out.append(rule)
            continue
        x = normalize_features(x_raw, mean, std).astype(np.float32)
        residual = np.tanh(model(jt.array(x)).numpy())
        out.append(rule * fuse_rule + residual * gamma)
    return np.vstack(out)


def score_seq_model(model_path, train_edges, queries, batch_size=512):
    from .seq_tower import score_seq_model as _score

    return _score(model_path, train_edges, queries, batch_size=batch_size)


def score_craft_residual(model_path, dataset_name, train_edges, queries, batch_size=256):
    from .craft_residual import score_craft_residual as _score

    return _score(model_path, dataset_name, train_edges, queries, batch_size=batch_size)


def cold_mask(train_edges, queries):
    seen_dst = {dst for _, dst, _ in train_edges}
    return np.asarray(
        [[1.0 if dst not in seen_dst else 0.0 for dst in candidates] for _, _, candidates in queries],
        dtype=np.float32,
    )
