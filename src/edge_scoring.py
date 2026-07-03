from pathlib import Path

import numpy as np

from .data_loader import iter_test_rows, iter_train_edges
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


def row_zscore(scores):
    scores = np.asarray(scores, dtype=np.float32)
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (scores - mean) / std


def softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    total = exp_scores.sum(axis=1, keepdims=True)
    total = np.where(total <= 0, 1.0, total)
    return exp_scores / total


def score_rule(dataset_name, train_edges, queries):
    ranker = RuleRankerV2(dataset_name)
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
    rule_idx = FEATURE_NAMES.index("rule_score")

    model.eval()
    out = []
    for start in range(0, len(queries), batch_size):
        chunk = queries[start:start + batch_size]
        x_raw = np.asarray(
            [
                [builder.vector(src, time, dst) for dst in candidates]
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
