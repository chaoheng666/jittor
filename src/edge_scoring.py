from pathlib import Path

import numpy as np

from .data_loader import iter_test_rows, iter_train_edges
from .rule_ranker_v2 import RuleRankerV2


def load_jittor():
    import jittor as jt

    return jt


def load_jt_ranker_parts():
    from .jt_ranker import (
        CandidateFeatureBuilder,
        FEATURE_NAMES,
        load_model,
        normalize_features,
        normalize_query_features,
    )

    return CandidateFeatureBuilder, FEATURE_NAMES, load_model, normalize_features, normalize_query_features


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
    CandidateFeatureBuilder, FEATURE_NAMES, load_model, normalize_features, normalize_query_features = load_jt_ranker_parts()
    model, meta = load_model(model_path)
    builder = CandidateFeatureBuilder(
        dataset_name,
        history_len=int(meta.get("history_len", 60)),
        node_values=meta.get("node_values"),
    )
    builder.fit(train_edges)

    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    query_mean = np.asarray(meta["query_mean"], dtype=np.float32)
    query_std = np.asarray(meta["query_std"], dtype=np.float32)
    use_craft_model = bool(meta.get("use_craft_model", True))
    rule_idx = FEATURE_NAMES.index("rule_score")

    model.eval()
    out = []
    for start in range(0, len(queries), batch_size):
        chunk = queries[start:start + batch_size]
        arrays = builder.arrays_for_queries(chunk)
        rule = arrays["x"][:, :, rule_idx]
        if not use_craft_model:
            out.append(rule)
            continue
        x = normalize_features(arrays["x"], mean, std).astype(np.float32)
        query = normalize_query_features(arrays["query"], query_mean, query_std).astype(np.float32)
        scores = model(
            jt.array(x),
            jt.array(arrays["candidate_idx"].astype(np.int32)),
            jt.array(arrays["history_idx"].astype(np.int32)),
            jt.array(arrays["history_delta"].astype(np.float32)),
            jt.array(arrays["history_mask"].astype(np.float32)),
            jt.array(query),
        ).numpy()
        out.append(scores)
    return np.vstack(out)
