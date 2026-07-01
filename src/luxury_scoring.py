import csv
from pathlib import Path

import numpy as np
import jittor as jt

from .data_loader import iter_test_rows, iter_train_edges
from .jt_ranker import CandidateFeatureBuilder, FEATURE_NAMES, load_model, normalize_features
from .seq_ranker import SequenceFeatureBuilder, load_seq_model


def iter_valid_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates = [int(row[f"c{i}"]) for i in range(1, 101)]
            yield int(row["src"]), int(row["time"]), int(row["label"]), candidates


def load_valid_queries(dataset_dir, max_rows=0):
    rows = []
    labels = []
    for src, time, label, candidates in iter_valid_rows(Path(dataset_dir) / "valid.csv"):
        if max_rows and len(rows) >= max_rows:
            break
        rows.append((src, time, candidates))
        labels.append(label)
    return rows, np.asarray(labels, dtype=np.int32)


def load_test_queries(dataset_dir):
    return [(src, time, candidates) for src, time, candidates in iter_test_rows(Path(dataset_dir) / "test.csv")]


def rank_of_label(scores, label):
    positive_score = scores[label]
    rank = 1
    for i, score in enumerate(scores):
        if i != label and score > positive_score:
            rank += 1
    return rank


def mrr(scores, labels):
    if len(labels) == 0:
        return 0.0
    rr = 0.0
    for row, label in zip(scores, labels):
        rr += 1.0 / rank_of_label(row, int(label))
    return rr / len(labels)


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


def build_feature_array(dataset_name, train_edges, queries):
    builder = CandidateFeatureBuilder(dataset_name)
    builder.fit(train_edges)
    rows = [builder.matrix(src, time, candidates) for src, time, candidates in queries]
    return np.asarray(rows, dtype=np.float32)


def rule_scores_from_features(x_raw):
    rule_idx = FEATURE_NAMES.index("rule_score")
    return x_raw[:, :, rule_idx]


def score_rule(dataset_name, train_edges, queries):
    return rule_scores_from_features(build_feature_array(dataset_name, train_edges, queries))


def load_cached_x(cache_dir, dataset_name, kind):
    return np.load(Path(cache_dir) / dataset_name / f"x_{kind}.npy", mmap_mode="r")


def cached_rule_scores(cache_dir, dataset_name, kind):
    return rule_scores_from_features(load_cached_x(cache_dir, dataset_name, kind))


def score_mlp_model(model_path, dataset_name, train_edges, queries, batch_size=512, score_mode="fused"):
    model, meta = load_model(model_path)
    x_raw = build_feature_array(dataset_name, train_edges, queries)
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    fuse_rule = float(meta.get("fuse_rule", 1.0))
    mlp_weight = float(meta.get("mlp_weight", 1.0))
    rule = rule_scores_from_features(x_raw)

    x = normalize_features(x_raw, mean, std).astype(np.float32)
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        residual = np.tanh(model(jt.array(x[start:end])).numpy())
        if score_mode == "residual":
            out.append(residual)
        else:
            out.append(rule[start:end] * fuse_rule + residual * mlp_weight)
    return np.vstack(out)


def score_mlp_model_cached(model_path, dataset_name, cache_dir, kind, batch_size=512, score_mode="fused"):
    model, meta = load_model(model_path)
    x_raw = load_cached_x(cache_dir, dataset_name, kind)
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    fuse_rule = float(meta.get("fuse_rule", 1.0))
    mlp_weight = float(meta.get("mlp_weight", 1.0))
    rule = rule_scores_from_features(x_raw)

    model.eval()
    out = []
    for start in range(0, len(x_raw), batch_size):
        end = min(start + batch_size, len(x_raw))
        x = normalize_features(np.asarray(x_raw[start:end], dtype=np.float32), mean, std).astype(np.float32)
        residual = np.tanh(model(jt.array(x)).numpy())
        if score_mode == "residual":
            out.append(residual)
        else:
            out.append(rule[start:end] * fuse_rule + residual * mlp_weight)
    return np.vstack(out)


def score_seq_model(model_path, dataset_name, train_edges, queries, batch_size=256, score_mode="fused"):
    model, meta = load_seq_model(model_path)
    feature_builder = CandidateFeatureBuilder(dataset_name)
    feature_builder.fit(train_edges)

    seq_builder = SequenceFeatureBuilder(meta.get("seq_len", 50))
    seq_builder.load_dst_values(meta["dst_values"])
    seq_builder.fit_history(train_edges)

    x_rows = []
    seq_dst = []
    seq_gap = []
    cand_idx = []
    for src, time, candidates in queries:
        sdst, sgap, cidx = seq_builder.build_query(src, time, candidates)
        x_rows.append(feature_builder.matrix(src, time, candidates))
        seq_dst.append(sdst)
        seq_gap.append(sgap)
        cand_idx.append(cidx)

    x_raw = np.asarray(x_rows, dtype=np.float32)
    seq_dst = np.asarray(seq_dst, dtype=np.int32)
    seq_gap = np.asarray(seq_gap, dtype=np.int32)
    cand_idx = np.asarray(cand_idx, dtype=np.int32)

    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    x = normalize_features(x_raw, mean, std).astype(np.float32)
    rule = rule_scores_from_features(x_raw)
    fuse_rule = float(meta.get("fuse_rule", 1.0))
    gamma = float(meta.get("gamma", 0.2))

    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        residual = np.tanh(model(
            jt.array(seq_dst[start:end]),
            jt.array(seq_gap[start:end]),
            jt.array(cand_idx[start:end]),
            jt.array(x[start:end]),
        ).numpy())
        if score_mode == "residual":
            out.append(residual)
        else:
            out.append(rule[start:end] * fuse_rule + residual * gamma)
    return np.vstack(out)


def score_seq_model_cached(model_path, dataset_name, cache_dir, kind, batch_size=256, score_mode="fused"):
    model, meta = load_seq_model(model_path)
    dataset_cache = Path(cache_dir) / dataset_name
    seq_len = int(meta.get("seq_len", 50))
    x_raw = np.load(dataset_cache / f"x_{kind}.npy", mmap_mode="r")
    seq_dst = np.load(dataset_cache / f"seq_l{seq_len}_{kind}_dst.npy", mmap_mode="r")
    seq_gap = np.load(dataset_cache / f"seq_l{seq_len}_{kind}_gap.npy", mmap_mode="r")
    cand_idx = np.load(dataset_cache / f"seq_l{seq_len}_{kind}_cand.npy", mmap_mode="r")

    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    rule = rule_scores_from_features(x_raw)
    fuse_rule = float(meta.get("fuse_rule", 1.0))
    gamma = float(meta.get("gamma", 0.2))

    model.eval()
    out = []
    for start in range(0, len(x_raw), batch_size):
        end = min(start + batch_size, len(x_raw))
        x = normalize_features(np.asarray(x_raw[start:end], dtype=np.float32), mean, std).astype(np.float32)
        residual = np.tanh(model(
            jt.array(seq_dst[start:end]),
            jt.array(seq_gap[start:end]),
            jt.array(cand_idx[start:end]),
            jt.array(x),
        ).numpy())
        if score_mode == "residual":
            out.append(residual)
        else:
            out.append(rule[start:end] * fuse_rule + residual * gamma)
    return np.vstack(out)


def load_train_edges(dataset_dir):
    return list(iter_train_edges(Path(dataset_dir) / "train.csv"))
