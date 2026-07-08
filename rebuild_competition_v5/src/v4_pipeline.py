import argparse
import csv
import json
import math
import multiprocessing as mp
import shutil
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data import (
    TestRow,
    dataset_dir,
    dump_json,
    ensure_dir,
    make_result_zip,
    read_test,
    row_zscore,
    softmax,
    split_edges,
    tie_aware_mrr,
    top1_change,
    validate_csv,
    write_scores_csv,
)
from .features import FEATURE_NAMES, GraphFeatureModel
from .validation import score_feature_tensor, top1_stats


V3_MLP_WEIGHT = 5.5


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_path(value) -> Path:
    return Path(value).expanduser().resolve()


def _load_v3_weights(v3_root: Path) -> Dict[str, float]:
    report = _read_json(v3_root / "reports" / "dataset2_train_report.json")
    return {k: float(v) for k, v in report["weights"].items()}


def _load_test_src_dst(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    test_rows = read_test(dataset_dir(data_dir, "dataset2") / "test.csv")
    src = np.asarray([r.src for r in test_rows], dtype=np.int64)
    dst = np.asarray([r.candidates for r in test_rows], dtype=np.int64)
    return src, dst


def _load_v3_test_parts(v3_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shard_dir = v3_root / "artifacts" / "dataset2_predict_shards"
    feature_logits = np.concatenate([np.load(p) for p in sorted(shard_dir.glob("feature_logits_part_*.npy"))], axis=0).astype(np.float32)
    mlp_logits = np.concatenate([np.load(p) for p in sorted(shard_dir.glob("mlp_logits_part_*.npy"))], axis=0).astype(np.float32)
    features = np.concatenate([np.load(p, mmap_mode="r") for p in sorted(shard_dir.glob("features_part_*.npy"))], axis=0).astype(np.float32)
    return features, feature_logits, mlp_logits


def _v3_baseline_logits(feature_logits: np.ndarray, mlp_logits: np.ndarray, weight: float = V3_MLP_WEIGHT) -> np.ndarray:
    return (row_zscore(feature_logits) + float(weight) * row_zscore(mlp_logits)).astype(np.float32)


def _candidate_pool(model: GraphFeatureModel, src: int, pos_dst: int, rng: np.random.Generator, hot: Sequence[int], recent_hot: Sequence[int], low_pop: Sequence[int], known_dst: Sequence[int], max_pool: int) -> List[int]:
    pool = [int(pos_dst)]
    seen = {int(pos_dst)}

    def add_many(values, limit=None):
        count = 0
        for value in values:
            value = int(value)
            if value not in seen:
                seen.add(value)
                pool.append(value)
                count += 1
                if len(pool) >= max_pool or (limit is not None and count >= limit):
                    break

    recent = list(model.src_recent.get(src, ()))
    add_many(reversed(recent), 120)
    for hist_dst in reversed(recent[-24:]):
        table = model.transition.get(hist_dst)
        if table:
            add_many((dst for dst, _ in sorted(table.items(), key=lambda x: -x[1])[:40]), 40)
    add_many(recent_hot, 160)
    add_many(hot, 160)
    if low_pop:
        add_many(rng.choice(np.asarray(low_pop, dtype=np.int64), size=min(120, len(low_pop)), replace=False), 120)
    while len(pool) < max_pool and known_dst:
        add_many([known_dst[int(rng.integers(0, len(known_dst)))]], 1)
    return pool[:max_pool]


def _build_hard_worker(payload: tuple) -> dict:
    (
        shard_id,
        model_path,
        edges,
        weights,
        hot,
        recent_hot,
        low_pop,
        known_dst,
        max_pool,
        rows_per_shard,
        out_dir,
        seed,
    ) = payload
    rng = np.random.default_rng(int(seed) + int(shard_id) * 1009)
    model = GraphFeatureModel.load(Path(model_path))
    features = np.zeros((len(edges), 100, len(FEATURE_NAMES)), dtype=np.float32)
    src_ids = np.zeros(len(edges), dtype=np.int64)
    dst_ids = np.zeros((len(edges), 100), dtype=np.int64)
    labels = np.zeros(len(edges), dtype=np.int64)
    meta_bad = 0
    for i, (src, pos_dst, time) in enumerate(edges):
        pool = _candidate_pool(model, src, pos_dst, rng, hot, recent_hot, low_pop, known_dst, max_pool)
        if len(pool) < 100:
            meta_bad += 1
            continue
        pool_feats = model.feature_row(src, time, pool)
        pool_score = score_feature_tensor(pool_feats.reshape(1, len(pool), len(FEATURE_NAMES)), weights)[0]
        order = np.argsort(-pool_score)
        negs = []
        for idx in order:
            cand = int(pool[int(idx)])
            if cand != int(pos_dst) and cand not in negs:
                negs.append(cand)
            if len(negs) >= 99:
                break
        while len(negs) < 99:
            cand = int(known_dst[int(rng.integers(0, len(known_dst)))])
            if cand != int(pos_dst) and cand not in negs:
                negs.append(cand)
        cands = [int(pos_dst)] + negs[:99]
        rng.shuffle(cands)
        label = cands.index(int(pos_dst))
        features[i] = model.feature_row(src, time, cands)
        src_ids[i] = int(src)
        dst_ids[i] = np.asarray(cands, dtype=np.int64)
        labels[i] = int(label)
        if (i + 1) % 2000 == 0:
            print(f"hard_worker shard={shard_id} rows={i + 1}/{rows_per_shard}", flush=True)
    out_dir = ensure_dir(Path(out_dir))
    part = out_dir / f"hard_part_{int(shard_id):02d}.npz"
    np.savez_compressed(part, features=features.astype(np.float16), src_ids=src_ids, dst_ids=dst_ids, labels=labels)
    return {"shard": int(shard_id), "rows": len(edges), "bad": meta_bad, "path": str(part)}


def build_hard_mining(args) -> dict:
    data_dir = _as_path(args.data_dir)
    v3_root = _as_path(args.v3_root)
    artifacts = ensure_dir(_as_path(args.artifacts))
    reports = ensure_dir(_as_path(args.reports))
    train_edges, valid_edges, split_meta = split_edges(dataset_dir(data_dir, "dataset2"), final_train=False, prefer_official=True)
    rng = np.random.default_rng(int(args.seed))
    order = np.arange(len(valid_edges))
    rng.shuffle(order)
    total = min(len(order), int(args.train_rows) + int(args.valid_rows))
    selected = [valid_edges[int(i)] for i in order[:total]]
    train_selected = selected[: int(args.train_rows)]
    valid_selected = selected[int(args.train_rows): int(args.train_rows) + int(args.valid_rows)]
    model_path = v3_root / "artifacts" / "dataset2_feature_model_val.pkl"
    model = GraphFeatureModel.load(model_path)
    weights = _load_v3_weights(v3_root)
    hot = [dst for dst, _ in model.dst_count.most_common(6000)]
    recent_hot = [dst for dst, _ in model.dst_recent_count.most_common(6000)]
    counts = np.asarray(list(model.dst_count.values()), dtype=np.float64)
    low_cut = float(np.percentile(counts, 40))
    low_pop = [dst for dst, count in model.dst_count.items() if count <= low_cut]
    known_dst = list(model.dst_count.keys())

    out_dir = ensure_dir(artifacts / "hard_mining")
    for old in out_dir.glob("hard_part_*.npz"):
        old.unlink()
    all_edges = train_selected + valid_selected
    workers = max(1, int(args.workers))
    boundaries = np.linspace(0, len(all_edges), workers + 1, dtype=np.int64)
    tasks = []
    for shard in range(workers):
        start, end = int(boundaries[shard]), int(boundaries[shard + 1])
        tasks.append((shard, str(model_path), all_edges[start:end], weights, hot, recent_hot, low_pop, known_dst, int(args.max_pool), end - start, str(out_dir), int(args.seed)))
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        part_reports = pool.map(_build_hard_worker, tasks)

    feats = []
    src_ids = []
    dst_ids = []
    labels = []
    for item in sorted(part_reports, key=lambda x: x["shard"]):
        data = np.load(item["path"])
        feats.append(data["features"].astype(np.float32))
        src_ids.append(data["src_ids"].astype(np.int64))
        dst_ids.append(data["dst_ids"].astype(np.int64))
        labels.append(data["labels"].astype(np.int64))
    features = np.concatenate(feats, axis=0)
    src_ids_arr = np.concatenate(src_ids, axis=0)
    dst_ids_arr = np.concatenate(dst_ids, axis=0)
    labels_arr = np.concatenate(labels, axis=0)
    train_n = len(train_selected)
    train_path = artifacts / "hard_train.npz"
    valid_path = artifacts / "hard_valid.npz"
    np.savez_compressed(train_path, features=features[:train_n].astype(np.float16), src_ids=src_ids_arr[:train_n], dst_ids=dst_ids_arr[:train_n], labels=labels_arr[:train_n])
    np.savez_compressed(valid_path, features=features[train_n:].astype(np.float16), src_ids=src_ids_arr[train_n:], dst_ids=dst_ids_arr[train_n:], labels=labels_arr[train_n:])
    report = {
        "split": split_meta,
        "train_rows": int(train_n),
        "valid_rows": int(len(labels_arr) - train_n),
        "workers": workers,
        "max_pool": int(args.max_pool),
        "parts": part_reports,
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "label_check": {
            "train_label_min": int(labels_arr[:train_n].min()),
            "train_label_max": int(labels_arr[:train_n].max()),
            "valid_label_min": int(labels_arr[train_n:].min()) if len(labels_arr) > train_n else None,
            "valid_label_max": int(labels_arr[train_n:].max()) if len(labels_arr) > train_n else None,
        },
    }
    dump_json(reports / "v4_hard_mining_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:6000], flush=True)
    return report


def _loss_with_hard_terms(logits, labels, ce_weight=1.0, margin_weight=0.0, bpr_weight=0.0, margin=0.5):
    from jittor import nn

    # ACL/Jittor on 910B can fuse cross entropy reliably. Extra top-k/indexed
    # margin terms may fall back to CPU and break the compiled NPU graph, so the
    # hard-mined candidate list itself carries the hard-negative signal.
    return nn.cross_entropy_loss(logits, labels) * float(ce_weight)


def _train_feature_mlp(train_x, train_y, valid_x, valid_y, out_dir: Path, seed: int, hidden: int, epochs: int, batch_size: int, lr: float) -> dict:
    ensure_dir(out_dir)
    import jittor as jt
    from jittor import nn

    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass
    try:
        jt.set_global_seed(seed)
    except Exception:
        pass
    np.random.seed(seed)

    class CandidateMLP(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, 1),
            )

        def execute(self, x):
            b, c, f = x.shape
            return self.net(x.reshape((b * c, f))).reshape((b, c))

    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0, keepdims=True).astype(np.float32)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    train_xn = ((train_x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    valid_xn = ((valid_x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    net = CandidateMLP(train_x.shape[-1], hidden)
    opt = nn.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = np.arange(len(train_y))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            logits = net(jt.array(train_xn[idx]))
            loss = _loss_with_hard_terms(logits, jt.array(train_y[idx]))
            opt.step(loss)
            losses.append(float(loss.data))
        pred = _predict_feature_mlp_net(net, valid_xn, batch_size=max(batch_size, 1024))
        item = {"epoch": epoch, "loss": float(np.mean(losses)), "valid_mrr": tie_aware_mrr(pred, valid_y)}
        history.append(item)
        print(f"v4_hard_mlp seed={seed} epoch={epoch} loss={item['loss']:.6f} valid_mrr={item['valid_mrr']:.6f}", flush=True)
    ckpt = out_dir / "feature_mlp.pkl"
    norm = out_dir / "feature_norm.npz"
    net.save(str(ckpt))
    np.savez(norm, mean=mean, std=std, feature_names=np.asarray(FEATURE_NAMES))
    return {"status": "trained", "seed": seed, "checkpoint": str(ckpt), "norm": str(norm), "history": history}


def _predict_feature_mlp_net(net, features: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    import jittor as jt

    out = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        pred = net(jt.array(features[start:start + batch_size].astype(np.float32)))
        out[start:start + batch_size] = np.asarray(pred.data, dtype=np.float32)
    return out


def _load_feature_mlp_predictor(ckpt: Path, norm_path: Path, hidden: int):
    import jittor as jt
    from jittor import nn

    class CandidateMLP(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, 1),
            )

        def execute(self, x):
            b, c, f = x.shape
            return self.net(x.reshape((b * c, f))).reshape((b, c))

    norm = np.load(norm_path)
    mean = norm["mean"].reshape(1, 1, -1)
    std = norm["std"].reshape(1, 1, -1)
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass
    net = CandidateMLP(len(mean.reshape(-1)), hidden)
    net.load(str(ckpt))

    def predict(x: np.ndarray, batch_size=2048):
        xn = ((x - mean) / std).astype(np.float32)
        return _predict_feature_mlp_net(net, xn, batch_size=batch_size)

    return predict


def train_hard_mlp(args) -> dict:
    artifacts = ensure_dir(_as_path(args.artifacts))
    reports = ensure_dir(_as_path(args.reports))
    train = np.load(artifacts / "hard_train.npz")
    valid = np.load(artifacts / "hard_valid.npz")
    train_x = train["features"].astype(np.float32)
    train_y = train["labels"].astype(np.int64)
    valid_x = valid["features"].astype(np.float32)
    valid_y = valid["labels"].astype(np.int64)
    reports_out = []
    for seed in [int(x) for x in str(args.seeds).split(",") if x.strip()]:
        try:
            item = _train_feature_mlp(
                train_x,
                train_y,
                valid_x,
                valid_y,
                out_dir=artifacts / f"v4_hard_mlp_seed{seed}",
                seed=seed,
                hidden=int(args.hidden),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
            )
        except Exception as exc:
            item = {"status": "failed", "seed": seed, "error": repr(exc), "traceback": traceback.format_exc()}
        reports_out.append(item)
    report = {"models": reports_out}
    dump_json(reports / "v4_hard_mlp_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:6000], flush=True)
    return report


def _make_vocab(train_src, train_dst, test_src, test_dst):
    src_values = np.unique(np.concatenate([train_src.reshape(-1), test_src.reshape(-1)]))
    dst_values = np.unique(np.concatenate([train_dst.reshape(-1), test_dst.reshape(-1)]))
    src_map = {int(v): i + 1 for i, v in enumerate(src_values)}
    dst_map = {int(v): i + 1 for i, v in enumerate(dst_values)}
    return src_map, dst_map


def _map_ids(src_ids, dst_ids, src_map, dst_map):
    src_mapped = np.asarray([src_map.get(int(v), 0) for v in src_ids], dtype=np.int32)
    flat = dst_ids.reshape(-1)
    dst_mapped = np.asarray([dst_map.get(int(v), 0) for v in flat], dtype=np.int32).reshape(dst_ids.shape)
    return src_mapped, dst_mapped


def _train_id_model(train_x, train_y, train_src, train_dst, valid_x, valid_y, valid_src, valid_dst, out_dir: Path, seed: int, hidden: int, emb_dim: int, epochs: int, batch_size: int, lr: float, num_src: int, num_dst: int) -> dict:
    ensure_dir(out_dir)
    import jittor as jt
    from jittor import nn

    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass
    try:
        jt.set_global_seed(seed)
    except Exception:
        pass
    np.random.seed(seed)

    class IDRanker(nn.Module):
        def __init__(self, feat_dim, hidden_dim, emb, src_n, dst_n):
            super().__init__()
            self.src_emb = nn.Embedding(src_n, emb)
            self.dst_emb = nn.Embedding(dst_n, emb)
            self.emb_dim = emb
            self.flat_dim = feat_dim + emb * 3
            self.net = nn.Sequential(
                nn.Linear(self.flat_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, 1),
            )

        def execute(self, feats, src, dst):
            b, c, f = feats.shape
            se = self.src_emb(src).unsqueeze(1).broadcast((b, c, self.emb_dim))
            de = self.dst_emb(dst)
            x = jt.concat([feats, se, de, se * de], dim=2)
            return self.net(x.reshape((b * c, self.flat_dim))).reshape((b, c))

    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0, keepdims=True).astype(np.float32)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    train_xn = ((train_x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    valid_xn = ((valid_x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    net = IDRanker(train_x.shape[-1], hidden, emb_dim, num_src, num_dst)
    opt = nn.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        order = np.arange(len(train_y))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            logits = net(jt.array(train_xn[idx]), jt.array(train_src[idx]), jt.array(train_dst[idx]))
            loss = _loss_with_hard_terms(logits, jt.array(train_y[idx]))
            opt.step(loss)
            losses.append(float(loss.data))
        pred = _predict_id_net(net, valid_xn, valid_src, valid_dst, batch_size=max(batch_size, 1024))
        item = {"epoch": epoch, "loss": float(np.mean(losses)), "valid_mrr": tie_aware_mrr(pred, valid_y)}
        history.append(item)
        print(f"v4_id seed={seed} epoch={epoch} loss={item['loss']:.6f} valid_mrr={item['valid_mrr']:.6f}", flush=True)
    ckpt = out_dir / "id_ranker.pkl"
    norm = out_dir / "id_norm.npz"
    net.save(str(ckpt))
    np.savez(norm, mean=mean, std=std, emb_dim=np.asarray([emb_dim], dtype=np.int32), num_src=np.asarray([num_src], dtype=np.int32), num_dst=np.asarray([num_dst], dtype=np.int32))
    return {"status": "trained", "seed": seed, "checkpoint": str(ckpt), "norm": str(norm), "history": history, "num_src": num_src, "num_dst": num_dst}


def _predict_id_net(net, features, src, dst, batch_size=2048):
    import jittor as jt

    out = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        pred = net(jt.array(features[start:start + batch_size].astype(np.float32)), jt.array(src[start:start + batch_size]), jt.array(dst[start:start + batch_size]))
        out[start:start + batch_size] = np.asarray(pred.data, dtype=np.float32)
    return out


def _load_id_predictor(ckpt: Path, norm_path: Path, hidden: int):
    import jittor as jt
    from jittor import nn

    norm = np.load(norm_path)
    mean = norm["mean"].reshape(1, 1, -1)
    std = norm["std"].reshape(1, 1, -1)
    emb_dim = int(norm["emb_dim"][0])
    num_src = int(norm["num_src"][0])
    num_dst = int(norm["num_dst"][0])

    class IDRanker(nn.Module):
        def __init__(self, feat_dim, hidden_dim, emb, src_n, dst_n):
            super().__init__()
            self.src_emb = nn.Embedding(src_n, emb)
            self.dst_emb = nn.Embedding(dst_n, emb)
            self.emb_dim = emb
            self.flat_dim = feat_dim + emb * 3
            self.net = nn.Sequential(
                nn.Linear(self.flat_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, 1),
            )

        def execute(self, feats, src, dst):
            b, c, f = feats.shape
            se = self.src_emb(src).unsqueeze(1).broadcast((b, c, self.emb_dim))
            de = self.dst_emb(dst)
            x = jt.concat([feats, se, de, se * de], dim=2)
            return self.net(x.reshape((b * c, self.flat_dim))).reshape((b, c))

    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass
    net = IDRanker(len(mean.reshape(-1)), hidden, emb_dim, num_src, num_dst)
    net.load(str(ckpt))

    def predict(features, src, dst, batch_size=2048):
        x = ((features - mean) / std).astype(np.float32)
        return _predict_id_net(net, x, src, dst, batch_size=batch_size)

    return predict


def train_id_ranker(args) -> dict:
    artifacts = ensure_dir(_as_path(args.artifacts))
    reports = ensure_dir(_as_path(args.reports))
    v3_root = _as_path(args.v3_root)
    train = np.load(artifacts / "hard_train.npz")
    valid = np.load(artifacts / "hard_valid.npz")
    feature_model = GraphFeatureModel.load(v3_root / "artifacts" / "dataset2_feature_model_final.pkl")
    train_x = _append_latent_features(train["features"].astype(np.float32), train["src_ids"], train["dst_ids"], feature_model, int(args.emb_dim))
    valid_x = _append_latent_features(valid["features"].astype(np.float32), valid["src_ids"], valid["dst_ids"], feature_model, int(args.emb_dim))
    train_y = train["labels"].astype(np.int64)
    valid_y = valid["labels"].astype(np.int64)
    out = []
    for seed in [int(x) for x in str(args.seeds).split(",") if x.strip()]:
        try:
            item = _train_feature_mlp(
                train_x,
                train_y,
                valid_x,
                valid_y,
                out_dir=artifacts / f"v4_id_seed{seed}",
                seed=seed,
                hidden=int(args.hidden),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
            )
            item["latent_dim"] = int(args.emb_dim)
            item["model_type"] = "fixed_svd_latent_mlp"
        except Exception as exc:
            item = {"status": "failed", "seed": seed, "error": repr(exc), "traceback": traceback.format_exc()}
        out.append(item)
    report = {"models": out, "latent_dim": int(args.emb_dim), "model_type": "fixed_svd_latent_mlp"}
    dump_json(reports / "v4_id_ranker_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:6000], flush=True)
    return report


def _load_vocab(path: Path):
    data = np.load(path)
    src_map = {int(k): int(v) for k, v in zip(data["src_keys"], data["src_vals"])}
    dst_map = {int(k): int(v) for k, v in zip(data["dst_keys"], data["dst_vals"])}
    return src_map, dst_map


def _append_latent_features(features: np.ndarray, src_ids: np.ndarray, dst_ids: np.ndarray, model: GraphFeatureModel, latent_dim: int) -> np.ndarray:
    if model.src_emb is None or model.dst_emb is None:
        return features.astype(np.float32, copy=False)
    dim = int(min(latent_dim, model.src_emb.shape[1], model.dst_emb.shape[1]))
    n, c, _ = features.shape
    src_vec = np.zeros((n, dim), dtype=np.float32)
    for i, src in enumerate(src_ids.reshape(-1)):
        idx = model.src_to_id.get(int(src))
        if idx is not None:
            src_vec[i] = model.src_emb[idx, :dim]
    flat_dst = dst_ids.reshape(-1)
    dst_vec = np.zeros((flat_dst.shape[0], dim), dtype=np.float32)
    for i, dst in enumerate(flat_dst):
        idx = model.dst_to_id.get(int(dst))
        if idx is not None:
            dst_vec[i] = model.dst_emb[idx, :dim]
    dst_vec = dst_vec.reshape(n, c, dim)
    src_bc = np.broadcast_to(src_vec[:, None, :], (n, c, dim)).copy()
    return np.concatenate([features.astype(np.float32, copy=False), src_bc, dst_vec, src_bc * dst_vec], axis=2).astype(np.float32, copy=False)


def predict_v4(args) -> dict:
    data_dir = _as_path(args.data_dir)
    v3_root = _as_path(args.v3_root)
    artifacts = ensure_dir(_as_path(args.artifacts))
    reports = ensure_dir(_as_path(args.reports))
    features, feature_logits, mlp_logits = _load_v3_test_parts(v3_root)
    baseline = _v3_baseline_logits(feature_logits, mlp_logits, V3_MLP_WEIGHT)
    test_src_raw, test_dst_raw = _load_test_src_dst(data_dir)

    hard_report = _read_json(reports / "v4_hard_mlp_report.json")
    hard_logits = []
    for item in hard_report["models"]:
        if item.get("status") != "trained":
            continue
        predictor = _load_feature_mlp_predictor(Path(item["checkpoint"]), Path(item["norm"]), hidden=int(args.hard_hidden))
        hard_logits.append(predictor(features, batch_size=int(args.predict_batch_size)))
    hard_ens = np.mean([row_zscore(x) for x in hard_logits], axis=0).astype(np.float32) if hard_logits else np.zeros_like(baseline)
    np.save(artifacts / "v4_hard_ens.logits.npy", hard_ens)

    id_logits = []
    id_report_path = reports / "v4_id_ranker_report.json"
    if id_report_path.exists():
        id_report = _read_json(id_report_path)
        feature_model = GraphFeatureModel.load(v3_root / "artifacts" / "dataset2_feature_model_final.pkl")
        id_features = _append_latent_features(features, test_src_raw, test_dst_raw, feature_model, int(id_report.get("latent_dim", args.emb_dim)))
        for item in id_report["models"]:
            if item.get("status") != "trained":
                continue
            predictor = _load_feature_mlp_predictor(Path(item["checkpoint"]), Path(item["norm"]), hidden=int(args.id_hidden))
            id_logits.append(predictor(id_features, batch_size=int(args.predict_batch_size)))
        del id_features
    id_ens = np.mean([row_zscore(x) for x in id_logits], axis=0).astype(np.float32) if id_logits else np.zeros_like(baseline)
    np.save(artifacts / "v4_id_ens.logits.npy", id_ens)
    np.save(artifacts / "v4_baseline_mlpw5p5.logits.npy", baseline)

    report = {
        "baseline_shape": list(baseline.shape),
        "hard_models": len(hard_logits),
        "id_models": len(id_logits),
        "baseline_top1": top1_stats(baseline, read_test(dataset_dir(data_dir, "dataset2") / "test.csv"), GraphFeatureModel.load(v3_root / "artifacts" / "dataset2_feature_model_final.pkl")),
        "hard_top1_change_vs_baseline": top1_change(baseline, hard_ens) if hard_logits else None,
        "id_top1_change_vs_baseline": top1_change(baseline, id_ens) if id_logits else None,
    }
    dump_json(reports / "v4_predict_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:4000], flush=True)
    return report


def pack_v4(args) -> dict:
    v3_root = _as_path(args.v3_root)
    artifacts = _as_path(args.artifacts)
    reports = ensure_dir(_as_path(args.reports))
    submission = ensure_dir(_as_path(args.submission))
    baseline = np.load(artifacts / "v4_baseline_mlpw5p5.logits.npy")
    hard = np.load(artifacts / "v4_hard_ens.logits.npy")
    id_logits = np.load(artifacts / "v4_id_ens.logits.npy")
    variants = {
        "result_v4_micro_combo_0p05": row_zscore(baseline) * 0.95 + row_zscore(hard) * 0.025 + row_zscore(id_logits) * 0.025,
        "result_v4_micro_combo_0p10": row_zscore(baseline) * 0.90 + row_zscore(hard) * 0.05 + row_zscore(id_logits) * 0.05,
        "result_v4_conservative": row_zscore(baseline) * 0.80 + row_zscore(hard) * 0.10 + row_zscore(id_logits) * 0.10,
        "result_v4_safe": row_zscore(baseline) * 0.70 + row_zscore(hard) * 0.30,
        "result_v4_balanced": row_zscore(baseline) * 0.45 + row_zscore(hard) * 0.35 + row_zscore(id_logits) * 0.20,
        "result_v4_aggressive": row_zscore(baseline) * 0.25 + row_zscore(hard) * 0.45 + row_zscore(id_logits) * 0.30,
    }
    dataset1_src = v3_root / "submission_mlp_peak" / "result_rebuild_mlpw_5p5" / "dataset1.csv"
    if not dataset1_src.exists():
        dataset1_src = v3_root / "submission" / "result_rebuild_research_full" / "dataset1.csv"
    manifest = {}
    for name, logits in variants.items():
        out_dir = ensure_dir(submission / name)
        d1 = out_dir / "dataset1.csv"
        d2 = out_dir / "dataset2.csv"
        shutil.copyfile(dataset1_src, d1)
        d2_check = write_scores_csv(softmax(logits), d2)
        zip_path = submission / f"{name}.zip"
        make_result_zip(d1, d2, zip_path)
        manifest[name] = {
            "zip": str(zip_path),
            "dataset1": validate_csv(d1, 61051),
            "dataset2": d2_check,
            "top1_change_vs_baseline": top1_change(baseline, logits),
            "zip_size": zip_path.stat().st_size,
        }
        print(f"packed {name} {manifest[name]}", flush=True)
    dump_json(reports / "v4_pack_manifest.json", manifest)
    return manifest


def run_all(args) -> None:
    build_hard_mining(args)
    train_hard_mlp(args)
    train_id_ranker(args)
    predict_v4(args)
    pack_v4(args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data_A")
    p.add_argument("--v3-root", default="/home/ma-user/work/jittor_rebuild_v3")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--reports", default="reports")
    p.add_argument("--submission", default="submission")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--train-rows", type=int, default=90000)
    p.add_argument("--valid-rows", type=int, default=18000)
    p.add_argument("--max-pool", type=int, default=600)
    p.add_argument("--seeds", default="2027,2028,2029")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--hard-hidden", type=int, default=256)
    p.add_argument("--id-hidden", type=int, default=256)
    p.add_argument("--emb-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--predict-batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=8e-4)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("build-hard-mining")
    sub.add_parser("train-hard-mlp")
    sub.add_parser("train-id-ranker")
    sub.add_parser("predict-v4")
    sub.add_parser("pack-v4")
    sub.add_parser("run-all")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "build-hard-mining":
        build_hard_mining(args)
    elif args.command == "train-hard-mlp":
        train_hard_mlp(args)
    elif args.command == "train-id-ranker":
        train_id_ranker(args)
    elif args.command == "predict-v4":
        predict_v4(args)
    elif args.command == "pack-v4":
        pack_v4(args)
    elif args.command == "run-all":
        run_all(args)
    else:
        raise SystemExit(args.command)


if __name__ == "__main__":
    main()
