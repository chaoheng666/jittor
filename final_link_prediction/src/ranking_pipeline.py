import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data import (
    dataset_dir,
    dump_json,
    ensure_dir,
    make_result_zip,
    read_test,
    read_train,
    row_rank_score,
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
from .baseline_support import (
    BASELINE_MLP_WEIGHT,
    _as_path,
    _build_hard_worker,
    _candidate_pool,
    _baseline_logits,
    _load_feature_mlp_predictor,
    _load_baseline_test_parts,
    _load_baseline_weights,
    _read_json,
    _train_feature_mlp,
)


def _fit_model(edges: Sequence[Tuple[int, int, int]], test_rows, out_path: Path, seed: int, svd_dim: int) -> GraphFeatureModel:
    model = GraphFeatureModel("dataset2", svd_dim=svd_dim, seed=seed).fit(edges, test_rows)
    model.save(out_path)
    return model


class SequenceContext:
    def __init__(self, model: GraphFeatureModel, edges: Sequence[Tuple[int, int, int]], dst_seq_len: int = 64):
        self.model = model
        self.dst_seq_len = int(dst_seq_len)
        self.dst_recent_src: Dict[int, List[int]] = {}
        temp: Dict[int, List[int]] = {}
        for src, dst, _time in sorted(edges, key=lambda x: (x[2], x[0], x[1])):
            arr = temp.setdefault(int(dst), [])
            arr.append(int(src))
            if len(arr) > self.dst_seq_len:
                del arr[0: len(arr) - self.dst_seq_len]
        self.dst_recent_src = temp
        self.audience_mean: Dict[int, np.ndarray] = {}
        self.audience_count: Dict[int, float] = {}
        if model.src_emb is None:
            return
        dim = model.src_emb.shape[1]
        for dst, srcs in self.dst_recent_src.items():
            vecs = []
            for src in srcs:
                idx = model.src_to_id.get(int(src))
                if idx is not None:
                    vecs.append(model.src_emb[idx])
            if vecs:
                arr = np.asarray(vecs, dtype=np.float32)
                mean = arr.mean(axis=0)
                norm = max(float(np.linalg.norm(mean)), 1e-6)
                self.audience_mean[int(dst)] = (mean / norm).astype(np.float32)
                self.audience_count[int(dst)] = float(np.log1p(len(vecs)))
        self.max_audience_count = max(self.audience_count.values(), default=1.0)


def _append_sequence_features(base: np.ndarray, src_ids: np.ndarray, dst_ids: np.ndarray, ctx: SequenceContext, src_seq_len: int = 64) -> np.ndarray:
    model = ctx.model
    if model.src_emb is None or model.dst_emb is None:
        return base.astype(np.float32, copy=False)
    dim = min(model.src_emb.shape[1], model.dst_emb.shape[1])
    n, c, _ = base.shape
    extra = np.zeros((n, c, 8), dtype=np.float32)
    zero_src = np.zeros(dim, dtype=np.float32)
    zero_dst = np.zeros(dim, dtype=np.float32)
    for i in range(n):
        src = int(src_ids[i])
        src_idx = model.src_to_id.get(src)
        src_vec = model.src_emb[src_idx] if src_idx is not None else zero_src
        recent = list(model.src_recent.get(src, ())) [-int(src_seq_len):]
        hist_vecs = []
        for hist_dst in recent:
            idx = model.dst_to_id.get(int(hist_dst))
            if idx is not None:
                hist_vecs.append(model.dst_emb[idx])
        if hist_vecs:
            hist = np.asarray(hist_vecs, dtype=np.float32)
            hist_mean = hist.mean(axis=0)
            hist_mean = hist_mean / max(float(np.linalg.norm(hist_mean)), 1e-6)
            hist_last = hist[-1]
            hist_last = hist_last / max(float(np.linalg.norm(hist_last)), 1e-6)
        else:
            hist = None
            hist_mean = zero_dst
            hist_last = zero_dst
        dst_vecs = np.zeros((c, dim), dtype=np.float32)
        aud_vecs = np.zeros((c, dim), dtype=np.float32)
        aud_count = np.zeros(c, dtype=np.float32)
        recent_src_hit = np.zeros(c, dtype=np.float32)
        for j, dst in enumerate(dst_ids[i]):
            dst = int(dst)
            didx = model.dst_to_id.get(dst)
            if didx is not None:
                v = model.dst_emb[didx]
                dst_vecs[j] = v / max(float(np.linalg.norm(v)), 1e-6)
            av = ctx.audience_mean.get(dst)
            if av is not None:
                aud_vecs[j] = av
                aud_count[j] = ctx.audience_count.get(dst, 0.0) / max(ctx.max_audience_count, 1.0)
            if src in ctx.dst_recent_src.get(dst, ()):
                recent_src_hit[j] = 1.0
        seq_mean = dst_vecs @ hist_mean
        seq_last = dst_vecs @ hist_last
        if hist is not None and len(hist):
            hist_norm = hist / np.maximum(np.linalg.norm(hist, axis=1, keepdims=True), 1e-6)
            seq_max = (dst_vecs @ hist_norm.T).max(axis=1)
        else:
            seq_max = np.zeros(c, dtype=np.float32)
        aud_dot = aud_vecs @ (src_vec / max(float(np.linalg.norm(src_vec)), 1e-6))
        extra[i, :, 0] = seq_mean
        extra[i, :, 1] = seq_max
        extra[i, :, 2] = seq_last
        extra[i, :, 3] = aud_dot
        extra[i, :, 4] = aud_count
        extra[i, :, 5] = recent_src_hit
        extra[i, :, 6] = row_rank_score(seq_max)[0]
        extra[i, :, 7] = row_rank_score(aud_dot)[0]
    return np.concatenate([base.astype(np.float32, copy=False), extra], axis=2).astype(np.float32, copy=False)


def _hard_lists_from_edges(
    model_path: Path,
    model: GraphFeatureModel,
    edges: Sequence[Tuple[int, int, int]],
    weights: Dict[str, float],
    out_dir: Path,
    prefix: str,
    workers: int,
    max_pool: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    hot = [dst for dst, _ in model.dst_count.most_common(8000)]
    recent_hot = [dst for dst, _ in model.dst_recent_count.most_common(8000)]
    counts = np.asarray(list(model.dst_count.values()), dtype=np.float64)
    low_cut = float(np.percentile(counts, 40)) if len(counts) else 0.0
    low_pop = [dst for dst, count in model.dst_count.items() if count <= low_cut]
    known_dst = list(model.dst_count.keys())

    ensure_dir(out_dir)
    for old in out_dir.glob(f"{prefix}_part_*.npz"):
        old.unlink()
    workers = max(1, int(workers))
    boundaries = np.linspace(0, len(edges), workers + 1, dtype=np.int64)
    tasks = []
    for shard in range(workers):
        start, end = int(boundaries[shard]), int(boundaries[shard + 1])
        tasks.append(
            (
                shard,
                str(model_path),
                list(edges[start:end]),
                weights,
                hot,
                recent_hot,
                low_pop,
                known_dst,
                int(max_pool),
                end - start,
                str(out_dir),
                int(seed),
            )
        )
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        part_reports = pool.map(_build_hard_worker, tasks)

    part_files = sorted(out_dir.glob("hard_part_*.npz"))
    features = []
    src_ids = []
    dst_ids = []
    labels = []
    for i, path in enumerate(part_files):
        renamed = out_dir / f"{prefix}_part_{i:02d}.npz"
        path.rename(renamed)
        data = np.load(renamed)
        features.append(data["features"])
        src_ids.append(data["src_ids"])
        dst_ids.append(data["dst_ids"])
        labels.append(data["labels"])
    return (
        np.concatenate(features, axis=0).astype(np.float16),
        np.concatenate(src_ids, axis=0),
        np.concatenate(dst_ids, axis=0),
        np.concatenate(labels, axis=0),
        part_reports,
    )


def build_context_features(args) -> dict:
    data_dir = _as_path(args.data_dir)
    baseline_root = _as_path(args.baseline_root)
    artifacts = ensure_dir(_as_path(args.artifacts))
    reports = ensure_dir(_as_path(args.reports))
    ds_dir = dataset_dir(data_dir, "dataset2")
    test_rows = read_test(ds_dir / "test.csv")
    split0, split1, split_meta = split_edges(ds_dir, final_train=False, prefer_official=True)
    split0 = sorted(split0, key=lambda x: (x[2], x[0], x[1]))
    split1 = sorted(split1, key=lambda x: (x[2], x[0], x[1]))
    if int(args.fit_edge_limit) > 0:
        keep = int(args.fit_edge_limit)
        split0 = split0[: min(len(split0), keep)]
        split1 = split1[: min(len(split1), max(1000, keep // 5))]
        test_rows = test_rows[: min(len(test_rows), 3000)]

    cut = max(1, min(len(split0) - 1, int(len(split0) * float(args.history_frac))))
    history_edges = split0[:cut]
    train_pool = split0[cut:]
    rng = np.random.default_rng(int(args.seed))
    train_order = rng.permutation(len(train_pool))[: min(int(args.train_rows), len(train_pool))]
    valid_order = rng.permutation(len(split1))[: min(int(args.valid_rows), len(split1))]
    train_edges = [train_pool[int(i)] for i in train_order]
    valid_edges = [split1[int(i)] for i in valid_order]

    model_dir = ensure_dir(artifacts / "models")
    block_model_path = model_dir / "dataset2_block_history_model.pkl"
    valid_model_path = model_dir / "dataset2_split0_valid_model.pkl"
    final_model_path = model_dir / "dataset2_all_train_final_model.pkl"
    print(f"[final] fitting block history model edges={len(history_edges)}", flush=True)
    block_model = _fit_model(history_edges, test_rows, block_model_path, int(args.seed), int(args.svd_dim))
    print(f"[final] fitting split0 validation model edges={len(split0)}", flush=True)
    valid_model = _fit_model(split0, test_rows, valid_model_path, int(args.seed) + 1, int(args.svd_dim))

    weights = _load_baseline_weights(baseline_root)
    hard_dir = ensure_dir(artifacts / "block_hard")
    print(f"[final] building train hard lists rows={len(train_edges)}", flush=True)
    train_x, train_src, train_dst, train_y, train_parts = _hard_lists_from_edges(
        block_model_path,
        block_model,
        train_edges,
        weights,
        hard_dir,
        "train",
        int(args.workers),
        int(args.max_pool),
        int(args.seed),
    )
    print("[final] appending train src-sequence and dst-audience tower features", flush=True)
    train_ctx = SequenceContext(block_model, history_edges, dst_seq_len=int(args.dst_seq_len))
    train_x = _append_sequence_features(train_x.astype(np.float32), train_src, train_dst, train_ctx, src_seq_len=int(args.src_seq_len)).astype(np.float16)
    print(f"[final] building valid hard lists rows={len(valid_edges)}", flush=True)
    valid_x, valid_src, valid_dst, valid_y, valid_parts = _hard_lists_from_edges(
        valid_model_path,
        valid_model,
        valid_edges,
        weights,
        hard_dir,
        "valid",
        int(args.workers),
        int(args.max_pool),
        int(args.seed) + 17,
    )
    print("[final] appending valid src-sequence and dst-audience tower features", flush=True)
    valid_ctx = SequenceContext(valid_model, split0, dst_seq_len=int(args.dst_seq_len))
    valid_x = _append_sequence_features(valid_x.astype(np.float32), valid_src, valid_dst, valid_ctx, src_seq_len=int(args.src_seq_len)).astype(np.float16)
    train_path = artifacts / "final_train.npz"
    valid_path = artifacts / "final_valid.npz"
    np.savez_compressed(train_path, features=train_x, src_ids=train_src, dst_ids=train_dst, labels=train_y)
    np.savez_compressed(valid_path, features=valid_x, src_ids=valid_src, dst_ids=valid_dst, labels=valid_y)

    print(f"[final] fitting final all-train model", flush=True)
    all_rows = read_train(ds_dir / "train.csv")
    all_edges = [(r.src, r.dst, r.time) for r in all_rows]
    if int(args.fit_edge_limit) > 0:
        all_edges = all_edges[: min(len(all_edges), int(args.fit_edge_limit))]
    _fit_model(all_edges, test_rows, final_model_path, int(args.seed) + 2, int(args.svd_dim))

    valid_feature_logits = score_feature_tensor(valid_x[:, :, : len(FEATURE_NAMES)].astype(np.float32), weights)
    report = {
        "split": split_meta,
        "history_frac": float(args.history_frac),
        "history_edges": len(history_edges),
        "train_pool_edges": len(train_pool),
        "train_rows": int(len(train_y)),
        "valid_rows": int(len(valid_y)),
        "svd_dim": int(args.svd_dim),
        "src_seq_len": int(args.src_seq_len),
        "dst_seq_len": int(args.dst_seq_len),
        "feature_dim": int(train_x.shape[-1]),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "block_model": str(block_model_path),
        "valid_model": str(valid_model_path),
        "final_model": str(final_model_path),
        "feature_valid_mrr": tie_aware_mrr(valid_feature_logits, valid_y),
        "label_check": {
            "train_min": int(train_y.min()),
            "train_max": int(train_y.max()),
            "valid_min": int(valid_y.min()),
            "valid_max": int(valid_y.max()),
        },
        "parts": {"train": train_parts, "valid": valid_parts},
    }
    dump_json(reports / "final_build_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:6000], flush=True)
    return report


def train_context_ranker(args) -> dict:
    artifacts = ensure_dir(_as_path(args.artifacts))
    reports = ensure_dir(_as_path(args.reports))
    train = np.load(artifacts / "final_train.npz")
    valid = np.load(artifacts / "final_valid.npz")
    train_x = train["features"].astype(np.float32)
    valid_x = valid["features"].astype(np.float32)
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
                out_dir=artifacts / f"final_mlp_seed{seed}",
                seed=seed,
                hidden=int(args.hidden),
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
            )
        except Exception as exc:
            item = {"status": "failed", "seed": seed, "error": repr(exc)}
        out.append(item)
    report = {"models": out}
    dump_json(reports / "final_model_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:6000], flush=True)
    return report


def _feature_tensor_sharded(model_path: Path, rows, out_dir: Path, workers: int) -> np.ndarray:
    ensure_dir(out_dir)
    for old in out_dir.glob("features_part_*.npy"):
        old.unlink()
    workers = max(1, int(workers))
    bounds = np.linspace(0, len(rows), workers + 1, dtype=np.int64)
    tasks = []
    for shard in range(workers):
        tasks.append((shard, str(model_path), rows[int(bounds[shard]): int(bounds[shard + 1])], str(out_dir)))
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        pool.map(_feature_worker, tasks)
    return np.concatenate([np.load(p, mmap_mode="r") for p in sorted(out_dir.glob("features_part_*.npy"))], axis=0).astype(np.float32)


def _feature_worker(payload: tuple) -> str:
    shard, model_path, rows, out_dir = payload
    model = GraphFeatureModel.load(Path(model_path))
    feats = model.feature_tensor(rows, progress_every=10000)
    path = Path(out_dir) / f"features_part_{int(shard):02d}.npy"
    np.save(path, feats.astype(np.float32))
    return str(path)


def predict_context_ranker(args) -> dict:
    data_dir = _as_path(args.data_dir)
    baseline_root = _as_path(args.baseline_root)
    artifacts = ensure_dir(_as_path(args.artifacts))
    reports = ensure_dir(_as_path(args.reports))
    build = _read_json(reports / "final_build_report.json")
    test_rows = read_test(dataset_dir(data_dir, "dataset2") / "test.csv")
    final_model_path = Path(build["final_model"])
    if str(args.reuse_baseline_features) == "1":
        features, feature_logits, mlp_logits = _load_baseline_test_parts(baseline_root)
    else:
        features = _feature_tensor_sharded(final_model_path, test_rows, artifacts / "final_test_features", int(args.workers))
        weights = _load_baseline_weights(baseline_root)
        feature_logits = score_feature_tensor(features, weights).astype(np.float32)
        _, _, mlp_logits = _load_baseline_test_parts(baseline_root)
    final_model = GraphFeatureModel.load(final_model_path)
    test_src = np.asarray([r.src for r in test_rows], dtype=np.int64)
    test_dst = np.asarray([r.candidates for r in test_rows], dtype=np.int64)
    print("[final] appending test src-sequence and dst-audience tower features", flush=True)
    final_ctx = SequenceContext(final_model, [(r.src, r.dst, r.time) for r in read_train(dataset_dir(data_dir, "dataset2") / "train.csv")], dst_seq_len=int(args.dst_seq_len))
    features = _append_sequence_features(features.astype(np.float32), test_src, test_dst, final_ctx, src_seq_len=int(args.src_seq_len))
    baseline = _baseline_logits(feature_logits, mlp_logits, BASELINE_MLP_WEIGHT)

    model_report = _read_json(reports / "final_model_report.json")
    logits = []
    for item in model_report["models"]:
        if item.get("status") != "trained":
            continue
        predictor = _load_feature_mlp_predictor(Path(item["checkpoint"]), Path(item["norm"]), hidden=int(args.hidden))
        logits.append(predictor(features, batch_size=int(args.predict_batch_size)))
    if not logits:
        raise RuntimeError("no trained context ranking models")
    context_logits = np.mean([row_zscore(x) for x in logits], axis=0).astype(np.float32)
    np.save(artifacts / "final_context_ensemble.logits.npy", context_logits)
    np.save(artifacts / "final_baseline.logits.npy", baseline)
    report = {
        "models": len(logits),
        "shape": list(context_logits.shape),
        "reuse_baseline_features": str(args.reuse_baseline_features),
        "top1_change_vs_baseline": top1_change(baseline, context_logits),
        "baseline_top1": top1_stats(baseline, test_rows, final_model),
        "context_top1": top1_stats(context_logits, test_rows, final_model),
    }
    dump_json(reports / "final_predict_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:5000], flush=True)
    return report


def package_final_result(args) -> dict:
    baseline_root = _as_path(args.baseline_root)
    artifacts = _as_path(args.artifacts)
    reports = ensure_dir(_as_path(args.reports))
    submission = ensure_dir(_as_path(args.submission))
    baseline = np.load(artifacts / "final_baseline.logits.npy")
    context_logits = np.load(artifacts / "final_context_ensemble.logits.npy")
    blend_weight = float(args.blend_weight)
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError(f"blend_weight must be between 0 and 1, got {blend_weight}")
    name = str(args.output_name or "result_final_blend_0p10")
    logits = row_zscore(baseline) * (1.0 - blend_weight) + row_zscore(context_logits) * blend_weight
    dataset1_src = baseline_root / "submission_mlp_peak" / "result_rebuild_mlpw_5p5" / "dataset1.csv"
    if not dataset1_src.exists():
        dataset1_src = baseline_root / "submission" / "result_rebuild_research_full" / "dataset1.csv"
    out_dir = ensure_dir(submission / name)
    d1 = out_dir / "dataset1.csv"
    d2 = out_dir / "dataset2.csv"
    shutil.copyfile(dataset1_src, d1)
    d2_check = write_scores_csv(softmax(logits), d2)
    zip_path = submission / f"{name}.zip"
    make_result_zip(d1, d2, zip_path)
    manifest = {
        name: {
            "zip": str(zip_path),
            "public_score": 1.2829 if abs(blend_weight - 0.10) < 1e-12 else None,
            "blend_weight": blend_weight,
            "dataset1": validate_csv(d1, 61051),
            "dataset2": d2_check,
            "top1_change_vs_baseline": top1_change(baseline, logits),
            "zip_size": zip_path.stat().st_size,
        }
    }
    print(f"packed {name} {manifest[name]}", flush=True)
    dump_json(reports / "final_pack_manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data_A")
    p.add_argument("--baseline-root", default="baseline_artifacts")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--reports", default="reports")
    p.add_argument("--submission", default="submission")
    p.add_argument("--seed", type=int, default=3026)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--history-frac", type=float, default=0.70)
    p.add_argument("--train-rows", type=int, default=500000)
    p.add_argument("--valid-rows", type=int, default=80000)
    p.add_argument("--max-pool", type=int, default=700)
    p.add_argument("--svd-dim", type=int, default=128)
    p.add_argument("--fit-edge-limit", type=int, default=0)
    p.add_argument("--src-seq-len", type=int, default=64)
    p.add_argument("--dst-seq-len", type=int, default=64)
    p.add_argument("--seeds", default="3101,3102,3103")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--predict-batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--reuse-baseline-features", default="0")
    p.add_argument("--blend-weight", type=float, default=0.10)
    p.add_argument("--output-name", default="result_final_blend_0p10")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("train")
    sub.add_parser("predict")
    sub.add_parser("package")
    sub.add_parser("run-all")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "build":
        build_context_features(args)
    elif args.command == "train":
        train_context_ranker(args)
    elif args.command == "predict":
        predict_context_ranker(args)
    elif args.command == "package":
        package_final_result(args)
    elif args.command == "run-all":
        build_context_features(args)
        train_context_ranker(args)
        predict_context_ranker(args)
        package_final_result(args)
    else:
        raise SystemExit(args.command)


if __name__ == "__main__":
    main()
