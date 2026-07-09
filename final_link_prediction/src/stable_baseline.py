import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .baseline_support import _as_path, _load_feature_mlp_predictor, _train_feature_mlp
from .data import (
    dataset_dir,
    dump_json,
    ensure_dir,
    read_test,
    read_train,
    softmax,
    split_edges,
    validate_csv,
    write_scores_csv,
)
from .features import FEATURE_NAMES, GraphFeatureModel
from .validation import (
    aggregate_mrr,
    attach_features,
    build_validation_sets,
    evaluate_components,
    score_feature_tensor,
    search_weights_multi,
    top1_stats,
)


def _make_training_feature_block(vsets, max_rows: int, seed: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    rows = []
    labels = []
    rng = np.random.default_rng(seed)
    usable = [v for v in vsets if not v.name.startswith("teacher_") and v.features is not None]
    for vset in usable:
        idx = np.arange(len(vset.labels))
        rng.shuffle(idx)
        take = min(len(idx), max(1, int(max_rows / max(len(usable), 1))))
        keep = np.sort(idx[:take])
        rows.append(vset.features[keep])
        labels.append(vset.labels[keep])
    if not rows:
        raise ValueError("no validation features available to train stable baseline MLP")
    x = np.concatenate(rows, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    if len(y) > max_rows:
        idx = rng.choice(np.arange(len(y)), size=int(max_rows), replace=False)
        x = x[idx]
        y = y[idx]
    return x, y, {"train_rows": int(len(y)), "feature_dim": int(x.shape[-1])}


def _score_feature_shard(payload: tuple) -> dict:
    shard_id, model_path, rows, weights, out_dir = payload
    model = GraphFeatureModel.load(Path(model_path))
    rows = list(rows)
    features = model.feature_tensor(rows, progress_every=0).astype(np.float32)
    feature_logits = score_feature_tensor(features, weights).astype(np.float32)
    out_dir = ensure_dir(Path(out_dir))
    feature_path = out_dir / f"features_part_{int(shard_id):02d}.npy"
    logits_path = out_dir / f"feature_logits_part_{int(shard_id):02d}.npy"
    np.save(feature_path, features)
    np.save(logits_path, feature_logits)
    return {
        "shard_id": int(shard_id),
        "rows": len(rows),
        "feature_path": str(feature_path),
        "logits_path": str(logits_path),
    }


def train_stable_dataset(args, dataset: str) -> dict:
    data_dir = _as_path(args.data_dir)
    baseline_root = ensure_dir(_as_path(args.baseline_root))
    artifacts = ensure_dir(baseline_root / "artifacts")
    reports = ensure_dir(baseline_root / "reports")
    ds_dir = dataset_dir(data_dir, dataset)
    train_edges, valid_edges, split_meta = split_edges(ds_dir, final_train=False, prefer_official=(dataset == "dataset2"))
    test_rows = read_test(ds_dir / "test.csv")
    model = GraphFeatureModel(
        dataset=dataset,
        svd_dim=int(args.stable_svd_dim),
        recent_limit=int(args.stable_recent_limit),
        transition_window=int(args.stable_transition_window),
        transition_topk=int(args.stable_transition_topk),
        seed=int(args.stable_seed),
    ).fit(train_edges, test_rows)
    model_path = artifacts / f"{dataset}_feature_model_val.pkl"
    model.save(model_path)

    vsets = build_validation_sets(dataset, train_edges, valid_edges, test_rows, int(args.stable_max_valid_events), int(args.stable_seed))
    attach_features(model, vsets)
    component_report = evaluate_components(vsets)
    weights, history = search_weights_multi(vsets, rounds=int(args.stable_search_rounds))
    aggregate, by_set = aggregate_mrr(vsets, weights)

    mlp_report = {"status": "disabled"}
    if dataset == "dataset2" and str(args.train_stable_mlp) == "1":
        try:
            train_x, train_y, train_meta = _make_training_feature_block(vsets, int(args.stable_mlp_train_rows), int(args.stable_seed))
            valid_x = vsets[0].features[: min(len(vsets[0].labels), 5000)] if vsets and vsets[0].features is not None else None
            valid_y = vsets[0].labels[: min(len(vsets[0].labels), 5000)] if vsets else None
            mlp_report = _train_feature_mlp(
                train_x,
                train_y,
                valid_x,
                valid_y,
                out_dir=artifacts / "dataset2_stable_mlp",
                seed=int(args.stable_seed),
                hidden=int(args.stable_mlp_hidden),
                epochs=int(args.stable_mlp_epochs),
                batch_size=int(args.stable_mlp_batch_size),
                lr=float(args.stable_mlp_lr),
            )
            mlp_report["hidden"] = int(args.stable_mlp_hidden)
            mlp_report["training_block"] = train_meta
        except Exception as exc:
            mlp_report = {"status": "failed", "error": repr(exc), "hidden": int(args.stable_mlp_hidden)}

    report = {
        "dataset": dataset,
        "split": split_meta,
        "train_edges_used": len(train_edges),
        "valid_edges_used": len(valid_edges),
        "test_rows": len(test_rows),
        "feature_names": FEATURE_NAMES,
        "artifact": str(model_path),
        "svd_dim": int(args.stable_svd_dim),
        "validation_sets": [{"name": v.name, "rows": len(v.rows), "weight": v.weight, "meta": v.meta} for v in vsets],
        "component_report": component_report,
        "weights": weights,
        "weight_history": history,
        "aggregate_mrr": aggregate,
        "by_set_mrr": by_set,
        "jittor": mlp_report,
    }
    dump_json(reports / f"{dataset}_train_report.json", report)
    print(json.dumps({k: report[k] for k in ["dataset", "aggregate_mrr", "by_set_mrr", "weights", "jittor"]}, indent=2, ensure_ascii=False), flush=True)
    return report


def predict_stable_dataset1(args) -> dict:
    data_dir = _as_path(args.data_dir)
    baseline_root = ensure_dir(_as_path(args.baseline_root))
    artifacts = ensure_dir(baseline_root / "artifacts")
    reports = ensure_dir(baseline_root / "reports")
    ds_dir = dataset_dir(data_dir, "dataset1")
    train_edges, _valid_edges, split_meta = split_edges(ds_dir, final_train=True)
    test_rows = read_test(ds_dir / "test.csv")
    report = json.loads((reports / "dataset1_train_report.json").read_text(encoding="utf-8"))
    weights: Dict[str, float] = {k: float(v) for k, v in report["weights"].items()}
    model = GraphFeatureModel(
        dataset="dataset1",
        svd_dim=int(report.get("svd_dim", args.stable_svd_dim)),
        recent_limit=int(args.stable_recent_limit),
        transition_window=int(args.stable_transition_window),
        transition_topk=int(args.stable_transition_topk),
        seed=int(args.stable_seed) + 100,
    ).fit(train_edges, test_rows)
    model_path = artifacts / "dataset1_feature_model_final.pkl"
    model.save(model_path)
    logits = model.score_rows(test_rows, weights, batch_size=int(args.stable_predict_batch_size)).astype(np.float32)
    np.save(artifacts / "dataset1_model_logits.npy", logits)
    out_dir = ensure_dir(baseline_root / "submission_mlp_peak" / "result_rebuild_mlpw_5p5")
    csv_path = out_dir / "dataset1.csv"
    check = write_scores_csv(softmax(logits), csv_path)
    pred_report = {
        "dataset": "dataset1",
        "split": split_meta,
        "weights": weights,
        "model": str(model_path),
        "logits": str(artifacts / "dataset1_model_logits.npy"),
        "csv": str(csv_path),
        "test_rows": len(test_rows),
        "top1_stats": top1_stats(logits, test_rows, model),
        "validation": check,
    }
    dump_json(reports / "dataset1_predict_report.json", pred_report)
    return pred_report


def _load_stable_mlp_predictor(report: dict, args):
    meta = report.get("jittor", {})
    if meta.get("status") != "trained":
        return None
    ckpt = Path(meta.get("checkpoint", ""))
    norm = Path(meta.get("norm", ""))
    if not ckpt.exists() or not norm.exists():
        return None
    return _load_feature_mlp_predictor(ckpt, norm, hidden=int(meta.get("hidden", args.stable_mlp_hidden)))


def predict_stable_dataset2(args) -> dict:
    data_dir = _as_path(args.data_dir)
    baseline_root = ensure_dir(_as_path(args.baseline_root))
    artifacts = ensure_dir(baseline_root / "artifacts")
    reports = ensure_dir(baseline_root / "reports")
    ds_dir = dataset_dir(data_dir, "dataset2")
    train_edges, _valid_edges, split_meta = split_edges(ds_dir, final_train=True)
    test_rows = read_test(ds_dir / "test.csv")
    report = json.loads((reports / "dataset2_train_report.json").read_text(encoding="utf-8"))
    weights: Dict[str, float] = {k: float(v) for k, v in report["weights"].items()}
    model = GraphFeatureModel(
        dataset="dataset2",
        svd_dim=int(report.get("svd_dim", args.stable_svd_dim)),
        recent_limit=int(args.stable_recent_limit),
        transition_window=int(args.stable_transition_window),
        transition_topk=int(args.stable_transition_topk),
        seed=int(args.stable_seed) + 100,
    ).fit(train_edges, test_rows)
    model_path = artifacts / "dataset2_feature_model_final.pkl"
    model.save(model_path)

    shard_dir = ensure_dir(artifacts / "dataset2_predict_shards")
    for old in shard_dir.glob("*.npy"):
        old.unlink()
    workers = max(1, int(args.stable_predict_workers))
    boundaries = np.linspace(0, len(test_rows), workers + 1, dtype=np.int64)
    tasks = []
    for shard_id in range(workers):
        start = int(boundaries[shard_id])
        end = int(boundaries[shard_id + 1])
        tasks.append((shard_id, str(model_path), test_rows[start:end], weights, str(shard_dir)))
    if workers > 1:
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            shard_reports = pool.map(_score_feature_shard, tasks)
    else:
        shard_reports = [_score_feature_shard(task) for task in tasks]
    shard_reports = sorted(shard_reports, key=lambda x: x["shard_id"])

    predictor = _load_stable_mlp_predictor(report, args)
    feature_logits_all = []
    mlp_logits_all = []
    for item in shard_reports:
        features = np.load(item["feature_path"], mmap_mode="r")
        feature_logits = np.load(item["logits_path"])
        if predictor is None:
            mlp_logits = np.zeros_like(feature_logits, dtype=np.float32)
        else:
            mlp_logits = predictor(features, batch_size=int(args.stable_predict_batch_size)).astype(np.float32)
        mlp_path = shard_dir / f"mlp_logits_part_{int(item['shard_id']):02d}.npy"
        np.save(mlp_path, mlp_logits)
        item["mlp_logits_path"] = str(mlp_path)
        feature_logits_all.append(feature_logits)
        mlp_logits_all.append(mlp_logits)
    feature_logits_arr = np.concatenate(feature_logits_all, axis=0)
    mlp_logits_arr = np.concatenate(mlp_logits_all, axis=0)
    combined = feature_logits_arr if predictor is None else feature_logits_arr + float(args.stable_mlp_output_weight) * mlp_logits_arr
    np.save(artifacts / "dataset2_model_logits.npy", combined.astype(np.float32))
    pred_report = {
        "dataset": "dataset2",
        "split": split_meta,
        "weights": weights,
        "model": str(model_path),
        "logits": str(artifacts / "dataset2_model_logits.npy"),
        "test_rows": len(test_rows),
        "top1_stats": top1_stats(combined, test_rows, model),
        "jittor_used_in_prediction": predictor is not None,
        "shards": shard_reports,
    }
    dump_json(reports / "dataset2_predict_report.json", pred_report)
    return pred_report


def build_stable_baseline(args) -> dict:
    train1 = train_stable_dataset(args, "dataset1")
    train2 = train_stable_dataset(args, "dataset2")
    pred1 = predict_stable_dataset1(args)
    pred2 = predict_stable_dataset2(args)
    payload = {"dataset1_train": train1, "dataset2_train": train2, "dataset1_predict": pred1, "dataset2_predict": pred2}
    dump_json(ensure_dir(_as_path(args.baseline_root) / "reports") / "stable_baseline_report.json", payload)
    return payload
