import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .data import (
    dataset_dir,
    dump_json,
    ensure_dir,
    read_test,
    read_train,
    read_zip_scores,
    split_edges,
    train_edges,
    validate_csv,
)
from .ensemble import model_logits_from_components, write_candidate_packages, write_research_package
from .features import FEATURE_NAMES, GraphFeatureModel
from .models import blend_feature_and_mlp_scores, make_jittor_feature_predictor, make_training_feature_block, train_jittor_feature_mlp
from .validation import (
    aggregate_mrr,
    attach_features,
    build_validation_sets,
    evaluate_components,
    score_feature_tensor,
    search_weights_multi,
    top1_stats,
)


def _score_feature_shard(payload: tuple) -> dict:
    shard_id, model_path, rows, weights, batch_size, out_dir = payload
    model = GraphFeatureModel.load(Path(model_path))
    rows = list(rows)
    feature_parts = []
    logit_parts = []
    for start in range(0, len(rows), int(batch_size)):
        chunk = rows[start:start + int(batch_size)]
        feats = model.feature_tensor(chunk, progress_every=0)
        feature_parts.append(feats.astype(np.float32))
        logit_parts.append(score_feature_tensor(feats, weights).astype(np.float32))
        print(f"worker shard={shard_id} rows={start + len(chunk)}/{len(rows)}", flush=True)
    features = np.concatenate(feature_parts, axis=0) if feature_parts else np.zeros((0, 100, len(FEATURE_NAMES)), dtype=np.float32)
    logits = np.concatenate(logit_parts, axis=0) if logit_parts else np.zeros((0, 100), dtype=np.float32)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / f"features_part_{int(shard_id):02d}.npy"
    logits_path = out_dir / f"feature_logits_part_{int(shard_id):02d}.npy"
    np.save(feature_path, features)
    np.save(logits_path, logits)
    return {
        "shard_id": int(shard_id),
        "rows": len(rows),
        "feature_path": str(feature_path),
        "logits_path": str(logits_path),
    }


def default_paths(args: argparse.Namespace) -> None:
    args.data_dir = Path(args.data_dir)
    args.teacher_zip = Path(args.teacher_zip)
    args.artifacts = Path(args.artifacts)
    args.reports = Path(args.reports)
    args.submission = Path(args.submission)
    for path in [args.artifacts, args.reports, args.submission]:
        ensure_dir(path)


def load_teacher_scores(args: argparse.Namespace, dataset: str) -> Optional[np.ndarray]:
    if not args.teacher_zip.exists():
        print(f"teacher zip missing, skip teacher: {args.teacher_zip}", flush=True)
        return None
    try:
        return read_zip_scores(args.teacher_zip, f"{dataset}.csv")
    except Exception as exc:
        print(f"teacher read failed dataset={dataset}: {exc!r}", flush=True)
        return None


def profile(args: argparse.Namespace) -> dict:
    default_paths(args)
    payload = {"data_dir": str(args.data_dir), "teacher_zip": str(args.teacher_zip), "datasets": {}}
    for dataset in ["dataset1", "dataset2"]:
        ddir = dataset_dir(args.data_dir, dataset)
        rows = read_train(ddir / "train.csv")
        tests = read_test(ddir / "test.csv")
        srcs = {r.src for r in rows}
        dsts = {r.dst for r in rows}
        test_candidates = [dst for row in tests for dst in row.candidates]
        known = sum(1 for dst in test_candidates if dst in dsts)
        split_counts = {}
        for r in rows:
            key = "none" if r.split in (None, "") else str(r.split)
            split_counts[key] = split_counts.get(key, 0) + 1
        train0, valid, split_meta = split_edges(ddir, final_train=False, prefer_official=(dataset == "dataset2"))
        payload["datasets"][dataset] = {
            "train_rows": len(rows),
            "test_rows": len(tests),
            "unique_src_train": len(srcs),
            "unique_dst_train": len(dsts),
            "split_counts": split_counts,
            "validation_split": split_meta,
            "test_candidate_total": len(test_candidates),
            "test_candidate_known_frac": known / max(len(test_candidates), 1),
            "test_unique_candidate": len(set(test_candidates)),
            "first_test_time": tests[0].time if tests else None,
            "last_test_time": tests[-1].time if tests else None,
        }
        teacher = load_teacher_scores(args, dataset)
        if teacher is not None and len(teacher) == len(tests):
            payload["datasets"][dataset]["teacher"] = {
                "top_prob_mean": float(np.max(teacher, axis=1).mean()),
                "top_prob_p95": float(np.percentile(np.max(teacher, axis=1), 95)),
            }
    dump_json(args.reports / "profile.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:8000], flush=True)
    return payload


def train_dataset(args: argparse.Namespace, dataset: str) -> dict:
    default_paths(args)
    ddir = dataset_dir(args.data_dir, dataset)
    smoke = bool(args.smoke)
    max_valid = int(args.max_valid_events_smoke if smoke else args.max_valid_events)
    svd_dim = int(args.svd_dim_smoke if smoke else args.svd_dim)
    train, valid, split_meta = split_edges(ddir, final_train=False, prefer_official=(dataset == "dataset2"))
    test_rows = read_test(ddir / "test.csv")
    if smoke:
        train = train[: min(len(train), int(args.max_train_edges_smoke))]
        valid = valid[: min(len(valid), max_valid * 2)]
        test_rows_for_fit = test_rows[: min(len(test_rows), 2000)]
    else:
        test_rows_for_fit = test_rows

    teacher = load_teacher_scores(args, dataset)
    if teacher is not None and smoke:
        teacher = teacher[: len(test_rows)]

    print(
        f"train_dataset dataset={dataset} smoke={smoke} train={len(train)} valid={len(valid)} test={len(test_rows)} svd_dim={svd_dim}",
        flush=True,
    )
    model = GraphFeatureModel(
        dataset=dataset,
        svd_dim=svd_dim,
        recent_limit=int(args.recent_limit),
        transition_window=int(args.transition_window),
        transition_topk=int(args.transition_topk_smoke if smoke else args.transition_topk),
        seed=int(args.seed),
    ).fit(train, test_rows_for_fit)
    artifact_path = args.artifacts / f"{dataset}_feature_model_val.pkl"
    model.save(artifact_path)

    vsets = build_validation_sets(dataset, train, valid, test_rows, max_valid, int(args.seed), teacher_scores=teacher)
    attach_features(model, vsets)
    component_report = evaluate_components(vsets)
    weights, history = search_weights_multi(vsets, rounds=int(args.search_rounds_smoke if smoke else args.search_rounds))
    agg, by_set = aggregate_mrr(vsets, weights)

    jittor_report = {"status": "disabled"}
    if bool(args.enable_jittor) and not smoke:
        try:
            train_x, train_y, train_meta = make_training_feature_block(
                model,
                vsets,
                max_rows=int(args.jittor_train_rows),
                seed=int(args.seed),
            )
            valid_x = vsets[0].features[: min(len(vsets[0].labels), 5000)] if vsets and vsets[0].features is not None else None
            valid_y = vsets[0].labels[: min(len(vsets[0].labels), 5000)] if vsets else None
            jittor_report = train_jittor_feature_mlp(
                train_x,
                train_y,
                valid_x,
                valid_y,
                out_dir=args.artifacts / f"{dataset}_jittor_mlp",
                hidden=int(args.jittor_hidden),
                epochs=int(args.jittor_epochs),
                batch_size=int(args.jittor_batch_size),
                lr=float(args.jittor_lr),
                seed=int(args.seed),
            )
            jittor_report["training_block"] = train_meta
        except Exception as exc:
            jittor_report = {"status": "failed_before_train", "error": repr(exc)}

    report = {
        "dataset": dataset,
        "smoke": smoke,
        "split": split_meta,
        "train_edges_used": len(train),
        "valid_edges_used": len(valid),
        "test_rows": len(test_rows),
        "feature_names": FEATURE_NAMES,
        "artifact": str(artifact_path),
        "svd_dim": svd_dim,
        "validation_sets": [
            {
                "name": v.name,
                "rows": len(v.rows),
                "weight": v.weight,
                "meta": v.meta,
            }
            for v in vsets
        ],
        "component_report": component_report,
        "weights": weights,
        "weight_history": history,
        "aggregate_mrr": agg,
        "by_set_mrr": by_set,
        "jittor": jittor_report,
    }
    out_path = args.reports / f"{dataset}_train_report{'_smoke' if smoke else ''}.json"
    dump_json(out_path, report)
    print(json.dumps({k: report[k] for k in ["dataset", "smoke", "aggregate_mrr", "by_set_mrr", "weights", "jittor"]}, indent=2, ensure_ascii=False), flush=True)
    return report


def predict_dataset(args: argparse.Namespace, dataset: str) -> dict:
    default_paths(args)
    ddir = dataset_dir(args.data_dir, dataset)
    report_path = args.reports / f"{dataset}_train_report.json"
    if not report_path.exists():
        smoke_path = args.reports / f"{dataset}_train_report_smoke.json"
        if smoke_path.exists():
            report_path = smoke_path
    if not report_path.exists():
        raise FileNotFoundError(f"missing train report for {dataset}: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    weights: Dict[str, float] = {k: float(v) for k, v in report.get("weights", {"rule": 1.0}).items()}
    train, _valid, split_meta = split_edges(ddir, final_train=True)
    test_rows = read_test(ddir / "test.csv")
    smoke = bool(args.smoke)
    if smoke:
        train = train[: min(len(train), int(args.max_train_edges_smoke))]
        test_rows = test_rows[: min(len(test_rows), 3000)]
    svd_dim = int(report.get("svd_dim", args.svd_dim_smoke if smoke else args.svd_dim))
    print(f"predict_dataset dataset={dataset} smoke={smoke} train={len(train)} test={len(test_rows)} weights={weights}", flush=True)
    model = GraphFeatureModel(
        dataset=dataset,
        svd_dim=svd_dim,
        recent_limit=int(args.recent_limit),
        transition_window=int(args.transition_window),
        transition_topk=int(args.transition_topk_smoke if smoke else args.transition_topk),
        seed=int(args.seed) + 100,
    ).fit(train, test_rows)
    model_path = args.artifacts / f"{dataset}_feature_model_final.pkl"
    model.save(model_path)
    jittor_meta = report.get("jittor", {})
    can_use_mlp = dataset == "dataset2" and jittor_meta.get("status") == "trained"

    def load_mlp_predictor():
        if not can_use_mlp:
            return None
        checkpoint = Path(jittor_meta.get("checkpoint", ""))
        norm = Path(jittor_meta.get("norm", ""))
        if checkpoint.exists() and norm.exists():
            try:
                predictor = make_jittor_feature_predictor(
                    checkpoint,
                    norm,
                    hidden=int(jittor_meta.get("hidden", args.jittor_hidden)),
                    batch_size=int(args.jittor_batch_size),
                )
                print(f"loaded jittor predictor checkpoint={checkpoint}", flush=True)
                return predictor
            except Exception as exc:
                print(f"jittor predictor load failed: {exc!r}", flush=True)
        return None

    logits = np.zeros((len(test_rows), 100), dtype=np.float32)
    mlp_used = False
    workers = max(1, int(args.predict_workers))
    if workers > 1 and len(test_rows) >= workers * 2000:
        shard_dir = args.artifacts / f"{dataset}_predict_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for old in shard_dir.glob("*.npy"):
            old.unlink()
        boundaries = np.linspace(0, len(test_rows), workers + 1, dtype=np.int64)
        tasks = []
        for shard_id in range(workers):
            start = int(boundaries[shard_id])
            end = int(boundaries[shard_id + 1])
            tasks.append((shard_id, str(model_path), test_rows[start:end], weights, int(args.predict_batch_size), str(shard_dir)))
        print(f"parallel feature scoring dataset={dataset} workers={workers} rows={len(test_rows)}", flush=True)
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            shard_reports = pool.map(_score_feature_shard, tasks)
        shard_reports = sorted(shard_reports, key=lambda x: x["shard_id"])
        mlp_predict = load_mlp_predictor()
        offset = 0
        for item in shard_reports:
            features = np.load(item["feature_path"], mmap_mode="r")
            feature_logits = np.load(item["logits_path"])
            mlp_logits = mlp_predict(features) if mlp_predict is not None else None
            logits[offset:offset + int(item["rows"])] = blend_feature_and_mlp_scores(
                feature_logits,
                mlp_logits,
                mlp_weight=float(args.mlp_final_weight),
            )
            mlp_used = mlp_used or mlp_logits is not None
            offset += int(item["rows"])
            print(f"merged shard={item['shard_id']} cumulative_rows={offset} mlp={mlp_logits is not None}", flush=True)
    else:
        mlp_predict = load_mlp_predictor()
        for start in range(0, len(test_rows), int(args.predict_batch_size)):
            chunk = test_rows[start:start + int(args.predict_batch_size)]
            feats = model.feature_tensor(chunk, progress_every=0)
            feature_logits = score_feature_tensor(feats, weights)
            mlp_logits = mlp_predict(feats) if mlp_predict is not None else None
            logits[start:start + len(chunk)] = blend_feature_and_mlp_scores(
                feature_logits,
                mlp_logits,
                mlp_weight=float(args.mlp_final_weight),
            )
            mlp_used = mlp_used or mlp_logits is not None
            print(f"scored dataset={dataset} rows={start + len(chunk)} mlp={mlp_predict is not None}", flush=True)
    logits_path = args.artifacts / f"{dataset}_model_logits.npy"
    np.save(logits_path, logits.astype(np.float32))
    pred_report = {
        "dataset": dataset,
        "smoke": smoke,
        "split": split_meta,
        "weights": weights,
        "model": str(model_path),
        "logits": str(logits_path),
        "test_rows": len(test_rows),
        "top1_stats": top1_stats(logits, test_rows, model),
        "jittor_used_in_prediction": mlp_used,
        "mlp_final_weight": float(args.mlp_final_weight),
        "predict_workers": workers,
    }
    dump_json(args.reports / f"{dataset}_predict_report{'_smoke' if smoke else ''}.json", pred_report)
    print(json.dumps(pred_report, indent=2, ensure_ascii=False)[:5000], flush=True)
    return pred_report


def ensemble(args: argparse.Namespace) -> dict:
    default_paths(args)
    d1_rows = read_test(dataset_dir(args.data_dir, "dataset1") / "test.csv")
    d2_rows = read_test(dataset_dir(args.data_dir, "dataset2") / "test.csv")
    if bool(args.smoke):
        d2_rows = d2_rows[:3000]
    d2_logits_path = args.artifacts / "dataset2_model_logits.npy"
    if not d2_logits_path.exists():
        raise FileNotFoundError(f"missing dataset2 logits: {d2_logits_path}")
    d2_logits = np.load(d2_logits_path)
    if len(d2_logits) != len(d2_rows):
        if bool(args.smoke) and len(d2_logits) >= len(d2_rows):
            d2_logits = d2_logits[: len(d2_rows)]
        else:
            raise ValueError(f"dataset2 logits rows {len(d2_logits)} != test rows {len(d2_rows)}")

    d1_logits = None
    d1_logits_path = args.artifacts / "dataset1_model_logits.npy"
    if bool(args.use_dataset1_blend) and d1_logits_path.exists():
        d1_logits = np.load(d1_logits_path)
        if len(d1_logits) != len(d1_rows):
            d1_logits = d1_logits[: len(d1_rows)]

    model_logits = model_logits_from_components(d2_logits)
    targets = [float(x) for x in str(args.target_changes).split(",") if x.strip()]
    manifest = write_candidate_packages(
        teacher_zip=args.teacher_zip,
        dataset1_test_rows=d1_rows,
        dataset2_test_rows=d2_rows,
        dataset2_model_logits=model_logits,
        out_root=args.submission,
        target_changes=targets,
        dataset1_model_logits=d1_logits,
        dataset1_alpha=float(args.dataset1_alpha),
        prefix=args.package_prefix,
    )
    research = write_research_package(
        teacher_zip=args.teacher_zip,
        dataset1_test_rows=d1_rows,
        dataset2_test_rows=d2_rows,
        dataset2_model_logits=model_logits,
        out_root=args.submission,
        prefix=f"{args.package_prefix}_research_full",
    )
    payload = {"packages": manifest, "research": research}
    dump_json(args.reports / "ensemble_manifest.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:8000], flush=True)
    return payload


def pack_check(args: argparse.Namespace) -> dict:
    default_paths(args)
    payload = {"submission_dir": str(args.submission), "zips": {}}
    for zip_path in sorted(args.submission.glob("*.zip")):
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(zf.namelist())
        payload["zips"][zip_path.name] = {"members": names, "size": zip_path.stat().st_size}
    for csv_path in sorted(args.submission.glob("*/dataset*.csv")):
        expected = 61051 if csv_path.name == "dataset1.csv" else 153420
        if bool(args.smoke):
            expected = None
        payload[str(csv_path)] = validate_csv(csv_path, expected_rows=expected)
    dump_json(args.reports / "pack_check.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:8000], flush=True)
    return payload


def run_all(args: argparse.Namespace) -> None:
    profile(args)
    train_dataset(args, "dataset1")
    train_dataset(args, "dataset2")
    predict_dataset(args, "dataset1")
    predict_dataset(args, "dataset2")
    ensemble(args)
    pack_check(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--teacher-zip", default="/home/ma-user/work/jittor/result_pairwise_w05.zip")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--submission", default="submission")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--svd-dim", type=int, default=160)
    parser.add_argument("--svd-dim-smoke", type=int, default=32)
    parser.add_argument("--recent-limit", type=int, default=160)
    parser.add_argument("--transition-window", type=int, default=16)
    parser.add_argument("--transition-topk", type=int, default=384)
    parser.add_argument("--transition-topk-smoke", type=int, default=64)
    parser.add_argument("--max-valid-events", type=int, default=30000)
    parser.add_argument("--max-valid-events-smoke", type=int, default=1500)
    parser.add_argument("--max-train-edges-smoke", type=int, default=120000)
    parser.add_argument("--search-rounds", type=int, default=5)
    parser.add_argument("--search-rounds-smoke", type=int, default=2)
    parser.add_argument("--predict-batch-size", type=int, default=16384)
    parser.add_argument("--predict-workers", type=int, default=4)
    parser.add_argument("--enable-jittor", action="store_true")
    parser.add_argument("--jittor-train-rows", type=int, default=80000)
    parser.add_argument("--jittor-hidden", type=int, default=192)
    parser.add_argument("--jittor-epochs", type=int, default=8)
    parser.add_argument("--jittor-batch-size", type=int, default=256)
    parser.add_argument("--jittor-lr", type=float, default=8e-4)
    parser.add_argument("--mlp-final-weight", type=float, default=0.20)
    parser.add_argument("--target-changes", default="0.01,0.03,0.05,0.08,0.12")
    parser.add_argument("--package-prefix", default="result_rebuild")
    parser.add_argument("--use-dataset1-blend", action="store_true")
    parser.add_argument("--dataset1-alpha", type=float, default=0.0)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("profile")
    p_train = sub.add_parser("train-dataset")
    p_train.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    p_predict = sub.add_parser("predict-dataset")
    p_predict.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    sub.add_parser("ensemble")
    sub.add_parser("pack-check")
    sub.add_parser("run-all")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "profile":
        profile(args)
    elif args.command == "train-dataset":
        train_dataset(args, args.dataset)
    elif args.command == "predict-dataset":
        predict_dataset(args, args.dataset)
    elif args.command == "ensemble":
        ensemble(args)
    elif args.command == "pack-check":
        pack_check(args)
    elif args.command == "run-all":
        run_all(args)
    else:
        raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
