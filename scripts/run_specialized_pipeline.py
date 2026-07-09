import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.common.submission import (
    dataset_map,
    make_zip,
    write_probability_chunks,
    write_report,
    write_zero_csv_for_dataset,
)
from src.dataset1.ranker import iter_dataset1_proba_chunks, train_dataset1
from src.dataset2.temporal_recommender import (
    iter_dataset2_proba_chunks,
    train_dataset2,
    train_dataset2_feature_reranker,
    train_dataset2_listwise_feature_ranker,
    train_dataset2_pairwise_classifier,
)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def require_positive_int(name, value):
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def require_non_negative_float(name, value):
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def train_selected(args, datasets, artifact_root):
    reports = {}
    if args.target in {"dataset1", "all"}:
        if "dataset1" not in datasets:
            raise RuntimeError("dataset1 not found")
        reports["dataset1_train"] = train_dataset1(datasets["dataset1"], artifact_root / "dataset1")
    if args.target in {"dataset2", "all"}:
        if "dataset2" not in datasets:
            raise RuntimeError("dataset2 not found")
        final_train = parse_bool(args.final_train)
        if args.d2_model_type == "pairwise":
            reports["dataset2_train"] = train_dataset2_pairwise_classifier(
                datasets["dataset2"],
                artifact_root / "dataset2",
                final_train=final_train,
                neg_count=args.d2_pairwise_neg_count,
                max_train_events=args.d2_pairwise_max_train_events,
                valid_max_events=args.d2_valid_max_events,
                model_kind=args.d2_feature_model_kind,
                fusion_model_weight=args.d2_fusion_model_weight,
                fusion_rule_weight=args.d2_fusion_rule_weight,
                unknown_policy=args.d2_unknown_policy,
                unknown_score=args.d2_unknown_score,
                unknown_margin=args.d2_unknown_margin,
                cold_prior_weight=args.d2_cold_prior_weight,
                seed=args.seed,
            )
            return reports
        if args.d2_model_type == "feature":
            reports["dataset2_train"] = train_dataset2_feature_reranker(
                datasets["dataset2"],
                artifact_root / "dataset2",
                final_train=final_train,
                neg_count=args.d2_rerank_neg_count,
                max_train_events=args.d2_feature_max_train_events,
                valid_max_events=args.d2_valid_max_events,
                model_kind=args.d2_feature_model_kind,
                fusion_model_weight=None if args.d2_feature_auto_weight else args.d2_fusion_model_weight,
                fusion_rule_weight=args.d2_fusion_rule_weight,
                unknown_policy=args.d2_unknown_policy,
                unknown_score=args.d2_unknown_score,
                unknown_margin=args.d2_unknown_margin,
                cold_prior_weight=args.d2_cold_prior_weight,
                seed=args.seed,
            )
            return reports
        if args.d2_model_type == "listwise":
            if final_train and parse_bool(args.d2_validate_before_final):
                reports["dataset2_validation_train"] = train_dataset2_listwise_feature_ranker(
                    datasets["dataset2"],
                    artifact_root / "dataset2_validation",
                    final_train=False,
                    neg_count=args.d2_listwise_neg_count,
                    max_train_events=args.d2_listwise_max_train_events,
                    valid_max_events=args.d2_valid_max_events,
                    hidden_dim=args.d2_listwise_hidden_dim,
                    epochs=args.d2_listwise_epochs,
                    batch_size=args.d2_listwise_batch_size,
                    lr=args.d2_listwise_lr,
                    margin_weight=args.d2_listwise_margin_weight,
                    fusion_model_weight=None if args.d2_feature_auto_weight else args.d2_fusion_model_weight,
                    fusion_rule_weight=args.d2_fusion_rule_weight,
                    unknown_policy=args.d2_unknown_policy,
                    unknown_score=args.d2_unknown_score,
                    unknown_margin=args.d2_unknown_margin,
                    cold_prior_weight=args.d2_cold_prior_weight,
                    new_pair_only=parse_bool(args.d2_listwise_new_pair_only),
                    seed=args.seed,
                )
            reports["dataset2_train"] = train_dataset2_listwise_feature_ranker(
                datasets["dataset2"],
                artifact_root / "dataset2",
                final_train=final_train,
                neg_count=args.d2_listwise_neg_count,
                max_train_events=args.d2_listwise_max_train_events,
                valid_max_events=args.d2_valid_max_events,
                hidden_dim=args.d2_listwise_hidden_dim,
                epochs=args.d2_listwise_epochs,
                batch_size=args.d2_listwise_batch_size,
                lr=args.d2_listwise_lr,
                margin_weight=args.d2_listwise_margin_weight,
                fusion_model_weight=None if args.d2_feature_auto_weight else args.d2_fusion_model_weight,
                fusion_rule_weight=args.d2_fusion_rule_weight,
                unknown_policy=args.d2_unknown_policy,
                unknown_score=args.d2_unknown_score,
                unknown_margin=args.d2_unknown_margin,
                cold_prior_weight=args.d2_cold_prior_weight,
                new_pair_only=parse_bool(args.d2_listwise_new_pair_only),
                seed=args.seed,
            )
            return reports
        if final_train and parse_bool(args.d2_validate_before_final):
            reports["dataset2_validation_train"] = train_dataset2(
                datasets["dataset2"],
                artifact_root / "dataset2_validation",
                final_train=False,
                cuda=parse_bool(args.cuda),
                softmax_mode=args.d2_softmax_mode,
                max_train_events=args.d2_temporal_max_train_events,
                neg_count=args.d2_neg_count,
                seq_len=args.d2_seq_len,
                emb_dim=args.d2_emb_dim,
                hidden_dim=args.d2_hidden_dim,
                dropout=args.d2_dropout,
                epochs=args.d2_epochs,
                batch_size=args.d2_batch_size,
                lr=args.d2_lr,
                weight_decay=args.d2_weight_decay,
                bpr_weight=args.d2_bpr_weight,
                all_dst_weight=args.d2_all_dst_weight,
                hard_negative_count=args.d2_hard_negative_count,
                sampled_correction=parse_bool(args.d2_sampled_correction),
                rerank_neg_count=args.d2_rerank_neg_count,
                rerank_weight=args.d2_rerank_weight,
                fusion_model_weight=args.d2_fusion_model_weight,
                fusion_rule_weight=args.d2_fusion_rule_weight,
                include_test_vocab=parse_bool(args.d2_include_test_vocab),
                unknown_policy=args.d2_unknown_policy,
                unknown_score=args.d2_unknown_score,
                unknown_margin=args.d2_unknown_margin,
                cold_prior_weight=args.d2_cold_prior_weight,
                valid_max_events=args.d2_valid_max_events,
                seed=args.seed,
            )
        reports["dataset2_train"] = train_dataset2(
            datasets["dataset2"],
            artifact_root / "dataset2",
            final_train=final_train,
            cuda=parse_bool(args.cuda),
            softmax_mode=args.d2_softmax_mode,
            max_train_events=args.d2_temporal_max_train_events,
            neg_count=args.d2_neg_count,
            seq_len=args.d2_seq_len,
            emb_dim=args.d2_emb_dim,
            hidden_dim=args.d2_hidden_dim,
            dropout=args.d2_dropout,
            epochs=args.d2_epochs,
            batch_size=args.d2_batch_size,
            lr=args.d2_lr,
            weight_decay=args.d2_weight_decay,
            bpr_weight=args.d2_bpr_weight,
            all_dst_weight=args.d2_all_dst_weight,
            hard_negative_count=args.d2_hard_negative_count,
            sampled_correction=parse_bool(args.d2_sampled_correction),
            rerank_neg_count=args.d2_rerank_neg_count,
            rerank_weight=args.d2_rerank_weight,
            fusion_model_weight=args.d2_fusion_model_weight,
            fusion_rule_weight=args.d2_fusion_rule_weight,
            include_test_vocab=parse_bool(args.d2_include_test_vocab),
            unknown_policy=args.d2_unknown_policy,
            unknown_score=args.d2_unknown_score,
            unknown_margin=args.d2_unknown_margin,
            cold_prior_weight=args.d2_cold_prior_weight,
            valid_max_events=args.d2_valid_max_events,
            seed=args.seed,
        )
    return reports


def predict_outputs(args, datasets, artifact_root, out_dir):
    reports = {}
    for dataset_name in sorted(datasets):
        dataset_dir = datasets[dataset_name]
        output_path = out_dir / f"{dataset_name}.csv"
        inactive_probe = parse_bool(args.zero_other) and args.target != "all" and dataset_name != args.target
        if inactive_probe:
            reports[dataset_name] = {
                "mode": "zero_probe",
                **write_zero_csv_for_dataset(dataset_dir, output_path),
            }
            continue
        if dataset_name == "dataset1":
            chunks = iter_dataset1_proba_chunks(
                dataset_dir,
                artifact_root / "dataset1",
                batch_size=args.batch_size,
                max_rows=args.max_rows,
            )
        elif dataset_name == "dataset2":
            chunks = iter_dataset2_proba_chunks(
                dataset_dir,
                artifact_root / "dataset2",
                batch_size=args.batch_size,
                max_rows=args.max_rows,
            )
        else:
            if parse_bool(args.zero_unknown_datasets):
                reports[dataset_name] = {
                    "mode": "zero_unknown_dataset",
                    **write_zero_csv_for_dataset(dataset_dir, output_path),
                }
                continue
            raise RuntimeError(f"unsupported dataset: {dataset_name}")
        reports[dataset_name] = {
            "mode": "predicted",
            **write_probability_chunks(output_path, chunks, validate=True),
        }
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--out-dir", default="submission_specialized")
    parser.add_argument("--zip", default="result_best.zip")
    parser.add_argument("--report", default="reports/specialized_pipeline.json")
    parser.add_argument("--target", choices=["dataset1", "dataset2", "all"], default="all")
    parser.add_argument("--train", default="1")
    parser.add_argument("--predict", default="1")
    parser.add_argument("--zero-other", default="0")
    parser.add_argument("--zero-unknown-datasets", default="0")
    parser.add_argument("--final-train", default="1")
    parser.add_argument("--cuda", default="1")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--d2-model-type", choices=["pairwise", "feature", "listwise", "temporal"], default="temporal")
    parser.add_argument("--d2-feature-model-kind", choices=["jittor_mlp", "sgd", "hgb", "torch_mlp"], default="jittor_mlp")
    parser.add_argument("--d2-feature-max-train-events", type=int, default=120000)
    parser.add_argument("--d2-feature-auto-weight", default="1")
    parser.add_argument("--d2-listwise-neg-count", type=int, default=99)
    parser.add_argument("--d2-listwise-max-train-events", type=int, default=160000)
    parser.add_argument("--d2-listwise-hidden-dim", type=int, default=160)
    parser.add_argument("--d2-listwise-epochs", type=int, default=10)
    parser.add_argument("--d2-listwise-batch-size", type=int, default=4096)
    parser.add_argument("--d2-listwise-lr", type=float, default=0.001)
    parser.add_argument("--d2-listwise-margin-weight", type=float, default=0.05)
    parser.add_argument("--d2-listwise-new-pair-only", default="1")
    parser.add_argument("--d2-pairwise-neg-count", type=int, default=8)
    parser.add_argument("--d2-pairwise-max-train-events", type=int, default=400000)
    parser.add_argument("--d2-softmax-mode", choices=["sampled", "full"], default="sampled")
    parser.add_argument("--d2-temporal-max-train-events", type=int, default=0)
    parser.add_argument("--d2-neg-count", type=int, default=4096)
    parser.add_argument("--d2-seq-len", type=int, default=80)
    parser.add_argument("--d2-emb-dim", type=int, default=96)
    parser.add_argument("--d2-hidden-dim", type=int, default=192)
    parser.add_argument("--d2-dropout", type=float, default=0.1)
    parser.add_argument("--d2-epochs", type=int, default=6)
    parser.add_argument("--d2-batch-size", type=int, default=2048)
    parser.add_argument("--d2-lr", type=float, default=0.001)
    parser.add_argument("--d2-weight-decay", type=float, default=1e-6)
    parser.add_argument("--d2-bpr-weight", type=float, default=0.05)
    parser.add_argument("--d2-all-dst-weight", type=float, default=0.20)
    parser.add_argument("--d2-hard-negative-count", type=int, default=512)
    parser.add_argument("--d2-sampled-correction", default="1")
    parser.add_argument("--d2-rerank-neg-count", type=int, default=99)
    parser.add_argument("--d2-rerank-weight", type=float, default=1.00)
    parser.add_argument("--d2-fusion-model-weight", type=float, default=0.05)
    parser.add_argument("--d2-fusion-rule-weight", type=float, default=1.0)
    parser.add_argument("--d2-include-test-vocab", default="1")
    parser.add_argument("--d2-unknown-policy", choices=["neutral", "mild_penalty", "boost_by_id_prior", "demote", "constant"], default="neutral")
    parser.add_argument("--d2-unknown-score", type=float, default=0.0)
    parser.add_argument("--d2-unknown-margin", type=float, default=0.0)
    parser.add_argument("--d2-cold-prior-weight", type=float, default=0.0)
    parser.add_argument("--d2-valid-max-events", type=int, default=20000)
    parser.add_argument("--d2-validate-before-final", default="1")
    args = parser.parse_args()
    args.batch_size = require_positive_int("--batch-size", args.batch_size)
    args.max_rows = max(int(args.max_rows), 0)
    args.d2_neg_count = require_positive_int("--d2-neg-count", args.d2_neg_count)
    args.d2_seq_len = require_positive_int("--d2-seq-len", args.d2_seq_len)
    args.d2_emb_dim = require_positive_int("--d2-emb-dim", args.d2_emb_dim)
    args.d2_hidden_dim = require_positive_int("--d2-hidden-dim", args.d2_hidden_dim)
    args.d2_epochs = require_positive_int("--d2-epochs", args.d2_epochs)
    args.d2_batch_size = require_positive_int("--d2-batch-size", args.d2_batch_size)
    args.d2_feature_max_train_events = max(int(args.d2_feature_max_train_events), 0)
    args.d2_feature_auto_weight = parse_bool(args.d2_feature_auto_weight)
    args.d2_listwise_neg_count = require_positive_int("--d2-listwise-neg-count", args.d2_listwise_neg_count)
    args.d2_listwise_max_train_events = max(int(args.d2_listwise_max_train_events), 0)
    args.d2_listwise_hidden_dim = require_positive_int("--d2-listwise-hidden-dim", args.d2_listwise_hidden_dim)
    args.d2_listwise_epochs = require_positive_int("--d2-listwise-epochs", args.d2_listwise_epochs)
    args.d2_listwise_batch_size = require_positive_int("--d2-listwise-batch-size", args.d2_listwise_batch_size)
    args.d2_listwise_lr = require_non_negative_float("--d2-listwise-lr", args.d2_listwise_lr)
    args.d2_listwise_margin_weight = require_non_negative_float(
        "--d2-listwise-margin-weight",
        args.d2_listwise_margin_weight,
    )
    args.d2_pairwise_neg_count = require_positive_int("--d2-pairwise-neg-count", args.d2_pairwise_neg_count)
    args.d2_pairwise_max_train_events = max(int(args.d2_pairwise_max_train_events), 0)
    args.d2_temporal_max_train_events = max(int(args.d2_temporal_max_train_events), 0)
    args.d2_bpr_weight = require_non_negative_float("--d2-bpr-weight", args.d2_bpr_weight)
    args.d2_all_dst_weight = require_non_negative_float("--d2-all-dst-weight", args.d2_all_dst_weight)
    args.d2_hard_negative_count = max(int(args.d2_hard_negative_count), 0)
    args.d2_rerank_neg_count = max(int(args.d2_rerank_neg_count), 0)
    args.d2_rerank_weight = require_non_negative_float("--d2-rerank-weight", args.d2_rerank_weight)
    args.d2_fusion_model_weight = require_non_negative_float(
        "--d2-fusion-model-weight",
        args.d2_fusion_model_weight,
    )
    args.d2_fusion_rule_weight = require_non_negative_float(
        "--d2-fusion-rule-weight",
        args.d2_fusion_rule_weight,
    )
    args.d2_unknown_margin = require_non_negative_float("--d2-unknown-margin", args.d2_unknown_margin)
    args.d2_cold_prior_weight = float(args.d2_cold_prior_weight)
    args.d2_valid_max_events = max(int(args.d2_valid_max_events), 0)

    datasets = dataset_map(args.data_dir)
    if not datasets:
        raise RuntimeError(f"no datasets found under {args.data_dir}")
    artifact_root = Path(args.artifact_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": args.target,
        "zero_other": parse_bool(args.zero_other),
        "train": {},
        "predict": {},
    }
    if parse_bool(args.train):
        payload["train"] = train_selected(args, datasets, artifact_root)
    if parse_bool(args.predict):
        payload["predict"] = predict_outputs(args, datasets, artifact_root, out_dir)
        make_zip(out_dir, args.zip)
        payload["zip"] = str(args.zip)
    write_report(args.report, payload)
    print(f"saved report {args.report}")
    if parse_bool(args.predict):
        print(f"packed {args.zip}")


if __name__ == "__main__":
    main()
