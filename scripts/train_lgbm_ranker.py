import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np


def load_lgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit(
            "train_lgbm_ranker.py requires lightgbm. Install it with: python -m pip install lightgbm"
        ) from exc
    return lgb


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
    total = 0.0
    for row, label in zip(scores, labels):
        total += 1.0 / rank_of_label(row, int(label))
    return total / len(labels)


def make_relevance(labels, candidate_count=100):
    rel = np.zeros((len(labels), candidate_count), dtype=np.float32)
    rel[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
    return rel.reshape(-1)


def lgb_mrr_metric(preds, dataset):
    labels = dataset.get_label().reshape(-1, 100)
    scores = preds.reshape(-1, 100)
    label_idx = labels.argmax(axis=1)
    return "mrr", mrr(scores, label_idx), True


def dataset_params(dataset_name):
    if dataset_name == "dataset2":
        return {
            "num_leaves": 63,
            "learning_rate": 0.02,
            "min_data_in_leaf": 300,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "num_boost_round": 3000,
            "early_stopping_rounds": 150,
        }
    return {
        "num_leaves": 31,
        "learning_rate": 0.03,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "num_boost_round": 2500,
        "early_stopping_rounds": 100,
    }


def load_feature_names(dataset_cache):
    path = dataset_cache / "feature_names.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten_queries(x_raw):
    return np.asarray(x_raw, dtype=np.float32).reshape(-1, x_raw.shape[-1])


def predict_queries(model, x_raw, batch_queries):
    out = np.empty((len(x_raw), 100), dtype=np.float32)
    for start in range(0, len(x_raw), batch_queries):
        end = min(start + batch_queries, len(x_raw))
        flat = flatten_queries(x_raw[start:end])
        out[start:end] = model.predict(flat, num_iteration=model.best_iteration).reshape(end - start, 100)
    return out


def train_dataset(args, dataset_name):
    lgb = load_lgbm()
    dataset_cache = Path(args.cache_dir) / dataset_name
    x_valid = np.load(dataset_cache / "x_valid.npy", mmap_mode="r")
    y_valid = np.load(dataset_cache / "y_valid.npy", mmap_mode="r")

    rows = len(y_valid)
    if args.max_rows:
        rows = min(rows, args.max_rows)
    if rows < 5:
        raise ValueError(f"{dataset_name}: not enough rows for LightGBM training")

    x_all = x_valid[:rows]
    y_all = np.asarray(y_valid[:rows], dtype=np.int32)
    cut = max(1, int(rows * (1.0 - args.eval_ratio)))
    if cut >= rows:
        cut = rows - 1

    x_train = flatten_queries(x_all[:cut])
    y_train = make_relevance(y_all[:cut])
    group_train = np.full(cut, 100, dtype=np.int32)

    x_eval = flatten_queries(x_all[cut:rows])
    y_eval = make_relevance(y_all[cut:rows])
    group_eval = np.full(rows - cut, 100, dtype=np.int32)

    feature_names = load_feature_names(dataset_cache)
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        group=group_train,
        feature_name=feature_names,
        free_raw_data=False,
    )
    eval_set = lgb.Dataset(
        x_eval,
        label=y_eval,
        group=group_eval,
        feature_name=feature_names,
        reference=train_set,
        free_raw_data=False,
    )

    profile = dataset_params(dataset_name)
    params = {
        "objective": "lambdarank",
        "metric": "None",
        "label_gain": [0, 1],
        "verbosity": -1,
        "seed": args.seed,
        "num_threads": args.num_threads,
        "force_col_wise": True,
        "num_leaves": profile["num_leaves"],
        "learning_rate": profile["learning_rate"],
        "min_data_in_leaf": profile["min_data_in_leaf"],
        "feature_fraction": profile["feature_fraction"],
        "bagging_fraction": profile["bagging_fraction"],
        "bagging_freq": profile["bagging_freq"],
    }
    callbacks = [
        lgb.early_stopping(profile["early_stopping_rounds"], verbose=True),
        lgb.log_evaluation(args.log_period),
    ]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=profile["num_boost_round"],
        valid_sets=[eval_set],
        valid_names=["eval"],
        feval=lgb_mrr_metric,
        callbacks=callbacks,
    )

    model_dir = Path(args.model_dir)
    score_dir = Path(args.score_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{dataset_name}_lgbm_ranker.txt"
    model.save_model(model_path)

    valid_scores = predict_queries(model, x_all, args.predict_batch_queries)
    valid_mrr = mrr(valid_scores, y_all)
    np.save(score_dir / f"{dataset_name}_lgbm_ranker_valid.npy", valid_scores.astype(np.float32))

    x_test = np.load(dataset_cache / "x_test.npy", mmap_mode="r")
    test_scores = predict_queries(model, x_test, args.predict_batch_queries)
    np.save(score_dir / f"{dataset_name}_lgbm_ranker_test.npy", test_scores.astype(np.float32))

    print(
        f"{dataset_name}: saved {model_path} rows={rows} "
        f"valid_mrr={valid_mrr:.8f} best_iter={model.best_iteration}"
    )


def find_dataset_names(cache_dir, dataset_arg):
    if dataset_arg != "all":
        return [name.strip() for name in dataset_arg.split(",") if name.strip()]
    return sorted(
        p.name for p in Path(cache_dir).iterdir()
        if p.is_dir() and (p / "x_valid.npy").exists() and (p / "x_test.npy").exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="feature_cache")
    parser.add_argument("--model-dir", default="models_lgbm")
    parser.add_argument("--score-dir", default="scores_lgbm")
    parser.add_argument("--dataset", default="all", help="all or comma-separated dataset names")
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--predict-batch-queries", type=int, default=4096)
    parser.add_argument("--log-period", type=int, default=50)
    args = parser.parse_args()

    names = find_dataset_names(args.cache_dir, args.dataset)
    if not names:
        raise ValueError(f"no cached datasets found in {args.cache_dir}")
    for name in names:
        train_dataset(args, name)


if __name__ == "__main__":
    main()
