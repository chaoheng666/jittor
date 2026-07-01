import argparse
import csv
import json
from pathlib import Path

import numpy as np


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


def load_rule_scores(dataset_cache, max_rows):
    x = np.load(dataset_cache / "x_valid.npy", mmap_mode="r")
    with open(dataset_cache / "feature_names.json", encoding="utf-8") as f:
        names = json.load(f)
    rule_idx = names.index("rule_score")
    scores = x[:, :, rule_idx]
    return scores[:max_rows] if max_rows else scores


def load_labels(dataset_cache, max_rows):
    labels = np.load(dataset_cache / "y_valid.npy", mmap_mode="r")
    return labels[:max_rows] if max_rows else labels


def summarize_fold(fold_name, cache_root, score_root, weights_root, max_rows):
    rows = []
    fold_cache = Path(cache_root) / fold_name
    fold_scores = Path(score_root) / fold_name
    weights_path = Path(weights_root) / fold_name / "ensemble_weights.json"
    weights = {}
    if weights_path.exists():
        with open(weights_path, encoding="utf-8") as f:
            weights = json.load(f).get("datasets", {})

    for dataset_cache in sorted(p for p in fold_cache.iterdir() if p.is_dir()):
        dataset = dataset_cache.name
        labels = load_labels(dataset_cache, max_rows)
        rule_scores = load_rule_scores(dataset_cache, len(labels))
        lgbm_path = fold_scores / f"{dataset}_lgbm_ranker_valid.npy"
        lgbm_mrr = ""
        if lgbm_path.exists():
            lgbm_scores = np.load(lgbm_path, mmap_mode="r")[:len(labels)]
            lgbm_mrr = f"{mrr(lgbm_scores, labels):.8f}"
        tgnn_path = fold_scores / f"{dataset}_tgnn_valid.npy"
        tgnn_mrr = ""
        if tgnn_path.exists():
            tgnn_scores = np.load(tgnn_path, mmap_mode="r")[:len(labels)]
            tgnn_mrr = f"{mrr(tgnn_scores, labels):.8f}"
        selected = []
        ensemble_mrr = ""
        if dataset in weights:
            ensemble_mrr = f"{weights[dataset].get('valid_mrr', 0.0):.8f}"
            selected = [
                f"{component['name']}:{component['weight']}"
                for component in weights[dataset].get("components", [])
            ]
        rows.append({
            "fold": fold_name,
            "dataset": dataset,
            "rows": str(len(labels)),
            "rule_mrr": f"{mrr(rule_scores, labels):.8f}",
            "lgbm_mrr": lgbm_mrr,
            "tgnn_mrr": tgnn_mrr,
            "ensemble_mrr": ensemble_mrr,
            "components": "|".join(selected),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="feature_cache_fast")
    parser.add_argument("--score-root", default="fast_scores")
    parser.add_argument("--weights-root", default="fast_models")
    parser.add_argument("--out", default="ablation_summary.csv")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--require-lgbm-better", action="store_true")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    folds = sorted(p.name for p in cache_root.glob("fold*") if p.is_dir())
    if not folds and cache_root.exists():
        folds = ["."]
    if not folds:
        raise ValueError(f"no fold caches found under {cache_root}")

    rows = []
    for fold_name in folds:
        rows.extend(summarize_fold(fold_name, args.cache_root, args.score_root, args.weights_root, args.max_rows))

    fieldnames = ["fold", "dataset", "rows", "rule_mrr", "lgbm_mrr", "tgnn_mrr", "ensemble_mrr", "components"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['fold']} {row['dataset']} rows={row['rows']} "
            f"rule={row['rule_mrr']} lgbm={row['lgbm_mrr']} tgnn={row['tgnn_mrr']} "
            f"ensemble={row['ensemble_mrr']} components={row['components']}"
        )
        if args.require_lgbm_better:
            if row["lgbm_mrr"] == "":
                raise SystemExit(f"missing LightGBM MRR for {row['fold']} {row['dataset']}")
            if float(row["lgbm_mrr"]) <= float(row["rule_mrr"]):
                raise SystemExit(
                    f"LightGBM did not beat rule for {row['fold']} {row['dataset']}: "
                    f"lgbm={row['lgbm_mrr']} rule={row['rule_mrr']}"
                )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
