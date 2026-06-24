import argparse
import csv
import json
from pathlib import Path

from data_loader import iter_train_edges
from rule_ranker_v2 import RuleRankerV2


def load_weights(path, dataset_name):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(dataset_name, data)


def iter_valid_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates = [int(row[f"c{i}"]) for i in range(1, 101)]
            yield int(row["src"]), int(row["time"]), int(row["label"]), candidates


def rank_of_label(scores, label):
    positive_score = scores[label]
    rank = 1
    for i, score in enumerate(scores):
        if i != label and score > positive_score:
            rank += 1
    return rank


def evaluate_dataset(valid_dir, dataset_name, weights):
    dataset_dir = valid_dir / dataset_name
    ranker = RuleRankerV2(dataset_name, weights)
    ranker.fit(iter_train_edges(dataset_dir / "train.csv"))

    rr_sum = 0.0
    rows = 0
    for src, time, label, candidates in iter_valid_rows(dataset_dir / "valid.csv"):
        scores = [ranker.score(src, time, dst) for dst in candidates]
        rr_sum += 1.0 / rank_of_label(scores, label)
        rows += 1
    if rows == 0:
        return 0.0, 0
    return rr_sum / rows, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-dir", default="validation")
    parser.add_argument("--dataset", choices=["all", "dataset1", "dataset2"], default="all")
    parser.add_argument("--weights")
    args = parser.parse_args()

    valid_dir = Path(args.valid_dir)
    names = ["dataset1", "dataset2"] if args.dataset == "all" else [args.dataset]
    for name in names:
        mrr, rows = evaluate_dataset(valid_dir, name, load_weights(args.weights, name))
        print(f"{name}_mrr={mrr:.8f} rows={rows}")


if __name__ == "__main__":
    main()
