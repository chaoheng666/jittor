import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.data_loader import find_dataset_dirs
from src.luxury_scoring import (
    cached_rule_scores,
    load_test_queries,
    load_train_edges,
    row_zscore,
    score_mlp_model,
    score_mlp_model_cached,
    score_rule,
    score_seq_model,
    score_seq_model_cached,
    softmax,
)


def score_component(component, dataset_name, train_edges, queries, cache_dir=""):
    ctype = component["type"]
    if cache_dir:
        if ctype == "rule":
            return cached_rule_scores(cache_dir, dataset_name, "test")
        if ctype == "mlp":
            return score_mlp_model_cached(
                component["path"], dataset_name, cache_dir, "test", score_mode=component.get("score_mode", "fused")
            )
        if ctype == "seq":
            return score_seq_model_cached(
                component["path"], dataset_name, cache_dir, "test", score_mode=component.get("score_mode", "fused")
            )
    if ctype == "rule":
        return score_rule(dataset_name, train_edges, queries)
    if ctype == "mlp":
        return score_mlp_model(
            component["path"], dataset_name, train_edges, queries, score_mode=component.get("score_mode", "fused")
        )
    if ctype == "seq":
        return score_seq_model(
            component["path"], dataset_name, train_edges, queries, score_mode=component.get("score_mode", "fused")
        )
    raise ValueError(f"unknown component type: {ctype}")


def write_dataset(dataset_dir, dataset_weights, output_path, cache_dir=""):
    dataset_name = dataset_dir.name
    if cache_dir:
        train_edges = None
        queries = None
        expected_rows = len(np.load(Path(cache_dir) / dataset_name / "x_test.npy", mmap_mode="r"))
    else:
        train_edges = load_train_edges(dataset_dir)
        queries = load_test_queries(dataset_dir)
        expected_rows = len(queries)
    total = None

    for component in dataset_weights["components"]:
        weight = float(component["weight"])
        scores = score_component(component, dataset_name, train_edges, queries, cache_dir)
        if len(scores) != expected_rows:
            raise ValueError(f"{dataset_name}:{component['name']} row mismatch {len(scores)} != {expected_rows}")
        scores = row_zscore(scores) * weight
        total = scores if total is None else total + scores
        print(f"{dataset_name}: loaded {component['name']} weight={weight}")

    probs = softmax(total)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in probs:
            writer.writerow([f"{p:.8f}" for p in row])
    return len(probs)


def make_zip(output_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_path in sorted(output_dir.glob("*.csv")):
            zf.write(csv_path, arcname=csv_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--weights", default="luxury_models/ensemble_weights.json")
    parser.add_argument("--out-dir", default="submission_luxury")
    parser.add_argument("--zip", default="result_luxury.zip")
    parser.add_argument("--cache-dir", default="")
    args = parser.parse_args()

    with open(args.weights, encoding="utf-8") as f:
        weights = json.load(f)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dirs = find_dataset_dirs(data_dir)
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        if dataset_name not in weights["datasets"]:
            print(f"skip {dataset_name}: no ensemble weights")
            continue
        rows = write_dataset(
            dataset_dir,
            weights["datasets"][dataset_name],
            out_dir / f"{dataset_name}.csv",
            args.cache_dir,
        )
        print(f"{dataset_name}: wrote {rows} rows")

    make_zip(out_dir, Path(args.zip))
    print(f"packed {args.zip}")


if __name__ == "__main__":
    main()
