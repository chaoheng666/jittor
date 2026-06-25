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
    load_test_queries,
    load_train_edges,
    row_zscore,
    score_mlp_model,
    score_rule,
    score_seq_model,
    softmax,
)


def score_component(component, dataset_name, train_edges, queries):
    ctype = component["type"]
    if ctype == "rule":
        return score_rule(dataset_name, train_edges, queries)
    if ctype == "mlp":
        return score_mlp_model(component["path"], dataset_name, train_edges, queries)
    if ctype == "seq":
        return score_seq_model(component["path"], dataset_name, train_edges, queries)
    if ctype == "craft":
        path = Path(component.get("test_path", ""))
        if not path.exists():
            valid_path = Path(component["path"])
            path = valid_path.with_name(valid_path.name.replace("_valid.npy", "_test.npy"))
        if not path.exists():
            raise FileNotFoundError(f"missing CRAFT test score cache: {path}")
        return np.load(path)
    raise ValueError(f"unknown component type: {ctype}")


def write_dataset(dataset_dir, dataset_weights, output_path):
    dataset_name = dataset_dir.name
    train_edges = load_train_edges(dataset_dir)
    queries = load_test_queries(dataset_dir)
    total = None

    for component in dataset_weights["components"]:
        weight = float(component["weight"])
        scores = score_component(component, dataset_name, train_edges, queries)
        if len(scores) != len(queries):
            raise ValueError(f"{dataset_name}:{component['name']} row mismatch {len(scores)} != {len(queries)}")
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
        rows = write_dataset(dataset_dir, weights["datasets"][dataset_name], out_dir / f"{dataset_name}.csv")
        print(f"{dataset_name}: wrote {rows} rows")

    make_zip(out_dir, Path(args.zip))
    print(f"packed {args.zip}")


if __name__ == "__main__":
    main()
