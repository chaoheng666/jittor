import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import find_dataset_dirs
from src.edge_scoring import (
    load_test_queries,
    load_train_edges,
    row_zscore,
    score_edge_mlp_model,
    score_rule,
    softmax,
)


def score_component(component, dataset_name, train_edges, queries):
    ctype = component["type"]
    if ctype == "rule":
        return score_rule(dataset_name, train_edges, queries)
    if ctype == "edge_mlp":
        return score_edge_mlp_model(component["path"], dataset_name, train_edges, queries)
    raise ValueError(f"unknown component type: {ctype}")


def write_dataset(dataset_dir, dataset_config, output_path):
    dataset_name = dataset_dir.name
    train_edges = load_train_edges(dataset_dir)
    queries = load_test_queries(dataset_dir)
    total = None

    for component in dataset_config["components"]:
        scores = score_component(component, dataset_name, train_edges, queries)
        if len(scores) != len(queries):
            raise ValueError(f"{dataset_name}:{component['name']} row mismatch {len(scores)} != {len(queries)}")
        scores = row_zscore(scores) * float(component.get("weight", 1.0))
        total = scores if total is None else total + scores
        print(f"{dataset_name}: loaded {component['name']} type={component['type']}")

    if total is None:
        raise ValueError(f"{dataset_name}: no rerank components selected")

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
    parser.add_argument("--config", default="competition_models_best/craft_rerank_config.json")
    parser.add_argument("--out-dir", default="submission_best")
    parser.add_argument("--zip", default="result_best.zip")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    if config.get("mode") not in {"edge_intensity", "craft_rerank"}:
        raise ValueError(f"unexpected config mode: {config.get('mode')}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset_dir in find_dataset_dirs(args.data_dir):
        dataset_name = dataset_dir.name
        dataset_config = config["datasets"].get(dataset_name)
        if not dataset_config:
            print(f"skip {dataset_name}: no rerank config")
            continue
        rows = write_dataset(dataset_dir, dataset_config, out_dir / f"{dataset_name}.csv")
        print(f"{dataset_name}: wrote {rows} rows")

    make_zip(out_dir, Path(args.zip))
    print(f"packed {args.zip}")


if __name__ == "__main__":
    main()
