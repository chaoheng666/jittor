import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.base_intensity_v3 import DEFAULT_BASE_WEIGHTS, BaseIntensityV3
from src.data_loader import find_dataset_dirs, iter_train_edges


def train_dataset(dataset_dir, out_dir):
    dataset_name = dataset_dir.name
    edges = list(iter_train_edges(dataset_dir / "train.csv"))
    model = BaseIntensityV3(dataset_name)
    model.fit(edges)
    artifact = {
        "dataset": dataset_name,
        "type": "base_intensity_v3",
        "enabled": True,
        "base_weights": DEFAULT_BASE_WEIGHTS.get(dataset_name, DEFAULT_BASE_WEIGHTS["dataset1"]),
        "num_edges": len(edges),
        "repeat_edge_fraction": model.rule.features.repeat_edge_fraction,
        "is_bipartite_like": model.rule.features.is_bipartite_like,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset_name}_base_intensity.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    print(f"{dataset_name}: saved {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--model-dir", default="models_v2/base_intensity")
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    dataset_dirs = find_dataset_dirs(args.data_dir)
    if args.dataset != "all":
        wanted = {name.strip() for name in args.dataset.split(",") if name.strip()}
        dataset_dirs = [path for path in dataset_dirs if path.name in wanted]
    if not dataset_dirs:
        raise ValueError(f"no datasets found in {args.data_dir}")
    out_dir = Path(args.model_dir)
    for dataset_dir in dataset_dirs:
        train_dataset(dataset_dir, out_dir)


if __name__ == "__main__":
    main()
