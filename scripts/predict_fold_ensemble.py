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


def row_zscore(scores):
    scores = np.asarray(scores, dtype=np.float32)
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (scores - mean) / std


def softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    total = exp_scores.sum(axis=1, keepdims=True)
    total = np.where(total <= 0, 1.0, total)
    return exp_scores / total


def discover_fold_weights(weights_root):
    root = Path(weights_root)
    paths = sorted(root.glob("fold*/ensemble_weights.json"))
    if not paths and (root / "ensemble_weights.json").exists():
        paths = [root / "ensemble_weights.json"]
    if not paths:
        raise ValueError(f"no ensemble_weights.json files found under {root}")
    return paths


def fold_cache_dir(cache_root, fold_name):
    candidate = Path(cache_root) / fold_name
    return str(candidate if candidate.exists() else Path(cache_root))


def cached_rule_scores(cache_dir, dataset_name):
    dataset_cache = Path(cache_dir) / dataset_name
    x = np.load(dataset_cache / "x_test.npy", mmap_mode="r")
    with open(dataset_cache / "feature_names.json", encoding="utf-8") as f:
        names = json.load(f)
    return x[:, :, names.index("rule_score")]


def score_component(component, dataset_name, cache_dir):
    ctype = component["type"]
    if ctype == "rule":
        return cached_rule_scores(cache_dir, dataset_name)
    if ctype == "lgbm":
        path = Path(component.get("test_path", ""))
        if not path.exists():
            valid_path = Path(component["path"])
            path = valid_path.with_name(valid_path.name.replace("_valid.npy", "_test.npy"))
        if not path.exists():
            raise FileNotFoundError(f"missing LightGBM test score cache: {path}")
        return np.load(path, mmap_mode="r")
    if ctype == "tgnn":
        path = Path(component.get("test_path", ""))
        if not path.exists():
            valid_path = Path(component["path"])
            path = valid_path.with_name(valid_path.name.replace("_valid.npy", "_test.npy"))
        if not path.exists():
            raise FileNotFoundError(f"missing TGNN test score cache: {path}")
        return np.load(path, mmap_mode="r")

    from scripts.predict_luxury_ensemble import score_component as jittor_score_component
    return jittor_score_component(component, dataset_name, None, None, cache_dir)


def score_fold(dataset_name, dataset_weights, cache_dir):
    expected_rows = len(np.load(Path(cache_dir) / dataset_name / "x_test.npy", mmap_mode="r"))
    total = None
    for component in dataset_weights["components"]:
        weight = float(component["weight"])
        scores = score_component(component, dataset_name, cache_dir)
        if len(scores) != expected_rows:
            raise ValueError(f"{dataset_name}:{component['name']} row mismatch {len(scores)} != {expected_rows}")
        scores = row_zscore(scores) * weight
        total = scores if total is None else total + scores
    return row_zscore(total)


def write_dataset(dataset_name, fold_weight_paths, cache_root, output_path, min_top1_diff):
    total = None
    used = 0
    first_cache_dir = None
    for weights_path in fold_weight_paths:
        with open(weights_path, encoding="utf-8") as f:
            weights = json.load(f)
        if dataset_name not in weights.get("datasets", {}):
            continue
        fold_name = weights_path.parent.name
        cache_dir = fold_cache_dir(cache_root, fold_name)
        if first_cache_dir is None:
            first_cache_dir = cache_dir
        scores = score_fold(dataset_name, weights["datasets"][dataset_name], cache_dir)
        total = scores if total is None else total + scores
        used += 1
        print(f"{dataset_name}: loaded fold={fold_name} weights={weights_path}")

    if used == 0:
        raise ValueError(f"{dataset_name}: no fold weights available")

    final_scores = total / used
    if min_top1_diff > 0 and first_cache_dir is not None:
        rule_scores = cached_rule_scores(first_cache_dir, dataset_name)
        diff = float((final_scores.argmax(axis=1) != rule_scores.argmax(axis=1)).mean())
        print(f"{dataset_name}: top1_diff_vs_rule={diff:.4f}")
        if diff < min_top1_diff:
            raise SystemExit(
                f"{dataset_name}: top1 diff {diff:.4f} is below required {min_top1_diff:.4f}"
            )

    probs = softmax(final_scores)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in probs:
            writer.writerow([f"{p:.8f}" for p in row])
    return len(probs), used


def make_zip(output_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_path in sorted(output_dir.glob("*.csv")):
            zf.write(csv_path, arcname=csv_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--cache-root", default="feature_cache_fast")
    parser.add_argument("--weights-root", default="fast_models")
    parser.add_argument("--out-dir", default="submission_fast")
    parser.add_argument("--zip", default="result_fast.zip")
    parser.add_argument("--min-top1-diff", type=float, default=0.0)
    args = parser.parse_args()

    fold_weight_paths = discover_fold_weights(args.weights_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset_dir in find_dataset_dirs(Path(args.data_dir)):
        rows, folds = write_dataset(
            dataset_dir.name,
            fold_weight_paths,
            args.cache_root,
            out_dir / f"{dataset_dir.name}.csv",
            args.min_top1_diff,
        )
        print(f"{dataset_dir.name}: wrote {rows} rows from {folds} folds")
    make_zip(out_dir, Path(args.zip))
    print(f"packed {args.zip}")


if __name__ == "__main__":
    main()
