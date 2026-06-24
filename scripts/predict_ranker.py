import argparse
import csv
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import jittor as jt

from src.data_loader import find_dataset_dirs, iter_test_rows, iter_train_edges
from src.jt_ranker import CandidateFeatureBuilder, FEATURE_NAMES, load_model, normalize_features


def softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - scores.max()
    exp_scores = np.exp(scores)
    total = exp_scores.sum()
    if total <= 0:
        return np.full(len(scores), 1.0 / len(scores))
    return exp_scores / total


def write_dataset_submission(dataset_dir, model_dir, output_path, mode, batch_size):
    model_path = Path(model_dir) / f"{dataset_dir.name}_jt_ranker.pkl"
    model, meta = load_model(model_path)
    mean = np.asarray(meta["mean"], dtype=np.float32)
    std = np.asarray(meta["std"], dtype=np.float32)
    fuse_rule = float(meta.get("fuse_rule", 1.0))
    rule_idx = FEATURE_NAMES.index("rule_score")

    feature_builder = CandidateFeatureBuilder(dataset_dir.name)
    feature_builder.fit(iter_train_edges(dataset_dir / "train.csv"))

    rows = []
    written = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for src, time, candidates in iter_test_rows(dataset_dir / "test.csv"):
            rows.append(feature_builder.matrix(src, time, candidates))
            if len(rows) >= batch_size:
                written += write_batch(writer, model, rows, mean, std, rule_idx, fuse_rule, mode)
                rows.clear()
        if rows:
            written += write_batch(writer, model, rows, mean, std, rule_idx, fuse_rule, mode)
    return written


def write_batch(writer, model, rows, mean, std, rule_idx, fuse_rule, mode):
    x_raw = np.asarray(rows, dtype=np.float32)
    x = normalize_features(x_raw, mean, std).astype(np.float32)
    model.eval()
    scores = model(jt.array(x)).numpy()
    if mode == "fuse":
        scores = scores + x_raw[:, :, rule_idx] * fuse_rule
    for row_scores in scores:
        probs = softmax(row_scores)
        writer.writerow([f"{p:.8f}" for p in probs])
    return len(rows)


def make_zip(output_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_path in sorted(output_dir.glob("*.csv")):
            zf.write(csv_path, arcname=csv_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--out-dir", default="submission_mlp")
    parser.add_argument("--zip", default="result_mlp.zip")
    parser.add_argument("--mode", choices=["mlp", "fuse"], default="fuse")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    if args.cuda:
        jt.flags.use_cuda = 1

    data_dir = Path(args.data_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = find_dataset_dirs(data_dir)
    if not dataset_dirs:
        raise ValueError(f"no dataset dirs found in {data_dir}")

    for dataset_dir in dataset_dirs:
        output_path = output_dir / f"{dataset_dir.name}.csv"
        rows = write_dataset_submission(dataset_dir, args.model_dir, output_path, args.mode, args.batch_size)
        print(f"{dataset_dir.name}: wrote {rows} rows to {output_path}")

    make_zip(output_dir, Path(args.zip))
    print(f"packed {args.zip}")


if __name__ == "__main__":
    main()
