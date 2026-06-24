import argparse
import csv
import json
import zipfile
from pathlib import Path

from data_loader import find_dataset_dirs, iter_test_rows, iter_train_edges
from rule_ranker import RuleRanker
from rule_ranker_v2 import RuleRankerV2


def load_weights(path, dataset_name):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(dataset_name, data)


def write_dataset_submission(dataset_dir, output_path, ranker_name, weights_path):
    if ranker_name == "v2":
        ranker = RuleRankerV2(dataset_dir.name, load_weights(weights_path, dataset_dir.name))
    else:
        ranker = RuleRanker()
    ranker.fit(iter_train_edges(dataset_dir / "train.csv"))

    rows = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for src, time, candidates in iter_test_rows(dataset_dir / "test.csv"):
            probs = ranker.predict_proba(src, time, candidates)
            writer.writerow([f"{p:.8f}" for p in probs])
            rows += 1
    return rows


def make_zip(output_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_path in sorted(output_dir.glob("*.csv")):
            zf.write(csv_path, arcname=csv_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--out-dir", default="submission")
    parser.add_argument("--zip", default="result.zip")
    parser.add_argument("--ranker", choices=["v1", "v2"], default="v2")
    parser.add_argument("--weights")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.out_dir)
    zip_path = Path(args.zip)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = find_dataset_dirs(data_dir)
    if not dataset_dirs:
        raise ValueError(f"no dataset dirs found in {data_dir}")

    for dataset_dir in dataset_dirs:
        output_path = output_dir / f"{dataset_dir.name}.csv"
        rows = write_dataset_submission(dataset_dir, output_path, args.ranker, args.weights)
        print(f"{dataset_dir.name}: wrote {rows} rows to {output_path}")

    make_zip(output_dir, zip_path)
    print(f"packed {zip_path}")


if __name__ == "__main__":
    main()
