import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.valid_builder import build_dataset, find_dataset_dirs


FOLDS = [
    ("fold0", 0.10),
    ("fold1", 0.15),
    ("fold2", 0.20),
]


def read_times(path):
    times = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(int(row["time"]))
    return times


def validate_time_order(dataset_out):
    train_times = read_times(dataset_out / "train.csv")
    valid_times = []
    with open(dataset_out / "valid.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid_times.append(int(row["time"]))
    if not train_times or not valid_times:
        raise ValueError(f"{dataset_out}: empty train or valid split")
    if max(train_times) > min(valid_times):
        raise ValueError(
            f"{dataset_out}: time leakage max_train={max(train_times)} min_valid={min(valid_times)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--fold-root", default="rank_folds")
    parser.add_argument("--max-valid", type=int, default=0)
    parser.add_argument("--hard-recent-limit", type=int, default=100)
    parser.add_argument("--hard-transition-limit", type=int, default=300)
    parser.add_argument("--hard-popular-limit", type=int, default=3000)
    parser.add_argument("--hard-popular-sample", type=int, default=350)
    parser.add_argument(
        "--valid-mode",
        choices=["test-prior", "recent-heavy", "popular-heavy", "transition-heavy", "mixed"],
        default="test-prior",
    )
    parser.add_argument("--cold-fraction", type=float, default=-1.0)
    parser.add_argument("--max-cold-pool", type=int, default=3000000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fold_root = Path(args.fold_root)
    fold_root.mkdir(parents=True, exist_ok=True)
    dataset_dirs = find_dataset_dirs(data_dir)
    if not dataset_dirs:
        raise ValueError(f"no dataset dirs found in {data_dir}")

    for fold_idx, (fold_name, valid_ratio) in enumerate(FOLDS):
        fold_dir = fold_root / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        for dataset_idx, dataset_dir in enumerate(dataset_dirs):
            dataset_out = fold_dir / dataset_dir.name
            dataset_out.mkdir(parents=True, exist_ok=True)
            train_rows, valid_rows, split_mode, test_prior = build_dataset(
                dataset_dir,
                dataset_out,
                valid_ratio,
                args.seed + fold_idx * 100 + dataset_idx,
                args.max_valid,
                args,
            )
            validate_time_order(dataset_out)
            print(
                f"{fold_name}/{dataset_dir.name}: train={train_rows} valid={valid_rows} "
                f"ratio={valid_ratio:.2f} split={split_mode} "
                f"test_cold_fraction={test_prior['cold_fraction']:.4f}"
            )


if __name__ == "__main__":
    main()
