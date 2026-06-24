import csv
from pathlib import Path


def find_dataset_dirs(data_dir):
    data_path = Path(data_dir)
    return sorted(
        p for p in data_path.iterdir()
        if p.is_dir() and (p / "train.csv").exists() and (p / "test.csv").exists()
    )


def iter_train_edges(train_path):
    with open(train_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield int(row["src"]), int(row["dst"]), int(row["time"])


def iter_test_rows(test_path):
    with open(test_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != 102:
            raise ValueError(f"{test_path} should have 102 columns, got {len(header)}")
        for line_no, row in enumerate(reader, start=2):
            if len(row) != 102:
                raise ValueError(f"{test_path}:{line_no} should have 102 columns, got {len(row)}")
            src = int(row[0])
            time = int(row[1])
            candidates = [int(x) for x in row[2:]]
            yield src, time, candidates
