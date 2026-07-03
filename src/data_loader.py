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


def load_train_edges(dataset_dir):
    return list(iter_train_edges(Path(dataset_dir) / "train.csv"))


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


def stream_candidates_from_test(test_path):
    yield from iter_test_rows(test_path)


def load_test_rows(dataset_dir):
    return [(src, time, candidates) for src, time, candidates in iter_test_rows(Path(dataset_dir) / "test.csv")]


def split_by_time(edges, history_ratio=0.8):
    rows = sorted(edges, key=lambda x: x[2])
    if len(rows) < 2:
        raise ValueError("need at least two edges for a time split")
    cut = int(len(rows) * float(history_ratio))
    cut = min(max(cut, 1), len(rows) - 1)
    return rows[:cut], rows[cut:]


def iter_eval_events(edges, history_ratio=0.8, max_events=0):
    _, valid_edges = split_by_time(edges, history_ratio)
    if max_events and max_events > 0:
        valid_edges = valid_edges[:max_events]
    for row in valid_edges:
        yield row


def build_large_pool_queries(valid_edges, sampler, pool_size=500):
    for src, dst, time in valid_edges:
        negs = sampler.large_pool(src, dst, max(int(pool_size) - 1, 0))
        yield src, time, [dst] + negs
