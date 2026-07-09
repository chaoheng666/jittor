import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_train_rows(train_path):
    rows = []
    with open(train_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_split = "split" in (reader.fieldnames or [])
        for row in reader:
            split = row.get("split", "") if has_split else ""
            rows.append((int(row["src"]), int(row["dst"]), int(row["time"]), str(split)))
    rows.sort(key=lambda x: x[2])
    return rows


def split_rows(rows):
    has_split = any(split != "" for *_, split in rows)
    if has_split:
        train = [(src, dst, time) for src, dst, time, split in rows if split == "0"]
        valid = [(src, dst, time) for src, dst, time, split in rows if split != "0"]
        if train and valid:
            return train, valid, "split_column"
    cut = int(len(rows) * 0.8)
    cut = min(max(cut, 1), len(rows) - 1)
    return [(s, d, t) for s, d, t, _ in rows[:cut]], [(s, d, t) for s, d, t, _ in rows[cut:]], "time_80_20"


def percentile_summary(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "avg": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def build_history_stats(edges):
    src_counts = Counter()
    dst_counts = Counter()
    seen_pairs = set()
    last_src_time = {}
    last_dst_time = {}
    for src, dst, time in edges:
        src_counts[src] += 1
        dst_counts[dst] += 1
        seen_pairs.add((src, dst))
        last_src_time[src] = max(last_src_time.get(src, time), time)
        last_dst_time[dst] = max(last_dst_time.get(dst, time), time)
    return {
        "src_counts": src_counts,
        "dst_counts": dst_counts,
        "seen_pairs": seen_pairs,
        "src_seen": set(src_counts),
        "dst_seen": set(dst_counts),
        "last_src_time": last_src_time,
        "last_dst_time": last_dst_time,
    }


def valid_bucket_stats(valid_edges, history):
    counts = Counter()
    for src, dst, _time in valid_edges:
        counts["events"] += 1
        if (src, dst) in history["seen_pairs"]:
            counts["repeated_pair"] += 1
        else:
            counts["new_pair"] += 1
        if dst not in history["dst_seen"]:
            counts["cold_dst"] += 1
        if src not in history["src_seen"]:
            counts["cold_src"] += 1
        elif history["src_counts"].get(src, 0) <= 0:
            counts["no_history_src"] += 1
    total = max(counts["events"], 1)
    return {
        "events": int(counts["events"]),
        "repeated_pair": int(counts["repeated_pair"]),
        "new_pair": int(counts["new_pair"]),
        "cold_dst": int(counts["cold_dst"]),
        "cold_src": int(counts["cold_src"]),
        "no_history_src": int(counts["no_history_src"]),
        "repeated_pair_ratio": counts["repeated_pair"] / total,
        "new_pair_ratio": counts["new_pair"] / total,
        "cold_dst_ratio": counts["cold_dst"] / total,
        "cold_src_ratio": counts["cold_src"] / total,
        "no_history_src_ratio": counts["no_history_src"] / total,
    }


def test_candidate_stats(test_path, history):
    row_counts = Counter()
    cell_counts = Counter()
    src_hist_lens = []
    candidate_dst_hist_lens = []
    with open(test_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) != 102:
            raise ValueError(f"{test_path}: expected 102 columns, got {len(header)}")
        for line_no, row in enumerate(reader, start=2):
            if len(row) != 102:
                raise ValueError(f"{test_path}:{line_no}: expected 102 columns, got {len(row)}")
            src = int(row[0])
            candidates = [int(value) for value in row[2:]]
            row_counts["rows"] += 1
            src_hist_len = int(history["src_counts"].get(src, 0))
            src_hist_lens.append(src_hist_len)
            if src not in history["src_seen"]:
                row_counts["src_unseen"] += 1
            if src_hist_len <= 0:
                row_counts["src_no_history"] += 1
            unseen_in_row = 0
            for dst in candidates:
                hist_len = int(history["dst_counts"].get(dst, 0))
                candidate_dst_hist_lens.append(hist_len)
                cell_counts["candidate_cells"] += 1
                if dst in history["dst_seen"]:
                    cell_counts["candidate_dst_seen"] += 1
                else:
                    cell_counts["candidate_dst_unseen"] += 1
                    unseen_in_row += 1
            if unseen_in_row:
                row_counts["rows_with_unseen_candidate_dst"] += 1
    cells = max(cell_counts["candidate_cells"], 1)
    rows = max(row_counts["rows"], 1)
    return {
        "rows": int(row_counts["rows"]),
        "candidate_cells": int(cell_counts["candidate_cells"]),
        "candidate_dst_unseen": int(cell_counts["candidate_dst_unseen"]),
        "candidate_dst_seen": int(cell_counts["candidate_dst_seen"]),
        "candidate_dst_unseen_ratio": cell_counts["candidate_dst_unseen"] / cells,
        "candidate_dst_seen_ratio": cell_counts["candidate_dst_seen"] / cells,
        "rows_with_unseen_candidate_dst": int(row_counts["rows_with_unseen_candidate_dst"]),
        "rows_with_unseen_candidate_dst_ratio": row_counts["rows_with_unseen_candidate_dst"] / rows,
        "src_unseen_rows": int(row_counts["src_unseen"]),
        "src_unseen_ratio": row_counts["src_unseen"] / rows,
        "src_no_history_rows": int(row_counts["src_no_history"]),
        "src_no_history_ratio": row_counts["src_no_history"] / rows,
        "src_history_length": percentile_summary(src_hist_lens),
        "candidate_dst_history_length": percentile_summary(candidate_dst_hist_lens),
    }


def profile_dataset2(data_dir):
    dataset_dir = Path(data_dir) / "dataset2"
    train_rows = read_train_rows(dataset_dir / "train.csv")
    train_edges, valid_edges, split_method = split_rows(train_rows)
    history = build_history_stats(train_edges)
    valid = valid_bucket_stats(valid_edges, history)
    test = test_candidate_stats(dataset_dir / "test.csv", history)
    return {
        "dataset": "dataset2",
        "split_method": split_method,
        "train_split0_events": len(train_edges),
        "valid_split1_events": len(valid_edges),
        "num_src": len(history["src_seen"]),
        "num_dst": len(history["dst_seen"]),
        "valid": valid,
        "test": test,
        "train_src_history_length": percentile_summary(history["src_counts"].values()),
        "train_dst_history_length": percentile_summary(history["dst_counts"].values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--out", default="reports/dataset2_data_profile.json")
    args = parser.parse_args()
    result = profile_dataset2(args.data_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
