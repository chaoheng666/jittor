import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


POP_BUCKETS = [
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101-500", 101, 500),
    (">500", 501, 10**12),
]


def bucket_name(count):
    count = int(count)
    if count <= 0:
        return "0"
    for name, lo, hi in POP_BUCKETS:
        if lo <= count <= hi:
            return name
    return ">500"


def percentiles(values):
    arr = np.asarray(values, dtype=np.int64)
    if arr.size == 0:
        return {}
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(arr, qs)
    return {str(q): float(v) for q, v in zip(qs, vals)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ma-user/work/jittor_rebuild_v5/data_A")
    parser.add_argument("--dataset", default="dataset2")
    parser.add_argument("--output", default="")
    parser.add_argument("--recent_tail_fraction", type=float, default=0.2)
    args = parser.parse_args()

    train_path = Path(args.data_dir) / args.dataset / "train.csv"
    split0 = []
    split1 = []
    with train_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = (int(row["time"]), int(row["src"]), int(row["dst"]))
            if str(row.get("split", "")) == "1":
                split1.append(item)
            else:
                split0.append(item)

    split0.sort()
    split1.sort()
    dst_counts = Counter()
    src_history = defaultdict(set)
    src_last = {}
    transition = defaultdict(Counter)
    for _t, src, dst in split0:
        prev = src_last.get(src)
        if prev is not None:
            transition[prev][dst] += 1
        src_history[src].add(dst)
        src_last[src] = dst
        dst_counts[dst] += 1

    tail_start = int(len(split0) * max(0.0, min(1.0, 1.0 - args.recent_tail_fraction)))
    recent_counts = Counter(dst for _t, _s, dst in split0[tail_start:])
    known_dsts = set(dst_counts)
    seen = 0
    repeated = 0
    transition_hit = 0
    pop_counts = []
    recent_vals = []
    pop_buckets = Counter()
    recent_buckets = Counter()
    for _t, src, dst in split1:
        if dst in known_dsts:
            seen += 1
        if dst in src_history.get(src, set()):
            repeated += 1
        last = src_last.get(src)
        if last is not None and dst in transition.get(last, {}):
            transition_hit += 1
        pc = int(dst_counts.get(dst, 0))
        rc = int(recent_counts.get(dst, 0))
        pop_counts.append(pc)
        recent_vals.append(rc)
        pop_buckets[bucket_name(pc)] += 1
        recent_buckets[bucket_name(rc)] += 1

    total = max(len(split1), 1)
    payload = {
        "dataset": args.dataset,
        "split0_rows": len(split0),
        "split1_rows": len(split1),
        "unique_split0_dst": len(known_dsts),
        "dst_seen_in_split0_ratio": seen / total,
        "cold_dst_ratio": 1.0 - seen / total,
        "src_history_hit_ratio": repeated / total,
        "new_pair_ratio": 1.0 - repeated / total,
        "transition_hit_ratio": transition_hit / total,
        "split1_pos_dst_split0_count_percentiles": percentiles(pop_counts),
        "split1_pos_dst_recent_count_percentiles": percentiles(recent_vals),
        "split1_pos_pop_buckets": {k: v / total for k, v in sorted(pop_buckets.items())},
        "split1_pos_recent_buckets": {k: v / total for k, v in sorted(recent_buckets.items())},
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
