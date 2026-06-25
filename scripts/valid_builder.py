import argparse
import csv
import random
from collections import Counter, defaultdict, deque
from pathlib import Path


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split")
            yield int(row["src"]), int(row["dst"]), int(row["time"]), split


def write_train(path, edges):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "dst", "time"])
        writer.writerows(edges)


def build_hard_candidates(train_edges, recent_limit=20, popular_limit=500):
    dst_counter = Counter()
    recent_by_src = defaultdict(lambda: deque(maxlen=recent_limit))
    transition = defaultdict(Counter)
    by_src = defaultdict(list)

    for src, dst, time in sorted(train_edges, key=lambda x: x[2]):
        dst_counter[dst] += 1
        recent_by_src[src].append(dst)
        by_src[src].append((time, dst))

    for rows in by_src.values():
        rows.sort()
        for i in range(1, len(rows)):
            transition[rows[i - 1][1]][rows[i][1]] += 1

    popular = [dst for dst, _ in dst_counter.most_common(popular_limit)]
    return recent_by_src, transition, popular


def sample_candidates(
    rng, src, positive, dst_values, dst_unique, recent_by_src, transition,
    popular, transition_limit, popular_sample, valid_mode
):
    seen = {positive}
    candidates = [positive]

    recent = list(recent_by_src.get(src, ()))
    recent_values = list(reversed(recent))
    transition_values = []
    if recent:
        transition_values = [dst for dst, _ in transition[recent[-1]].most_common(transition_limit)]
    popular_values = popular[:popular_sample]

    if valid_mode == "recent-heavy":
        hard_values = recent_values + transition_values + popular_values
    elif valid_mode == "popular-heavy":
        hard_values = popular_values + recent_values + transition_values
    elif valid_mode == "transition-heavy":
        hard_values = transition_values + recent_values + popular_values
    else:
        hard_values = []
        max_len = max(len(recent_values), len(transition_values), len(popular_values))
        for i in range(max_len):
            if i < len(recent_values):
                hard_values.append(recent_values[i])
            if i < len(transition_values):
                hard_values.append(transition_values[i])
            if i < len(popular_values):
                hard_values.append(popular_values[i])

    for dst in hard_values:
        if len(candidates) >= 100:
            break
        if dst in seen:
            continue
        seen.add(dst)
        candidates.append(dst)

    tries = 0
    while len(candidates) < 100 and tries < 10000:
        dst = rng.choice(dst_values)
        tries += 1
        if dst in seen:
            continue
        seen.add(dst)
        candidates.append(dst)
    for dst in dst_unique:
        if len(candidates) >= 100:
            break
        if dst not in seen:
            seen.add(dst)
            candidates.append(dst)
    rng.shuffle(candidates)
    return candidates.index(positive), candidates


def write_valid(
    path, valid_edges, train_edges, seed, max_valid,
    hard_recent_limit, hard_transition_limit, hard_popular_limit, hard_popular_sample,
    valid_mode
):
    rng = random.Random(seed)
    dst_values = [dst for _, dst, _ in train_edges]
    dst_unique = sorted(set(dst_values))
    if not dst_unique:
        raise ValueError("training split has no destination nodes")
    recent_by_src, transition, popular = build_hard_candidates(
        train_edges, hard_recent_limit, hard_popular_limit
    )

    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "time", "label"] + [f"c{i}" for i in range(1, 101)])
        for src, dst, time in valid_edges:
            if max_valid and rows >= max_valid:
                break
            label, candidates = sample_candidates(
                rng, src, dst, dst_values, dst_unique, recent_by_src, transition,
                popular, hard_transition_limit, hard_popular_sample, valid_mode
            )
            writer.writerow([src, time, label] + candidates)
            rows += 1
    return rows


def build_dataset1(dataset_dir, output_dir, valid_ratio, seed, max_valid, args):
    rows = [(src, dst, time) for src, dst, time, _ in read_rows(dataset_dir / "train.csv")]
    rows.sort(key=lambda x: x[2])
    cut = int(len(rows) * (1.0 - valid_ratio))
    train_edges = rows[:cut]
    valid_edges = rows[cut:]
    write_train(output_dir / "train.csv", train_edges)
    valid_rows = write_valid(
        output_dir / "valid.csv", valid_edges, train_edges, seed, max_valid,
        args.hard_recent_limit, args.hard_transition_limit,
        args.hard_popular_limit, args.hard_popular_sample, args.valid_mode
    )
    return len(train_edges), valid_rows


def build_dataset2(dataset_dir, output_dir, seed, max_valid, args):
    train_edges = []
    valid_edges = []
    for src, dst, time, split in read_rows(dataset_dir / "train.csv"):
        if split == "1":
            valid_edges.append((src, dst, time))
        else:
            train_edges.append((src, dst, time))
    write_train(output_dir / "train.csv", train_edges)
    valid_rows = write_valid(
        output_dir / "valid.csv", valid_edges, train_edges, seed, max_valid,
        args.hard_recent_limit, args.hard_transition_limit,
        args.hard_popular_limit, args.hard_popular_sample, args.valid_mode
    )
    return len(train_edges), valid_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--out-dir", default="validation")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--max-valid", type=int, default=50000)
    parser.add_argument("--hard-recent-limit", type=int, default=20)
    parser.add_argument("--hard-transition-limit", type=int, default=50)
    parser.add_argument("--hard-popular-limit", type=int, default=500)
    parser.add_argument("--hard-popular-sample", type=int, default=200)
    parser.add_argument(
        "--valid-mode",
        choices=["recent-heavy", "popular-heavy", "transition-heavy", "mixed"],
        default="mixed",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ["dataset1", "dataset2"]:
        dataset_out = out_dir / name
        dataset_out.mkdir(parents=True, exist_ok=True)
        if name == "dataset1":
            train_rows, valid_rows = build_dataset1(
                data_dir / name, dataset_out, args.valid_ratio, args.seed, args.max_valid, args
            )
        else:
            train_rows, valid_rows = build_dataset2(
                data_dir / name, dataset_out, args.seed + 1, args.max_valid, args
            )
        print(f"{name}: train={train_rows}, valid={valid_rows}, out={dataset_out}")


if __name__ == "__main__":
    main()
