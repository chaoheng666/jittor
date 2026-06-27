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


def build_test_prior(dataset_dir, train_edges, max_cold_pool, seed):
    train_dsts = {dst for _, dst, _ in train_edges}
    test_path = dataset_dir / "test.csv"
    if not test_path.exists():
        return {
            "cold_values": [],
            "cold_fraction": 0.0,
        }

    rng = random.Random(seed)
    total_count = 0
    cold_count = 0
    cold_values = []
    with open(test_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            for value in row[2:]:
                total_count += 1
                dst = int(value)
                if dst in train_dsts:
                    continue
                cold_count += 1
                if len(cold_values) < max_cold_pool:
                    cold_values.append(dst)
                else:
                    replace_idx = rng.randrange(cold_count)
                    if replace_idx < max_cold_pool:
                        cold_values[replace_idx] = dst

    return {
        "cold_values": cold_values,
        "cold_fraction": cold_count / max(total_count, 1),
    }


def build_hard_candidates(train_edges, recent_limit=80, popular_limit=2000):
    dst_counter = Counter()
    recent_by_src = defaultdict(lambda: deque(maxlen=recent_limit))
    transition = defaultdict(Counter)
    cooc = defaultdict(Counter)
    by_src = defaultdict(list)

    for src, dst, time in sorted(train_edges, key=lambda x: x[2]):
        dst_counter[dst] += 1
        recent_by_src[src].append(dst)
        by_src[src].append((time, dst))

    for rows in by_src.values():
        rows.sort()
        dsts = [dst for _, dst in rows]
        for i in range(1, len(rows)):
            transition[rows[i - 1][1]][rows[i][1]] += 1
        recent_dsts = dsts[-300:]
        for i, dst in enumerate(recent_dsts):
            start = max(0, i - 8)
            for j in range(start, i):
                prev = recent_dsts[j]
                if prev != dst:
                    cooc[prev][dst] += 1.0 / (i - j)

    popular = [dst for dst, _ in dst_counter.most_common(popular_limit)]
    return recent_by_src, transition, cooc, popular


def add_unique(candidates, seen, values, limit):
    added = 0
    for dst in values:
        if added >= limit:
            break
        if dst in seen:
            continue
        seen.add(dst)
        candidates.append(dst)
        added += 1
    return added


def sample_from_pool(rng, candidates, seen, values, limit, max_tries=20000):
    if not values or limit <= 0:
        return 0
    added = 0
    tries = 0
    while added < limit and tries < max_tries:
        tries += 1
        dst = rng.choice(values)
        if dst in seen:
            continue
        seen.add(dst)
        candidates.append(dst)
        added += 1
    return added


def sample_candidates(
    rng,
    src,
    positive,
    dst_values,
    dst_unique,
    recent_by_src,
    transition,
    cooc,
    popular,
    transition_limit,
    popular_sample,
    valid_mode,
    test_prior,
    cold_fraction_override,
):
    seen = {positive}
    candidates = [positive]

    recent = list(recent_by_src.get(src, ()))
    recent_values = list(reversed(recent))
    transition_values = []
    cooc_values = []
    if recent:
        transition_values = [dst for dst, _ in transition[recent[-1]].most_common(transition_limit)]
        cooc_values = [dst for dst, _ in cooc[recent[-1]].most_common(transition_limit)]
    popular_values = popular[:popular_sample]

    if valid_mode == "recent-heavy":
        known_hard = recent_values + transition_values + cooc_values + popular_values
    elif valid_mode == "popular-heavy":
        known_hard = popular_values + recent_values + transition_values + cooc_values
    elif valid_mode == "transition-heavy":
        known_hard = transition_values + cooc_values + recent_values + popular_values
    else:
        known_hard = []
        max_len = max(len(recent_values), len(transition_values), len(cooc_values), len(popular_values))
        for i in range(max_len):
            if i < len(recent_values):
                known_hard.append(recent_values[i])
            if i < len(transition_values):
                known_hard.append(transition_values[i])
            if i < len(cooc_values):
                known_hard.append(cooc_values[i])
            if i < len(popular_values):
                known_hard.append(popular_values[i])

    if valid_mode == "test-prior":
        cold_fraction = (
            float(cold_fraction_override)
            if cold_fraction_override >= 0
            else float(test_prior.get("cold_fraction", 0.0))
        )
        target_cold = int(round(99 * max(0.0, min(cold_fraction, 0.95))))
        target_known = 99 - target_cold
        add_unique(candidates, seen, known_hard, target_known)
        sample_from_pool(rng, candidates, seen, test_prior.get("cold_values", []), target_cold)
    else:
        add_unique(candidates, seen, known_hard, 99)

    while len(candidates) < 100:
        added = sample_from_pool(rng, candidates, seen, dst_values, 100 - len(candidates), max_tries=10000)
        if added == 0:
            break
    for dst in dst_unique:
        if len(candidates) >= 100:
            break
        if dst not in seen:
            seen.add(dst)
            candidates.append(dst)
    cold_values = test_prior.get("cold_values", [])
    while len(candidates) < 100 and cold_values:
        if sample_from_pool(rng, candidates, seen, cold_values, 1, max_tries=1000) == 0:
            break

    if len(candidates) != 100:
        raise ValueError(f"could only build {len(candidates)} candidates for src={src}")
    rng.shuffle(candidates)
    return candidates.index(positive), candidates


def write_valid(
    path,
    dataset_dir,
    valid_edges,
    train_edges,
    seed,
    max_valid,
    hard_recent_limit,
    hard_transition_limit,
    hard_popular_limit,
    hard_popular_sample,
    valid_mode,
    cold_fraction,
    max_cold_pool,
):
    rng = random.Random(seed)
    dst_values = [dst for _, dst, _ in train_edges]
    dst_unique = sorted(set(dst_values))
    if not dst_unique:
        raise ValueError("training split has no destination nodes")

    recent_by_src, transition, cooc, popular = build_hard_candidates(
        train_edges, hard_recent_limit, hard_popular_limit
    )
    test_prior = build_test_prior(dataset_dir, train_edges, max_cold_pool, seed)

    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "time", "label"] + [f"c{i}" for i in range(1, 101)])
        for src, dst, time in valid_edges:
            if max_valid and rows >= max_valid:
                break
            label, candidates = sample_candidates(
                rng,
                src,
                dst,
                dst_values,
                dst_unique,
                recent_by_src,
                transition,
                cooc,
                popular,
                hard_transition_limit,
                hard_popular_sample,
                valid_mode,
                test_prior,
                cold_fraction,
            )
            writer.writerow([src, time, label] + candidates)
            rows += 1
    return rows, test_prior


def split_dataset(rows, valid_ratio):
    has_split = any(split is not None and split != "" for *_, split in rows)
    if has_split:
        train_edges = [(src, dst, time) for src, dst, time, split in rows if split != "1"]
        valid_edges = [(src, dst, time) for src, dst, time, split in rows if split == "1"]
        if valid_edges:
            return train_edges, valid_edges, "split-column"

    rows_3 = [(src, dst, time) for src, dst, time, _ in rows]
    rows_3.sort(key=lambda x: x[2])
    cut = int(len(rows_3) * (1.0 - valid_ratio))
    return rows_3[:cut], rows_3[cut:], "time-ratio"


def build_dataset(dataset_dir, output_dir, valid_ratio, seed, max_valid, args):
    rows = list(read_rows(dataset_dir / "train.csv"))
    train_edges, valid_edges, split_mode = split_dataset(rows, valid_ratio)
    write_train(output_dir / "train.csv", train_edges)
    valid_rows, test_prior = write_valid(
        output_dir / "valid.csv",
        dataset_dir,
        valid_edges,
        train_edges,
        seed,
        max_valid,
        args.hard_recent_limit,
        args.hard_transition_limit,
        args.hard_popular_limit,
        args.hard_popular_sample,
        args.valid_mode,
        args.cold_fraction,
        args.max_cold_pool,
    )
    return len(train_edges), valid_rows, split_mode, test_prior


def find_dataset_dirs(data_dir):
    return sorted(
        p for p in Path(data_dir).iterdir()
        if p.is_dir() and (p / "train.csv").exists() and (p / "test.csv").exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--out-dir", default="validation")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--max-valid", type=int, default=50000)
    parser.add_argument("--hard-recent-limit", type=int, default=80)
    parser.add_argument("--hard-transition-limit", type=int, default=200)
    parser.add_argument("--hard-popular-limit", type=int, default=2000)
    parser.add_argument("--hard-popular-sample", type=int, default=300)
    parser.add_argument(
        "--valid-mode",
        choices=["test-prior", "recent-heavy", "popular-heavy", "transition-heavy", "mixed"],
        default="test-prior",
    )
    parser.add_argument(
        "--cold-fraction",
        type=float,
        default=-1.0,
        help="Override cold candidate fraction; -1 uses the test candidate prior.",
    )
    parser.add_argument("--max-cold-pool", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = find_dataset_dirs(data_dir)
    if not dataset_dirs:
        raise ValueError(f"no dataset dirs found in {data_dir}")

    for idx, dataset_dir in enumerate(dataset_dirs):
        dataset_out = out_dir / dataset_dir.name
        dataset_out.mkdir(parents=True, exist_ok=True)
        train_rows, valid_rows, split_mode, test_prior = build_dataset(
            dataset_dir, dataset_out, args.valid_ratio, args.seed + idx, args.max_valid, args
        )
        print(
            f"{dataset_dir.name}: train={train_rows}, valid={valid_rows}, "
            f"split={split_mode}, test_cold_fraction={test_prior['cold_fraction']:.4f}, "
            f"out={dataset_out}"
        )


if __name__ == "__main__":
    main()
