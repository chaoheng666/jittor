import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import find_dataset_dirs, iter_test_rows, iter_train_edges, split_by_time
from src.feature_builder import FeatureBuilder


def entropy(counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for value in counter.values():
        p = value / total
        out -= p * math.log(max(p, 1e-12))
    return out


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(float(p) * (len(values) - 1)))
    return values[idx]


def js_divergence(left, right):
    keys = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    if left_total <= 0 or right_total <= 0:
        return 0.0
    out = 0.0
    for key in keys:
        p = left.get(key, 0) / left_total
        q = right.get(key, 0) / right_total
        m = 0.5 * (p + q)
        if p > 0:
            out += 0.5 * p * math.log(p / m)
        if q > 0:
            out += 0.5 * q * math.log(q / m)
    return out


def repeat_gap_median(edges):
    by_pair = defaultdict(list)
    for src, dst, time in edges:
        by_pair[(src, dst)].append(time)
    gaps = []
    for times in by_pair.values():
        times.sort()
        gaps.extend(max(times[i] - times[i - 1], 0) for i in range(1, len(times)))
    return percentile(gaps, 0.5)


def candidate_stats(test_path, hist_dst, full_dst):
    total = 0
    unseen_hist = 0
    unseen_full = 0
    duplicate_rows = 0
    rows = 0
    for _, _, candidates in iter_test_rows(test_path):
        rows += 1
        total += len(candidates)
        unseen_hist += sum(dst not in hist_dst for dst in candidates)
        unseen_full += sum(dst not in full_dst for dst in candidates)
        duplicate_rows += int(len(set(candidates)) < len(candidates))
    return {
        "test_rows": rows,
        "candidate_total": total,
        "candidate_unseen_frac": unseen_full / max(total, 1),
        "candidate_unseen_frac_vs_history": unseen_hist / max(total, 1),
        "candidate_duplicate_row_frac": duplicate_rows / max(rows, 1),
    }


def src_entropy_rows(valid_edges):
    by_src = defaultdict(Counter)
    for src, dst, _ in valid_edges:
        by_src[src][dst] += 1
    rows = []
    values = []
    for src, counter in sorted(by_src.items()):
        value = entropy(counter)
        values.append(value)
        rows.append({
            "src": src,
            "future_events": sum(counter.values()),
            "future_unique_dst": len(counter),
            "entropy": value,
        })
    return rows, values


def dst_pop_windows(edges, windows=5):
    rows = []
    if not edges:
        return rows
    sorted_edges = sorted(edges, key=lambda x: x[2])
    n = len(sorted_edges)
    prev = None
    for idx in range(windows):
        start = int(n * idx / windows)
        end = int(n * (idx + 1) / windows)
        chunk = sorted_edges[start:end]
        counter = Counter(dst for _, dst, _ in chunk)
        top = counter.most_common(1)
        shift = js_divergence(prev, counter) if prev is not None else 0.0
        rows.append({
            "window": idx,
            "start_time": chunk[0][2] if chunk else "",
            "end_time": chunk[-1][2] if chunk else "",
            "events": len(chunk),
            "unique_dst": len(counter),
            "top_dst": top[0][0] if top else "",
            "top_dst_count": top[0][1] if top else 0,
            "dst_pop_shift_js": shift,
        })
        prev = counter
    return rows


def two_hop_summary(history_edges, valid_edges, max_eval):
    if max_eval and max_eval > 0:
        valid_edges = valid_edges[:max_eval]
    fb = FeatureBuilder()
    fb.fit(history_edges)
    covered = 0
    cn_values = []
    aa_values = []
    ra_values = []
    for src, dst, time in valid_edges:
        feats = fb.features(src, time, dst)
        cn = feats.get("temporal_cn", 0.0)
        aa = feats.get("temporal_aa", 0.0)
        ra = feats.get("temporal_ra", 0.0)
        covered += int(cn > 0.0)
        cn_values.append(cn)
        aa_values.append(aa)
        ra_values.append(ra)
    return {
        "two_hop_eval_edges": len(valid_edges),
        "two_hop_coverage": covered / max(len(valid_edges), 1),
        "temporal_cn_mean": sum(cn_values) / max(len(cn_values), 1),
        "temporal_cn_p90": percentile(cn_values, 0.9),
        "temporal_aa_mean": sum(aa_values) / max(len(aa_values), 1),
        "temporal_ra_mean": sum(ra_values) / max(len(ra_values), 1),
    }


def analyze_dataset(dataset_dir, args):
    edges = list(iter_train_edges(dataset_dir / "train.csv"))
    history, valid = split_by_time(edges, args.history_ratio)
    srcs = {src for src, _, _ in edges}
    dsts = {dst for _, dst, _ in edges}
    hist_dst = {dst for _, dst, _ in history}
    full_dst = set(dsts)
    pairs = Counter((src, dst) for src, dst, _ in edges)
    hist_pairs = {(src, dst) for src, dst, _ in history}
    repeat_extra = sum(count - 1 for count in pairs.values()) / max(len(edges), 1)
    valid_repeat = sum((src, dst) in hist_pairs for src, dst, _ in valid) / max(len(valid), 1)
    overlap_ratio = len(srcs & dsts) / max(min(len(srcs), len(dsts)), 1)
    ent_rows, ent_values = src_entropy_rows(valid)
    cand = candidate_stats(dataset_dir / "test.csv", hist_dst, full_dst)
    two_hop = two_hop_summary(history, valid, args.max_two_hop_eval)
    pop_rows = dst_pop_windows(edges, args.windows)
    dst_recent_top = set(dst for dst, _ in Counter(dst for _, dst, _ in history[-max(len(history) // 5, 1):]).most_common(args.recent_hit_k))
    dst_recent_hit = sum(dst in dst_recent_top for _, dst, _ in valid) / max(len(valid), 1)

    summary = {
        "dataset": dataset_dir.name,
        "num_edges": len(edges),
        "num_src": len(srcs),
        "num_dst": len(dsts),
        "src_dst_overlap_ratio": overlap_ratio,
        "is_bipartite_like": int(overlap_ratio < 0.01),
        "repeat_edge_ratio_all": repeat_extra,
        "repeat_edge_ratio_valid": valid_repeat,
        "candidate_unseen_frac": cand["candidate_unseen_frac"],
        "candidate_unseen_frac_vs_history": cand["candidate_unseen_frac_vs_history"],
        "src_next_entropy_mean": sum(ent_values) / max(len(ent_values), 1),
        "src_next_entropy_p50": percentile(ent_values, 0.5),
        "src_next_entropy_p90": percentile(ent_values, 0.9),
        "dst_pop_shift": sum(row["dst_pop_shift_js"] for row in pop_rows[1:]) / max(len(pop_rows) - 1, 1),
        "two_hop_coverage": two_hop["two_hop_coverage"],
        "repeat_gap_median": repeat_gap_median(edges),
        f"dst_recent_hit@{args.recent_hit_k}": dst_recent_hit,
        **cand,
        **two_hop,
    }
    return summary, ent_rows, pop_rows, two_hop


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--out-dir", default="reports/data_stats")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--history-ratio", type=float, default=0.8)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--recent-hit-k", type=int, default=100)
    parser.add_argument("--max-two-hop-eval", type=int, default=5000)
    args = parser.parse_args()

    dataset_dirs = find_dataset_dirs(args.data_dir)
    if args.dataset != "all":
        wanted = {name.strip() for name in args.dataset.split(",") if name.strip()}
        dataset_dirs = [path for path in dataset_dirs if path.name in wanted]
    if not dataset_dirs:
        raise ValueError(f"no datasets found in {args.data_dir}")

    out_dir = Path(args.out_dir)
    summaries = []
    entropy_rows = []
    pop_rows_all = []
    two_hop_rows = []
    for dataset_dir in dataset_dirs:
        summary, ent_rows, pop_rows, two_hop = analyze_dataset(dataset_dir, args)
        summaries.append(summary)
        entropy_rows.extend({"dataset": dataset_dir.name, **row} for row in ent_rows)
        pop_rows_all.extend({"dataset": dataset_dir.name, **row} for row in pop_rows)
        two_hop_rows.append({"dataset": dataset_dir.name, **two_hop})
        print(
            f"{dataset_dir.name}: repeat_valid={summary['repeat_edge_ratio_valid']:.6f} "
            f"candidate_unseen={summary['candidate_unseen_frac']:.6f} "
            f"entropy_mean={summary['src_next_entropy_mean']:.6f} "
            f"two_hop={summary['two_hop_coverage']:.6f}"
        )

    write_csv(out_dir / "summary.csv", summaries, list(summaries[0].keys()))
    write_csv(out_dir / "src_entropy.csv", entropy_rows, list(entropy_rows[0].keys()) if entropy_rows else ["dataset"])
    write_csv(out_dir / "dst_pop_windows.csv", pop_rows_all, list(pop_rows_all[0].keys()) if pop_rows_all else ["dataset"])
    write_csv(out_dir / "two_hop.csv", two_hop_rows, list(two_hop_rows[0].keys()))
    test_stats = [
        {
            key: value for key, value in row.items()
            if key in {
                "dataset",
                "test_rows",
                "candidate_total",
                "candidate_unseen_frac",
                "candidate_unseen_frac_vs_history",
                "candidate_duplicate_row_frac",
            }
        }
        for row in summaries
    ]
    write_csv(out_dir / "test_candidate_stats.csv", test_stats, list(test_stats[0].keys()))


if __name__ == "__main__":
    main()
