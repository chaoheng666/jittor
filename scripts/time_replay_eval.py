import argparse
import csv
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from scripts.validate_large_pool import (
    build_test_prior,
    discover_components,
    evaluate_component,
    official_like_negatives,
)
from src.base_intensity_v3 import BaseIntensityV3
from src.data_loader import find_dataset_dirs, iter_train_edges
from src.samplers import MixedNegativeSampler


def replay_dataset(dataset_dir, args):
    edges = sorted(iter_train_edges(dataset_dir / "train.csv"), key=lambda x: x[2])
    n = len(edges)
    rows = []
    for block in range(1, args.blocks):
        start = int(n * block / args.blocks)
        end = int(n * (block + 1) / args.blocks)
        history = edges[:start]
        eval_edges = edges[start:end]
        if args.max_block_events and args.max_block_events > 0:
            eval_edges = eval_edges[:args.max_block_events]
        if not history or not eval_edges:
            continue
        sampler = MixedNegativeSampler(history, seed=args.seed + block)
        test_prior = None
        if args.candidate_mode == "test-prior":
            test_prior = build_test_prior(dataset_dir, history, args.max_cold_pool, args.seed + block)
        rng = random.Random(args.seed + block + sum(ord(ch) for ch in dataset_dir.name))
        queries = []
        skipped = 0
        for src, dst, time in eval_edges:
            need = args.pool_size - 1
            if args.candidate_mode == "test-prior":
                negatives = official_like_negatives(sampler, rng, src, dst, need, test_prior)
            else:
                negatives = sampler.large_pool(src, dst, need)
            if len(negatives) == need:
                queries.append((src, time, [dst] + negatives))
            else:
                skipped += 1
        if skipped:
            print(f"{dataset_dir.name}:block={block}: skipped {skipped} incomplete replay rows")
        if not queries:
            continue
        base = BaseIntensityV3(dataset_dir.name)
        base.fit(history)
        base_rows = []
        rule_rows = []
        for src, time, candidates in queries:
            base_scores, rule_scores = base.score_many_with_rule(src, time, candidates)
            base_rows.append(base_scores)
            rule_rows.append(rule_scores)
        precomputed = {
            "base_intensity_v3": np.asarray(base_rows, dtype=np.float32),
            "manual_rule": np.asarray(rule_rows, dtype=np.float32),
        }
        for component in discover_components(args.model_root, dataset_dir.name, legacy_top_k=args.legacy_top_k):
            metrics = evaluate_component(component, dataset_dir.name, history, queries, args.batch_size, precomputed=precomputed)
            row = {
                "dataset": dataset_dir.name,
                "block": block,
                "component": component["name"],
                "type": component["type"],
                "history_edges": len(history),
                "eval_edges": len(eval_edges),
                "pool_size": args.pool_size,
                **metrics,
            }
            rows.append(row)
            print(
                f"{dataset_dir.name}:block={block}:{component['name']} "
                f"mrr={row['large_pool_mrr']:.6f} enabled={row['enabled']}"
            )
    return rows


def add_summary(rows):
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["component"])
        grouped.setdefault(key, []).append(row)
    out = []
    for (dataset, component), vals in grouped.items():
        enabled_vals = [row for row in vals if int(row["enabled"]) == 1]
        mrr = [float(row["large_pool_mrr"]) for row in enabled_vals]
        out.append({
            "dataset": dataset,
            "component": component,
            "blocks": len(vals),
            "enabled_blocks": len(enabled_vals),
            "failed_blocks": len(vals) - len(enabled_vals),
            "time_replay_mrr": float(np.mean(mrr)) if mrr else 0.0,
            "time_replay_mrr_min": float(np.min(mrr)) if mrr else 0.0,
        })
    return out


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--model-root", default="models_v2")
    parser.add_argument("--out", default="reports/time_replay.csv")
    parser.add_argument("--summary-out", default="reports/time_replay_summary.csv")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--max-block-events", type=int, default=1000)
    parser.add_argument("--pool-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--candidate-mode", choices=["test-prior", "mixed"], default="mixed")
    parser.add_argument("--max-cold-pool", type=int, default=3000000)
    parser.add_argument("--legacy-top-k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    dataset_dirs = find_dataset_dirs(args.data_dir)
    if args.dataset != "all":
        wanted = {name.strip() for name in args.dataset.split(",") if name.strip()}
        dataset_dirs = [path for path in dataset_dirs if path.name in wanted]
    rows = []
    workers = min(max(int(args.workers), 1), max(len(dataset_dirs), 1))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for dataset_rows in executor.map(replay_dataset, dataset_dirs, [args] * len(dataset_dirs)):
                rows.extend(dataset_rows)
    else:
        for dataset_dir in dataset_dirs:
            rows.extend(replay_dataset(dataset_dir, args))
    fieldnames = [
        "dataset", "block", "component", "type", "history_edges", "eval_edges",
        "pool_size", "large_pool_mrr", "hit10", "queries", "enabled", "error",
    ]
    write_csv(Path(args.out), rows, fieldnames)
    summary = add_summary(rows)
    write_csv(
        Path(args.summary_out),
        summary,
        [
            "dataset", "component", "blocks", "enabled_blocks", "failed_blocks",
            "time_replay_mrr", "time_replay_mrr_min",
        ],
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
