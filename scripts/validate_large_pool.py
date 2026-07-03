import argparse
import csv
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.base_intensity_v3 import BaseIntensityV3
from src.data_loader import find_dataset_dirs, iter_test_rows, iter_train_edges, split_by_time
from src.fusion import discover_disabled_component, score_component
from src.legacy_selection import legacy_component_candidates
from src.metrics import hit_at_k, reciprocal_rank
from src.samplers import MixedNegativeSampler


def read_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_test_prior(dataset_dir, history, max_cold_pool, seed):
    seen_dst = {dst for _, dst, _ in history}
    rng = random.Random(seed)
    total_count = 0
    cold_count = 0
    cold_values = []
    for _, _, candidates in iter_test_rows(dataset_dir / "test.csv"):
        for dst in candidates:
            total_count += 1
            if dst in seen_dst:
                continue
            cold_count += 1
            if len(cold_values) < max_cold_pool:
                cold_values.append(dst)
            else:
                replace_idx = rng.randrange(cold_count)
                if replace_idx < max_cold_pool:
                    cold_values[replace_idx] = dst
    return {
        "cold_fraction": cold_count / max(total_count, 1),
        "cold_values": cold_values,
    }


def add_unique(out, seen, values, limit):
    added = 0
    for dst in values:
        if added >= limit:
            break
        if dst in seen:
            continue
        seen.add(dst)
        out.append(dst)
        added += 1
    return added


def sample_unique(rng, out, seen, values, limit, max_tries=20000):
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
        out.append(dst)
        added += 1
    return added


def official_like_negatives(sampler, rng, src, positive, count, test_prior):
    count = int(count)
    if count <= 0:
        return []
    seen = {positive}
    out = []
    cold_fraction = max(0.0, min(float(test_prior.get("cold_fraction", 0.0)), 0.95))
    target_cold = int(round(count * cold_fraction))
    target_known = count - target_cold

    recent = list(reversed(list(sampler.recent_by_src.get(src, ()))))
    hard = recent
    if recent:
        last_dst = recent[0]
        hard = (
            hard
            + [dst for dst, _ in sampler.transition[last_dst].most_common(sampler.transition_limit)]
            + [dst for dst, _ in sampler.cooc[last_dst].most_common(sampler.transition_limit)]
        )
    hard = hard + sampler.popular

    add_unique(out, seen, hard, target_known)
    sample_unique(rng, out, seen, test_prior.get("cold_values", []), target_cold)

    if len(out) < count:
        mixed = sampler.large_pool(src, positive, count - len(out))
        add_unique(out, seen, mixed, count - len(out))
    if len(out) < count:
        add_unique(out, seen, sampler.dst_unique, count - len(out))
    return out[:count]


def discover_components(model_root, dataset_name, legacy_top_k=1):
    model_root = Path(model_root)
    components = [
        {"name": "manual_rule", "type": "manual_rule", "enabled": True, "weight": 1.0},
        {"name": "base_intensity_v3", "type": "base_intensity_v3", "enabled": True, "weight": 1.0},
    ]
    seq_json = read_json(model_root / "seq" / f"{dataset_name}_seq_nextdst.json")
    seq_pkl = model_root / "seq" / f"{dataset_name}_seq_nextdst.pkl"
    if seq_pkl.exists():
        components.append({"name": "seq_nextdst", "type": "seq_nextdst", "path": str(seq_pkl), "enabled": True, "weight": 1.0})
    elif seq_json and not seq_json.get("enabled", False):
        components.append(discover_disabled_component("seq_nextdst", "seq_nextdst", seq_json.get("disabled_reason", "disabled")))

    craft_json = read_json(model_root / "craft" / f"{dataset_name}_craft_residual.json")
    craft_pkl = model_root / "craft" / f"{dataset_name}_craft_residual.pkl"
    if craft_pkl.exists():
        components.append({"name": "craft_residual", "type": "craft_residual", "path": str(craft_pkl), "enabled": True, "weight": 1.0})
    elif craft_json and not craft_json.get("enabled", False):
        components.append(discover_disabled_component("craft_residual", "craft_residual", craft_json.get("disabled_reason", "disabled")))

    for legacy in legacy_component_candidates(model_root / "legacy", dataset_name, top_k=legacy_top_k):
        components.append({
            "name": legacy["name"],
            "type": "edge_mlp_legacy",
            "path": str(legacy["path"]),
            "enabled": True,
            "weight": 1.0,
        })
    return components


def evaluate_component(component, dataset_name, history, queries, batch_size, precomputed=None):
    if precomputed and component["name"] in precomputed:
        scores = precomputed[component["name"]]
        rr = [reciprocal_rank(row, 0) for row in scores]
        hits = [hit_at_k(row, 0, 10) for row in scores]
        return {
            "large_pool_mrr": float(np.mean(rr)) if rr else 0.0,
            "hit10": float(np.mean(hits)) if hits else 0.0,
            "queries": len(queries),
            "enabled": 1,
            "error": "",
        }
    if not component.get("enabled", True):
        return {
            "large_pool_mrr": 0.0,
            "hit10": 0.0,
            "queries": len(queries),
            "enabled": 0,
            "error": component.get("disabled_reason", "disabled"),
        }
    try:
        scores = score_component(component, dataset_name, history, queries, batch_size=batch_size)
        rr = [reciprocal_rank(row, 0) for row in scores]
        hits = [hit_at_k(row, 0, 10) for row in scores]
        return {
            "large_pool_mrr": float(np.mean(rr)) if rr else 0.0,
            "hit10": float(np.mean(hits)) if hits else 0.0,
            "queries": len(queries),
            "enabled": 1,
            "error": "",
        }
    except Exception as exc:
        return {
            "large_pool_mrr": 0.0,
            "hit10": 0.0,
            "queries": len(queries),
            "enabled": 0,
            "error": str(exc).replace("\n", " ")[:300],
        }


def validate_dataset(dataset_dir, args):
    edges = list(iter_train_edges(dataset_dir / "train.csv"))
    history, valid = split_by_time(edges, args.history_ratio)
    if args.max_eval_edges and args.max_eval_edges > 0:
        valid = valid[:args.max_eval_edges]
    sampler = MixedNegativeSampler(history, seed=args.seed)
    rng = random.Random(args.seed + sum(ord(ch) for ch in dataset_dir.name))
    test_prior = None
    if args.candidate_mode == "test-prior":
        test_prior = build_test_prior(dataset_dir, history, args.max_cold_pool, args.seed)
        print(
            f"{dataset_dir.name}: test-prior cold_fraction="
            f"{test_prior['cold_fraction']:.6f}"
        )
    queries = []
    skipped = 0
    for src, dst, time in valid:
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
        print(f"{dataset_dir.name}: skipped {skipped} incomplete validation rows")
    precomputed = {}
    base = BaseIntensityV3(dataset_dir.name)
    base.fit(history)
    base_rows = []
    rule_rows = []
    for src, time, candidates in queries:
        base_scores, rule_scores = base.score_many_with_rule(src, time, candidates)
        base_rows.append(base_scores)
        rule_rows.append(rule_scores)
    precomputed["base_intensity_v3"] = np.asarray(base_rows, dtype=np.float32)
    precomputed["manual_rule"] = np.asarray(rule_rows, dtype=np.float32)
    rows = []
    for component in discover_components(args.model_root, dataset_dir.name, legacy_top_k=args.legacy_top_k):
        metrics = evaluate_component(component, dataset_dir.name, history, queries, args.batch_size, precomputed=precomputed)
        row = {
            "dataset": dataset_dir.name,
            "component": component["name"],
            "type": component["type"],
            "pool_size": args.pool_size,
            **metrics,
        }
        rows.append(row)
        print(
            f"{dataset_dir.name}:{component['name']} "
            f"mrr={row['large_pool_mrr']:.6f} hit10={row['hit10']:.6f} "
            f"enabled={row['enabled']}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--model-root", default="models_v2")
    parser.add_argument("--out", default="reports/val_large_pool.csv")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--history-ratio", type=float, default=0.8)
    parser.add_argument("--max-eval-edges", type=int, default=2000)
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
            for dataset_rows in executor.map(validate_dataset, dataset_dirs, [args] * len(dataset_dirs)):
                rows.extend(dataset_rows)
    else:
        for dataset_dir in dataset_dirs:
            rows.extend(validate_dataset(dataset_dir, args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "component", "type", "pool_size", "large_pool_mrr", "hit10", "queries", "enabled", "error"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
