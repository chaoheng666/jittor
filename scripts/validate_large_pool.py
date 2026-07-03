import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.base_intensity_v3 import BaseIntensityV3
from src.data_loader import find_dataset_dirs, iter_train_edges, split_by_time
from src.fusion import discover_disabled_component, score_component
from src.metrics import hit_at_k, reciprocal_rank
from src.samplers import MixedNegativeSampler


def read_json(path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_components(model_root, dataset_name):
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

    legacy = sorted((model_root / "legacy").rglob(f"{dataset_name}_edge_ranker.pkl"))
    if legacy:
        components.append({"name": "edge_mlp_legacy", "type": "edge_mlp_legacy", "path": str(legacy[0]), "enabled": True, "weight": 1.0})
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
    queries = []
    for src, dst, time in valid:
        negatives = sampler.large_pool(src, dst, args.pool_size - 1)
        if negatives:
            queries.append((src, time, [dst] + negatives))
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
    for component in discover_components(args.model_root, dataset_dir.name):
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
    parser.add_argument("--pool-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    dataset_dirs = find_dataset_dirs(args.data_dir)
    if args.dataset != "all":
        wanted = {name.strip() for name in args.dataset.split(",") if name.strip()}
        dataset_dirs = [path for path in dataset_dirs if path.name in wanted]
    rows = []
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
