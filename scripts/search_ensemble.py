import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.luxury_scoring import (
    load_train_edges,
    load_valid_queries,
    mrr,
    row_zscore,
    score_mlp_model,
    score_rule,
    score_seq_model,
)


def discover_components(dataset_name, model_root, score_dir):
    model_root = Path(model_root)
    score_dir = Path(score_dir)
    components = []
    for path in sorted(model_root.rglob(f"{dataset_name}_jt_ranker.pkl")):
        components.append({"name": path.parent.name, "type": "mlp", "path": str(path)})
    for path in sorted(model_root.rglob(f"{dataset_name}_seq_ranker.pkl")):
        components.append({"name": path.parent.name, "type": "seq", "path": str(path)})
    if score_dir.exists():
        for path in sorted(score_dir.glob(f"{dataset_name}_*_valid.npy")):
            name = path.name.removeprefix(f"{dataset_name}_").removesuffix("_valid.npy")
            test_path = score_dir / f"{dataset_name}_{name}_test.npy"
            components.append({
                "name": name,
                "type": "craft",
                "path": str(path),
                "test_path": str(test_path),
            })
    return components


def component_scores(component, dataset_name, dataset_dir, train_edges, queries, max_rows):
    ctype = component["type"]
    if ctype == "rule":
        return score_rule(dataset_name, train_edges, queries)
    if ctype == "mlp":
        return score_mlp_model(component["path"], dataset_name, train_edges, queries)
    if ctype == "seq":
        return score_seq_model(component["path"], dataset_name, train_edges, queries)
    if ctype == "craft":
        scores = np.load(component["path"])
        return scores[:max_rows] if max_rows else scores
    raise ValueError(f"unknown component type: {ctype}")


def search_dataset(args, dataset_name):
    dataset_dir = Path(args.valid_dir) / dataset_name
    train_edges = load_train_edges(dataset_dir)
    queries, labels = load_valid_queries(dataset_dir, args.max_rows)

    base_components = [{"name": "rule", "type": "rule"}]
    base_components.extend(discover_components(dataset_name, args.model_root, args.score_dir))

    scored = []
    for component in base_components:
        scores = component_scores(component, dataset_name, dataset_dir, train_edges, queries, args.max_rows)
        if len(scores) != len(labels):
            print(f"skip {dataset_name}:{component['name']} row mismatch {len(scores)} != {len(labels)}")
            continue
        component_mrr = mrr(scores, labels)
        scored.append((component_mrr, component, row_zscore(scores)))
        print(f"{dataset_name} component={component['name']} type={component['type']} mrr={component_mrr:.8f}")

    scored.sort(key=lambda x: x[0], reverse=True)
    rule_item = next((item for item in scored if item[1]["type"] == "rule"), scored[0])
    current = rule_item[2].copy()
    best_mrr = mrr(current, labels)
    selected = [{
        "name": rule_item[1]["name"],
        "type": rule_item[1]["type"],
        "weight": 1.0,
    }]
    print(f"{dataset_name} start ensemble_mrr={best_mrr:.8f}")

    tried_weights = [float(x) for x in args.weight_grid.split(",") if x.strip()]
    for _, component, scores in scored:
        if component["type"] == "rule":
            continue
        best_weight = 0.0
        best_candidate_mrr = best_mrr
        for weight in tried_weights:
            candidate = current + scores * weight
            candidate_mrr = mrr(candidate, labels)
            if candidate_mrr > best_candidate_mrr + 1e-12:
                best_candidate_mrr = candidate_mrr
                best_weight = weight
        if best_weight > 0:
            current = current + scores * best_weight
            best_mrr = best_candidate_mrr
            item = dict(component)
            item["weight"] = best_weight
            selected.append(item)
            print(f"{dataset_name} add {component['name']} weight={best_weight} mrr={best_mrr:.8f}")
        else:
            print(f"{dataset_name} drop {component['name']}")

    return {
        "dataset": dataset_name,
        "valid_mrr": best_mrr,
        "components": selected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-dir", default="validation")
    parser.add_argument("--model-root", default="luxury_models")
    parser.add_argument("--score-dir", default="luxury_scores")
    parser.add_argument("--out", default="luxury_models/ensemble_weights.json")
    parser.add_argument("--dataset", choices=["all", "dataset1", "dataset2"], default="all")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--weight-grid", default="0.05,0.1,0.2,0.3,0.5,0.8,1.0")
    args = parser.parse_args()

    names = ["dataset1", "dataset2"] if args.dataset == "all" else [args.dataset]
    result = {"datasets": {}}
    for name in names:
        result["datasets"][name] = search_dataset(args, name)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
