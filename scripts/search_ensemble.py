import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.luxury_scoring import (
    cached_rule_scores,
    load_train_edges,
    load_valid_queries,
    mrr,
    row_zscore,
    score_mlp_model,
    score_mlp_model_cached,
    score_rule,
    score_seq_model,
    score_seq_model_cached,
)


def discover_components(dataset_name, model_root, score_dir):
    model_root = Path(model_root)
    score_dir = Path(score_dir)
    components = []
    if score_dir.exists():
        for path in sorted(score_dir.glob(f"{dataset_name}_lgbm_*_valid.npy")):
            name = path.name.removeprefix(f"{dataset_name}_").removesuffix("_valid.npy")
            test_path = score_dir / f"{dataset_name}_{name}_test.npy"
            components.append({
                "name": name,
                "type": "lgbm",
                "path": str(path),
                "test_path": str(test_path),
            })
        for path in sorted(score_dir.glob(f"{dataset_name}_tgnn_valid.npy")):
            name = path.name.removeprefix(f"{dataset_name}_").removesuffix("_valid.npy")
            test_path = score_dir / f"{dataset_name}_{name}_test.npy"
            components.append({
                "name": name,
                "type": "tgnn",
                "path": str(path),
                "test_path": str(test_path),
            })
    for path in sorted(model_root.rglob(f"{dataset_name}_jt_ranker.pkl")):
        name = path.parent.relative_to(model_root).as_posix()
        components.append({"name": f"{name}:residual", "type": "mlp", "path": str(path), "score_mode": "residual"})
    for path in sorted(model_root.rglob(f"{dataset_name}_seq_ranker.pkl")):
        name = path.parent.relative_to(model_root).as_posix()
        components.append({"name": f"{name}:residual", "type": "seq", "path": str(path), "score_mode": "residual"})
    return components


def component_scores(component, dataset_name, dataset_dir, train_edges, queries, max_rows, cache_dir=""):
    ctype = component["type"]
    if cache_dir:
        if ctype == "rule":
            scores = cached_rule_scores(cache_dir, dataset_name, "valid")
            return scores[:max_rows] if max_rows else scores
        if ctype == "mlp":
            scores = score_mlp_model_cached(
                component["path"], dataset_name, cache_dir, "valid", score_mode=component.get("score_mode", "fused")
            )
            return scores[:max_rows] if max_rows else scores
        if ctype == "seq":
            scores = score_seq_model_cached(
                component["path"], dataset_name, cache_dir, "valid", score_mode=component.get("score_mode", "fused")
            )
            return scores[:max_rows] if max_rows else scores
        if ctype == "lgbm":
            scores = np.load(component["path"], mmap_mode="r")
            return scores[:max_rows] if max_rows else scores
        if ctype == "tgnn":
            scores = np.load(component["path"], mmap_mode="r")
            return scores[:max_rows] if max_rows else scores
    if ctype == "rule":
        return score_rule(dataset_name, train_edges, queries)
    if ctype == "mlp":
        return score_mlp_model(
            component["path"], dataset_name, train_edges, queries, score_mode=component.get("score_mode", "fused")
        )
    if ctype == "seq":
        return score_seq_model(
            component["path"], dataset_name, train_edges, queries, score_mode=component.get("score_mode", "fused")
        )
    if ctype == "lgbm":
        scores = np.load(component["path"], mmap_mode="r")
        return scores[:max_rows] if max_rows else scores
    if ctype == "tgnn":
        scores = np.load(component["path"], mmap_mode="r")
        return scores[:max_rows] if max_rows else scores
    raise ValueError(f"unknown component type: {ctype}")


def search_dataset(args, dataset_name):
    dataset_dir = Path(args.valid_dir) / dataset_name
    if args.cache_dir:
        labels = np.load(Path(args.cache_dir) / dataset_name / "y_valid.npy", mmap_mode="r")
        if args.max_rows:
            labels = labels[:args.max_rows]
        train_edges = None
        queries = None
    else:
        train_edges = load_train_edges(dataset_dir)
        queries, labels = load_valid_queries(dataset_dir, args.max_rows)

    base_components = [{"name": "rule", "type": "rule"}]
    base_components.extend(discover_components(dataset_name, args.model_root, args.score_dir))

    scored = []
    for component in base_components:
        scores = component_scores(
            component, dataset_name, dataset_dir, train_edges, queries, args.max_rows, args.cache_dir
        )
        if len(scores) != len(labels):
            print(f"skip {dataset_name}:{component['name']} row mismatch {len(scores)} != {len(labels)}")
            continue
        component_mrr = mrr(scores, labels)
        scored.append((component_mrr, component, row_zscore(scores)))
        print(f"{dataset_name} component={component['name']} type={component['type']} mrr={component_mrr:.8f}")

    if not scored:
        raise ValueError(f"{dataset_name}: no valid ensemble components")

    scored.sort(key=lambda x: x[0], reverse=True)
    start_item = scored[0]
    current = start_item[2].copy()
    best_mrr = mrr(current, labels)
    selected_component = dict(start_item[1])
    selected_component["weight"] = 1.0
    selected = [selected_component]
    print(
        f"{dataset_name} start component={start_item[1]['name']} "
        f"type={start_item[1]['type']} ensemble_mrr={best_mrr:.8f}"
    )

    tried_weights = [float(x) for x in args.weight_grid.split(",") if x.strip()]
    for _, component, scores in scored[1:]:
        if component == start_item[1]:
            continue
        best_weight = 0.0
        best_candidate_mrr = best_mrr
        for weight in tried_weights:
            candidate = current + scores * weight
            candidate_mrr = mrr(candidate, labels)
            if candidate_mrr > best_candidate_mrr + 1e-12:
                best_candidate_mrr = candidate_mrr
                best_weight = weight
        if best_weight != 0.0:
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


def find_dataset_names(valid_dir, dataset_arg):
    if dataset_arg != "all":
        return [name.strip() for name in dataset_arg.split(",") if name.strip()]
    valid_path = Path(valid_dir)
    return sorted(
        p.name for p in valid_path.iterdir()
        if p.is_dir() and (p / "train.csv").exists() and (p / "valid.csv").exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-dir", default="validation")
    parser.add_argument("--model-root", default="luxury_models")
    parser.add_argument("--score-dir", default="luxury_scores")
    parser.add_argument("--out", default="luxury_models/ensemble_weights.json")
    parser.add_argument("--dataset", default="all", help="all or comma-separated dataset names")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument(
        "--weight-grid",
        default="-1.0,-0.8,-0.5,-0.3,-0.2,-0.15,-0.1,-0.08,-0.05,-0.02,0.02,0.05,0.08,0.1,0.15,0.2,0.3,0.5,0.8,1.0",
    )
    args = parser.parse_args()

    names = find_dataset_names(args.valid_dir, args.dataset)
    if not names:
        raise ValueError(f"no validation datasets found in {args.valid_dir}")
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
