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
    score_edge_mlp_model,
    score_rule,
)


def discover_components(dataset_name, model_root, score_dir):
    model_root = Path(model_root)
    score_dir = Path(score_dir)
    components = []
    for path in sorted(model_root.rglob(f"{dataset_name}_edge_ranker.pkl")):
        name = path.parent.relative_to(model_root).as_posix()
        components.append({"name": name, "type": "edge_mlp", "path": str(path)})
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
    if ctype == "edge_mlp":
        return score_edge_mlp_model(component["path"], dataset_name, train_edges, queries)
    if ctype == "craft":
        scores = np.load(component["path"])
        return scores[:max_rows] if max_rows else scores
    raise ValueError(f"unknown component type: {ctype}")


def top1_diff_rate(scores, reference_scores):
    scores = np.asarray(scores)
    reference_scores = np.asarray(reference_scores)
    if len(scores) == 0:
        return 0.0
    return float(np.mean(np.argmax(scores, axis=1) != np.argmax(reference_scores, axis=1)))


def search_dataset(args, dataset_name):
    dataset_dir = Path(args.valid_dir) / dataset_name
    train_edges = load_train_edges(dataset_dir)
    queries, labels = load_valid_queries(dataset_dir, args.max_rows)

    rule_component = {"name": "rule", "type": "rule"}
    rule_scores = component_scores(rule_component, dataset_name, dataset_dir, train_edges, queries, args.max_rows)
    rule_mrr = mrr(rule_scores, labels)
    rule_z = row_zscore(rule_scores)
    print(f"{dataset_name} component=rule type=rule mrr={rule_mrr:.8f}")

    base_components = discover_components(dataset_name, args.model_root, args.score_dir)

    scored = []
    for component in base_components:
        scores = component_scores(component, dataset_name, dataset_dir, train_edges, queries, args.max_rows)
        if len(scores) != len(labels):
            print(f"skip {dataset_name}:{component['name']} row mismatch {len(scores)} != {len(labels)}")
            continue
        component_mrr = mrr(scores, labels)
        component_z = row_zscore(scores)
        search_scores = component_z
        residualized = False
        if args.residualize_against_rule:
            search_scores = component_z - rule_z
            residualized = True
        scored.append((component_mrr, component, search_scores, residualized))
        diff = top1_diff_rate(component_z, rule_z)
        print(
            f"{dataset_name} component={component['name']} type={component['type']} "
            f"mrr={component_mrr:.8f} top1_diff_vs_rule={diff:.4f}"
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    if args.start == "best" and scored and scored[0][0] > rule_mrr:
        start_item = scored.pop(0)
        current = rule_z + start_item[2] if start_item[3] else start_item[2].copy()
        best_mrr = mrr(current, labels)
        selected = []
        if start_item[3]:
            selected.append({"name": "rule", "type": "rule", "weight": 1.0})
        selected_component = dict(start_item[1])
        selected_component["weight"] = 1.0
        selected_component["residualize"] = bool(start_item[3])
        selected.append(selected_component)
        print(
            f"{dataset_name} start component={start_item[1]['name']} "
            f"type={start_item[1]['type']} ensemble_mrr={best_mrr:.8f}"
        )
    else:
        current = rule_z.copy()
        best_mrr = rule_mrr
        selected = [{"name": "rule", "type": "rule", "weight": 1.0}]
        print(f"{dataset_name} start component=rule type=rule ensemble_mrr={best_mrr:.8f}")

    tried_weights = [float(x) for x in args.weight_grid.split(",") if x.strip()]
    for _, component, scores, residualized in scored:
        best_weight = 0.0
        best_candidate_mrr = best_mrr
        best_candidate_diff = top1_diff_rate(current, rule_z)
        for weight in tried_weights:
            candidate = current + scores * weight
            candidate_mrr = mrr(candidate, labels)
            candidate_diff = top1_diff_rate(candidate, rule_z)
            if candidate_diff > args.max_top1_diff:
                continue
            if candidate_mrr > best_candidate_mrr + args.min_add_gain:
                best_candidate_mrr = candidate_mrr
                best_weight = weight
                best_candidate_diff = candidate_diff
        if abs(best_weight) > 0:
            current = current + scores * best_weight
            best_mrr = best_candidate_mrr
            item = dict(component)
            item["weight"] = best_weight
            item["residualize"] = bool(residualized)
            selected.append(item)
            print(
                f"{dataset_name} add {component['name']} weight={best_weight} "
                f"mrr={best_mrr:.8f} top1_diff_vs_rule={best_candidate_diff:.4f}"
            )
        else:
            print(f"{dataset_name} drop {component['name']}")

    return {
        "dataset": dataset_name,
        "valid_mrr": best_mrr,
        "rule_mrr": rule_mrr,
        "top1_diff_vs_rule": top1_diff_rate(current, rule_z),
        "residualize_against_rule": bool(args.residualize_against_rule),
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
    parser.add_argument("--weight-grid", default="0.01,0.02,0.05,0.08,0.1,0.15,0.2")
    parser.add_argument("--start", choices=["rule", "best"], default="rule")
    parser.add_argument("--min-add-gain", type=float, default=0.0005)
    parser.add_argument("--max-top1-diff", type=float, default=0.35)
    parser.add_argument("--no-residualize-against-rule", action="store_true")
    args = parser.parse_args()
    args.residualize_against_rule = not args.no_residualize_against_rule

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
