import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import jittor as jt
import numpy as np

from src.data_loader import find_dataset_dirs


def load_meta(path):
    data = jt.load(str(path))
    return data.get("meta", {})


def row_zscore(scores):
    scores = np.asarray(scores, dtype=np.float64)
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (scores - mean) / std


def mrr(scores, targets):
    ranks = []
    for row, target in zip(scores, targets):
        ranks.append(1 + int(np.sum(row > row[target])))
    ranks = np.asarray(ranks, dtype=np.float64)
    return float(np.mean(1.0 / ranks)) if len(ranks) else 0.0


def discover_models(model_root, dataset_name):
    model_root = Path(model_root)
    rows = []
    for path in sorted(model_root.rglob(f"{dataset_name}_edge_ranker.pkl")):
        meta = load_meta(path)
        name = path.parent.relative_to(model_root).as_posix()
        eval_cache_path = meta.get("eval_cache_path") or str(path.with_name(f"{dataset_name}_edge_ranker_eval.npz"))
        rows.append({
            "name": name,
            "type": "edge_mlp",
            "path": str(path),
            "eval_cache_path": eval_cache_path,
            "validation_mrr": float(meta.get("validation_mrr", 0.0)),
            "validation_hit1": float(meta.get("validation_hit1", 0.0)),
            "validation_seen_mrr": float(meta.get("validation_seen_mrr", 0.0)),
            "validation_unseen_mrr": float(meta.get("validation_unseen_mrr", 0.0)),
            "rule_mrr": float(meta.get("rule_mrr", 0.0)),
            "rule_hit1": float(meta.get("rule_hit1", 0.0)),
            "train_queries": int(meta.get("train_queries", 0)),
            "eval_queries": int(meta.get("eval_queries", 0)),
            "use_craft_model": bool(meta.get("use_craft_model", True)),
        })
    return rows


def load_eval_cache(model):
    path = Path(model["eval_cache_path"])
    if not path.exists():
        return None
    return np.load(path)


def same_eval_grid(left, right):
    if left is None or right is None:
        return False
    return (
        np.array_equal(left["targets"], right["targets"])
        and np.array_equal(left["keys"], right["keys"])
        and np.array_equal(left["candidates"], right["candidates"])
    )


def maybe_make_ensemble(best, models):
    best_cache = load_eval_cache(best)
    if best_cache is None:
        return None
    best_score = best["validation_mrr"]
    best_payload = None
    for other in models[1:6]:
        other_cache = load_eval_cache(other)
        if not same_eval_grid(best_cache, other_cache):
            continue
        targets = best_cache["targets"]
        left = row_zscore(best_cache["scores"])
        right = row_zscore(other_cache["scores"])
        for weight in (0.25, 0.5, 0.75):
            scores = left * weight + right * (1.0 - weight)
            score = mrr(scores, targets)
            if score > best_score + 1e-12:
                best_score = score
                best_payload = {
                    "validation_mrr": float(score),
                    "components": [
                        {
                            "name": best["name"],
                            "type": "edge_mlp",
                            "path": best["path"],
                            "weight": float(weight),
                        },
                        {
                            "name": other["name"],
                            "type": "edge_mlp",
                            "path": other["path"],
                            "weight": float(1.0 - weight),
                        },
                    ],
                    "members": [best, other],
                }
    return best_payload


def select_dataset(model_root, dataset_name, min_validation_gain):
    models = discover_models(model_root, dataset_name)
    if not models:
        print(f"{dataset_name}: no CRAFT reranker found, use rule-only")
        return {
            "dataset": dataset_name,
            "selection_metric": "rule_only_no_model",
            "components": [{"name": "rule", "type": "rule", "weight": 1.0}],
            "candidates": [],
        }

    models.sort(key=lambda row: row["validation_mrr"], reverse=True)
    best = models[0]
    gain = best["validation_mrr"] - best["rule_mrr"]
    print(
        f"{dataset_name}: best={best['name']} "
        f"validation_mrr={best['validation_mrr']:.6f} "
        f"rule_mrr={best['rule_mrr']:.6f} gain={gain:.6f} "
        f"seen={best['validation_seen_mrr']:.6f} unseen={best['validation_unseen_mrr']:.6f}"
    )

    if gain < min_validation_gain or not best["use_craft_model"]:
        print(f"{dataset_name}: gain below {min_validation_gain}, use rule-only")
        return {
            "dataset": dataset_name,
            "selection_metric": "rule_only_low_gain",
            "best_model": best,
            "components": [{"name": "rule", "type": "rule", "weight": 1.0}],
            "candidates": models,
        }

    ensemble = maybe_make_ensemble(best, models)
    if ensemble:
        print(
            f"{dataset_name}: enabled top-2 ensemble "
            f"validation_mrr={ensemble['validation_mrr']:.6f}"
        )
        return {
            "dataset": dataset_name,
            "selection_metric": "validation_mrr_top2_ensemble",
            "best_model": best,
            "ensemble": ensemble,
            "components": ensemble["components"],
            "candidates": models,
        }

    return {
        "dataset": dataset_name,
        "selection_metric": "validation_mrr",
        "best_model": best,
        "components": [{
            "name": best["name"],
            "type": "edge_mlp",
            "path": best["path"],
            "weight": 1.0,
        }],
        "candidates": models,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--model-root", default="competition_models_best")
    parser.add_argument("--out", default="competition_models_best/craft_rerank_config.json")
    parser.add_argument("--dataset", default="all", help="all or comma-separated dataset names")
    parser.add_argument("--min-validation-gain", "--min-future-gain", dest="min_validation_gain", type=float, default=0.0005)
    args = parser.parse_args()

    dataset_dirs = find_dataset_dirs(args.data_dir)
    if args.dataset == "all":
        names = [path.name for path in dataset_dirs]
    else:
        names = [name.strip() for name in args.dataset.split(",") if name.strip()]
    if not names:
        raise ValueError(f"no datasets found in {args.data_dir}")

    result = {
        "mode": "craft_rerank",
        "min_validation_gain": float(args.min_validation_gain),
        "datasets": {},
    }
    for name in names:
        result["datasets"][name] = select_dataset(args.model_root, name, args.min_validation_gain)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
