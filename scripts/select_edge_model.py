import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import jittor as jt

from src.data_loader import find_dataset_dirs


def load_meta(path):
    data = jt.load(str(path))
    return data.get("meta", {})


def discover_models(model_root, dataset_name):
    model_root = Path(model_root)
    rows = []
    for path in sorted(model_root.rglob(f"{dataset_name}_edge_ranker.pkl")):
        meta = load_meta(path)
        name = path.parent.relative_to(model_root).as_posix()
        rows.append({
            "name": name,
            "type": "edge_mlp",
            "path": str(path),
            "future_rule_acc": float(meta.get("future_rule_acc", 0.0)),
            "future_residual_acc": float(meta.get("future_residual_acc", 0.0)),
            "future_fused_acc": float(meta.get("future_fused_acc", 0.0)),
            "future_eval_loss": float(meta.get("future_eval_loss", 0.0)),
            "train_pairs": int(meta.get("train_pairs", 0)),
            "eval_pairs": int(meta.get("eval_pairs", 0)),
            "use_edge_mlp": bool(meta.get("use_edge_mlp", True)),
        })
    return rows


def select_dataset(model_root, dataset_name, min_future_gain):
    models = discover_models(model_root, dataset_name)
    if not models:
        print(f"{dataset_name}: no edge MLP found, use rule-only")
        return {
            "dataset": dataset_name,
            "selection_metric": "rule_only_no_model",
            "components": [{"name": "rule", "type": "rule", "weight": 1.0}],
            "candidates": [],
        }

    models.sort(key=lambda row: row["future_fused_acc"], reverse=True)
    best = models[0]
    gain = best["future_fused_acc"] - best["future_rule_acc"]
    print(
        f"{dataset_name}: best={best['name']} "
        f"future_fused_acc={best['future_fused_acc']:.6f} "
        f"future_rule_acc={best['future_rule_acc']:.6f} gain={gain:.6f}"
    )

    if gain < min_future_gain or not best["use_edge_mlp"]:
        print(f"{dataset_name}: gain below {min_future_gain}, use rule-only")
        return {
            "dataset": dataset_name,
            "selection_metric": "rule_only_low_gain",
            "best_model": best,
            "components": [{"name": "rule", "type": "rule", "weight": 1.0}],
            "candidates": models,
        }

    component = {
        "name": best["name"],
        "type": "edge_mlp",
        "path": best["path"],
        "weight": 1.0,
    }
    return {
        "dataset": dataset_name,
        "selection_metric": "future_fused_acc",
        "best_model": best,
        "components": [component],
        "candidates": models,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--model-root", default="competition_models_best")
    parser.add_argument("--out", default="competition_models_best/edge_intensity_config.json")
    parser.add_argument("--dataset", default="all", help="all or comma-separated dataset names")
    parser.add_argument("--min-future-gain", type=float, default=0.0005)
    args = parser.parse_args()

    dataset_dirs = find_dataset_dirs(args.data_dir)
    if args.dataset == "all":
        names = [path.name for path in dataset_dirs]
    else:
        names = [name.strip() for name in args.dataset.split(",") if name.strip()]
    if not names:
        raise ValueError(f"no datasets found in {args.data_dir}")

    result = {
        "mode": "edge_intensity",
        "min_future_gain": float(args.min_future_gain),
        "datasets": {},
    }
    for name in names:
        result["datasets"][name] = select_dataset(args.model_root, name, args.min_future_gain)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
