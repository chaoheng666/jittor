import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import find_dataset_dirs
from src.fusion import save_fusion_config


DEFAULT_WEIGHTS = {
    "dataset1": {
        "base_intensity_v3": 0.55,
        "seq_nextdst": 0.15,
        "craft_residual": 0.05,
        "edge_mlp_legacy": 0.10,
        "manual_rule": 0.15,
        "cold_penalty": 0.05,
    },
    "dataset2": {
        "base_intensity_v3": 0.45,
        "seq_nextdst": 0.30,
        "craft_residual": 0.05,
        "edge_mlp_legacy": 0.05,
        "manual_rule": 0.20,
        "cold_penalty": 0.20,
    },
}


def read_metric_table(path, key_field="component", value_field="large_pool_mrr"):
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset = row["dataset"]
            component = row[key_field]
            out.setdefault(dataset, {})[component] = {
                **row,
                value_field: float(row.get(value_field, 0.0) or 0.0),
                "enabled": int(row.get("enabled", 1) or 0),
            }
    return out


def maybe_component(dataset_name, name, ctype, path, default_weight, val_rows, base_mrr, rule_mrr, time_rows):
    exists = path is None or Path(path).exists()
    metric = val_rows.get(name, {})
    mrr = float(metric.get("large_pool_mrr", 0.0) or 0.0)
    enabled = exists and int(metric.get("enabled", 1)) == 1
    weight = float(default_weight) if enabled else 0.0
    reason = ""

    if ctype in {"seq_nextdst", "craft_residual", "edge_mlp_legacy"}:
        threshold = max(base_mrr, rule_mrr) - 1e-12
        if not enabled:
            reason = "missing_or_validation_failed"
            weight = 0.0
        elif mrr < threshold:
            reason = "large_pool_not_better_than_base_or_rule"
            weight = 0.0

    if ctype == "seq_nextdst" and weight > 0.0:
        base_time = float(time_rows.get("base_intensity_v3", {}).get("time_replay_mrr", 0.0) or 0.0)
        seq_time = float(time_rows.get("seq_nextdst", {}).get("time_replay_mrr", 0.0) or 0.0)
        if base_time and seq_time < base_time - 0.002:
            weight = min(weight, 0.10)
            reason = "time_replay_seq_degraded"

    component = {
        "name": name,
        "type": ctype,
        "enabled": bool(exists),
        "weight": weight,
        "validation_mrr": mrr,
    }
    if path is not None:
        component["path"] = str(path)
    if reason:
        component["disabled_reason"] = reason
    return component


def select_dataset(dataset_name, model_root, val_metrics, time_metrics):
    defaults = DEFAULT_WEIGHTS.get(dataset_name, DEFAULT_WEIGHTS["dataset1"])
    val_rows = val_metrics.get(dataset_name, {})
    time_rows = time_metrics.get(dataset_name, {})
    base_mrr = float(val_rows.get("base_intensity_v3", {}).get("large_pool_mrr", 0.0) or 0.0)
    rule_mrr = float(val_rows.get("manual_rule", {}).get("large_pool_mrr", 0.0) or 0.0)
    model_root = Path(model_root)
    legacy = sorted((model_root / "legacy").rglob(f"{dataset_name}_edge_ranker.pkl"))
    legacy_path = legacy[0] if legacy else None
    seq_path = model_root / "seq" / f"{dataset_name}_seq_nextdst.pkl"
    craft_path = model_root / "craft" / f"{dataset_name}_craft_residual.pkl"

    components = [
        maybe_component(dataset_name, "base_intensity_v3", "base_intensity_v3", None, defaults["base_intensity_v3"], val_rows, base_mrr, rule_mrr, time_rows),
        maybe_component(dataset_name, "seq_nextdst", "seq_nextdst", seq_path, defaults["seq_nextdst"], val_rows, base_mrr, rule_mrr, time_rows),
        maybe_component(dataset_name, "craft_residual", "craft_residual", craft_path, defaults["craft_residual"], val_rows, base_mrr, rule_mrr, time_rows),
        maybe_component(dataset_name, "edge_mlp_legacy", "edge_mlp_legacy", legacy_path, defaults["edge_mlp_legacy"], val_rows, base_mrr, rule_mrr, time_rows),
        maybe_component(dataset_name, "manual_rule", "manual_rule", None, defaults["manual_rule"], val_rows, base_mrr, rule_mrr, time_rows),
    ]

    if base_mrr <= 0.0 and rule_mrr <= 0.0:
        for component in components:
            if component["type"] in {"seq_nextdst", "craft_residual", "edge_mlp_legacy"}:
                component["weight"] = 0.0
                component["disabled_reason"] = "no_validation_metrics"

    return {
        "dataset": dataset_name,
        "components": components,
        "temperature": 1.0,
        "cold_penalty": float(defaults["cold_penalty"]),
        "fallback_policy": "zero_deep_component_on_validation_or_sanity_failure",
        "validation_metrics": {
            "large_pool": val_rows,
            "time_replay": time_rows,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--model-root", default="models_v2")
    parser.add_argument("--validation", default="reports/val_large_pool.csv")
    parser.add_argument("--time-replay", default="reports/time_replay_summary.csv")
    parser.add_argument("--out", default="models_v2/fusion_config.json")
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    val_metrics = read_metric_table(args.validation, value_field="large_pool_mrr")
    time_metrics = read_metric_table(args.time_replay, value_field="time_replay_mrr")
    dataset_dirs = find_dataset_dirs(args.data_dir)
    if args.dataset != "all":
        wanted = {name.strip() for name in args.dataset.split(",") if name.strip()}
        dataset_dirs = [path for path in dataset_dirs if path.name in wanted]
    result = {
        "mode": "fusion_v2",
        "model_root": args.model_root,
        "datasets": {},
    }
    for dataset_dir in dataset_dirs:
        result["datasets"][dataset_dir.name] = select_dataset(dataset_dir.name, args.model_root, val_metrics, time_metrics)
        weights = {
            component["type"]: component["weight"]
            for component in result["datasets"][dataset_dir.name]["components"]
        }
        print(f"{dataset_dir.name}: weights={weights}")
    save_fusion_config(args.out, result)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
