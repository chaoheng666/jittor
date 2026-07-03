import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import find_dataset_dirs
from src.fusion import save_fusion_config
from src.legacy_selection import legacy_component_candidates


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


WEAK_COMPONENT_SCALE = 0.35
TIME_REPLAY_DROP_TOL = 0.002
LEARNED_TYPES = {"seq_nextdst", "craft_residual", "edge_mlp_legacy"}


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


def optional_int(row, key):
    if key not in row or row.get(key) in {"", None}:
        return None
    return int(float(row[key]))


def maybe_component(dataset_name, name, ctype, path, default_weight, val_rows, base_mrr, rule_mrr, time_rows):
    requires_path = ctype in LEARNED_TYPES
    exists = (path is None and not requires_path) or (path is not None and Path(path).exists())
    metric = val_rows.get(name, {})
    mrr = float(metric.get("large_pool_mrr", 0.0) or 0.0)
    enabled = exists and int(metric.get("enabled", 1)) == 1
    weight = float(default_weight) if enabled else 0.0
    reason = ""

    if ctype in LEARNED_TYPES:
        threshold = max(base_mrr, rule_mrr) - 1e-12
        if not enabled:
            reason = "missing_or_validation_failed"
            weight = 0.0
        elif mrr <= 0.0:
            reason = "non_positive_validation_mrr"
            weight = 0.0
        elif threshold > 0.0 and mrr < threshold:
            reason = "large_pool_below_base_or_rule_reduced_weight"
            weight = float(default_weight) * WEAK_COMPONENT_SCALE

    if ctype in LEARNED_TYPES and weight > 0.0 and time_rows:
        base_time = float(time_rows.get("base_intensity_v3", {}).get("time_replay_mrr", 0.0) or 0.0)
        rule_time = float(time_rows.get("manual_rule", {}).get("time_replay_mrr", 0.0) or 0.0)
        time_threshold = max(base_time, rule_time)
        time_metric = time_rows.get(name)
        if time_threshold > 0.0:
            if not time_metric:
                weight = min(weight, float(default_weight) * WEAK_COMPONENT_SCALE)
                reason = "missing_time_replay_metric_reduced_weight"
            else:
                blocks = optional_int(time_metric, "blocks")
                enabled_blocks = optional_int(time_metric, "enabled_blocks")
                if enabled_blocks is not None and enabled_blocks <= 0:
                    weight = 0.0
                    reason = "time_replay_failed"
                elif enabled_blocks is not None and blocks is not None and enabled_blocks < blocks:
                    weight = min(weight, float(default_weight) * WEAK_COMPONENT_SCALE)
                    reason = "partial_time_replay_failed_reduced_weight"

                component_time = float(time_metric.get("time_replay_mrr", 0.0) or 0.0)
                if weight > 0.0 and component_time < time_threshold - TIME_REPLAY_DROP_TOL:
                    weight = min(weight, float(default_weight) * WEAK_COMPONENT_SCALE)
                    reason = "time_replay_below_base_or_rule_reduced_weight"

                if ctype == "seq_nextdst" and weight > 0.0 and base_time > 0.0:
                    if component_time < base_time - TIME_REPLAY_DROP_TOL:
                        weight = min(weight, 0.10)
                        reason = "time_replay_seq_degraded"

    component = {
        "name": name,
        "type": ctype,
        "enabled": bool(enabled and weight != 0.0),
        "weight": weight,
        "validation_mrr": mrr,
    }
    if path is not None:
        component["path"] = str(path)
    if reason:
        component["disabled_reason"] = reason
    return component


def select_legacy_component(dataset_name, model_root, default_weight, val_rows, base_mrr, rule_mrr, time_rows, legacy_top_k):
    candidates = []
    for legacy in legacy_component_candidates(model_root / "legacy", dataset_name, top_k=legacy_top_k):
        candidates.append(maybe_component(
            dataset_name,
            legacy["name"],
            "edge_mlp_legacy",
            legacy["path"],
            default_weight,
            val_rows,
            base_mrr,
            rule_mrr,
            time_rows,
        ))
    if not candidates:
        return maybe_component(
            dataset_name,
            "edge_mlp_legacy",
            "edge_mlp_legacy",
            None,
            default_weight,
            val_rows,
            base_mrr,
            rule_mrr,
            time_rows,
        )
    candidates.sort(
        key=lambda component: (
            float(component.get("weight", 0.0) or 0.0) > 0.0,
            float(component.get("validation_mrr", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return candidates[0]


def select_dataset(dataset_name, model_root, val_metrics, time_metrics, legacy_top_k=3):
    defaults = DEFAULT_WEIGHTS.get(dataset_name, DEFAULT_WEIGHTS["dataset1"])
    val_rows = val_metrics.get(dataset_name, {})
    time_rows = time_metrics.get(dataset_name, {})
    base_mrr = float(val_rows.get("base_intensity_v3", {}).get("large_pool_mrr", 0.0) or 0.0)
    rule_mrr = float(val_rows.get("manual_rule", {}).get("large_pool_mrr", 0.0) or 0.0)
    model_root = Path(model_root)
    seq_path = model_root / "seq" / f"{dataset_name}_seq_nextdst.pkl"
    craft_path = model_root / "craft" / f"{dataset_name}_craft_residual.pkl"

    components = [
        maybe_component(dataset_name, "base_intensity_v3", "base_intensity_v3", None, defaults["base_intensity_v3"], val_rows, base_mrr, rule_mrr, time_rows),
        maybe_component(dataset_name, "seq_nextdst", "seq_nextdst", seq_path, defaults["seq_nextdst"], val_rows, base_mrr, rule_mrr, time_rows),
        maybe_component(dataset_name, "craft_residual", "craft_residual", craft_path, defaults["craft_residual"], val_rows, base_mrr, rule_mrr, time_rows),
        select_legacy_component(dataset_name, model_root, defaults["edge_mlp_legacy"], val_rows, base_mrr, rule_mrr, time_rows, legacy_top_k),
        maybe_component(dataset_name, "manual_rule", "manual_rule", None, defaults["manual_rule"], val_rows, base_mrr, rule_mrr, time_rows),
    ]

    if base_mrr <= 0.0 and rule_mrr <= 0.0:
        for component in components:
            if component["type"] in LEARNED_TYPES:
                component["weight"] = 0.0
                component["disabled_reason"] = "no_validation_metrics"

    return {
        "dataset": dataset_name,
        "components": components,
        "temperature": 1.0,
        "cold_penalty": float(defaults["cold_penalty"]),
        "fallback_policy": "zero_failed_or_missing_deep_component_reduce_weak_or_unstable_component",
        "validation_metrics": {
            "large_pool": val_rows,
            "time_replay": time_rows,
        },
    }


def learned_weight_sum(dataset_config):
    return sum(
        float(component.get("weight", 0.0) or 0.0)
        for component in dataset_config.get("components", [])
        if component.get("type") in LEARNED_TYPES
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--model-root", default="models_v2")
    parser.add_argument("--validation", default="reports/val_large_pool.csv")
    parser.add_argument("--time-replay", default="reports/time_replay_summary.csv")
    parser.add_argument("--out", default="models_v2/fusion_config.json")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--require-learned", action="store_true")
    parser.add_argument("--min-learned-weight", type=float, default=1e-9)
    parser.add_argument("--legacy-top-k", type=int, default=3)
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
    missing_learned = []
    for dataset_dir in dataset_dirs:
        result["datasets"][dataset_dir.name] = select_dataset(
            dataset_dir.name,
            args.model_root,
            val_metrics,
            time_metrics,
            legacy_top_k=args.legacy_top_k,
        )
        weights = {
            component["type"]: component["weight"]
            for component in result["datasets"][dataset_dir.name]["components"]
        }
        print(f"{dataset_dir.name}: weights={weights}")
        if args.require_learned and learned_weight_sum(result["datasets"][dataset_dir.name]) < args.min_learned_weight:
            missing_learned.append(dataset_dir.name)
    if missing_learned:
        joined = ", ".join(missing_learned)
        raise RuntimeError(f"learned components have zero usable weight for: {joined}")
    save_fusion_config(args.out, result)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
