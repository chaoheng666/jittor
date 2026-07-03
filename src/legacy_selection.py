from pathlib import Path


def _component_name(legacy_root, path):
    rel = path.parent.relative_to(legacy_root).as_posix()
    return f"edge_mlp_legacy:{rel}"


def _load_jittor_meta(path):
    try:
        import jittor as jt

        data = jt.load(str(path))
    except Exception:
        return {}
    return data.get("meta", {}) if isinstance(data, dict) else {}


def discover_legacy_models(legacy_root, dataset_name):
    legacy_root = Path(legacy_root)
    rows = []
    for path in sorted(legacy_root.rglob(f"{dataset_name}_edge_ranker.pkl")):
        meta = _load_jittor_meta(path)
        fused = float(meta.get("future_fused_acc", 0.0) or 0.0)
        rule = float(meta.get("future_rule_acc", 0.0) or 0.0)
        rows.append({
            "name": _component_name(legacy_root, path),
            "path": path,
            "has_meta": bool(meta),
            "use_edge_mlp": bool(meta.get("use_edge_mlp", True)),
            "future_fused_acc": fused,
            "future_gain": fused - rule,
            "eval_pairs": int(meta.get("eval_pairs", 0) or 0),
        })
    return rows


def ranked_legacy_models(legacy_root, dataset_name):
    models = discover_legacy_models(legacy_root, dataset_name)
    if not models:
        return []
    if not any(row["has_meta"] for row in models):
        return models
    models.sort(
        key=lambda row: (
            row["use_edge_mlp"],
            row["future_fused_acc"],
            row["future_gain"],
            row["eval_pairs"],
        ),
        reverse=True,
    )
    return models


def legacy_component_candidates(legacy_root, dataset_name, top_k=1):
    top_k = max(int(top_k), 1)
    return ranked_legacy_models(legacy_root, dataset_name)[:top_k]


def best_legacy_model_path(legacy_root, dataset_name):
    candidates = legacy_component_candidates(legacy_root, dataset_name, top_k=1)
    return candidates[0]["path"] if candidates else None
