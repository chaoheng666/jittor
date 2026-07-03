import json
from pathlib import Path

import numpy as np

from .base_intensity_v3 import BaseIntensityV3
from .edge_scoring import (
    cold_mask,
    score_base_intensity_model,
    score_craft_residual,
    score_edge_mlp_model,
    score_rule,
    score_seq_model,
)
from .metrics import row_zscore, softmax
from .rule_ranker_v2 import RuleRankerV2


def component_enabled(component):
    if not component.get("enabled", True):
        return False
    return float(component.get("weight", 1.0)) != 0.0


def score_component(component, dataset_name, train_edges, queries, batch_size=512):
    ctype = component.get("type")
    if not component_enabled(component):
        return np.zeros((len(queries), len(queries[0][2]) if queries else 0), dtype=np.float32)
    if ctype in {"rule", "manual_rule"}:
        return score_rule(dataset_name, train_edges, queries)
    if ctype == "base_intensity_v3":
        return score_base_intensity_model(dataset_name, train_edges, queries, weights=component.get("base_weights"))
    if ctype in {"edge_mlp", "edge_mlp_legacy"}:
        return score_edge_mlp_model(component["path"], dataset_name, train_edges, queries, batch_size=batch_size)
    if ctype == "seq_nextdst":
        return score_seq_model(component["path"], train_edges, queries, batch_size=batch_size)
    if ctype == "craft_residual":
        return score_craft_residual(component["path"], dataset_name, train_edges, queries, batch_size=max(1, batch_size // 2))
    raise ValueError(f"unknown component type: {ctype}")


def score_fusion(dataset_config, dataset_name, train_edges, queries, batch_size=512):
    if not queries:
        return np.zeros((0, 0), dtype=np.float32), {}
    total = None
    component_scores = {}
    base_model = None
    rule_model = None
    components = dataset_config.get("components", [])
    needs_base = any(component.get("type") == "base_intensity_v3" for component in components)
    needs_rule = any(component.get("type") in {"rule", "manual_rule"} for component in components)
    if needs_base and needs_rule:
        base_component = next(component for component in components if component.get("type") == "base_intensity_v3")
        base_model = BaseIntensityV3(dataset_name, weights=base_component.get("base_weights"))
        base_model.fit(train_edges)
        base_rows = []
        rule_rows = []
        for src, time, candidates in queries:
            base_scores, rule_scores = base_model.score_many_with_rule(src, time, candidates)
            base_rows.append(base_scores)
            rule_rows.append(rule_scores)
        component_scores["base_intensity_v3"] = np.asarray(base_rows, dtype=np.float32)
        component_scores["manual_rule"] = np.asarray(rule_rows, dtype=np.float32)

    for component in components:
        name = component.get("name", component.get("type", "component"))
        ctype = component.get("type")
        if name in component_scores:
            scores = component_scores[name]
        elif ctype == "base_intensity_v3":
            if base_model is None:
                base_model = BaseIntensityV3(dataset_name, weights=component.get("base_weights"))
                base_model.fit(train_edges)
            scores = np.asarray(
                [base_model.score_many(src, time, candidates) for src, time, candidates in queries],
                dtype=np.float32,
            )
        elif ctype in {"rule", "manual_rule"}:
            if base_model is not None:
                scores = np.asarray(
                    [base_model.rule.score_many(src, time, candidates) for src, time, candidates in queries],
                    dtype=np.float32,
                )
            else:
                if rule_model is None:
                    rule_model = RuleRankerV2(dataset_name)
                    rule_model.fit(train_edges)
                scores = np.asarray(
                    [rule_model.score_many(src, time, candidates) for src, time, candidates in queries],
                    dtype=np.float32,
                )
        else:
            scores = score_component(component, dataset_name, train_edges, queries, batch_size=batch_size)
        component_scores[name] = scores
        weight = float(component.get("weight", 1.0))
        if weight == 0.0:
            continue
        weighted = row_zscore(scores) * weight
        total = weighted if total is None else total + weighted
    if total is None:
        total = np.zeros((len(queries), len(queries[0][2])), dtype=np.float32)
    penalty = float(dataset_config.get("cold_penalty", 0.0))
    if penalty:
        total = total - penalty * cold_mask(train_edges, queries)
    return total, component_scores


def predict_proba(dataset_config, dataset_name, train_edges, queries, batch_size=512):
    scores, component_scores = score_fusion(dataset_config, dataset_name, train_edges, queries, batch_size=batch_size)
    probs = softmax(scores, temperature=float(dataset_config.get("temperature", 1.0)))
    return probs, scores, component_scores


def load_fusion_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_fusion_config(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def discover_disabled_component(name, ctype, reason):
    return {
        "name": name,
        "type": ctype,
        "enabled": False,
        "weight": 0.0,
        "disabled_reason": reason,
    }
