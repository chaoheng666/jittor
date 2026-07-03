import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.data_loader import find_dataset_dirs, iter_test_rows, iter_train_edges
from src.fusion import load_fusion_config, save_fusion_config, score_fusion
from src.metrics import topk_unseen_stats
from src.rule_ranker_v2 import RuleRankerV2


def load_queries(dataset_dir, max_rows=0):
    rows = []
    for idx, row in enumerate(iter_test_rows(dataset_dir / "test.csv")):
        if max_rows and idx >= max_rows:
            break
        rows.append(row)
    return rows


def component_weight(config, ctype):
    for component in config.get("components", []):
        if component.get("type") == ctype:
            return float(component.get("weight", 0.0))
    return 0.0


def set_component_weight(config, ctype, value):
    for component in config.get("components", []):
        if component.get("type") == ctype:
            component["weight"] = float(value)


def sanity_dataset(dataset_dir, dataset_config, args):
    dataset_name = dataset_dir.name
    train_edges = list(iter_train_edges(dataset_dir / "train.csv"))
    seen_dst = {dst for _, dst, _ in train_edges}
    queries = load_queries(dataset_dir, args.max_rows)
    candidates = [row[2] for row in queries]
    candidate_total = sum(len(row) for row in candidates)
    candidate_unseen = sum(dst not in seen_dst for row in candidates for dst in row)
    candidate_unseen_frac = candidate_unseen / max(candidate_total, 1)

    scores, component_scores = score_fusion(dataset_config, dataset_name, train_edges, queries, batch_size=args.batch_size)
    unseen_stats = topk_unseen_stats(scores, candidates, seen_dst, ks=(1, 5))

    rule_scores = component_scores.get("manual_rule")
    if rule_scores is None:
        rule = RuleRankerV2(dataset_name)
        rule.fit(train_edges)
        rule_scores = np.asarray([rule.score_many(src, time, cands) for src, time, cands in queries], dtype=np.float32)
    fusion_top1 = np.argmax(scores, axis=1) if len(scores) else []
    rule_top1 = np.argmax(rule_scores, axis=1) if len(rule_scores) else []
    top1_rule_agreement = float(np.mean(fusion_top1 == rule_top1)) if len(rule_scores) else 0.0

    thresholds = {
        "top1_unseen_max": candidate_unseen_frac + args.top1_margin,
        "top5_unseen_max": candidate_unseen_frac + args.top5_margin,
        "top1_rule_agreement_min": args.min_rule_agreement,
    }
    passed = (
        unseen_stats["top1_unseen_frac_pred"] <= thresholds["top1_unseen_max"]
        and unseen_stats["top5_unseen_frac_pred"] <= thresholds["top5_unseen_max"]
        and top1_rule_agreement >= thresholds["top1_rule_agreement_min"]
    )
    metrics = {
        "dataset": dataset_name,
        "rows": len(queries),
        "candidate_unseen_frac": candidate_unseen_frac,
        "top1_unseen_frac_pred": unseen_stats["top1_unseen_frac_pred"],
        "top5_unseen_frac_pred": unseen_stats["top5_unseen_frac_pred"],
        "top1_rule_agreement": top1_rule_agreement,
        "passed": bool(passed),
        **thresholds,
    }
    return metrics


def adjust_dataset_config(dataset_config, metrics):
    dataset_config.setdefault("sanity_metrics", {}).update(metrics)
    if metrics["passed"]:
        return
    dataset_config["fallback_reason"] = "official_candidate_sanity_failed"
    set_component_weight(dataset_config, "craft_residual", 0.0)
    if metrics["top1_unseen_frac_pred"] > metrics["top1_unseen_max"]:
        set_component_weight(dataset_config, "seq_nextdst", min(component_weight(dataset_config, "seq_nextdst"), 0.10))
        dataset_config["cold_penalty"] = float(dataset_config.get("cold_penalty", 0.0)) + 0.10
    if metrics["top1_rule_agreement"] < metrics["top1_rule_agreement_min"]:
        set_component_weight(dataset_config, "craft_residual", 0.0)
        set_component_weight(dataset_config, "seq_nextdst", min(component_weight(dataset_config, "seq_nextdst"), 0.10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--config", default="models_v2/fusion_config.json")
    parser.add_argument("--out", default="reports/official_candidate_sanity.json")
    parser.add_argument("--adjusted-config", default="")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top1-margin", type=float, default=0.08)
    parser.add_argument("--top5-margin", type=float, default=0.10)
    parser.add_argument("--min-rule-agreement", type=float, default=0.40)
    args = parser.parse_args()

    config = load_fusion_config(args.config)
    dataset_dirs = find_dataset_dirs(args.data_dir)
    if args.dataset != "all":
        wanted = {name.strip() for name in args.dataset.split(",") if name.strip()}
        dataset_dirs = [path for path in dataset_dirs if path.name in wanted]

    report = {"datasets": {}}
    for dataset_dir in dataset_dirs:
        dataset_config = config["datasets"].get(dataset_dir.name)
        if not dataset_config:
            continue
        metrics = sanity_dataset(dataset_dir, dataset_config, args)
        report["datasets"][dataset_dir.name] = metrics
        adjust_dataset_config(dataset_config, metrics)
        print(
            f"{dataset_dir.name}: sanity passed={metrics['passed']} "
            f"top1_unseen={metrics['top1_unseen_frac_pred']:.6f} "
            f"candidate_unseen={metrics['candidate_unseen_frac']:.6f} "
            f"rule_agree={metrics['top1_rule_agreement']:.6f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if args.adjusted_config:
        save_fusion_config(args.adjusted_config, config)
        print(f"saved adjusted config {args.adjusted_config}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
