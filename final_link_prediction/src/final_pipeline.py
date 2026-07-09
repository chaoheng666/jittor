import argparse
from argparse import Namespace
from copy import copy
from typing import Iterable, List

from .data import dump_json, ensure_dir
from .baseline_support import _as_path
from .ranking_pipeline import build_context_features, package_final_result, predict_context_ranker, train_context_ranker
from .stable_baseline import build_stable_baseline


def _parse_blends(value: str) -> List[float]:
    blends = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            blends.append(float(item))
    if not blends:
        raise ValueError("at least one blend weight is required")
    return blends


def _blend_name(weight: float) -> str:
    if abs(float(weight) - 1.0) < 1e-12:
        return "result_final_pure"
    label = f"{weight:.2f}".replace(".", "p")
    return f"result_final_blend_{label}"


def _pack_one(args: Namespace, weight: float, name: str | None = None) -> dict:
    item_args = copy(args)
    item_args.blend_weight = float(weight)
    item_args.output_name = name or _blend_name(float(weight))
    return package_final_result(item_args)


def package_sweep(args: Namespace, weights: Iterable[float]) -> dict:
    manifest = {}
    for weight in weights:
        manifest.update(_pack_one(args, float(weight)))
    dump_json(ensure_dir(_as_path(args.reports)) / "final_pack_manifest.json", manifest)
    return manifest


def run_all(args: Namespace) -> None:
    if str(args.build_baseline) == "1":
        build_stable_baseline(args)
    build_context_features(args)
    train_context_ranker(args)
    predict_context_ranker(args)
    _pack_one(args, float(args.blend_weight), args.output_name)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Final Jittor competition pipeline")
    p.add_argument("--data-dir", default="data_A")
    p.add_argument("--baseline-root", default="baseline_artifacts")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--reports", default="reports")
    p.add_argument("--submission", default="submission")
    p.add_argument("--build-baseline", default="1")
    p.add_argument("--stable-seed", type=int, default=2026)
    p.add_argument("--stable-svd-dim", type=int, default=160)
    p.add_argument("--stable-recent-limit", type=int, default=160)
    p.add_argument("--stable-transition-window", type=int, default=16)
    p.add_argument("--stable-transition-topk", type=int, default=384)
    p.add_argument("--stable-max-valid-events", type=int, default=30000)
    p.add_argument("--stable-search-rounds", type=int, default=5)
    p.add_argument("--stable-predict-workers", type=int, default=4)
    p.add_argument("--stable-predict-batch-size", type=int, default=16384)
    p.add_argument("--train-stable-mlp", default="1")
    p.add_argument("--stable-mlp-train-rows", type=int, default=80000)
    p.add_argument("--stable-mlp-hidden", type=int, default=192)
    p.add_argument("--stable-mlp-epochs", type=int, default=8)
    p.add_argument("--stable-mlp-batch-size", type=int, default=256)
    p.add_argument("--stable-mlp-lr", type=float, default=8e-4)
    p.add_argument("--stable-mlp-output-weight", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=3026)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--history-frac", type=float, default=0.70)
    p.add_argument("--train-rows", type=int, default=500000)
    p.add_argument("--valid-rows", type=int, default=80000)
    p.add_argument("--max-pool", type=int, default=700)
    p.add_argument("--svd-dim", type=int, default=128)
    p.add_argument("--fit-edge-limit", type=int, default=0)
    p.add_argument("--src-seq-len", type=int, default=64)
    p.add_argument("--dst-seq-len", type=int, default=64)
    p.add_argument("--seeds", default="3101,3102,3103")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--predict-batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--reuse-baseline-features", default="0")
    p.add_argument("--blend-weight", type=float, default=0.10)
    p.add_argument("--output-name", default="result_final_blend_0p10")
    p.add_argument("--sweep-blends", default="0.02,0.05,0.10,0.20,0.35,1.00")

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("baseline")
    sub.add_parser("build")
    sub.add_parser("train")
    sub.add_parser("predict")
    sub.add_parser("package")
    sub.add_parser("package-sweep")
    sub.add_parser("all")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "baseline":
        build_stable_baseline(args)
    elif args.command == "build":
        build_context_features(args)
    elif args.command == "train":
        train_context_ranker(args)
    elif args.command == "predict":
        predict_context_ranker(args)
    elif args.command == "package":
        _pack_one(args, float(args.blend_weight), args.output_name)
    elif args.command == "package-sweep":
        package_sweep(args, _parse_blends(args.sweep_blends))
    elif args.command == "all":
        run_all(args)
    else:
        raise SystemExit(args.command)


if __name__ == "__main__":
    main()
