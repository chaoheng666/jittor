import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import find_dataset_dirs, load_test_rows, load_train_edges
from src.fusion import load_fusion_config, predict_proba
from src.metrics import probability_report


def write_dataset(dataset_dir, dataset_config, output_path, batch_size, max_rows=0):
    dataset_name = dataset_dir.name
    train_edges = load_train_edges(dataset_dir)
    queries = load_test_rows(dataset_dir)
    if max_rows and max_rows > 0:
        queries = queries[:max_rows]
    probs, _, _ = predict_proba(dataset_config, dataset_name, train_edges, queries, batch_size=batch_size)
    report = probability_report(probs, expected_cols=100)
    if not report["valid"]:
        raise RuntimeError(f"{dataset_name}: invalid probability export {report}")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in probs:
            writer.writerow([f"{p:.8f}" for p in row])
    return report


def make_zip(output_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_path in sorted(output_dir.glob("*.csv")):
            zf.write(csv_path, arcname=csv_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--config", default="models_v2/fusion_config.json")
    parser.add_argument("--out-dir", default="submission_best")
    parser.add_argument("--zip", default="result_best.zip")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--report", default="reports/export_check.json")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    config = load_fusion_config(args.config)
    if config.get("mode") not in {"fusion_v2", "edge_intensity"}:
        raise ValueError(f"unexpected config mode: {config.get('mode')}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for dataset_dir in find_dataset_dirs(args.data_dir):
        dataset_name = dataset_dir.name
        dataset_config = config["datasets"].get(dataset_name)
        if not dataset_config:
            print(f"skip {dataset_name}: no fusion config")
            continue
        report = write_dataset(dataset_dir, dataset_config, out_dir / f"{dataset_name}.csv", args.batch_size, args.max_rows)
        reports[dataset_name] = report
        print(f"{dataset_name}: wrote {report['rows']} rows")
    make_zip(out_dir, Path(args.zip))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(f"packed {args.zip}")


if __name__ == "__main__":
    main()
