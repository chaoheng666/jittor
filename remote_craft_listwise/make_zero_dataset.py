import argparse
import json
from pathlib import Path

import pandas as pd


def count_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/ma-user/work/jittor_rebuild_v5/data_A")
    parser.add_argument("--dataset", default="dataset1")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--report_dir", default="reports")
    args = parser.parse_args()

    test_path = Path(args.data_dir) / args.dataset / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    header = pd.read_csv(test_path, nrows=0)
    cand_cols = [c for c in header.columns if c.startswith("c")]
    rows = count_rows(test_path)
    cols = len(cand_cols)
    if cols <= 0:
        raise RuntimeError(f"no candidate columns found in {test_path}")

    out_path = Path(args.output_dir) / args.dataset / f"{args.dataset}_result.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    zero_line = ",".join(["0.00000000"] * cols) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(rows):
            f.write(zero_line)

    report = {
        "dataset": args.dataset,
        "test_file": str(test_path),
        "output_file": str(out_path),
        "rows": rows,
        "cols": cols,
        "value": 0.0,
    }
    report_path = Path(args.report_dir) / f"{args.dataset}_zero_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
