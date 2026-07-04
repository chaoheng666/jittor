import csv
import json
import zipfile
from pathlib import Path

import numpy as np

from src.data_loader import find_dataset_dirs, iter_test_rows
from src.metrics import probability_report


def write_probability_chunks(path, chunks, validate=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    cols = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for chunk in chunks:
            chunk = np.asarray(chunk, dtype=np.float64)
            if chunk.ndim != 2:
                raise ValueError(f"expected 2D probability chunk, got shape={chunk.shape}")
            if validate:
                report = probability_report(chunk, expected_cols=100)
                if not report["valid"]:
                    raise RuntimeError(f"invalid probability export for {path}: {report}")
            cols = int(chunk.shape[1])
            rows += int(chunk.shape[0])
            for row in chunk:
                writer.writerow([f"{float(value):.8f}" for value in row])
    return {"rows": rows, "cols": cols, "valid": bool(validate)}


def write_zero_csv_for_dataset(dataset_dir, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    zero_row = ["0.00000000"] * 100
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for _ in iter_test_rows(Path(dataset_dir) / "test.csv"):
            writer.writerow(zero_row)
            rows += 1
    return {"rows": rows, "cols": 100, "valid": False}


def make_zip(output_dir, zip_path):
    output_dir = Path(output_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_path in sorted(output_dir.glob("*.csv")):
            zf.write(csv_path, arcname=csv_path.name)


def write_report(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def dataset_map(data_dir):
    return {path.name: path for path in find_dataset_dirs(data_dir)}
