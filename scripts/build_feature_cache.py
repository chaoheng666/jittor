import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from numpy.lib.format import open_memmap

from src.data_loader import iter_train_edges
from src.jt_ranker import CandidateFeatureBuilder, FEATURE_NAMES
from src.seq_ranker import SequenceFeatureBuilder


_FEATURE_BUILDER = None
_SEQ_BUILDERS = {}
_SEQ_LENS = []


def iter_valid_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates = [int(row[f"c{i}"]) for i in range(1, 101)]
            yield int(row["src"]), int(row["time"]), int(row["label"]), candidates


def iter_test_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            yield int(row[0]), int(row[1]), [int(x) for x in row[2:]]


def count_rows(path, has_label):
    count = 0
    extra_dsts = set()
    iterator = iter_valid_rows(path) if has_label else iter_test_rows(path)
    for row in iterator:
        candidates = row[3] if has_label else row[2]
        extra_dsts.update(candidates)
        count += 1
    return count, extra_dsts


def chunked(iterator, chunk_size):
    rows = []
    start = 0
    for row in iterator:
        rows.append(row)
        if len(rows) >= chunk_size:
            yield start, rows
            start += len(rows)
            rows = []
    if rows:
        yield start, rows


def build_chunk(task):
    start, rows, has_label = task
    feature_rows = []
    labels = []
    seq_out = {
        seq_len: {
            "dst": [],
            "gap": [],
            "cand": [],
        }
        for seq_len in _SEQ_LENS
    }

    for row in rows:
        if has_label:
            src, time, label, candidates = row
            labels.append(label)
        else:
            src, time, candidates = row
        feature_rows.append(_FEATURE_BUILDER.matrix(src, time, candidates))

        for seq_len in _SEQ_LENS:
            seq_dst, seq_gap, cand_idx = _SEQ_BUILDERS[seq_len].build_query(src, time, candidates)
            seq_out[seq_len]["dst"].append(seq_dst)
            seq_out[seq_len]["gap"].append(seq_gap)
            seq_out[seq_len]["cand"].append(cand_idx)

    seq_arrays = {}
    for seq_len, values in seq_out.items():
        seq_arrays[seq_len] = {
            "dst": np.asarray(values["dst"], dtype=np.int32),
            "gap": np.asarray(values["gap"], dtype=np.int32),
            "cand": np.asarray(values["cand"], dtype=np.int32),
        }

    return (
        start,
        np.asarray(feature_rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int32) if has_label else None,
        seq_arrays,
    )


def write_metadata(dataset_cache, dataset_name, seq_lens, dst_values_by_len):
    with open(dataset_cache / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    with open(dataset_cache / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"dataset_name": dataset_name, "seq_lens": seq_lens}, f, indent=2)
    for seq_len, dst_values in dst_values_by_len.items():
        with open(dataset_cache / f"seq_l{seq_len}_dst_values.json", "w", encoding="utf-8") as f:
            json.dump(dst_values, f)


def build_kind(dataset_cache, kind, iterator, row_count, has_label, seq_lens, workers, chunk_size):
    feature_dim = len(FEATURE_NAMES)
    x_path = dataset_cache / f"x_{kind}.npy"
    x_mm = open_memmap(x_path, mode="w+", dtype=np.float32, shape=(row_count, 100, feature_dim))

    y_mm = None
    if has_label:
        y_mm = open_memmap(dataset_cache / f"y_{kind}.npy", mode="w+", dtype=np.int32, shape=(row_count,))

    seq_mm = {}
    for seq_len in seq_lens:
        seq_mm[seq_len] = {
            "dst": open_memmap(
                dataset_cache / f"seq_l{seq_len}_{kind}_dst.npy",
                mode="w+",
                dtype=np.int32,
                shape=(row_count, seq_len),
            ),
            "gap": open_memmap(
                dataset_cache / f"seq_l{seq_len}_{kind}_gap.npy",
                mode="w+",
                dtype=np.int32,
                shape=(row_count, seq_len),
            ),
            "cand": open_memmap(
                dataset_cache / f"seq_l{seq_len}_{kind}_cand.npy",
                mode="w+",
                dtype=np.int32,
                shape=(row_count, 100),
            ),
        }

    tasks = ((start, rows, has_label) for start, rows in chunked(iterator, chunk_size))
    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            results = pool.imap_unordered(build_chunk, tasks, chunksize=1)
            for start, x_chunk, y_chunk, seq_chunk in results:
                end = start + len(x_chunk)
                x_mm[start:end] = x_chunk
                if has_label:
                    y_mm[start:end] = y_chunk
                for seq_len in seq_lens:
                    seq_mm[seq_len]["dst"][start:end] = seq_chunk[seq_len]["dst"]
                    seq_mm[seq_len]["gap"][start:end] = seq_chunk[seq_len]["gap"]
                    seq_mm[seq_len]["cand"][start:end] = seq_chunk[seq_len]["cand"]
                print(f"{kind}: cached rows {end}/{row_count}", flush=True)
    else:
        for task in tasks:
            start, x_chunk, y_chunk, seq_chunk = build_chunk(task)
            end = start + len(x_chunk)
            x_mm[start:end] = x_chunk
            if has_label:
                y_mm[start:end] = y_chunk
            for seq_len in seq_lens:
                seq_mm[seq_len]["dst"][start:end] = seq_chunk[seq_len]["dst"]
                seq_mm[seq_len]["gap"][start:end] = seq_chunk[seq_len]["gap"]
                seq_mm[seq_len]["cand"][start:end] = seq_chunk[seq_len]["cand"]
            print(f"{kind}: cached rows {end}/{row_count}", flush=True)

    x_mm.flush()
    if y_mm is not None:
        y_mm.flush()
    for values in seq_mm.values():
        for mmap in values.values():
            mmap.flush()


def build_dataset_cache(args, dataset_dir):
    global _FEATURE_BUILDER, _SEQ_BUILDERS, _SEQ_LENS

    dataset_name = dataset_dir.name
    valid_dir = Path(args.valid_dir) / dataset_name
    dataset_cache = Path(args.cache_dir) / dataset_name
    dataset_cache.mkdir(parents=True, exist_ok=True)

    valid_rows, valid_extra = count_rows(valid_dir / "valid.csv", has_label=True)
    test_rows, test_extra = count_rows(dataset_dir / "test.csv", has_label=False)
    extra_dsts = valid_extra | test_extra
    valid_train_edges = list(iter_train_edges(valid_dir / "train.csv"))
    full_train_edges = list(iter_train_edges(dataset_dir / "train.csv"))
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]

    print(
        f"{dataset_name}: valid_train={len(valid_train_edges)} full_train={len(full_train_edges)} valid={valid_rows} "
        f"test={test_rows} extra_dsts={len(extra_dsts)} seq_lens={seq_lens}",
        flush=True,
    )

    _FEATURE_BUILDER = CandidateFeatureBuilder(dataset_name)
    _FEATURE_BUILDER.fit(valid_train_edges)
    _FEATURE_BUILDER.fit_candidate_priors(iter_test_rows(dataset_dir / "test.csv"))

    _SEQ_LENS = seq_lens
    _SEQ_BUILDERS = {}
    dst_values_by_len = {}
    for seq_len in seq_lens:
        builder = SequenceFeatureBuilder(seq_len)
        builder.fit(valid_train_edges, extra_dsts)
        _SEQ_BUILDERS[seq_len] = builder
        dst_values_by_len[seq_len] = builder.dst_values

    write_metadata(dataset_cache, dataset_name, seq_lens, dst_values_by_len)
    workers = args.workers
    if os.name != "posix":
        workers = 1
    workers = max(1, workers)

    build_kind(
        dataset_cache,
        "valid",
        iter_valid_rows(valid_dir / "valid.csv"),
        valid_rows,
        True,
        seq_lens,
        workers,
        args.chunk_size,
    )

    _FEATURE_BUILDER = CandidateFeatureBuilder(dataset_name)
    _FEATURE_BUILDER.fit(full_train_edges)
    _FEATURE_BUILDER.fit_candidate_priors(iter_test_rows(dataset_dir / "test.csv"))
    _SEQ_BUILDERS = {}
    for seq_len in seq_lens:
        builder = SequenceFeatureBuilder(seq_len)
        builder.load_dst_values(dst_values_by_len[seq_len])
        builder.fit_history(full_train_edges)
        _SEQ_BUILDERS[seq_len] = builder

    build_kind(
        dataset_cache,
        "test",
        iter_test_rows(dataset_dir / "test.csv"),
        test_rows,
        False,
        seq_lens,
        workers,
        args.chunk_size,
    )


def find_dataset_dirs(data_dir, valid_dir):
    data_dir = Path(data_dir)
    valid_dir = Path(valid_dir)
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_dir()
        and (p / "test.csv").exists()
        and (valid_dir / p.name / "train.csv").exists()
        and (valid_dir / p.name / "valid.csv").exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--valid-dir", default="validation")
    parser.add_argument("--fold-root", default="", help="Build one cache under cache-dir for each fold directory.")
    parser.add_argument("--cache-dir", default="feature_cache")
    parser.add_argument("--dataset", default="all", help="all or comma-separated dataset names")
    parser.add_argument("--seq-lens", default="50")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    if args.fold_root:
        fold_root = Path(args.fold_root)
        folds = sorted(p for p in fold_root.iterdir() if p.is_dir())
        if not folds:
            raise ValueError(f"no fold dirs found in {fold_root}")
        for fold_dir in folds:
            fold_args = argparse.Namespace(**vars(args))
            fold_args.valid_dir = str(fold_dir)
            fold_args.cache_dir = str(Path(args.cache_dir) / fold_dir.name)
            if fold_args.dataset == "all":
                dataset_dirs = find_dataset_dirs(args.data_dir, fold_args.valid_dir)
            else:
                names = [name.strip() for name in fold_args.dataset.split(",") if name.strip()]
                dataset_dirs = [Path(args.data_dir) / name for name in names]
            if not dataset_dirs:
                raise ValueError(f"no dataset dirs found for fold {fold_dir}")
            for dataset_dir in dataset_dirs:
                build_dataset_cache(fold_args, dataset_dir)
        return

    if args.dataset == "all":
        dataset_dirs = find_dataset_dirs(args.data_dir, args.valid_dir)
    else:
        names = [name.strip() for name in args.dataset.split(",") if name.strip()]
        dataset_dirs = [Path(args.data_dir) / name for name in names]
    if not dataset_dirs:
        raise ValueError(f"no dataset dirs found for {args.data_dir} and {args.valid_dir}")

    for dataset_dir in dataset_dirs:
        build_dataset_cache(args, dataset_dir)


if __name__ == "__main__":
    main()
