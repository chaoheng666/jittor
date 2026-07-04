import json
from pathlib import Path

import numpy as np

from src.base_intensity_v3 import BaseIntensityV3
from src.data_loader import iter_test_rows, load_test_rows, load_train_edges
from src.metrics import row_zscore, softmax


DEFAULT_DATASET1_CONFIG = {
    "dataset": "dataset1",
    "type": "dataset1_rule_base",
    "base_weight": 0.55,
    "rule_weight": 0.15,
    "cold_penalty": 0.05,
    "temperature": 1.0,
}


def cold_mask(train_edges, queries):
    seen_dst = {dst for _, dst, _ in train_edges}
    return np.asarray(
        [[1.0 if dst not in seen_dst else 0.0 for dst in candidates] for _, _, candidates in queries],
        dtype=np.float32,
    )


def train_dataset1(dataset_dir, artifact_dir, config=None):
    """Persist the conservative dataset1 configuration.

    Dataset1 is repeat-heavy and already scores well with the rule/base stack.
    The training step is intentionally a lightweight artifact creation step so
    single-dataset probes do not spend time on weak deep components.
    """

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(DEFAULT_DATASET1_CONFIG)
    if config:
        payload.update(config)
    payload["train_edges"] = len(load_train_edges(dataset_dir))
    out_path = artifact_dir / "model.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def load_dataset1_config(artifact_dir):
    path = Path(artifact_dir) / "model.json"
    if not path.exists():
        return dict(DEFAULT_DATASET1_CONFIG)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    config = dict(DEFAULT_DATASET1_CONFIG)
    config.update(payload)
    return config


def iter_dataset1_proba_chunks(dataset_dir, artifact_dir, batch_size=512, max_rows=0):
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    train_edges = load_train_edges(dataset_dir)
    config = load_dataset1_config(artifact_dir)
    model = BaseIntensityV3("dataset1")
    model.fit(train_edges)
    batch = []
    emitted = 0
    limit = max(int(max_rows or 0), 0)
    for row in iter_test_rows(Path(dataset_dir) / "test.csv"):
        if limit and emitted + len(batch) >= limit:
            break
        batch.append(row)
        if len(batch) >= batch_size:
            yield _score_batch(model, train_edges, batch, config)
            emitted += len(batch)
            batch = []
    if batch:
        yield _score_batch(model, train_edges, batch, config)


def _score_batch(model, train_edges, queries, config):
    base_rows = []
    rule_rows = []
    for src, time, candidates in queries:
        base_scores, rule_scores = model.score_many_with_rule(src, time, candidates)
        base_rows.append(base_scores)
        rule_rows.append(rule_scores)
    base = np.asarray(base_rows, dtype=np.float32)
    rule = np.asarray(rule_rows, dtype=np.float32)
    total = (
        row_zscore(base) * float(config.get("base_weight", 0.55))
        + row_zscore(rule) * float(config.get("rule_weight", 0.15))
    )
    penalty = float(config.get("cold_penalty", 0.0))
    if penalty:
        total = total - penalty * cold_mask(train_edges, queries)
    return softmax(total, temperature=float(config.get("temperature", 1.0)))
