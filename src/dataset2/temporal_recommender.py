import json
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

from src.data_loader import iter_test_rows, iter_train_edges
from src.base_intensity_v3 import BaseIntensityV3
from src.metrics import row_zscore, softmax


try:
    import jittor as jt
    from jittor import nn
except Exception:  # pragma: no cover - exercised only when Jittor is missing.
    jt = None
    nn = None


def jittor_available():
    return jt is not None and nn is not None


def read_edges_with_split(dataset_dir):
    rows = []
    with open(Path(dataset_dir) / "train.csv", newline="", encoding="utf-8") as f:
        import csv

        reader = csv.DictReader(f)
        has_split = "split" in (reader.fieldnames or [])
        for row in reader:
            split = row.get("split", "")
            if not has_split:
                split = ""
            rows.append((int(row["src"]), int(row["dst"]), int(row["time"]), split))
    rows.sort(key=lambda x: x[2])
    return rows


def split_dataset2_edges(dataset_dir, final_train=False):
    rows = read_edges_with_split(dataset_dir)
    if not rows:
        raise ValueError(f"{dataset_dir}: no training rows")
    if final_train:
        train = [(s, d, t) for s, d, t, _ in rows]
        return train, []
    has_split = any(split != "" for *_, split in rows)
    if has_split:
        train = [(s, d, t) for s, d, t, split in rows if str(split) == "0"]
        valid = [(s, d, t) for s, d, t, split in rows if str(split) != "0"]
        if train and valid:
            return train, valid
    cut = int(len(rows) * 0.8)
    cut = min(max(cut, 1), len(rows) - 1)
    train = [(s, d, t) for s, d, t, _ in rows[:cut]]
    valid = [(s, d, t) for s, d, t, _ in rows[cut:]]
    return train, valid


def build_id_maps(edges):
    src_values = sorted({src for src, _, _ in edges})
    dst_values = sorted({dst for _, dst, _ in edges})
    src_to_id = {value: idx + 1 for idx, value in enumerate(src_values)}
    dst_to_id = {value: idx + 1 for idx, value in enumerate(dst_values)}
    return src_to_id, dst_to_id


def _time_feature(time_value, last_time, hist_len, time_min, time_scale, gap_scale, hist_scale):
    global_pos = (float(time_value) - float(time_min)) / max(float(time_scale), 1.0)
    if last_time is None:
        gap = 0.0
    else:
        gap = math.log1p(max(float(time_value) - float(last_time), 0.0)) / max(float(gap_scale), 1.0)
    hist_value = math.log1p(float(hist_len)) / max(float(hist_scale), 1.0)
    return [global_pos, gap, hist_value]


def build_samples(history_edges, supervision_edges, src_to_id, dst_to_id, seq_len, update_with_supervision=True):
    history_by_src = defaultdict(lambda: deque(maxlen=seq_len))
    last_time_by_src = {}
    count_by_src = Counter()
    # Validation features are normalized from the observable history only.  When
    # history is empty during self-supervised training, fall back to the training
    # supervision window.
    time_source = history_edges if history_edges else supervision_edges
    all_times = [time for _, _, time in time_source]
    time_min = min(all_times) if all_times else 0
    time_max = max(all_times) if all_times else 1
    gaps = []
    for src, dst, time in sorted(history_edges, key=lambda x: x[2]):
        dst_id = dst_to_id.get(dst, 0)
        if dst_id:
            history_by_src[src].append(dst_id)
        if src in last_time_by_src:
            gaps.append(max(time - last_time_by_src[src], 0))
        last_time_by_src[src] = time
        count_by_src[src] += 1
    gap_scale = math.log1p(float(np.percentile(gaps, 90))) if gaps else 1.0
    hist_scale = math.log1p(max(count_by_src.values(), default=1))

    src_rows = []
    hist_rows = []
    time_rows = []
    labels = []
    pair_seen = set((src, dst) for src, dst, _ in history_edges)
    is_new_pair = []
    skipped = 0
    skipped_cold_dst = 0
    skipped_other = 0

    for src, dst, time in sorted(supervision_edges, key=lambda x: x[2]):
        src_id = src_to_id.get(src, 0)
        dst_id = dst_to_id.get(dst, 0)
        hist = list(history_by_src.get(src, ()))
        if src_id == 0 or dst_id == 0 or not hist:
            skipped += 1
            if dst_id == 0:
                skipped_cold_dst += 1
            else:
                skipped_other += 1
            if update_with_supervision and dst_id:
                history_by_src[src].append(dst_id)
                last_time_by_src[src] = time
                count_by_src[src] += 1
                pair_seen.add((src, dst))
            continue
        hist_vec = np.zeros(seq_len, dtype=np.int32)
        hist_vec[-len(hist[-seq_len:]):] = hist[-seq_len:]
        src_rows.append(src_id)
        hist_rows.append(hist_vec)
        time_rows.append(_time_feature(
            time,
            last_time_by_src.get(src),
            len(hist),
            time_min,
            max(time_max - time_min, 1),
            gap_scale,
            hist_scale,
        ))
        labels.append(dst_id - 1)
        is_new_pair.append(0 if (src, dst) in pair_seen else 1)
        if update_with_supervision:
            history_by_src[src].append(dst_id)
            last_time_by_src[src] = time
            count_by_src[src] += 1
            pair_seen.add((src, dst))

    return {
        "src": np.asarray(src_rows, dtype=np.int32),
        "hist": np.asarray(hist_rows, dtype=np.int32),
        "time": np.asarray(time_rows, dtype=np.float32),
        "label": np.asarray(labels, dtype=np.int32),
        "is_new_pair": np.asarray(is_new_pair, dtype=np.int8),
        "skipped": int(skipped),
        "skipped_cold_dst": int(skipped_cold_dst),
        "skipped_other": int(skipped_other),
    }


def source_histories_for_prediction(edges, dst_to_id, seq_len):
    histories = defaultdict(lambda: deque(maxlen=seq_len))
    last_time = {}
    count_by_src = Counter()
    times = []
    gaps = []
    for src, dst, time in sorted(edges, key=lambda x: x[2]):
        times.append(time)
        dst_id = dst_to_id.get(dst, 0)
        if dst_id:
            histories[src].append(dst_id)
        if src in last_time:
            gaps.append(max(time - last_time[src], 0))
        last_time[src] = time
        count_by_src[src] += 1
    return {
        "histories": histories,
        "last_time": last_time,
        "count_by_src": count_by_src,
        "time_min": min(times) if times else 0,
        "time_scale": max((max(times) - min(times)) if times else 1, 1),
        "gap_scale": math.log1p(float(np.percentile(gaps, 90))) if gaps else 1.0,
        "hist_scale": math.log1p(max(count_by_src.values(), default=1)),
    }


if nn is not None:
    class TemporalRecommender(nn.Module):
        def __init__(self, num_src, num_dst, emb_dim=96, hidden_dim=192, dropout=0.1):
            super().__init__()
            self.src_emb = nn.Embedding(num_src + 1, emb_dim)
            self.dst_emb = nn.Embedding(num_dst + 1, emb_dim)
            self.dst_bias = nn.Embedding(num_dst + 1, 1)
            self.time_proj = nn.Sequential(
                nn.Linear(3, emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, emb_dim),
            )
            self.state_proj = nn.Sequential(
                nn.Linear(emb_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, emb_dim),
            )

        def encode_state(self, src_ids, hist_ids, time_feats):
            src_vec = self.src_emb(src_ids)
            hist_mask = (hist_ids > 0).float32()
            hist_emb = self.dst_emb(hist_ids) * hist_mask.unsqueeze(-1)
            denom = jt.maximum(hist_mask.sum(dim=1, keepdims=True), jt.ones((hist_ids.shape[0], 1)))
            hist_vec = hist_emb.sum(dim=1) / denom
            time_vec = self.time_proj(time_feats)
            return self.state_proj(jt.concat([src_vec, hist_vec, time_vec], dim=1))

        def execute(self, src_ids, hist_ids, time_feats, cand_ids=None):
            state = self.encode_state(src_ids, hist_ids, time_feats)
            if cand_ids is None:
                dst_weight = self.dst_emb.weight[1:]
                bias = self.dst_bias.weight[1:].reshape((1, -1))
                return jt.matmul(state, dst_weight.transpose(1, 0)) + bias
            if len(cand_ids.shape) == 1:
                cand_emb = self.dst_emb(cand_ids)
                bias = self.dst_bias(cand_ids).reshape((1, -1))
                return jt.matmul(state, cand_emb.transpose(1, 0)) + bias
            cand_emb = self.dst_emb(cand_ids)
            bias = self.dst_bias(cand_ids).squeeze(-1)
            cand_mask = (cand_ids > 0).float32()
            scores = (state.unsqueeze(1) * cand_emb).sum(dim=2) + bias
            return scores * cand_mask - (1.0 - cand_mask) * 1e6
else:
    class TemporalRecommender:
        def __init__(self, *args, **kwargs):
            raise ImportError("jittor is required for TemporalRecommender")


def _batch_indices(size, batch_size, rng, shuffle=True):
    idx = np.arange(size)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, size, batch_size):
        yield idx[start:start + batch_size]


def _shared_candidate_set(labels, num_dst, neg_count, rng):
    pos_ids = labels.astype(np.int32) + 1
    negs = rng.integers(1, num_dst + 1, size=int(neg_count), dtype=np.int32)
    candidates = np.unique(np.concatenate([pos_ids, negs])).astype(np.int32)
    label_positions = np.searchsorted(candidates, pos_ids).astype(np.int32)
    return candidates, label_positions


def _save_model(path, model, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    jt.save({"state_dict": model.state_dict(), "meta": meta}, str(path))


def load_model(path):
    if not jittor_available():
        raise ImportError("jittor is required to load dataset2 temporal model")
    payload = jt.load(str(path))
    meta = payload["meta"]
    model = TemporalRecommender(
        int(meta["num_src"]),
        int(meta["num_dst"]),
        emb_dim=int(meta.get("emb_dim", 96)),
        hidden_dim=int(meta.get("hidden_dim", 192)),
        dropout=0.0,
    )
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(payload["state_dict"])
    else:
        model.load_parameters(payload["state_dict"])
    model.eval()
    return model, meta


def _bucket_metrics(rr_values):
    if not rr_values:
        return {"mrr": 0.0, "events": 0}
    return {"mrr": float(np.mean(rr_values)), "events": int(len(rr_values))}


def evaluate_full_mrr(model, samples, batch_size, max_events=20000):
    if len(samples["label"]) == 0:
        cold_events = int(samples.get("skipped_cold_dst", 0))
        return {
            "overall": {"mrr": 0.0, "events": 0},
            "repeated": {"mrr": 0.0, "events": 0},
            "new_pair": {"mrr": 0.0, "events": 0},
            "cold_dst": {"mrr": 0.0, "events": cold_events},
            "skipped_other": int(samples.get("skipped_other", 0)),
        }
    count = min(len(samples["label"]), int(max_events) if max_events else len(samples["label"]))
    idx = np.arange(len(samples["label"]))[:count]
    rr = []
    repeated_rr = []
    new_rr = []
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        logits = model(
            jt.array(samples["src"][batch_idx]),
            jt.array(samples["hist"][batch_idx]),
            jt.array(samples["time"][batch_idx]),
        ).numpy()
        labels = samples["label"][batch_idx]
        pos = logits[np.arange(len(labels)), labels]
        ranks = 1 + (logits > pos[:, None]).sum(axis=1)
        batch_rr = 1.0 / ranks
        rr.extend(batch_rr.tolist())
        is_new = samples["is_new_pair"][batch_idx].astype(bool)
        if is_new.any():
            new_rr.extend(batch_rr[is_new].tolist())
        if (~is_new).any():
            repeated_rr.extend(batch_rr[~is_new].tolist())
    cold_events = int(samples.get("skipped_cold_dst", 0))
    return {
        "overall": _bucket_metrics(rr),
        "repeated": _bucket_metrics(repeated_rr),
        "new_pair": _bucket_metrics(new_rr),
        "cold_dst": {"mrr": 0.0, "events": cold_events},
        "skipped_other": int(samples.get("skipped_other", 0)),
    }


def train_dataset2(
    dataset_dir,
    artifact_dir,
    final_train=False,
    cuda=True,
    softmax_mode="sampled",
    neg_count=4096,
    seq_len=80,
    emb_dim=96,
    hidden_dim=192,
    dropout=0.1,
    epochs=6,
    batch_size=512,
    lr=0.001,
    weight_decay=1e-6,
    bpr_weight=0.05,
    fusion_model_weight=1.0,
    fusion_rule_weight=0.25,
    valid_max_events=20000,
    seed=2026,
):
    if not jittor_available():
        raise ImportError("Jittor is required for dataset2 temporal recommender training")
    neg_count = int(neg_count)
    seq_len = int(seq_len)
    epochs = int(epochs)
    batch_size = int(batch_size)
    if neg_count < 1:
        raise ValueError(f"neg_count must be >= 1, got {neg_count}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    bpr_weight = float(bpr_weight)
    if bpr_weight < 0.0:
        raise ValueError(f"bpr_weight must be >= 0, got {bpr_weight}")
    fusion_model_weight = float(fusion_model_weight)
    fusion_rule_weight = float(fusion_rule_weight)
    if cuda:
        jt.flags.use_cuda = 1
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_edges, valid_edges = split_dataset2_edges(dataset_dir, final_train=final_train)
    src_to_id, dst_to_id = build_id_maps(train_edges)
    train_samples = build_samples([], train_edges, src_to_id, dst_to_id, seq_len, update_with_supervision=True)
    valid_samples = None
    if valid_edges:
        valid_samples = build_samples(train_edges, valid_edges, src_to_id, dst_to_id, seq_len, update_with_supervision=False)
    if len(train_samples["label"]) < 100:
        raise RuntimeError(f"dataset2: too few training samples after history filtering: {len(train_samples['label'])}")

    num_src = len(src_to_id)
    num_dst = len(dst_to_id)
    model = TemporalRecommender(num_src, num_dst, emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = nn.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    best_mrr = -1.0
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pkl"
    selected_meta = None

    use_full = str(softmax_mode).lower() == "full"
    effective_neg_count = min(neg_count, max(num_dst - 1, 1))
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        steps = 0
        for idx in _batch_indices(len(train_samples["label"]), batch_size, rng, shuffle=True):
            src = jt.array(train_samples["src"][idx])
            hist = jt.array(train_samples["hist"][idx])
            time_feats = jt.array(train_samples["time"][idx])
            labels = train_samples["label"][idx]
            if use_full:
                logits = model(src, hist, time_feats)
                loss = nn.cross_entropy_loss(logits, jt.array(labels.astype(np.int32)))
                if bpr_weight:
                    neg_ids = rng.integers(1, num_dst + 1, size=(len(idx), 1), dtype=np.int32)
                    cand = np.concatenate([(labels.astype(np.int32) + 1)[:, None], neg_ids], axis=1)
                    pair_scores = model(src, hist, time_feats, jt.array(cand))
                    diff = pair_scores[:, 0] - pair_scores[:, 1]
                    loss = loss + float(bpr_weight) * jt.log(1.0 + jt.exp(-diff)).mean()
            else:
                cand, label_positions = _shared_candidate_set(labels, num_dst, effective_neg_count, rng)
                logits = model(src, hist, time_feats, jt.array(cand))
                loss = nn.cross_entropy_loss(logits, jt.array(label_positions))
                if bpr_weight and cand.shape[0] > 1:
                    positive_mask = np.zeros((len(idx), len(cand)), dtype=np.float32)
                    positive_mask[np.arange(len(idx)), label_positions] = 1.0
                    positive_mask_jt = jt.array(positive_mask)
                    neg_logits = logits - positive_mask_jt * 1e6
                    max_neg = neg_logits.max(dim=1)
                    if isinstance(max_neg, tuple):
                        max_neg = max_neg[0]
                    pos_score = (logits * positive_mask_jt).sum(dim=1)
                    diff = pos_score - max_neg
                    loss = loss + float(bpr_weight) * jt.log(1.0 + jt.exp(-diff)).mean()
            optimizer.step(loss)
            loss_sum += float(loss.numpy())
            steps += 1

        metrics = {
            "overall": {"mrr": 0.0, "events": 0},
            "repeated": {"mrr": 0.0, "events": 0},
            "new_pair": {"mrr": 0.0, "events": 0},
            "cold_dst": {"mrr": 0.0, "events": 0},
            "skipped_other": 0,
        }
        if valid_samples is not None:
            model.eval()
            metrics = evaluate_full_mrr(model, valid_samples, batch_size=max(32, min(256, batch_size)), max_events=valid_max_events)
            score_for_selection = metrics["overall"]["mrr"]
        else:
            score_for_selection = float(epoch)
        print(
            f"dataset2: epoch={epoch} loss={loss_sum / max(steps, 1):.6f} "
            f"overall_mrr={metrics['overall']['mrr']:.6f} "
            f"repeated_mrr={metrics['repeated']['mrr']:.6f} "
            f"new_pair_mrr={metrics['new_pair']['mrr']:.6f} "
            f"cold_dst_mrr={metrics['cold_dst']['mrr']:.6f} "
            f"events=(all:{metrics['overall']['events']} rep:{metrics['repeated']['events']} "
            f"new:{metrics['new_pair']['events']} cold:{metrics['cold_dst']['events']})"
        )
        if score_for_selection > best_mrr:
            best_mrr = score_for_selection
            selected_meta = {
                "dataset": "dataset2",
                "type": "dataset2_temporal_recommender",
                "num_src": num_src,
                "num_dst": num_dst,
                "src_to_id": {str(k): int(v) for k, v in src_to_id.items()},
                "dst_to_id": {str(k): int(v) for k, v in dst_to_id.items()},
                "seq_len": int(seq_len),
                "emb_dim": int(emb_dim),
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "softmax_mode": "full" if use_full else "sampled",
                "neg_count": int(effective_neg_count),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "best_valid_mrr": float(metrics["overall"]["mrr"]),
                "best_repeated_mrr": float(metrics["repeated"]["mrr"]),
                "best_new_pair_mrr": float(metrics["new_pair"]["mrr"]),
                "best_cold_dst_mrr": float(metrics["cold_dst"]["mrr"]),
                "validation_metrics": metrics,
                "train_samples": int(len(train_samples["label"])),
                "train_skipped": int(train_samples["skipped"]),
                "train_skipped_cold_dst": int(train_samples["skipped_cold_dst"]),
                "train_skipped_other": int(train_samples["skipped_other"]),
                "valid_samples": int(len(valid_samples["label"])) if valid_samples is not None else 0,
                "valid_skipped": int(valid_samples["skipped"]) if valid_samples is not None else 0,
                "valid_skipped_cold_dst": int(valid_samples["skipped_cold_dst"]) if valid_samples is not None else 0,
                "valid_skipped_other": int(valid_samples["skipped_other"]) if valid_samples is not None else 0,
                "final_train": bool(final_train),
                "unknown_score": -8.0,
                "fusion_model_weight": fusion_model_weight,
                "fusion_rule_weight": fusion_rule_weight,
            }
            _save_model(model_path, model, selected_meta)

    with open(artifact_dir / "model.json", "w", encoding="utf-8") as f:
        json.dump(selected_meta or {}, f, indent=2, ensure_ascii=False)
    return selected_meta or {}


def iter_dataset2_proba_chunks(dataset_dir, artifact_dir, batch_size=512, max_rows=0):
    if not jittor_available():
        raise ImportError("Jittor is required for dataset2 temporal recommender prediction")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    model, meta = load_model(Path(artifact_dir) / "model.pkl")
    seq_len = int(meta.get("seq_len", 80))
    src_to_id = {int(k): int(v) for k, v in meta.get("src_to_id", {}).items()}
    dst_to_id = {int(k): int(v) for k, v in meta.get("dst_to_id", {}).items()}
    train_edges = list(iter_train_edges(Path(dataset_dir) / "train.csv"))
    pred_state = source_histories_for_prediction(train_edges, dst_to_id, seq_len)
    rule_model = BaseIntensityV3("dataset2")
    rule_model.fit(train_edges)

    unknown_score = float(meta.get("unknown_score", -8.0))
    fusion_model_weight = float(meta.get("fusion_model_weight", 1.0))
    fusion_rule_weight = float(meta.get("fusion_rule_weight", 0.25))
    batch = []
    emitted = 0
    limit = max(int(max_rows or 0), 0)
    for row in iter_test_rows(Path(dataset_dir) / "test.csv"):
        if limit and emitted + len(batch) >= limit:
            break
        batch.append(row)
        if len(batch) >= batch_size:
            yield _score_query_batch(
                model,
                rule_model,
                pred_state,
                src_to_id,
                dst_to_id,
                seq_len,
                batch,
                unknown_score,
                fusion_model_weight,
                fusion_rule_weight,
            )
            emitted += len(batch)
            batch = []
    if batch:
        yield _score_query_batch(
            model,
            rule_model,
            pred_state,
            src_to_id,
            dst_to_id,
            seq_len,
            batch,
            unknown_score,
            fusion_model_weight,
            fusion_rule_weight,
        )


def _score_query_batch(
    model,
    rule_model,
    pred_state,
    src_to_id,
    dst_to_id,
    seq_len,
    chunk,
    unknown_score,
    fusion_model_weight,
    fusion_rule_weight,
):
    src_values = [src for src, _, _ in chunk]
    src_ids = np.asarray([src_to_id.get(src, 0) for src in src_values], dtype=np.int32)
    hist_arr = np.zeros((len(chunk), seq_len), dtype=np.int32)
    time_arr = np.zeros((len(chunk), 3), dtype=np.float32)
    cand_arr = np.zeros((len(chunk), 100), dtype=np.int32)
    known_mask = np.zeros((len(chunk), 100), dtype=bool)
    for row_idx, (src, time, candidates) in enumerate(chunk):
        hist = list(pred_state["histories"].get(src, ()))
        if hist:
            hist_arr[row_idx, -len(hist[-seq_len:]):] = hist[-seq_len:]
        time_arr[row_idx] = _time_feature(
            time,
            pred_state["last_time"].get(src),
            len(hist),
            pred_state["time_min"],
            pred_state["time_scale"],
            pred_state["gap_scale"],
            pred_state["hist_scale"],
        )
        for cand_idx, dst in enumerate(candidates):
            dst_id = dst_to_id.get(dst, 0)
            cand_arr[row_idx, cand_idx] = dst_id
            known_mask[row_idx, cand_idx] = dst_id > 0
    scores = model(jt.array(src_ids), jt.array(hist_arr), jt.array(time_arr), jt.array(cand_arr)).numpy()
    if not known_mask.all():
        scores = np.where(known_mask, scores, unknown_score).astype(np.float32)
    if fusion_rule_weight or fusion_model_weight != 1.0:
        fused = row_zscore(scores) * float(fusion_model_weight)
        if fusion_rule_weight:
            rule_scores = np.asarray(
                [rule_model.score_many(src, time, candidates) for src, time, candidates in chunk],
                dtype=np.float32,
            )
            fused = fused + row_zscore(rule_scores) * float(fusion_rule_weight)
        scores = fused
    return softmax(scores.astype(np.float32), temperature=1.0)
