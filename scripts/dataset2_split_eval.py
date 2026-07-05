import argparse
import csv
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.dataset2.temporal_recommender import (
    FastDataset2RuleScorer,
    _history_arrays_for_src,
    _time_feature,
    build_id_maps,
    jittor_available,
    load_model,
    source_histories_for_prediction,
    split_dataset2_edges,
)
from src.metrics import row_zscore


BUCKETS = ("overall", "repeated", "new_pair", "cold_dst", "cold_src", "no_history_src")
VALID_MODES = {"rule_only", "model_only", "fusion", "pop", "recent"}
VALID_EVAL_SETS = {"all-dst", "pseudo100", "hard-pseudo100"}


def empty_bucket():
    return {"mrr": 0.0, "events": 0}


def empty_mode_raw():
    return {bucket: {} for bucket in BUCKETS}


def add_rr(bucket, rr):
    bucket["mrr_sum"] = bucket.get("mrr_sum", 0.0) + float(rr)
    bucket["events"] = bucket.get("events", 0) + 1


def add_event(raw, mode, bucket_names, rr):
    add_rr(raw[mode]["overall"], rr)
    for bucket in bucket_names:
        if bucket != "overall":
            add_rr(raw[mode][bucket], rr)


def finalize_bucket(bucket):
    events = int(bucket.get("events", 0))
    if events <= 0:
        return empty_bucket()
    return {"mrr": float(bucket.get("mrr_sum", 0.0) / events), "events": events}


def finalize_mode(raw_mode):
    return {bucket: finalize_bucket(raw_mode[bucket]) for bucket in BUCKETS}


def reciprocal_rank_from_scores(scores, positive_index):
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[int(positive_index)]
    return 1.0 / (1 + int(np.sum(scores > pos)))


def parse_csv_set(value, valid, arg_name):
    parsed = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = set(parsed) - set(valid)
    if unknown:
        raise ValueError(f"{arg_name}: unknown values {sorted(unknown)}")
    return parsed


def build_history_views(history_edges, seq_len):
    src_recent = defaultdict(lambda: deque(maxlen=seq_len))
    dst_counts = Counter()
    dst_last_time = {}
    src_seen = set()
    seen_pairs = set()
    for src, dst, time in sorted(history_edges, key=lambda x: x[2]):
        src_recent[src].append(dst)
        dst_counts[dst] += 1
        dst_last_time[dst] = time
        src_seen.add(src)
        seen_pairs.add((src, dst))
    popular_dst = [dst for dst, _ in dst_counts.most_common()]
    return {
        "src_recent": src_recent,
        "dst_counts": dst_counts,
        "dst_last_time": dst_last_time,
        "src_seen": src_seen,
        "seen_pairs": seen_pairs,
        "popular_dst": popular_dst,
    }


def event_buckets(src, dst, dst_to_id, history_views):
    buckets = []
    if (src, dst) in history_views["seen_pairs"]:
        buckets.append("repeated")
    else:
        buckets.append("new_pair")
    if src not in history_views["src_seen"]:
        buckets.append("cold_src")
    elif not history_views["src_recent"].get(src):
        buckets.append("no_history_src")
    if dst not in dst_to_id:
        buckets.append("cold_dst")
    return buckets


def build_all_dst_candidate_ids(dst_to_id, positive_dst):
    positive_id = dst_to_id.get(positive_dst)
    if positive_id is None:
        return None
    num_dst = len(dst_to_id)
    return np.arange(1, num_dst + 1, dtype=np.int32), int(positive_id - 1)


def build_random_candidate_ids(dst_to_id, positive_dst, pool_size, rng):
    positive_id = dst_to_id.get(positive_dst)
    if positive_id is None:
        return None
    num_dst = len(dst_to_id)
    count = min(max(int(pool_size), 2), num_dst)
    if count >= num_dst:
        return build_all_dst_candidate_ids(dst_to_id, positive_dst)
    neg_need = count - 1
    negs = set()
    while len(negs) < neg_need:
        value = int(rng.integers(1, num_dst + 1))
        if value != positive_id:
            negs.add(value)
    candidates = np.asarray([positive_id] + sorted(negs), dtype=np.int32)
    rng.shuffle(candidates)
    positive_index = int(np.where(candidates == positive_id)[0][0])
    return candidates, positive_index


def build_hard_candidate_ids(dst_to_id, positive_dst, src, history_views, pool_size, rng):
    positive_id = dst_to_id.get(positive_dst)
    if positive_id is None:
        return None
    num_dst = len(dst_to_id)
    count = min(max(int(pool_size), 2), num_dst)
    if count >= num_dst:
        return build_all_dst_candidate_ids(dst_to_id, positive_dst)

    selected = [positive_id]
    seen_ids = {positive_id}
    for dst in reversed(history_views["src_recent"].get(src, ())):
        dst_id = dst_to_id.get(dst)
        if dst_id and dst_id not in seen_ids:
            selected.append(dst_id)
            seen_ids.add(dst_id)
        if len(selected) >= count:
            break
    for dst in history_views["popular_dst"]:
        dst_id = dst_to_id.get(dst)
        if dst_id and dst_id not in seen_ids:
            selected.append(dst_id)
            seen_ids.add(dst_id)
        if len(selected) >= count:
            break
    while len(selected) < count:
        value = int(rng.integers(1, num_dst + 1))
        if value not in seen_ids:
            selected.append(value)
            seen_ids.add(value)
    candidates = np.asarray(selected, dtype=np.int32)
    rng.shuffle(candidates)
    positive_index = int(np.where(candidates == positive_id)[0][0])
    return candidates, positive_index


def build_candidate_ids(eval_set, dst_to_id, positive_dst, src, history_views, pseudo_size, rng):
    if eval_set == "all-dst":
        return build_all_dst_candidate_ids(dst_to_id, positive_dst)
    if eval_set == "pseudo100":
        return build_random_candidate_ids(dst_to_id, positive_dst, pseudo_size, rng)
    if eval_set == "hard-pseudo100":
        return build_hard_candidate_ids(dst_to_id, positive_dst, src, history_views, pseudo_size, rng)
    raise ValueError(f"unsupported eval_set: {eval_set}")


def selected_events(valid_edges, max_events):
    if not max_events or max_events >= len(valid_edges):
        return list(valid_edges)
    indices = np.linspace(0, len(valid_edges) - 1, int(max_events), dtype=np.int64)
    return [valid_edges[int(idx)] for idx in indices]


class ModelScorer:
    def __init__(self, artifact_dir, history_edges, batch_size=256, cuda=True):
        if not jittor_available():
            raise ImportError("Jittor is required for model_only/fusion split evaluation")
        import jittor as jt

        if cuda:
            jt.flags.use_cuda = 1
        self.jt = jt
        self.model, self.meta = load_model(Path(artifact_dir) / "model.pkl")
        self.seq_len = int(self.meta.get("seq_len", 80))
        self.src_to_id = {int(k): int(v) for k, v in self.meta.get("src_to_id", {}).items()}
        self.dst_to_id = {int(k): int(v) for k, v in self.meta.get("dst_to_id", {}).items()}
        self.state = source_histories_for_prediction(history_edges, self.dst_to_id, self.seq_len)
        self.batch_size = int(batch_size)

    def score(self, rows, candidate_id_rows):
        out = []
        for start in range(0, len(rows), self.batch_size):
            chunk = rows[start:start + self.batch_size]
            cand_chunk = candidate_id_rows[start:start + self.batch_size]
            max_candidates = max(len(cands) for cands in cand_chunk)
            src_ids = np.asarray([self.src_to_id.get(src, 0) for src, _, _ in chunk], dtype=np.int32)
            hist_arr = np.zeros((len(chunk), self.seq_len), dtype=np.int32)
            gap_arr = np.zeros((len(chunk), self.seq_len), dtype=np.float32)
            time_arr = np.zeros((len(chunk), 3), dtype=np.float32)
            cand_arr = np.zeros((len(chunk), max_candidates), dtype=np.int32)
            valid_model_rows = np.zeros(len(chunk), dtype=bool)
            for row_idx, ((src, time, _), cand_ids) in enumerate(zip(chunk, cand_chunk)):
                hist_vec, gap_vec, hist_len = _history_arrays_for_src(self.state, src, time, self.seq_len)
                hist_arr[row_idx] = hist_vec
                gap_arr[row_idx] = gap_vec
                valid_model_rows[row_idx] = src_ids[row_idx] > 0 and hist_len > 0
                time_arr[row_idx] = _time_feature(
                    time,
                    self.state["last_time"].get(src),
                    hist_len,
                    self.state["time_min"],
                    self.state["time_scale"],
                    self.state["gap_scale"],
                    self.state["hist_scale"],
                )
                cand_arr[row_idx, :len(cand_ids)] = cand_ids
            shared_candidates = all(
                len(cands) == len(cand_chunk[0]) and np.array_equal(cands, cand_chunk[0])
                for cands in cand_chunk
            )
            if shared_candidates:
                scores = self.model(
                    self.jt.array(src_ids),
                    self.jt.array(hist_arr),
                    self.jt.array(gap_arr),
                    self.jt.array(time_arr),
                    self.jt.array(cand_chunk[0]),
                ).numpy()
            else:
                scores = self.model(
                    self.jt.array(src_ids),
                    self.jt.array(hist_arr),
                    self.jt.array(gap_arr),
                    self.jt.array(time_arr),
                    self.jt.array(cand_arr),
                ).numpy()
            scores[~valid_model_rows, :] = 0.0
            for row_scores, cand_ids in zip(scores, cand_chunk):
                out.append(np.asarray(row_scores[:len(cand_ids)], dtype=np.float32))
        return out


class RuleScorer:
    def __init__(self, history_edges):
        self.model = FastDataset2RuleScorer()
        self.model.fit(history_edges)
        self.id_to_dst = None

    def set_id_to_dst(self, id_to_dst):
        self.id_to_dst = id_to_dst

    def score(self, rows, candidate_id_rows):
        out = []
        for row_idx, (src, time, _positive_dst) in enumerate(rows):
            candidates = [self.id_to_dst[int(dst_id)] for dst_id in candidate_id_rows[row_idx]]
            out.append(np.asarray(self.model.score_many(src, time, candidates), dtype=np.float32))
        return out


class PopScorer:
    def __init__(self, history_edges, id_to_dst):
        counts = Counter(dst for _, dst, _ in history_edges)
        self.score_by_id = np.zeros(len(id_to_dst) + 1, dtype=np.float32)
        for dst_id, dst in id_to_dst.items():
            self.score_by_id[int(dst_id)] = np.log1p(float(counts.get(dst, 0)))

    def score(self, rows, candidate_id_rows):
        return [self.score_by_id[np.asarray(cands, dtype=np.int32)] for cands in candidate_id_rows]


class RecentScorer:
    def __init__(self, history_edges, id_to_dst, seq_len=200):
        self.id_to_dst = id_to_dst
        self.src_recent = defaultdict(lambda: deque(maxlen=seq_len))
        self.dst_last_time = {}
        times = []
        for src, dst, time in sorted(history_edges, key=lambda x: x[2]):
            self.src_recent[src].append(dst)
            self.dst_last_time[dst] = time
            times.append(time)
        self.time_min = min(times) if times else 0
        self.time_scale = max((max(times) - min(times)) if times else 1, 1)

    def score(self, rows, candidate_id_rows):
        out = []
        for row_idx, (src, time, _positive_dst) in enumerate(rows):
            recent_rank = {}
            for rank, dst in enumerate(reversed(self.src_recent.get(src, ())), start=1):
                recent_rank.setdefault(dst, rank)
            row_scores = []
            for dst_id in candidate_id_rows[row_idx]:
                dst = self.id_to_dst[int(dst_id)]
                rank = recent_rank.get(dst)
                src_recent_score = 3.0 / float(rank) if rank else 0.0
                last_time = self.dst_last_time.get(dst)
                if last_time is None:
                    global_recent = 0.0
                else:
                    global_recent = 1.0 / (1.0 + max(float(time) - float(last_time), 0.0) / self.time_scale)
                row_scores.append(src_recent_score + global_recent)
            out.append(np.asarray(row_scores, dtype=np.float32))
        return out


def evaluate(args):
    rng = np.random.default_rng(args.seed)
    mode_list = parse_csv_set(args.modes, VALID_MODES, "--modes")
    eval_sets = parse_csv_set(args.eval_sets, VALID_EVAL_SETS, "--eval-sets")

    dataset_dir = Path(args.data_dir) / "dataset2"
    history_edges, valid_edges = split_dataset2_edges(dataset_dir, final_train=False)
    src_to_id, dst_to_id = build_id_maps(history_edges)
    id_to_dst = {idx: dst for dst, idx in dst_to_id.items()}
    history_views = build_history_views(history_edges, seq_len=args.seq_len)

    scorers = {}
    if "rule_only" in mode_list or "fusion" in mode_list:
        scorers["rule_only"] = RuleScorer(history_edges)
        scorers["rule_only"].set_id_to_dst(id_to_dst)
    if "pop" in mode_list:
        scorers["pop"] = PopScorer(history_edges, id_to_dst)
    if "recent" in mode_list:
        scorers["recent"] = RecentScorer(history_edges, id_to_dst, seq_len=max(args.seq_len, 200))
    if "model_only" in mode_list or "fusion" in mode_list:
        scorers["model_only"] = ModelScorer(
            args.artifact_dir,
            history_edges,
            batch_size=args.batch_size,
            cuda=args.cuda,
        )

    raw = {
        eval_set: {mode: empty_mode_raw() for mode in mode_list}
        for eval_set in eval_sets
    }
    pending = {eval_set: {"rows": [], "candidate_ids": [], "positive_indices": [], "buckets": []} for eval_set in eval_sets}

    eval_edges = selected_events(valid_edges, args.max_events)
    processed = 0
    for src, dst, time in eval_edges:
        processed += 1
        buckets = event_buckets(src, dst, dst_to_id, history_views)
        for eval_set in eval_sets:
            candidate_data = build_candidate_ids(
                eval_set,
                dst_to_id,
                dst,
                src,
                history_views,
                args.pseudo_size,
                rng,
            )
            if candidate_data is None:
                for mode in mode_list:
                    add_event(raw[eval_set], mode, buckets, 0.0)
                continue
            cand_ids, positive_index = candidate_data
            group = pending[eval_set]
            group["rows"].append((src, time, dst))
            group["candidate_ids"].append(cand_ids)
            group["positive_indices"].append(positive_index)
            group["buckets"].append(buckets)
            if len(group["rows"]) >= args.batch_size:
                flush_rows(raw[eval_set], mode_list, group, scorers, args)
                group["rows"].clear()
                group["candidate_ids"].clear()
                group["positive_indices"].clear()
                group["buckets"].clear()

    for eval_set in eval_sets:
        group = pending[eval_set]
        if group["rows"]:
            flush_rows(raw[eval_set], mode_list, group, scorers, args)

    return {
        "dataset": "dataset2",
        "history_source": "split=0",
        "history_edges": len(history_edges),
        "valid_edges_seen": processed,
        "known_sources": len(src_to_id),
        "known_destinations": len(dst_to_id),
        "pseudo_size": int(args.pseudo_size),
        "eval_sets": {
            eval_set: {
                "modes": {mode: finalize_mode(raw[eval_set][mode]) for mode in mode_list},
            }
            for eval_set in eval_sets
        },
    }


def flush_rows(raw, mode_list, group, scorers, args):
    rows = group["rows"]
    candidate_id_rows = group["candidate_ids"]
    positive_indices = group["positive_indices"]
    bucket_rows = group["buckets"]

    rule_scores = None
    model_scores = None
    pop_scores = None
    recent_scores = None
    if "rule_only" in mode_list or "fusion" in mode_list:
        rule_scores = scorers["rule_only"].score(rows, candidate_id_rows)
    if "model_only" in mode_list or "fusion" in mode_list:
        model_scores = scorers["model_only"].score(rows, candidate_id_rows)
    if "pop" in mode_list:
        pop_scores = scorers["pop"].score(rows, candidate_id_rows)
    if "recent" in mode_list:
        recent_scores = scorers["recent"].score(rows, candidate_id_rows)

    for idx, positive_index in enumerate(positive_indices):
        buckets = bucket_rows[idx]
        model_unsuitable = "cold_src" in buckets or "no_history_src" in buckets
        if "rule_only" in mode_list:
            rr = reciprocal_rank_from_scores(rule_scores[idx], positive_index)
            add_event(raw, "rule_only", buckets, rr)
        if "model_only" in mode_list:
            rr = 0.0 if model_unsuitable else reciprocal_rank_from_scores(model_scores[idx], positive_index)
            add_event(raw, "model_only", buckets, rr)
        if "fusion" in mode_list:
            fused = (
                row_zscore(np.asarray([model_scores[idx]], dtype=np.float32))[0] * float(args.fusion_model_weight)
                + row_zscore(np.asarray([rule_scores[idx]], dtype=np.float32))[0] * float(args.fusion_rule_weight)
            )
            rr = reciprocal_rank_from_scores(fused, positive_index)
            add_event(raw, "fusion", buckets, rr)
        if "pop" in mode_list:
            rr = reciprocal_rank_from_scores(pop_scores[idx], positive_index)
            add_event(raw, "pop", buckets, rr)
        if "recent" in mode_list:
            rr = reciprocal_rank_from_scores(recent_scores[idx], positive_index)
            add_event(raw, "recent", buckets, rr)


def write_outputs(result, out_json, out_csv):
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for eval_set, eval_payload in result["eval_sets"].items():
            for mode, metrics in eval_payload["modes"].items():
                for bucket in BUCKETS:
                    rows.append({
                        "eval_set": eval_set,
                        "mode": mode,
                        "bucket": bucket,
                        "mrr": metrics[bucket]["mrr"],
                        "events": metrics[bucket]["events"],
                    })
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["eval_set", "mode", "bucket", "mrr", "events"])
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--artifact-dir", default="artifacts/dataset2")
    parser.add_argument("--modes", default="rule_only,model_only,fusion,pop,recent")
    parser.add_argument("--eval-sets", default="all-dst,pseudo100,hard-pseudo100")
    parser.add_argument("--out", default="reports/dataset2_split_eval.json")
    parser.add_argument("--csv-out", default="reports/dataset2_split_eval.csv")
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument("--pseudo-size", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fusion-model-weight", type=float, default=1.0)
    parser.add_argument("--fusion-rule-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_events < 0:
        raise ValueError("--max-events must be >= 0")
    if args.pseudo_size < 2:
        raise ValueError("--pseudo-size must be >= 2")
    if args.seq_len < 1:
        raise ValueError("--seq-len must be >= 1")

    result = evaluate(args)
    write_outputs(result, args.out, args.csv_out)
    for eval_set, eval_payload in result["eval_sets"].items():
        print(f"[{eval_set}]")
        for mode, metrics in eval_payload["modes"].items():
            parts = []
            for bucket in BUCKETS:
                parts.append(f"{bucket}={metrics[bucket]['mrr']:.6f}/{metrics[bucket]['events']}")
            print(f"{mode}: " + " ".join(parts))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
