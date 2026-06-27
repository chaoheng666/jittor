import argparse
import csv
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

try:
    import jittor as jt
    from jittor_geometric.data import TemporalData
    from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler
    from jittor_geometric.nn.models.craft import CRAFT
except ImportError as exc:
    raise SystemExit(
        "train_craft_ranker.py requires jittor and jittor_geometric. "
        "Install the official baseline dependencies first."
    ) from exc

from src.luxury_scoring import mrr


def read_train(path):
    src, dst, time = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src.append(int(row["src"]))
            dst.append(int(row["dst"]))
            time.append(int(row["time"]))
    return (
        np.asarray(src, dtype=np.int32),
        np.asarray(dst, dtype=np.int32),
        np.asarray(time, dtype=np.int32),
    )


def read_valid(path, max_rows=0):
    src, time, labels, candidates = [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_rows and len(src) >= max_rows:
                break
            src.append(int(row["src"]))
            time.append(int(row["time"]))
            labels.append(int(row["label"]))
            candidates.append([int(row[f"c{i}"]) for i in range(1, 101)])
    return (
        np.asarray(src, dtype=np.int32),
        np.asarray(time, dtype=np.int32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(candidates, dtype=np.int32),
    )


def read_test(path):
    src, time, candidates = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            time.append(int(row[1]))
            candidates.append([int(x) for x in row[2:]])
    return (
        np.asarray(src, dtype=np.int32),
        np.asarray(time, dtype=np.int32),
        np.asarray(candidates, dtype=np.int32),
    )


def build_temporal_data(src, dst, time):
    edge_ids = np.arange(len(src), dtype=np.int32) + 1
    return TemporalData(
        src=jt.Var(src),
        dst=jt.Var(dst),
        t=jt.Var(time),
        edge_ids=jt.Var(edge_ids),
    )


def score_candidates(model, sampler, src, time, candidates, num_neighbors, batch_size):
    model.eval()
    out = []
    for start in range(0, len(src), batch_size):
        end = min(start + batch_size, len(src))
        batch_src = src[start:end]
        batch_time = time[start:end]
        batch_cand = candidates[start:end]

        src_neighb_seq, _, src_times = sampler.get_historical_neighbors_left(
            node_ids=batch_src,
            node_interact_times=batch_time,
            num_neighbors=num_neighbors,
        )
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)
        test_dst = jt.Var(batch_cand)

        dst_last_neighbor, _, dst_last_update_time = sampler.get_historical_neighbors_left(
            node_ids=test_dst.flatten().numpy(),
            node_interact_times=np.broadcast_to(
                batch_time[:, np.newaxis], (len(batch_time), test_dst.shape[1])
            ).flatten(),
            num_neighbors=1,
        )
        dst_last_update_time = np.asarray(dst_last_update_time).reshape(len(test_dst), -1)
        dst_last_update_time[dst_last_neighbor.reshape(len(test_dst), -1) == 0] = -100000

        src_seq_adj = jt.Var(src_neighb_seq) - model.dst_min_idx + 1
        test_dst_adj = test_dst - model.dst_min_idx + 1
        src_seq_adj = jt.where(src_seq_adj < 0, jt.zeros_like(src_seq_adj), src_seq_adj)

        logits = model.forward(
            src_seq_adj,
            jt.Var(neighbor_num),
            jt.Var(src_times),
            jt.Var(batch_time),
            test_dst=test_dst_adj,
            dst_last_update_times=jt.Var(dst_last_update_time),
        )
        out.append(logits.squeeze(-1).numpy())
    return np.vstack(out)


def train_one_dataset(args, dataset_name):
    dataset_dir = Path(args.valid_dir) / dataset_name
    src, dst, time = read_train(dataset_dir / "train.csv")
    valid_src, valid_time, labels, valid_candidates = read_valid(dataset_dir / "valid.csv", args.max_rows)

    max_node = max(int(src.max()), int(dst.max()), int(valid_candidates.max()))
    if (Path(args.data_dir) / dataset_name / "test.csv").exists():
        _, _, test_candidates = read_test(Path(args.data_dir) / dataset_name / "test.csv")
        max_node = max(max_node, int(test_candidates.max()))
    node_size = max_node + 1
    dst_min = min(int(dst.min()), int(valid_candidates.min()))
    src_min = int(src.min())

    train_data = build_temporal_data(src, dst, time)
    train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=1.0)
    sampler = get_neighbor_sampler(train_data, "recent", seed=args.seed)

    model = CRAFT(
        n_layers=2,
        n_heads=2,
        hidden_size=args.hidden_size,
        hidden_dropout_prob=args.dropout,
        attn_dropout_prob=args.dropout,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        n_nodes=node_size,
        max_seq_length=args.num_neighbors,
        loss_type="BPR",
        use_pos=True,
        input_cat_time_intervals=False,
        output_cat_time_intervals=True,
        output_cat_repeat_times=True,
        num_output_layer=1,
        emb_dropout_prob=args.dropout,
        skip_connection=True,
    )
    model.set_min_idx(src_min, dst_min)
    optimizer = jt.nn.Adam(list(model.parameters()), lr=args.lr)

    best_mrr = -1.0
    best_path = Path(args.model_dir) / f"{dataset_name}_craft.pkl"
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_data in train_loader:
            batch_src = jt.array(batch_data.src)
            batch_dst = jt.array(batch_data.dst)
            batch_time = jt.array(batch_data.t)
            neg_dst = jt.array(batch_data.neg_dst)

            src_neighb_seq, _, src_times = sampler.get_historical_neighbors_left(
                node_ids=batch_src.numpy(),
                node_interact_times=batch_time.numpy(),
                num_neighbors=args.num_neighbors,
            )
            neighbor_num = (src_neighb_seq != 0).sum(axis=1)
            if neighbor_num.sum() == 0:
                continue

            test_dst = jt.cat([jt.Var(batch_dst).unsqueeze(-1), jt.Var(neg_dst).unsqueeze(-1)], dim=-1)
            dst_last_neighbor, _, dst_last_update_time = sampler.get_historical_neighbors_left(
                node_ids=test_dst.flatten().numpy(),
                node_interact_times=np.broadcast_to(
                    batch_time.numpy()[:, np.newaxis], (len(batch_time), test_dst.shape[1])
                ).flatten(),
                num_neighbors=1,
            )
            dst_last_update_time = np.asarray(dst_last_update_time).reshape(len(test_dst), -1)
            dst_last_update_time[dst_last_neighbor.reshape(len(test_dst), -1) == 0] = -100000
            loss, _, _ = model.calculate_loss(
                src_neighb_seq=jt.Var(src_neighb_seq),
                src_neighb_seq_len=jt.Var(neighbor_num),
                src_neighb_interact_times=jt.Var(src_times),
                cur_pred_times=jt.Var(batch_time),
                test_dst=test_dst,
                dst_last_update_times=jt.Var(dst_last_update_time),
            )
            optimizer.step(loss)
            losses.append(float(loss.numpy()))

        valid_scores = score_candidates(
            model, sampler, valid_src, valid_time, valid_candidates,
            args.num_neighbors, args.batch_size
        )
        valid_mrr = mrr(valid_scores, labels)
        print(
            f"{dataset_name} epoch={epoch} loss={np.mean(losses):.6f} "
            f"craft_mrr={valid_mrr:.8f}"
        )
        if valid_mrr > best_mrr:
            best_mrr = valid_mrr
            jt.save(model.state_dict(), str(best_path))

    if best_path.exists():
        model.load_state_dict(jt.load(str(best_path)))

    Path(args.score_dir).mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"craft_n{args.num_neighbors}_h{args.hidden_size}_s{args.seed}"
    valid_scores = score_candidates(
        model, sampler, valid_src, valid_time, valid_candidates,
        args.num_neighbors, args.batch_size
    )
    np.save(Path(args.score_dir) / f"{dataset_name}_{run_name}_valid.npy", valid_scores.astype(np.float32))

    test_path = Path(args.data_dir) / dataset_name / "test.csv"
    if test_path.exists():
        test_src, test_time, test_candidates = read_test(test_path)
        test_scores = score_candidates(
            model, sampler, test_src, test_time, test_candidates,
            args.num_neighbors, args.batch_size
        )
        np.save(Path(args.score_dir) / f"{dataset_name}_{run_name}_test.npy", test_scores.astype(np.float32))
    print(f"{dataset_name}: saved CRAFT scores best_mrr={best_mrr:.8f}")


def find_dataset_names(valid_dir, dataset_arg):
    if dataset_arg != "all":
        return [name.strip() for name in dataset_arg.split(",") if name.strip()]
    valid_path = Path(valid_dir)
    return sorted(
        p.name for p in valid_path.iterdir()
        if p.is_dir() and (p / "train.csv").exists() and (p / "valid.csv").exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_A")
    parser.add_argument("--valid-dir", default="validation")
    parser.add_argument("--model-dir", default="models_craft")
    parser.add_argument("--score-dir", default="luxury_scores")
    parser.add_argument("--dataset", default="all", help="all or comma-separated dataset names")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-neighbors", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    if args.cuda:
        jt.flags.use_cuda = 1

    os.makedirs(args.model_dir, exist_ok=True)
    names = find_dataset_names(args.valid_dir, args.dataset)
    if not names:
        raise ValueError(f"no validation datasets found in {args.valid_dir}")
    for name in names:
        train_one_dataset(args, name)


if __name__ == "__main__":
    main()
