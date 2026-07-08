from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .data import TestRow, ensure_dir
from .features import FEATURE_NAMES, GraphFeatureModel
from .validation import ValidationSet, score_feature_tensor


def make_training_feature_block(
    model: GraphFeatureModel,
    vsets: Sequence[ValidationSet],
    max_rows: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    rows = []
    labels = []
    rng = np.random.default_rng(seed)
    for vset in vsets:
        if vset.name.startswith("teacher_"):
            continue
        if vset.features is None:
            continue
        idx = np.arange(len(vset.labels))
        rng.shuffle(idx)
        take = min(len(idx), max(1, int(max_rows / max(len(vsets), 1))))
        keep = np.sort(idx[:take])
        rows.append(vset.features[keep])
        labels.append(vset.labels[keep])
    if not rows:
        raise ValueError("no validation features available to train dense listwise model")
    x = np.concatenate(rows, axis=0).astype(np.float32)
    y = np.concatenate(labels, axis=0).astype(np.int64)
    if len(y) > max_rows:
        idx = rng.choice(np.arange(len(y)), size=int(max_rows), replace=False)
        x = x[idx]
        y = y[idx]
    meta = {"train_rows": int(len(y)), "feature_dim": int(x.shape[-1])}
    return x, y, meta


def train_jittor_feature_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: Optional[np.ndarray],
    valid_y: Optional[np.ndarray],
    out_dir: Path,
    hidden: int = 128,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 2026,
) -> dict:
    ensure_dir(out_dir)
    report = {
        "available": False,
        "status": "not_started",
        "hidden": int(hidden),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
    }
    try:
        import jittor as jt
        from jittor import nn
    except Exception as exc:
        report.update({"status": "jittor_import_failed", "error": repr(exc)})
        return report

    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass

    np.random.seed(seed)
    try:
        jt.set_global_seed(seed)
    except Exception:
        pass

    class CandidateMLP(nn.Module):
        def __init__(self, dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, 1),
            )

        def execute(self, x):
            b, c, f = x.shape
            return self.net(x.reshape((b * c, f))).reshape((b, c))

    try:
        feature_mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0, keepdims=True).astype(np.float32)
        feature_std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0, keepdims=True).astype(np.float32)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        train_xn = ((train_x - feature_mean.reshape(1, 1, -1)) / feature_std.reshape(1, 1, -1)).astype(np.float32)
        valid_xn = None
        if valid_x is not None:
            valid_xn = ((valid_x - feature_mean.reshape(1, 1, -1)) / feature_std.reshape(1, 1, -1)).astype(np.float32)

        net = CandidateMLP(train_x.shape[-1], int(hidden))
        opt = nn.Adam(net.parameters(), lr=float(lr), weight_decay=1e-5)
        history = []
        rng = np.random.default_rng(seed)
        for epoch in range(1, int(epochs) + 1):
            order = np.arange(len(train_y))
            rng.shuffle(order)
            losses = []
            for start in range(0, len(order), int(batch_size)):
                idx = order[start:start + int(batch_size)]
                xb = jt.array(train_xn[idx])
                yb = jt.array(train_y[idx])
                logits = net(xb)
                loss = nn.cross_entropy_loss(logits, yb)
                opt.step(loss)
                losses.append(float(loss.data))
            item = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else None}
            if valid_xn is not None and valid_y is not None and len(valid_y):
                pred = predict_jittor_array(net, valid_xn, batch_size=max(int(batch_size), 512))
                from .data import tie_aware_mrr

                item["valid_mrr"] = tie_aware_mrr(pred, valid_y)
            history.append(item)
            print(f"jittor_mlp epoch={epoch} loss={item['loss']} valid_mrr={item.get('valid_mrr')}", flush=True)

        ckpt = out_dir / "jittor_feature_mlp.pkl"
        np.savez(
            out_dir / "jittor_feature_norm.npz",
            mean=feature_mean.astype(np.float32),
            std=feature_std.astype(np.float32),
            feature_names=np.asarray(FEATURE_NAMES),
        )
        net.save(str(ckpt))
        report.update({
            "available": True,
            "status": "trained",
            "history": history,
            "checkpoint": str(ckpt),
            "norm": str(out_dir / "jittor_feature_norm.npz"),
        })
    except Exception as exc:
        report.update({"status": "jittor_training_failed", "error": repr(exc)})
    return report


def predict_jittor_array(net, features: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    import jittor as jt

    out = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
    for start in range(0, len(features), int(batch_size)):
        xb = jt.array(features[start:start + int(batch_size)].astype(np.float32))
        logits = net(xb)
        out[start:start + int(batch_size)] = np.asarray(logits.data, dtype=np.float32)
    return out


def load_and_predict_jittor_feature_mlp(
    checkpoint: Path,
    norm_path: Path,
    features: np.ndarray,
    hidden: int = 128,
    batch_size: int = 1024,
) -> Optional[np.ndarray]:
    try:
        import jittor as jt
        from jittor import nn
    except Exception:
        return None

    class CandidateMLP(nn.Module):
        def __init__(self, dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, 1),
            )

        def execute(self, x):
            b, c, f = x.shape
            return self.net(x.reshape((b * c, f))).reshape((b, c))

    norm = np.load(norm_path)
    mean = norm["mean"].reshape(1, 1, -1)
    std = norm["std"].reshape(1, 1, -1)
    x = ((features - mean) / std).astype(np.float32)
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass
    net = CandidateMLP(features.shape[-1], int(hidden))
    net.load(str(checkpoint))
    return predict_jittor_array(net, x, batch_size=batch_size)


def make_jittor_feature_predictor(
    checkpoint: Path,
    norm_path: Path,
    hidden: int = 128,
    batch_size: int = 1024,
):
    import jittor as jt
    from jittor import nn

    class CandidateMLP(nn.Module):
        def __init__(self, dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Relu(),
                nn.Linear(hidden_dim, 1),
            )

        def execute(self, x):
            b, c, f = x.shape
            return self.net(x.reshape((b * c, f))).reshape((b, c))

    norm = np.load(norm_path)
    mean = norm["mean"].reshape(1, 1, -1)
    std = norm["std"].reshape(1, 1, -1)
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass
    net = CandidateMLP(len(mean.reshape(-1)), int(hidden))
    net.load(str(checkpoint))

    def predict(features: np.ndarray) -> np.ndarray:
        x = ((features - mean) / std).astype(np.float32)
        return predict_jittor_array(net, x, batch_size=batch_size)

    return predict


def blend_feature_and_mlp_scores(feature_scores: np.ndarray, mlp_scores: Optional[np.ndarray], mlp_weight: float) -> np.ndarray:
    if mlp_scores is None:
        return feature_scores.astype(np.float32)
    from .data import row_zscore

    return (row_zscore(feature_scores) + float(mlp_weight) * row_zscore(mlp_scores)).astype(np.float32)
