from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import (
    TestRow,
    copy_member_from_zip,
    ensure_dir,
    make_result_zip,
    read_zip_scores,
    row_zscore,
    score_summary,
    softmax,
    top1_change,
    validate_csv,
    write_scores_csv,
)


def model_logits_from_components(feature_logits: np.ndarray, optional_logits: Optional[np.ndarray] = None, optional_weight: float = 0.25) -> np.ndarray:
    logits = row_zscore(feature_logits)
    if optional_logits is not None:
        logits = logits + float(optional_weight) * row_zscore(optional_logits)
    return logits.astype(np.float32)


def blend_logits_with_teacher(teacher_probs: np.ndarray, model_logits: np.ndarray, alpha: float) -> np.ndarray:
    teacher_logits = row_zscore(np.log(np.clip(teacher_probs, 1e-12, 1.0)))
    model_logits = row_zscore(model_logits)
    return ((1.0 - float(alpha)) * teacher_logits + float(alpha) * model_logits).astype(np.float32)


def find_alpha_for_top1_change(
    teacher_probs: np.ndarray,
    model_logits: np.ndarray,
    target_change: float,
    max_alpha: float = 0.65,
    steps: int = 80,
) -> Tuple[float, float]:
    best_alpha = 0.0
    best_change = 0.0
    best_gap = float("inf")
    for alpha in np.linspace(0.0, float(max_alpha), int(steps) + 1):
        blended = blend_logits_with_teacher(teacher_probs, model_logits, float(alpha))
        change = top1_change(teacher_probs, blended)
        gap = abs(change - float(target_change))
        if gap < best_gap:
            best_gap = gap
            best_alpha = float(alpha)
            best_change = float(change)
    return best_alpha, best_change


def copy_or_blend_dataset1(
    teacher_zip: Path,
    dataset1_model_logits: Optional[np.ndarray],
    dataset1_test_rows: Sequence[TestRow],
    out_csv: Path,
    alpha: float,
) -> dict:
    if dataset1_model_logits is None or alpha <= 0:
        copy_member_from_zip(teacher_zip, "dataset1.csv", out_csv)
        return {"mode": "teacher_copy", "validation": validate_csv(out_csv, expected_rows=len(dataset1_test_rows))}
    teacher = read_zip_scores(teacher_zip, "dataset1.csv")
    blended = blend_logits_with_teacher(teacher, dataset1_model_logits, alpha)
    probs = softmax(blended, temperature=1.0)
    check = write_scores_csv(probs, out_csv)
    return {
        "mode": "teacher_blend",
        "alpha": alpha,
        "top1_change_vs_teacher": top1_change(teacher, probs),
        "summary": score_summary(probs, dataset1_test_rows),
        "validation": check,
    }


def write_candidate_packages(
    teacher_zip: Path,
    dataset1_test_rows: Sequence[TestRow],
    dataset2_test_rows: Sequence[TestRow],
    dataset2_model_logits: np.ndarray,
    out_root: Path,
    target_changes: Sequence[float] = (0.01, 0.03, 0.05, 0.08, 0.12),
    dataset1_model_logits: Optional[np.ndarray] = None,
    dataset1_alpha: float = 0.0,
    prefix: str = "result_rebuild",
) -> Dict[str, dict]:
    ensure_dir(out_root)
    teacher2 = read_zip_scores(teacher_zip, "dataset2.csv")
    if len(teacher2) != len(dataset2_model_logits):
        if len(dataset2_model_logits) < len(teacher2):
            teacher2 = teacher2[: len(dataset2_model_logits)]
            dataset2_test_rows = dataset2_test_rows[: len(dataset2_model_logits)]
        else:
            raise ValueError(f"teacher2 rows {len(teacher2)} < model rows {len(dataset2_model_logits)}")
    manifest: Dict[str, dict] = {}
    for target in target_changes:
        alpha, actual_change = find_alpha_for_top1_change(teacher2, dataset2_model_logits, target)
        package_name = f"{prefix}_top1_{int(round(target * 100)):02d}p"
        package_dir = ensure_dir(out_root / package_name)
        dataset1_csv = package_dir / "dataset1.csv"
        dataset2_csv = package_dir / "dataset2.csv"
        zip_path = out_root / f"{package_name}.zip"

        d1_report = copy_or_blend_dataset1(teacher_zip, dataset1_model_logits, dataset1_test_rows, dataset1_csv, dataset1_alpha)
        blended_logits = blend_logits_with_teacher(teacher2, dataset2_model_logits, alpha)
        probs2 = softmax(blended_logits, temperature=1.0)
        d2_check = write_scores_csv(probs2, dataset2_csv)
        make_result_zip(dataset1_csv, dataset2_csv, zip_path)
        zip_members = sorted(zip_path.name for zip_path in [zip_path])
        manifest[package_name] = {
            "zip": str(zip_path),
            "target_top1_change": float(target),
            "actual_top1_change_vs_teacher_dataset2": float(actual_change),
            "alpha_dataset2": float(alpha),
            "dataset1": d1_report,
            "dataset2": {
                "summary": score_summary(probs2, dataset2_test_rows),
                "validation": d2_check,
            },
            "members": zip_members,
        }
        print(
            f"candidate package={package_name} alpha={alpha:.5f} top1_change={actual_change:.5f} zip={zip_path}",
            flush=True,
        )
    return manifest


def write_research_package(
    teacher_zip: Path,
    dataset1_test_rows: Sequence[TestRow],
    dataset2_test_rows: Sequence[TestRow],
    dataset2_model_logits: np.ndarray,
    out_root: Path,
    prefix: str = "result_rebuild_research_full",
) -> dict:
    package_dir = ensure_dir(out_root / prefix)
    dataset1_csv = package_dir / "dataset1.csv"
    dataset2_csv = package_dir / "dataset2.csv"
    copy_member_from_zip(teacher_zip, "dataset1.csv", dataset1_csv)
    if len(dataset2_test_rows) != len(dataset2_model_logits):
        dataset2_test_rows = dataset2_test_rows[: len(dataset2_model_logits)]
    probs2 = softmax(dataset2_model_logits, temperature=1.0)
    d2_check = write_scores_csv(probs2, dataset2_csv)
    zip_path = out_root / f"{prefix}.zip"
    make_result_zip(dataset1_csv, dataset2_csv, zip_path)
    return {
        "zip": str(zip_path),
        "mode": "no_teacher_constraint_dataset2",
        "dataset1": validate_csv(dataset1_csv, expected_rows=len(dataset1_test_rows)),
        "dataset2": {
            "summary": score_summary(probs2, dataset2_test_rows),
            "validation": d2_check,
        },
    }
