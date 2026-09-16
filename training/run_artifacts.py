"""训练 run 目录中的可重建运行产物管理。"""

from __future__ import annotations

import shutil
from pathlib import Path

TRAINING_METRICS_CSV = "trainning_data.csv"  # 外部既有契约，保留历史拼写

_RUN_DIRS = ("weights", "baseline_eval", "trained_eval")
_RUN_FILES = (
    "train.log",
    TRAINING_METRICS_CSV,
    "report.json",
    "results.csv",
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxF1_curve.png",
    "MaskPR_curve.png",
    "MaskP_curve.png",
    "MaskR_curve.png",
    "MaskF1_curve.png",
    "labels.jpg",
    "labels_correlogram.jpg",
    "args.yaml",
)
_RUN_GLOBS = ("train_batch*.jpg", "val_batch*.jpg", "events.out.tfevents.*")


def ensure_input_survives_reset(input_path: str | Path, save_dir: Path) -> None:
    """拒绝把训练输入放进启动时会清理的输出目录。"""
    source = Path(input_path).resolve()
    save_dir = Path(save_dir).resolve()
    for name in _RUN_DIRS:
        artifact_dir = save_dir / name
        try:
            source.relative_to(artifact_dir)
        except ValueError:
            continue
        raise ValueError(
            f"预训练模型位于本次训练会清理的输出目录: {source}；"
            "请提供独立的模型归档地址"
        )


def reset_run_artifacts(save_dir: Path) -> None:
    """清理同一 trainNum 的旧运行产物，保留 images/labels 与任务配置。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    for name in _RUN_DIRS:
        path = save_dir / name
        if path.is_dir():
            shutil.rmtree(path)
    for name in _RUN_FILES:
        path = save_dir / name
        if path.is_file():
            path.unlink()
    for pattern in _RUN_GLOBS:
        for path in save_dir.glob(pattern):
            if path.is_file():
                path.unlink()


__all__ = [
    "TRAINING_METRICS_CSV",
    "ensure_input_survives_reset",
    "reset_run_artifacts",
]
