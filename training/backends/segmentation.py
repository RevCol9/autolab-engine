"""YOLO 实例分割训练后端。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from training.backends.base import TrainBackend
from training.backends.detection import DetectionBackend
from training.data_yaml import labels_dir_for_job, prepare_data_yaml_for_job
from training.reporting import metric_lookup, trainer_lr, trainer_metric

_DETECTION_METRICS = DetectionBackend()


def _is_segmentation_label_line(line: str) -> bool:
    """YOLO-seg：class + 至少 3 个点（6 个坐标）。"""
    parts = line.split()
    return len(parts) >= 7


def _validate_segmentation_labels(labels_dir: Path, *, sample_limit: int = 32) -> None:
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"分割标签目录不存在: {labels_dir}")
    txt_files = sorted(labels_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"分割标签目录为空: {labels_dir}")
    checked = 0
    for path in txt_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                if not _is_segmentation_label_line(text):
                    raise ValueError(
                        f"标签格式不符合 YOLO 分割（需多边形点集）: {path.name} -> {text[:80]!r}"
                    )
                break
        checked += 1
        if checked >= sample_limit:
            break


class SegmentationBackend(TrainBackend):
    task = "segmentation"
    default_model = "yolo11n-seg.pt"

    def validate_job(self, param: Mapping[str, Any]) -> None:
        prepare_data_yaml_for_job(dict(param))
        labels_dir = labels_dir_for_job(dict(param))
        _validate_segmentation_labels(labels_dir)

    def build_epoch_row(
        self,
        trainer: Any,
        metrics: Mapping[str, Any],
        resource: Mapping[str, Any],
        *,
        epoch_index: int,
        epochs: int,
        epoch_spend: float,
        total_spend: float,
        remaining: float,
    ) -> Dict[str, Any]:
        row = _DETECTION_METRICS.build_epoch_row(
            trainer,
            metrics,
            resource,
            epoch_index=epoch_index,
            epochs=epochs,
            epoch_spend=epoch_spend,
            total_spend=total_spend,
            remaining=remaining,
        )
        row.update(
            {
                "train/seg_loss": trainer_metric(trainer, "train/seg_loss"),
                "val/seg_loss": metrics.get("val/seg_loss", 0),
                "metrics/precision(M)": metric_lookup(metrics, "precision_m"),
                "metrics/recall(M)": metric_lookup(metrics, "recall_m"),
                "metrics/mAP50(M)": metric_lookup(metrics, "map50_m"),
                "metrics/mAP50-95(M)": metric_lookup(metrics, "map5095_m"),
            }
        )
        return row


BACKEND = SegmentationBackend()
