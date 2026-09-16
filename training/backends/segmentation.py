"""YOLO 实例分割训练后端。"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from training.backends.detection import DetectionBackend
from training.reporting import metric_lookup, trainer_metric


class SegmentationBackend(DetectionBackend):
    task = "segmentation"
    default_model = "yolo11n-seg.pt"

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
        row = super().build_epoch_row(
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
