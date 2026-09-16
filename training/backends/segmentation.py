"""YOLO 实例分割训练后端。"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from training.backends.base import TrainBackend
from training.backends.detection import DetectionBackend
from training.data_yaml import images_dir_for_job, labels_dir_for_job, prepare_data_yaml_for_job
from training.label_validation import validate_image_dir, validate_yolo_labels
from training.reporting import metric_lookup, trainer_lr, trainer_metric

_DETECTION_METRICS = DetectionBackend()


class SegmentationBackend(TrainBackend):
    task = "segmentation"
    default_model = "yolo11n-seg.pt"

    def validate_job(self, param: Mapping[str, Any]) -> None:
        payload = dict(param)
        prepare_data_yaml_for_job(payload)
        validate_image_dir(images_dir_for_job(payload))
        validate_yolo_labels(labels_dir_for_job(payload), task=self.task)

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
