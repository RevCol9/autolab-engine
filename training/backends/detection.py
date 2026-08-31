"""YOLO 检测训练后端。"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from training.backends.base import TrainBackend
from training.data_yaml import prepare_data_yaml_for_job
from training.reporting import metric_lookup, trainer_lr, trainer_metric


class DetectionBackend(TrainBackend):
    task = "detection"
    default_model = "yolo11n.pt"

    def validate_job(self, param: Mapping[str, Any]) -> None:
        prepare_data_yaml_for_job(dict(param))

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
        return {
            "epoch": epoch_index,
            "train/box_loss": trainer_metric(trainer, "train/box_loss"),
            "train/cls_loss": trainer_metric(trainer, "train/cls_loss"),
            "train/dfl_loss": trainer_metric(trainer, "train/dfl_loss"),
            "metrics/precision(B)": metric_lookup(metrics, "precision"),
            "metrics/recall(B)": metric_lookup(metrics, "recall"),
            "metrics/mAP50(B)": metric_lookup(metrics, "map50"),
            "metrics/mAP50-95(B)": metric_lookup(metrics, "map5095"),
            "val/box_loss": metrics.get("val/box_loss", 0),
            "val/cls_loss": metrics.get("val/cls_loss", 0),
            "val/dfl_loss": metrics.get("val/dfl_loss", 0),
            "lr/pg0": trainer_lr(trainer, 0),
            "lr/pg1": trainer_lr(trainer, 1),
            "lr/pg2": trainer_lr(trainer, 2),
            "epoch_spend_time": round(epoch_spend, 4),
            "total_spend_time": round(total_spend, 4),
            "remaining_time": round(remaining, 4),
            "resource_cpu": resource.get("cpu", 0),
            "resource_mem": resource.get("mem", 0),
            "resource_gpu": resource.get("gpu", 0),
            "resource_gpu_mem_used_mb": resource.get("gpuMemUsedMb", 0),
            "resource_gpu_mem_total_mb": resource.get("gpuMemTotalMb", 0),
            "resource_disk": resource.get("disk", 0),
        }


BACKEND = DetectionBackend()
