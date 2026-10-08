"""Ultralytics trainers whose checkpoint lifecycle is .niii-model only."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils.torch_utils import unwrap_model

from toolkit.model_crypto import load_yolo_container
from training.encrypted_checkpoints import save_training_model


class _EncryptedTrainerMixin:
    _training_task: str
    _container_task: str

    def __init__(
        self,
        *args,
        key_dir: str | Path | None = None,
        allow_cpu_for_tests: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.args.resume:
            raise ValueError("加密训练首版不支持 resume=True；请从加密 best 重新微调")
        if self.world_size != 1 and not allow_cpu_for_tests:
            raise RuntimeError("加密训练首版只支持单 CUDA 设备")
        if self.device.type != "cuda" and not allow_cpu_for_tests:
            raise RuntimeError("加密训练首版必须使用 CUDA，不回退 CPU")
        if self.save_period > 0:
            raise ValueError("加密训练首版不支持周期 checkpoint，请设置 save_period=-1")
        self._encrypted_key_dir = key_dir
        self._allow_cpu_for_tests = allow_cpu_for_tests
        self.last = self.wdir / "last.niii-model"
        self.best = self.wdir / "best.niii-model"

    def save_model(self) -> bool:
        """Write last/best directly from EMA as authenticated ciphertext."""
        ema = unwrap_model(self.ema.ema)
        for name, tensor in ema.state_dict().items():
            if tensor.is_floating_point() or tensor.is_complex():
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"训练 EMA 含非有限 tensor，拒绝保存: {name}")
        save_training_model(
            ema,
            self.last,
            task=self._training_task,
            key_dir=self._encrypted_key_dir,
        )
        if self.best_fitness == self.fitness:
            save_training_model(
                ema,
                self.best,
                task=self._training_task,
                key_dir=self._encrypted_key_dir,
            )
        return True

    def final_eval(self) -> None:
        """Validate by reloading authenticated best/last ciphertext, never a .pt file."""
        encrypted = self.best if self.best.is_file() else self.last
        if not encrypted.is_file():
            raise RuntimeError("训练结束但未生成加密 best/last 模型")
        loaded = load_yolo_container(
            encrypted,
            self.device,
            self._container_task,
            key_dir=self._encrypted_key_dir,
            allow_cpu_for_tests=self._allow_cpu_for_tests,
        )
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        self.metrics = self.validator(model=loaded.model)
        self.metrics.pop("fitness", None)
        self.epoch += 1
        self.run_callbacks("on_fit_epoch_end")
        self.epoch -= 1

    def _handle_nan_recovery(self, epoch: int) -> bool:
        """Fail closed; v1 deliberately has no optimizer-state checkpoint recovery."""
        loss_nan = self.loss is not None and not self.loss.isfinite()
        fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
        fitness_collapse = bool(
            self.best_fitness and self.best_fitness > 0 and self.fitness == 0
        )
        if loss_nan or fitness_nan or fitness_collapse:
            raise RuntimeError(
                "加密训练检测到 NaN/Inf 或 fitness collapse；"
                "首版不从不完整 optimizer checkpoint 恢复"
            )
        return False


class EncryptedDetectionTrainer(_EncryptedTrainerMixin, DetectionTrainer):
    _training_task = "detection"
    _container_task = "detect"


class EncryptedSegmentationTrainer(_EncryptedTrainerMixin, SegmentationTrainer):
    _training_task = "segmentation"
    _container_task = "segment"


def create_encrypted_trainer(
    model,
    *,
    task: str,
    overrides: Mapping[str, Any],
    key_dir: str | Path | None = None,
    callbacks=None,
    allow_cpu_for_tests: bool = False,
):
    """Create the controlled single-device trainer around an in-memory YOLO model."""
    resume = overrides.get("resume")
    if resume is True or (
        resume is not None
        and str(resume).strip().lower() in {"1", "true", "yes", "y", "on"}
    ):
        raise ValueError("加密训练首版不支持 resume=True；请从加密 best 重新微调")
    trainer_type = {
        "detection": EncryptedDetectionTrainer,
        "segmentation": EncryptedSegmentationTrainer,
    }.get(str(task).strip().lower())
    if trainer_type is None:
        raise ValueError(f"不支持的加密训练任务: {task!r}")
    training_model = unwrap_model(model)
    is_fused = getattr(training_model, "is_fused", None)
    if callable(is_fused) and is_fused():
        raise ValueError(
            "加密训练拒绝 fused 预训练模型；推理/验证会原地融合模型，"
            "请从 .niii-model 重新认证加载后再创建 Trainer"
        )
    trainer = trainer_type(
        overrides={**dict(overrides), "pretrained": False, "resume": False},
        _callbacks=callbacks,
        key_dir=key_dir,
        allow_cpu_for_tests=allow_cpu_for_tests,
    )
    model_yaml = getattr(training_model, "yaml", None)
    if not isinstance(model_yaml, dict):
        raise ValueError("加密训练输入缺少受控 YOLO yaml")
    # Match the dataset class count exactly, while transferring compatible weights.
    trainer.model = trainer.get_model(cfg=model_yaml, weights=training_model, verbose=False)
    return trainer


__all__ = [
    "EncryptedDetectionTrainer",
    "EncryptedSegmentationTrainer",
    "create_encrypted_trainer",
]
