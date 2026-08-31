"""训练后端抽象：检测 / 分割等任务可插拔注册。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping


class TrainBackend(ABC):
    """单种训练任务的后端实现（Ultralytics YOLO 系列）。"""

    @property
    @abstractmethod
    def task(self) -> str:
        """任务标识，对应 config/training/{task}.yaml 与 API 路由。"""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """未传 model 且非续训时的默认预训练权重。"""

    @abstractmethod
    def validate_job(self, param: Mapping[str, Any]) -> None:
        """启动前校验数据集等；失败抛 ValueError / FileNotFoundError。"""

    @abstractmethod
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
        """构造 trainning_data.csv 单行。"""
