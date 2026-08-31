"""推理引擎抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from PIL import Image

from app.settings import ModelConfig


class BaseEngine(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Image.Image, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def classes(self) -> List[str]:
        return []

    def unload(self) -> None:
        """释放引擎占用的 GPU 显存；默认无操作。"""
        return
