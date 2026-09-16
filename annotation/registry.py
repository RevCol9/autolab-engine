"""推理模型运行时：配置解析、引擎缓存与设备互斥。"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from fastapi import HTTPException

from annotation.bootstrap import SETTINGS
from annotation.engines.base import BaseEngine
from annotation.engines.yolo import YoloDetectEngine, YoloSegmentEngine
from annotation.settings import EngineKind, ModelConfig, Settings, YoloTask
from shared.gpu_lock import GpuDeviceLock

GPU_INFER_TIMEOUT = float(os.environ.get("NIII_GPU_INFER_TIMEOUT", "120"))
GPU_LOAD_TIMEOUT = float(os.environ.get("NIII_GPU_LOAD_TIMEOUT", "180"))
BATCH_MAX_IMAGES = 32

EngineFactory = Callable[[ModelConfig], BaseEngine]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnloadResult:
    unloaded: bool
    loaded: tuple[str, ...]
    active_key: Optional[str]


class EngineRegistry:
    """在一个小 interface 后封装缓存、线程锁和跨进程设备锁。"""

    def __init__(
        self,
        settings: Settings,
        *,
        engine_factory: Optional[EngineFactory] = None,
    ) -> None:
        self._settings = settings
        self._engine_factory = engine_factory
        self._engines: dict[str, BaseEngine] = {}
        self._active_key: Optional[str] = None
        self._lock = threading.RLock()

    def get_model_config(self, model_key: Optional[str]) -> ModelConfig:
        key = (model_key or self._settings.default_model or "").strip()
        if not self._settings.models:
            raise HTTPException(status_code=500, detail="config/annotation.yaml 未配置 models")
        if not key:
            return self._settings.models[0]
        for model in self._settings.models:
            if model.key == key:
                return model
        raise HTTPException(status_code=400, detail=f"未知 model_key: {key}")

    def loaded_keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._engines)

    def active_key(self) -> Optional[str]:
        with self._lock:
            return self._active_key

    def is_loaded(self, model_key: str) -> bool:
        with self._lock:
            return model_key in self._engines

    def _build_engine(self, config: ModelConfig) -> BaseEngine:
        if self._engine_factory is not None:
            return self._engine_factory(config)
        if config.engine == EngineKind.LOCATE_ANYTHING.value:
            from annotation.engines.locate_engine import LocateEngine

            return LocateEngine(config, self._settings)
        if config.engine == EngineKind.SAM3.value:
            from annotation.engines.sam3_engine import Sam3Engine

            return Sam3Engine(config, self._settings)
        if config.engine != EngineKind.YOLO.value:
            raise ValueError(f"不支持的 engine: {config.engine}")
        if config.task == YoloTask.DETECT.value:
            return YoloDetectEngine(config)
        if config.task == YoloTask.SEGMENT.value:
            return YoloSegmentEngine(config)
        raise ValueError(f"不支持的 YOLO task: {config.task}")

    def _ensure_loaded(self, config: ModelConfig) -> BaseEngine:
        cached = self._engines.get(config.key)
        if cached is not None:
            self._active_key = config.key
            return cached
        if not config.path:
            raise HTTPException(
                status_code=503,
                detail=f"模型 {config.key} 尚未配置路径（models[].path 为空）",
            )
        engine = self._build_engine(config)
        engine.load()
        self._engines[config.key] = engine
        self._active_key = config.key
        return engine

    @contextmanager
    def use_engine(
        self,
        model: str | ModelConfig | None,
        *,
        timeout: float = GPU_INFER_TIMEOUT,
    ) -> Iterator[tuple[ModelConfig, BaseEngine]]:
        """在完整的线程/设备锁临界区内返回已加载引擎。"""
        config = model if isinstance(model, ModelConfig) else self.get_model_config(model)
        with self._lock:
            if not config.requires_cuda:
                yield config, self._ensure_loaded(config)
                return
            device_lock = GpuDeviceLock(config.device)
            if not device_lock.acquire(blocking=True, timeout=timeout):
                raise HTTPException(
                    status_code=503,
                    detail="GPU 正被训练或其它任务占用，请稍后重试",
                )
            try:
                yield config, self._ensure_loaded(config)
            finally:
                device_lock.release()

    def load(
        self,
        model_key: Optional[str] = None,
        *,
        timeout: float = GPU_LOAD_TIMEOUT,
    ) -> BaseEngine:
        with self.use_engine(model_key, timeout=timeout) as (_, engine):
            return engine

    def unload(self, model_key: str) -> UnloadResult:
        config = self.get_model_config(model_key)
        with self._lock:
            engine = self._engines.pop(config.key, None)
            if engine is None:
                return UnloadResult(False, tuple(self._engines), self._active_key)
            try:
                engine.unload()
            except Exception as exc:
                logger.warning("unload %s failed: %s", config.key, exc)
            finally:
                if self._active_key == config.key:
                    self._active_key = next(iter(self._engines), None)
            return UnloadResult(True, tuple(self._engines), self._active_key)

    def unload_all(self) -> int:
        with self._lock:
            engines = tuple(self._engines.values())
            self._engines.clear()
            self._active_key = None
            for engine in engines:
                try:
                    engine.unload()
                except Exception as exc:
                    logger.warning("unload all failed: %s", exc)
            return len(engines)


MODEL_RUNTIME = EngineRegistry(SETTINGS)

__all__ = [
    "BATCH_MAX_IMAGES",
    "EngineRegistry",
    "GPU_INFER_TIMEOUT",
    "GPU_LOAD_TIMEOUT",
    "MODEL_RUNTIME",
    "UnloadResult",
]
