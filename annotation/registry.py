"""引擎注册表：进程内缓存、跨进程 GPU 锁、模型配置与加载。"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

from fastapi import HTTPException

from annotation.bootstrap import SETTINGS
from annotation.engines.base import BaseEngine
from annotation.engines.yolo import YoloDetectEngine, YoloSegmentEngine
from shared.gpu_lock import GpuDeviceLock
from annotation.settings import ModelConfig

VLM_ENGINES = frozenset({"locateanything", "sam3"})
_engines: Dict[str, BaseEngine] = {}
_active_key: Optional[str] = None
_gpu_lock = threading.RLock()

GPU_INFER_TIMEOUT = float(os.environ.get("NIII_GPU_INFER_TIMEOUT", "120"))
GPU_LOAD_TIMEOUT = float(os.environ.get("NIII_GPU_LOAD_TIMEOUT", "180"))
BATCH_MAX_IMAGES = 32


def is_vlm_engine(cfg: ModelConfig) -> bool:
    return (cfg.engine or "yolo").lower() in VLM_ENGINES


def set_active_key(key: Optional[str]) -> None:
    global _active_key
    _active_key = key


def active_key() -> Optional[str]:
    return _active_key


@contextmanager
def cross_gpu_session(cfg: ModelConfig, *, timeout: float = GPU_INFER_TIMEOUT) -> Iterator[None]:
    """与训练进程共享的 GPU 文件锁；超时返回 503。"""
    lock = GpuDeviceLock(cfg.device)
    if not lock.acquire(blocking=True, timeout=timeout):
        raise HTTPException(
            status_code=503,
            detail="GPU 正被训练或其它任务占用，请稍后重试",
        )
    try:
        yield
    finally:
        lock.release()


def get_model_config(model_key: Optional[str]) -> ModelConfig:
    key = (model_key or SETTINGS.default_model or "").strip()
    if not SETTINGS.models:
        raise HTTPException(status_code=500, detail="config.yaml 未配置 models")
    if not key:
        return SETTINGS.models[0]
    for m in SETTINGS.models:
        if m.key == key:
            return m
    raise HTTPException(status_code=400, detail=f"未知 model_key: {key}")


def build_engine(config: ModelConfig) -> BaseEngine:
    eng = (config.engine or "yolo").lower()
    if eng == "locateanything":
        from annotation.engines.locate_engine import LocateEngine

        return LocateEngine(config, SETTINGS)
    if eng == "sam3":
        from annotation.engines.sam3_engine import Sam3Engine

        return Sam3Engine(config, SETTINGS)
    task = (config.task or "detect").lower()
    if task == "detect":
        return YoloDetectEngine(config)
    if task == "segment":
        return YoloSegmentEngine(config)
    raise HTTPException(status_code=400, detail=f"不支持的 task: {task}")


def ensure_model_engine(model_key: Optional[str] = None) -> BaseEngine:
    """将模型加载到进程内缓存；调用方须已持有进程内锁与跨进程 GPU 锁。"""
    global _active_key
    cfg = get_model_config(model_key)
    if cfg.key in _engines:
        _active_key = cfg.key
        return _engines[cfg.key]
    if not cfg.path:
        raise HTTPException(
            status_code=503,
            detail=f"模型 {cfg.key} 尚未配置路径（models[].path 为空）",
        )
    engine = build_engine(cfg)
    engine.load()
    _engines[cfg.key] = engine
    _active_key = cfg.key
    return engine
