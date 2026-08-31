"""推理运行时：引擎缓存、卸载与健康探测。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from annotation.engines.base import BaseEngine
from shared.gpu_lock import GpuDeviceLock, parse_device_index
from annotation.settings import ModelConfig, Settings

logger = logging.getLogger(__name__)


def model_entry(cfg: ModelConfig, *, loaded: bool, include_path: bool = False) -> Dict[str, Any]:
    """对外模型条目；默认不暴露权重绝对路径。"""
    entry = {
        "key": cfg.key,
        "name": cfg.name,
        "task": cfg.task,
        "engine": cfg.engine,
        "device": cfg.device,
        "conf": cfg.conf,
        "iou": cfg.iou,
        "imgsz": cfg.imgsz,
        "max_det": cfg.max_det,
        "path_configured": bool(cfg.path),
        "loaded": loaded,
    }
    if include_path:
        entry["path"] = cfg.path
    return entry


def probe_cuda() -> Dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        names: List[str] = []
        if available:
            for i in range(count):
                try:
                    names.append(torch.cuda.get_device_name(i))
                except Exception:
                    names.append(f"cuda:{i}")
        return {"available": available, "device_count": count, "device_names": names}
    except Exception as exc:
        return {"available": False, "device_count": 0, "device_names": [], "error": str(exc)}


def probe_default_weight(settings: Settings) -> Dict[str, Any]:
    key = settings.default_model
    for m in settings.models:
        if m.key != key:
            continue
        if not m.path:
            return {"model_key": key, "ok": False, "reason": "path_not_configured"}
        exists = Path(m.path).is_file()
        return {"model_key": key, "ok": exists, "reason": None if exists else "weight_missing"}
    return {"model_key": key, "ok": False, "reason": "model_not_in_config"}


def probe_gpu_locks(settings: Settings) -> List[Dict[str, Any]]:
    indices = {parse_device_index(m.device) for m in settings.models if m.device}
    indices.add(parse_device_index(settings.cuda_visible_devices.split(",")[0]))
    out: List[Dict[str, Any]] = []
    for idx in sorted(indices):
        lock = GpuDeviceLock(idx)
        out.append({"device": idx, "busy": lock.is_held_by_other()})
    return out


def unload_engine(
    model_key: str,
    engines: Dict[str, BaseEngine],
    active_key: Optional[str],
) -> Tuple[List[str], Optional[str]]:
    """卸载单模型；返回 (剩余 loaded 列表, 新 active_key)。"""
    engine = engines.pop(model_key, None)
    if engine is None:
        return list(engines.keys()), active_key
    try:
        engine.unload()
    except Exception as exc:
        logger.warning("unload %s: %s", model_key, exc)
    new_active = active_key
    if active_key == model_key:
        new_active = next(iter(engines), None)
    return list(engines.keys()), new_active


def unload_all_engines(
    engines: Dict[str, BaseEngine],
) -> None:
    keys = list(engines.keys())
    for key in keys:
        engine = engines.pop(key, None)
        if engine is None:
            continue
        try:
            engine.unload()
        except Exception as exc:
            logger.warning("unload all %s: %s", key, exc)
