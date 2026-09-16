"""推理运行时：引擎缓存、卸载与健康探测。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from annotation.settings import ModelConfig, Settings
from shared.gpu_lock import GpuDeviceLock


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
        path = Path(m.path)
        if m.engine == "locateanything" and not settings.locate_local_files_only:
            exists = True
            kind = "model_id_or_directory"
        else:
            kind = m.path_kind
            exists = path.is_file() if kind == "file" else path.is_dir()
        return {
            "model_key": key,
            "ok": exists,
            "kind": kind,
            "reason": None if exists else "weight_missing",
        }
    return {"model_key": key, "ok": False, "reason": "model_not_in_config"}


def probe_gpu_locks(settings: Settings) -> List[Dict[str, Any]]:
    devices = {model.device for model in settings.models if model.requires_cuda}
    out: List[Dict[str, Any]] = []
    for device in sorted(devices):
        lock = GpuDeviceLock(device)
        out.append(
            {
                "device": device,
                "physical_device": lock.device_key,
                "busy": lock.is_held_by_other(),
            }
        )
    return out
