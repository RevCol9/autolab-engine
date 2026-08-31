"""从 config/training/ 加载训练服务与 Ultralytics 默认超参。

加载顺序：base.yaml + {task}.yaml（默认 task=detection），后者覆盖同名顶层键。
环境变量 TRAINING_TASK 可选 detection | segmentation。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from shared.config_yaml import load_merged_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "config"
TRAINING_CONFIG_DIR = CONFIG_ROOT / "training"

# 不传给 Ultralytics model.train() 的顶层键
_TRAINING_META_KEYS = frozenset({"server", "gpu", "device"})

_VALID_TRAINING_TASKS = frozenset({"detection", "segmentation"})


def training_task() -> str:
    task = os.getenv("TRAINING_TASK", "detection").strip().lower() or "detection"
    if task not in _VALID_TRAINING_TASKS:
        raise ValueError(f"TRAINING_TASK 无效: {task!r}，可选 {sorted(_VALID_TRAINING_TASKS)}")
    return task


def _training_base_candidates() -> tuple[Path, Path]:
    return (
        TRAINING_CONFIG_DIR / "base.yaml",
        TRAINING_CONFIG_DIR / "base.example.yaml",
    )


def _training_task_candidates(task: str) -> tuple[Path, Path]:
    return (
        TRAINING_CONFIG_DIR / f"{task}.yaml",
        TRAINING_CONFIG_DIR / f"{task}.example.yaml",
    )


def _resolve_existing(candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"配置文件不存在: {candidates[0]}")


def resolve_training_config_paths() -> tuple[Path, Path]:
    """返回 (base.yaml, task.yaml) 路径。"""
    if explicit := os.getenv("TRAINING_CONFIG_PATH", "").strip():
        task_path = Path(explicit).expanduser().resolve()
        if not task_path.is_file():
            raise FileNotFoundError(f"TRAINING_CONFIG_PATH 指向的文件不存在: {task_path}")
        base_path = _resolve_existing(_training_base_candidates())
        return base_path, task_path
    task = training_task()
    return (
        _resolve_existing(_training_base_candidates()),
        _resolve_existing(_training_task_candidates(task)),
    )


@lru_cache(maxsize=1)
def load_training_config() -> Dict[str, Any]:
    base_path, task_path = resolve_training_config_paths()
    return load_merged_yaml([base_path, task_path])


def training_config_path() -> str:
    base_path, task_path = resolve_training_config_paths()
    return f"{base_path} + {task_path}"


def training_ultralytics_defaults() -> Dict[str, Any]:
    """Ultralytics 默认超参（排除 server / device / gpu）。"""
    return {
        key: value
        for key, value in load_training_config().items()
        if key not in _TRAINING_META_KEYS
    }


def default_training_device() -> str:
    data = load_training_config()
    if "device" in data and data["device"] is not None:
        return str(data["device"]).strip() or "0"
    gpu = data.get("gpu")
    if isinstance(gpu, dict) and gpu.get("device") is not None:
        return str(gpu["device"]).strip() or "0"
    return "0"


def resolve_training_device(explicit: Optional[str] = None) -> str:
    if explicit is not None and str(explicit).strip() != "":
        return str(explicit).strip()
    return default_training_device()


def training_server_host() -> str:
    server = load_training_config().get("server") or {}
    if isinstance(server, dict):
        return str(server.get("host") or "0.0.0.0")
    return "0.0.0.0"


def training_server_port() -> int:
    server = load_training_config().get("server") or {}
    if isinstance(server, dict):
        return int(server.get("port") or 21011)
    return 21011


def training_log_level() -> str:
    server = load_training_config().get("server") or {}
    if isinstance(server, dict):
        return str(server.get("log_level") or "info").lower()
    return "info"


def apply_training_runtime_env() -> None:
    """训练进程 GPU 可见性（config/training/base.yaml → gpu.cuda_visible_devices）。"""
    gpu = load_training_config().get("gpu")
    if not isinstance(gpu, dict):
        return
    visible = gpu.get("cuda_visible_devices")
    if visible is not None and str(visible).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible)
        os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", str(visible))
