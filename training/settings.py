"""从 config/training/ 加载训练服务与 Ultralytics 默认超参。

加载顺序：base.yaml + {task}.yaml，后者覆盖同名顶层键。
服务级设置（server/device）仅读 base.yaml；任务超参按 train_task 分别加载。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from shared.config_yaml import load_merged_yaml, load_yaml_file
from shared.device import parse_device_index

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "config"
TRAINING_CONFIG_DIR = CONFIG_ROOT / "training"

_TRAINING_META_KEYS = frozenset({"server", "gpu", "device", "train_task"})

_VALID_TRAINING_TASKS = frozenset({"detection", "segmentation"})


def _validate_task(task: str) -> str:
    key = (task or "detection").strip().lower()
    if key not in _VALID_TRAINING_TASKS:
        raise ValueError(f"未知训练任务: {task!r}，可选 {sorted(_VALID_TRAINING_TASKS)}")
    return key


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


@lru_cache(maxsize=1)
def load_training_base_config() -> Dict[str, Any]:
    return load_yaml_file(_resolve_existing(_training_base_candidates()))


@lru_cache(maxsize=8)
def load_training_config_for_task(task: str) -> Dict[str, Any]:
    task_key = _validate_task(task)
    task_env = f"TRAINING_{task_key.upper()}_CONFIG_PATH"
    explicit = os.getenv(task_env, "").strip()
    legacy_explicit = os.getenv("TRAINING_CONFIG_PATH", "").strip()
    if explicit or legacy_explicit:
        explicit = explicit or legacy_explicit
        task_path = Path(explicit).expanduser().resolve()
        if not task_path.is_file():
            raise FileNotFoundError(f"训练任务配置文件不存在: {task_path}")
        override = load_yaml_file(task_path)
        declared_task = str(override.get("train_task") or "").strip().lower()
        if legacy_explicit and not os.getenv(task_env, "").strip():
            if not declared_task:
                raise ValueError(
                    "TRAINING_CONFIG_PATH 指向的配置必须声明 train_task；"
                    f"也可改用 {task_env} 明确绑定路由"
                )
            if declared_task != task_key:
                raise ValueError(
                    f"TRAINING_CONFIG_PATH 的 train_task={declared_task!r} "
                    f"与请求任务 {task_key!r} 不一致"
                )
        elif declared_task and declared_task != task_key:
            raise ValueError(
                f"{task_env} 的 train_task={declared_task!r} 与请求任务不一致"
            )
        return load_merged_yaml([_resolve_existing(_training_base_candidates()), task_path])
    return load_merged_yaml(
        [
            _resolve_existing(_training_base_candidates()),
            _resolve_existing(_training_task_candidates(task_key)),
        ]
    )


def training_config_path(task: str = "detection") -> str:
    base = _resolve_existing(_training_base_candidates())
    task_path = _resolve_existing(_training_task_candidates(_validate_task(task)))
    return f"{base} + {task_path}"


def training_ultralytics_defaults(task: str = "detection") -> Dict[str, Any]:
    """指定任务的 Ultralytics 默认超参（排除 server / device / gpu）。"""
    return {
        key: value
        for key, value in load_training_config_for_task(task).items()
        if key not in _TRAINING_META_KEYS
    }


def default_training_device() -> str:
    data = load_training_base_config()
    if "device" in data and data["device"] is not None:
        return str(data["device"]).strip() or "0"
    gpu = data.get("gpu")
    if isinstance(gpu, dict) and gpu.get("device") is not None:
        return str(gpu["device"]).strip() or "0"
    return "0"


def resolve_training_device(explicit: Optional[str] = None) -> str:
    value = (
        str(explicit).strip()
        if explicit is not None and str(explicit).strip() != ""
        else default_training_device()
    )
    parse_device_index(value)
    return value


def training_server_host() -> str:
    server = load_training_base_config().get("server") or {}
    if isinstance(server, dict):
        return str(server.get("host") or "0.0.0.0")
    return "0.0.0.0"


def training_server_port() -> int:
    server = load_training_base_config().get("server") or {}
    if isinstance(server, dict):
        return int(server.get("port") or 21011)
    return 21011


def training_log_level() -> str:
    server = load_training_base_config().get("server") or {}
    if isinstance(server, dict):
        return str(server.get("log_level") or "info").lower()
    return "info"


def apply_training_runtime_env() -> None:
    """训练进程 GPU 可见性（config/training/base.yaml → gpu.cuda_visible_devices）。"""
    gpu = load_training_base_config().get("gpu")
    if not isinstance(gpu, dict):
        return
    visible = gpu.get("cuda_visible_devices")
    if visible is not None and str(visible).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible)
        os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", str(visible))
