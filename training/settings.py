"""从 training/config.yaml 加载训练服务与 Ultralytics 默认超参。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

TRAINING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_ROOT.parent

DEFAULT_TRAINING_CONFIG_CANDIDATES = (
    TRAINING_ROOT / "config.yaml",
    TRAINING_ROOT / "config.example.yaml",
)

# 不传给 Ultralytics model.train() 的顶层键
_TRAINING_META_KEYS = frozenset({"server", "gpu", "device"})


def resolve_training_config_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"训练配置文件不存在: {path}")
        return path
    env_path = os.getenv("TRAINING_CONFIG_PATH", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"TRAINING_CONFIG_PATH 指向的文件不存在: {path}")
        return path
    for candidate in DEFAULT_TRAINING_CONFIG_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"未找到训练配置: {TRAINING_ROOT / 'config.yaml'} 或 config.example.yaml"
    )


@lru_cache(maxsize=1)
def load_training_config() -> Dict[str, Any]:
    path = resolve_training_config_path()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"训练配置根节点必须为 mapping: {path}")
    return data


def training_config_path() -> str:
    return str(resolve_training_config_path())


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
    """训练进程可选 GPU 可见性（training/config.yaml → gpu.cuda_visible_devices）。"""
    gpu = load_training_config().get("gpu")
    if not isinstance(gpu, dict):
        return
    visible = gpu.get("cuda_visible_devices")
    if visible is not None and str(visible).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible)
        os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", str(visible))
