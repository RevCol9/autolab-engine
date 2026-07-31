"""从 config.yaml 加载运行时配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_CANDIDATES = (
    ROOT_DIR / "config.yaml",
    ROOT_DIR / "config.example.yaml",
)


def _dig(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass
class ModelConfig:
    key: str
    name: str
    task: str
    path: str
    device: str = "cuda:0"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 1280
    max_det: int = 300


@dataclass
class Settings:
    config_path: str
    host: str = "0.0.0.0"
    port: int = 21010
    log_level: str = "INFO"
    default_model: str = ""
    models: List[ModelConfig] = field(default_factory=list)
    coord_space: str = "pixel"


def resolve_config_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        return path
    env_path = os.getenv("CONFIG_PATH", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CONFIG_PATH 指向的文件不存在: {path}")
        return path
    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"未找到配置文件: {ROOT_DIR / 'config.yaml'} 或 {ROOT_DIR / 'config.example.yaml'}"
    )


def parse_models(raw: Any) -> List[ModelConfig]:
    out: List[ModelConfig] = []
    seen = set()
    if not isinstance(raw, list):
        return out
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or f"model-{i}").strip()
        if not key or key in seen:
            continue
        task = str(item.get("task") or "detect").strip().lower()
        out.append(
            ModelConfig(
                key=key,
                name=str(item.get("name") or key).strip(),
                task=task,
                path=str(item.get("path") or "").strip(),
                device=str(item.get("device") or "cuda:0").strip(),
                conf=float(item.get("conf", 0.25)),
                iou=float(item.get("iou", 0.45)),
                imgsz=int(item.get("imgsz", 1280)),
                max_det=int(item.get("max_det", 300)),
            )
        )
        seen.add(key)
    return out


def load_settings(config_path: Optional[str] = None) -> Settings:
    path = resolve_config_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置根节点必须为 mapping: {path}")

    models = parse_models(_dig(data, "models", default=[]))
    default_model = str(_dig(data, "default_model", default="") or "").strip()
    if not default_model and models:
        default_model = models[0].key
    if default_model and models and not any(m.key == default_model for m in models):
        raise ValueError(f"default_model={default_model!r} 不在 models 中: {[m.key for m in models]}")

    return Settings(
        config_path=str(path),
        host=str(_dig(data, "server", "host", default="0.0.0.0")),
        port=int(_dig(data, "server", "port", default=21010)),
        log_level=str(_dig(data, "server", "log_level", default="INFO")).upper(),
        default_model=default_model,
        models=models,
        coord_space=str(_dig(data, "inference", "coord_space", default="pixel")).lower(),
    )
