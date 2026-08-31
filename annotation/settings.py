"""从 config.yaml 加载运行时配置。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

ANNOTATION_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_CANDIDATES = (
    ANNOTATION_DIR / "config.yaml",
    ANNOTATION_DIR / "config.example.yaml",
)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _dig(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_max_memory(raw: Optional[Any]) -> Optional[Dict[Union[int, str], str]]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, dict):
        out: Dict[Union[int, str], str] = {}
        for key, value in raw.items():
            out[int(key) if str(key).isdigit() else key] = str(value).strip()
        return out or None
    text = str(raw).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(\.\d+)?\s*(GiB|GB|MiB|MB)", text, flags=re.IGNORECASE):
        return {"_all": re.sub(r"\s+", "", text)}
    mapping: Dict[Union[int, str], str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        mapping[int(key.strip()) if key.strip().isdigit() else key.strip()] = value.strip()
    return mapping or None


def expand_max_memory(
    raw: Optional[Dict[Union[int, str], str]],
    visible_gpu_count: int,
) -> Optional[Dict[Union[int, str], str]]:
    if not raw:
        return None
    if "_all" in raw:
        size = raw["_all"]
        n = max(int(visible_gpu_count), 1)
        return {i: size for i in range(n)}
    return raw


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
    engine: str = "yolo"  # yolo | locateanything | sam3


@dataclass
class Settings:
    config_path: str
    host: str = "0.0.0.0"
    port: int = 21010
    log_level: str = "INFO"
    default_model: str = ""
    models: List[ModelConfig] = field(default_factory=list)
    coord_space: str = "pixel"

    cuda_visible_devices: str = "0"
    device_map: Optional[str] = None
    max_memory_raw: Optional[Any] = None
    max_memory: Optional[Dict[Union[int, str], str]] = None

    locate_dtype: str = "float16"
    locate_local_files_only: bool = True
    in_token_limit: int = 4096
    max_image_side: int = 1280

    generation_mode: str = "hybrid"
    max_new_tokens: int = 512
    temperature: float = 0.0

    postprocess_enable: bool = True
    postprocess_nms_iou: float = 0.5
    postprocess_runaway_nms_iou: float = 0.35
    postprocess_min_box_area_ratio: float = 0.0003
    postprocess_max_box_area_ratio: float = 0.85
    postprocess_max_boxes: int = 32
    bilingual_prompt: bool = True


def resolve_config_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        return path
    env_path = (
        os.getenv("ANNOTATION_CONFIG_PATH", "").strip()
        or os.getenv("CONFIG_PATH", "").strip()
    )
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CONFIG_PATH 指向的文件不存在: {path}")
        return path
    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"未找到配置文件: {ANNOTATION_DIR / 'config.yaml'} 或 config.example.yaml"
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
        engine = str(item.get("engine") or "yolo").strip().lower()
        if engine == "yolo" and task in {"vlm", "locate", "sam"}:
            pass
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
                engine=engine,
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

    device_map_val = str(_dig(data, "gpu", "device_map", default="") or "").strip()
    if device_map_val.lower() in {"", "null", "none", "~"}:
        device_map_val = ""

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
        cuda_visible_devices=str(_dig(data, "gpu", "cuda_visible_devices", default="0")),
        device_map=device_map_val or None,
        max_memory_raw=_dig(data, "gpu", "max_memory", default=None),
        max_memory=parse_max_memory(_dig(data, "gpu", "max_memory", default=None)),
        locate_dtype=str(_dig(data, "locate", "dtype", default="float16")),
        locate_local_files_only=_as_bool(_dig(data, "locate", "local_files_only", default=True), True),
        in_token_limit=int(_dig(data, "locate", "in_token_limit", default=4096)),
        max_image_side=int(_dig(data, "locate", "max_image_side", default=1280)),
        generation_mode=str(_dig(data, "locate", "generation_mode", default="hybrid")),
        max_new_tokens=int(_dig(data, "locate", "max_new_tokens", default=512)),
        temperature=float(_dig(data, "locate", "temperature", default=0.0)),
        postprocess_enable=_as_bool(_dig(data, "postprocess", "enable", default=True), True),
        postprocess_nms_iou=float(_dig(data, "postprocess", "nms_iou", default=0.5)),
        postprocess_runaway_nms_iou=float(_dig(data, "postprocess", "runaway_nms_iou", default=0.35)),
        postprocess_min_box_area_ratio=float(
            _dig(data, "postprocess", "min_box_area_ratio", default=0.0003)
        ),
        postprocess_max_box_area_ratio=float(
            _dig(data, "postprocess", "max_box_area_ratio", default=0.85)
        ),
        postprocess_max_boxes=int(_dig(data, "postprocess", "max_boxes", default=32)),
        bilingual_prompt=_as_bool(_dig(data, "postprocess", "bilingual_prompt", default=True), True),
    )


def apply_runtime_env(settings: Settings) -> None:
    if settings.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.cuda_visible_devices)
        os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", str(settings.cuda_visible_devices))
    os.environ["LOG_LEVEL"] = settings.log_level
