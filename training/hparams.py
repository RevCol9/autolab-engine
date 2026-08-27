"""训练超参：API JSON → Ultralytics YAML 配置。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from training.data_yaml import prepare_data_yaml_for_job
from training.paths import baseline_pt_from_last_train, train_save_dir

TRAIN_DEFAULTS_PATH = Path(__file__).resolve().parent / "config" / "train_defaults.yaml"

# Java/API 字段 → Ultralytics 字段
API_FIELD_ALIASES: Dict[str, str] = {
    "batch_size": "batch",
    "image_size": "imgsz",
}

# 仅用于任务调度，不传入 model.train()
JOB_META_KEYS = frozenset(
    {
        "action",
        "projectId",
        "taskId",
        "trainNum",
        "is_continue",
        "last_train",
    }
)

# 写入 train_config.yaml、但不传给 Ultralytics
SCRIPT_META_KEYS = frozenset({"save_dir"})

# Ultralytics model.train() 支持的常用键（其余 unknown 键也会透传）
TRAIN_KWARG_KEYS = frozenset(
    {
        "epochs",
        "batch",
        "imgsz",
        "device",
        "project",
        "name",
        "exist_ok",
        "save",
        "amp",
        "verbose",
        "patience",
        "workers",
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "close_mosaic",
        "cos_lr",
        "rect",
        "resume",
        "fraction",
        "freeze",
        "plots",
        "val",
        "seed",
        "deterministic",
        "single_cls",
        "cache",
    }
)


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def resolve_model_path(param: Mapping[str, Any]) -> str:
    """解析起始权重：续训用 last_train 基线，否则用 model。"""
    if _truthy(param.get("is_continue")):
        last_train = param.get("last_train")
        if not last_train:
            raise ValueError("is_continue=true 时必须提供 last_train")
        return baseline_pt_from_last_train(str(last_train))
    model = param.get("model") or "yolov8n.pt"
    text = str(model).strip()
    if "/" not in text and "\\" not in text:
        return text.lower()
    return text


def load_train_defaults() -> Dict[str, Any]:
    if not TRAIN_DEFAULTS_PATH.is_file():
        raise FileNotFoundError(f"训练默认配置不存在: {TRAIN_DEFAULTS_PATH}")
    with TRAIN_DEFAULTS_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"训练默认配置格式错误: {TRAIN_DEFAULTS_PATH}")
    return data


def normalize_api_param(param: Mapping[str, Any]) -> Dict[str, Any]:
    """将 HTTP/API 扁平 JSON 转为 Ultralytics 字段名。"""
    out: Dict[str, Any] = {}
    for key, value in param.items():
        if key in JOB_META_KEYS:
            continue
        target = API_FIELD_ALIASES.get(key, key)
        out[target] = value
    return out


def build_job_train_config(
    param: Mapping[str, Any],
    *,
    device: str = "0",
) -> Dict[str, Any]:
    """合并默认配置、API 参数与任务路径，生成 train_config.yaml 内容。"""
    project_id = str(param["projectId"])
    task_id = str(param["taskId"])
    train_num = str(param["trainNum"])

    data_yaml = prepare_data_yaml_for_job(dict(param))
    save_path = train_save_dir(project_id, task_id, train_num)
    save_path.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = deepcopy(load_train_defaults())
    config.update(normalize_api_param(param))
    config.update(
        {
            "model": resolve_model_path(param),
            "data": str(data_yaml),
            "project": str(save_path.parent),
            "name": save_path.name,
            "save_dir": str(save_path),
            "device": str(device or config.get("device") or "0"),
        }
    )
    return config


def write_train_config(config: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, allow_unicode=True, sort_keys=False)
    return path


def write_job_train_config(param: Mapping[str, Any], *, device: str = "0") -> Path:
    config = build_job_train_config(param, device=device)
    save_dir = Path(str(config["save_dir"]))
    return write_train_config(config, save_dir / "train_config.yaml")


def load_train_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"train_config.yaml 不存在: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"train_config.yaml 格式错误: {cfg_path}")
    return data


def split_train_config(config: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """拆分为脚本元数据与 Ultralytics train kwargs。"""
    meta: Dict[str, Any] = {}
    train_kwargs: Dict[str, Any] = {}
    for key, value in config.items():
        if key in SCRIPT_META_KEYS:
            meta[key] = value
        elif key in {"model", "data"}:
            continue
        elif key in JOB_META_KEYS:
            continue
        train_kwargs[key] = value
    return meta, train_kwargs
