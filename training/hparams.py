"""训练超参：config.yaml + API JSON → Ultralytics train_config.yaml。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from training.settings import resolve_training_device, training_ultralytics_defaults
from training.data_yaml import prepare_data_yaml_for_job
from training.paths import baseline_pt_from_last_train, train_save_dir

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
        "device",
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
    """从 training/config.yaml 读取 Ultralytics 默认超参。"""
    defaults = training_ultralytics_defaults()
    if not defaults:
        raise ValueError(
            "training/config.yaml 缺少 Ultralytics 超参；请参考 training/config.example.yaml"
        )
    return deepcopy(defaults)


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
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """合并 config 默认、API 参数与任务路径，生成 train_config.yaml 内容。"""
    project_id = str(param["projectId"])
    task_id = str(param["taskId"])
    train_num = str(param["trainNum"])

    data_yaml = prepare_data_yaml_for_job(dict(param))
    save_path = train_save_dir(project_id, task_id, train_num)
    save_path.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = load_train_defaults()
    config.update(normalize_api_param(param))
    config.update(
        {
            "model": resolve_model_path(param),
            "data": str(data_yaml),
            "project": str(save_path.parent),
            "name": save_path.name,
            "save_dir": str(save_path),
            "device": resolve_training_device(device),
        }
    )
    return config


def write_train_config(config: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, allow_unicode=True, sort_keys=False)
    return path


def write_job_train_config(param: Mapping[str, Any], *, device: Optional[str] = None) -> Path:
    config = build_job_train_config(param, device=device)
    save_dir = Path(str(config["save_dir"]))
    return write_train_config(config, save_dir / "train_config.yaml")


def load_job_config(path: str | Path) -> Dict[str, Any]:
    """读取 train_config.yaml（子进程 closed_loop_train 使用）。"""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"train_config.yaml 不存在: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"train_config.yaml 格式错误: {cfg_path}")
    required = ("model", "data", "save_dir")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError(f"train_config.yaml 缺少字段: {missing}")
    return data
