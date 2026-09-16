"""训练超参：config/training/*.yaml + API JSON → 任务级 train_config.yaml。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from training.backends import get_backend
from training.dataset import TrainingDataset
from training.paths import (
    baseline_pt_from_last_train,
    resolve_pretrained_model_path,
    train_save_dir,
)
from training.run_artifacts import ensure_input_survives_reset
from training.settings import resolve_training_device, training_ultralytics_defaults

API_FIELD_ALIASES: Dict[str, str] = {
    "batch_size": "batch",
    "image_size": "imgsz",
}

JOB_META_KEYS = frozenset(
    {
        "action",
        "projectId",
        "taskId",
        "trainNum",
        "is_continue",
        "last_train",
        "pretrained_model_path",
        "device",
        "train_task",
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


def resolve_model_path(
    param: Mapping[str, Any],
    *,
    task: str,
    defaults: Optional[Mapping[str, Any]] = None,
) -> str:
    pretrained_model_path = param.get("pretrained_model_path")
    if _truthy(param.get("is_continue")):
        if pretrained_model_path is not None:
            raise ValueError(
                "is_continue=true 与 pretrained_model_path 不能同时使用"
            )
        last_train = param.get("last_train")
        if not last_train:
            raise ValueError("is_continue=true 时必须提供 last_train")
        return baseline_pt_from_last_train(str(last_train))
    if pretrained_model_path is not None:
        if param.get("model"):
            raise ValueError("model 与 pretrained_model_path 不能同时使用")
        return resolve_pretrained_model_path(pretrained_model_path)
    backend = get_backend(task)
    model = param.get("model") or (defaults or {}).get("model") or backend.default_model
    text = str(model).strip()
    if "/" not in text and "\\" not in text:
        return text.lower()
    return resolve_pretrained_model_path(text)


def load_training_defaults(task: str) -> Dict[str, Any]:
    defaults = training_ultralytics_defaults(task)
    if not defaults:
        raise ValueError(
            f"config/training/{task}.yaml 缺少 Ultralytics 超参；"
            f"请参考 config/training/{task}.example.yaml"
        )
    return deepcopy(defaults)


def normalize_api_param(param: Mapping[str, Any]) -> Dict[str, Any]:
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
    task: str,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    project_id = str(param["projectId"])
    task_id = str(param["taskId"])
    train_num = str(param["trainNum"])

    save_path = train_save_dir(project_id, task_id, train_num)
    config: Dict[str, Any] = load_training_defaults(task)
    config.update(normalize_api_param(param))
    model_path = resolve_model_path(param, task=task, defaults=config)
    explicit_model = str(param.get("model") or "")
    if param.get("pretrained_model_path") or any(
        separator in explicit_model for separator in ("/", "\\")
    ):
        ensure_input_survives_reset(model_path, save_path)

    data_yaml = TrainingDataset.from_job(param).prepare(task=task)
    save_path.mkdir(parents=True, exist_ok=True)
    config.update(
        {
            "train_task": task,
            "model": model_path,
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


def write_job_train_config(
    param: Mapping[str, Any],
    *,
    task: str,
    device: Optional[str] = None,
) -> Path:
    config = build_job_train_config(param, task=task, device=device)
    save_dir = Path(str(config["save_dir"]))
    return write_train_config(config, save_dir / "train_config.yaml")


def load_job_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"train_config.yaml 不存在: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"train_config.yaml 格式错误: {cfg_path}")
    required = ("model", "data", "save_dir", "train_task")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError(f"train_config.yaml 缺少字段: {missing}")
    return data
