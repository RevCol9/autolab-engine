"""Encrypted YOLO training snapshots; no PyTorch checkpoint is ever written."""

from __future__ import annotations

from pathlib import Path

from ultralytics.utils.torch_utils import unwrap_model

from toolkit.model_crypto.container_v1 import write_model_container
from toolkit.model_crypto.envelope import load_kek
from toolkit.model_crypto.yolo_contract import (
    normalize_container_task,
)


def save_training_model(
    model,
    destination: str | Path,
    *,
    task: str,
    key_dir: str | Path | None = None,
) -> Path:
    """Write or atomically replace an encrypted inference-weight snapshot."""
    container_task = normalize_container_task(task)
    destination = Path(destination)
    network = unwrap_model(model)
    model_yaml = getattr(network, "yaml", None)
    model_names = getattr(network, "names", None)
    if not isinstance(model_yaml, dict) or not isinstance(model_names, (dict, list, tuple)):
        raise ValueError("训练模型缺少受控 YOLO yaml/names 元数据")
    if not isinstance(model_names, dict):
        model_names = dict(enumerate(model_names))

    kek, key_id = load_kek(key_dir=key_dir)
    return write_model_container(
        destination,
        state_dict=network.state_dict(),
        model_yaml=model_yaml,
        model_names=model_names,
        task=container_task,
        kek=kek,
        key_id=key_id,
        replace_existing=destination.exists(),
    )


__all__ = ["save_training_model"]
