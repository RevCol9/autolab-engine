"""Encrypted YOLO training snapshots; no PyTorch checkpoint is ever written."""

from __future__ import annotations

from pathlib import Path

from ultralytics.utils.torch_utils import unwrap_model

from toolkit.model_crypto.container_v1 import write_model_container
from toolkit.model_crypto.envelope import load_kek
from toolkit.model_crypto.errors import ModelCryptoError
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
    try:
        container_task = normalize_container_task(task)
    except ValueError as exc:
        raise ModelCryptoError(ModelCryptoError.TASK_MISMATCH, str(exc)) from exc
    destination = Path(destination)
    network = unwrap_model(model)
    model_yaml = getattr(network, "yaml", None)
    model_names = getattr(network, "names", None)
    if not isinstance(model_yaml, dict) or not isinstance(model_names, (dict, list, tuple)):
        raise ModelCryptoError(
            ModelCryptoError.ARCH_UNSUPPORTED,
            "训练模型缺少受控 YOLO yaml/names 元数据",
        )
    if not isinstance(model_names, dict):
        model_names = dict(enumerate(model_names))

    try:
        kek, key_id = load_kek(key_dir=key_dir)
    except FileNotFoundError as exc:
        raise ModelCryptoError(ModelCryptoError.KEY_NOT_FOUND, str(exc)) from exc
    except PermissionError as exc:
        raise ModelCryptoError(ModelCryptoError.KEY_INSECURE, str(exc)) from exc
    except ValueError as exc:
        raise ModelCryptoError(ModelCryptoError.KEY_INVALID, str(exc)) from exc
    try:
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
    except ModelCryptoError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ModelCryptoError(ModelCryptoError.WRITE_FAILED, str(exc)) from exc


__all__ = ["save_training_model"]
