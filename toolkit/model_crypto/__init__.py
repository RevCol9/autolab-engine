"""Authenticated `.niii-model` containers and controlled YOLO loading.

Exports are resolved lazily so key-management commands do not import Torch or
Ultralytics.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ModelCryptoError": ("toolkit.model_crypto.errors", "ModelCryptoError"),
    "create_yolo_architecture": (
        "toolkit.model_crypto.runtime",
        "create_yolo_architecture",
    ),
    "load_yolo_container": ("toolkit.model_crypto.runtime", "load_yolo_container"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
