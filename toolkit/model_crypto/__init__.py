"""Authenticated `.niii-model` containers and controlled YOLO loading.

Exports are resolved lazily so key-management commands do not import Torch or
Ultralytics.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ENV_MODEL_CRYPTO_CONFIG": ("toolkit.model_crypto.config", "ENV_MODEL_CRYPTO_CONFIG"),
    "ENV_MODEL_KEK": ("toolkit.model_crypto.config", "ENV_MODEL_KEK"),
    "ENV_MODEL_KEK_ID": ("toolkit.model_crypto.config", "ENV_MODEL_KEK_ID"),
    "ENV_MODEL_KEYS_DIR": ("toolkit.model_crypto.config", "ENV_MODEL_KEYS_DIR"),
    "FORMAT_VERSION": ("toolkit.model_crypto.container_v1", "FORMAT_VERSION"),
    "ModelContainer": ("toolkit.model_crypto.container_v1", "ModelContainer"),
    "ModelCryptoConfig": ("toolkit.model_crypto.config", "ModelCryptoConfig"),
    "create_yolo_architecture": (
        "toolkit.model_crypto.runtime",
        "create_yolo_architecture",
    ),
    "default_model_crypto_config": (
        "toolkit.model_crypto.config",
        "default_model_crypto_config",
    ),
    "init_kek": ("toolkit.model_crypto.envelope", "init_kek"),
    "load_kek": ("toolkit.model_crypto.envelope", "load_kek"),
    "load_yolo_container": ("toolkit.model_crypto.runtime", "load_yolo_container"),
    "parse_kek_material": ("toolkit.model_crypto.envelope", "parse_kek_material"),
    "read_model_container": (
        "toolkit.model_crypto.container_v1",
        "read_model_container",
    ),
    "write_model_container": (
        "toolkit.model_crypto.container_v1",
        "write_model_container",
    ),
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
