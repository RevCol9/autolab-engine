"""Authenticated `.niii-model` containers and controlled YOLO loading."""

from toolkit.model_crypto.config import (
    ENV_MODEL_CRYPTO_CONFIG,
    ENV_MODEL_KEK,
    ENV_MODEL_KEK_ID,
    ENV_MODEL_KEYS_DIR,
    ModelCryptoConfig,
    default_model_crypto_config,
)
from toolkit.model_crypto.container_v1 import (
    FORMAT_VERSION,
    ModelContainer,
    read_model_container,
    write_model_container,
)
from toolkit.model_crypto.envelope import init_kek, load_kek, parse_kek_material
from toolkit.model_crypto.runtime import create_yolo_architecture, load_yolo_container

__all__ = [
    "ENV_MODEL_CRYPTO_CONFIG",
    "ENV_MODEL_KEK",
    "ENV_MODEL_KEK_ID",
    "ENV_MODEL_KEYS_DIR",
    "FORMAT_VERSION",
    "ModelContainer",
    "ModelCryptoConfig",
    "create_yolo_architecture",
    "default_model_crypto_config",
    "init_kek",
    "load_kek",
    "load_yolo_container",
    "parse_kek_material",
    "read_model_container",
    "write_model_container",
]
